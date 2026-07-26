from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tracking2.cnn_checkpoints import sha256


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require_measured(path: Path, artifact: dict, experiment: str) -> None:
    if artifact.get("status") != "MEASURED":
        raise RuntimeError(f"{path} is not measured")
    if artifact.get("experiment") != experiment:
        raise RuntimeError(
            f"{path} has experiment={artifact.get('experiment')!r}, "
            f"expected {experiment!r}"
        )


def require_exact_keys(
    path: Path,
    rows: list[dict],
    fields: tuple[str, ...],
    expected: set[tuple],
    label: str,
) -> None:
    observed = [tuple(row.get(field) for field in fields) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"{path} has duplicate {label} rows")
    if set(observed) != expected:
        missing = expected - set(observed)
        extra = set(observed) - expected
        raise RuntimeError(
            f"{path} has an incomplete {label} grid; "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def require_distinct_checkpoint_hashes(
    path: Path, hashes: dict[int, str]
) -> None:
    if len(set(hashes.values())) != len(hashes):
        raise RuntimeError(
            f"{path} reuses identical checkpoint weights across epochs"
        )


def require_complete_statistics(
    path: Path,
    artifact: dict,
    *,
    expected_epoch: int,
    expected_cuts: list[int],
    expected_ranks: list[int],
    expected_radii: list[float],
    expected_draws: int,
    checkpoint_hash: str,
    dataset_fingerprints: dict,
    source_revision: str | None,
    source_archive_sha256: str | None,
) -> None:
    require_measured(path, artifact, "lw_post_cnn_suffix_statistics")
    checkpoint = artifact["checkpoint"]
    if checkpoint["epoch"] != expected_epoch:
        raise RuntimeError(f"{path} uses checkpoint epoch {checkpoint['epoch']}")
    if checkpoint["sha256"] != checkpoint_hash:
        raise RuntimeError(f"{path} uses the wrong checkpoint")
    if artifact["config"]["surrogate_draws"] != expected_draws:
        raise RuntimeError(f"{path} has the wrong draw count")
    if artifact["config"]["cuts"] != expected_cuts:
        raise RuntimeError(f"{path} has the wrong requested cuts")
    if artifact["config"]["pca_ranks"] != expected_ranks:
        raise RuntimeError(f"{path} has the wrong requested PCA ranks")
    if artifact["config"]["mean_noise_radii"] != expected_radii:
        raise RuntimeError(f"{path} has the wrong mean-noise radii")
    if artifact["config"]["relax_epochs"] != 10:
        raise RuntimeError(f"{path} has the wrong relaxation budget")
    if artifact.get("dataset") != dataset_fingerprints:
        raise RuntimeError(
            f"{path} did not use the checkpoint trajectory's ordered dataset"
        )
    provenance = artifact.get("provenance", {})
    if source_revision is not None and (
        provenance.get("source_revision") != source_revision
    ):
        raise RuntimeError(f"{path} does not share the trajectory's source revision")
    if source_archive_sha256 is not None and (
        provenance.get("source_archive_sha256") != source_archive_sha256
    ):
        raise RuntimeError(f"{path} does not share the trajectory's source lineage")
    expected_protocol = {
        "data_backend": "torchvision",
        "fake_data": False,
        "train_size": 50000,
        "test_size": 10000,
        "pca_fit_size": 10000,
        "widths": [32, 64, 128, 128],
        "batch_size": 256,
        "relax_learning_rate": 0.01,
        "checkpoint_epoch": expected_epoch,
        "seed": 0,
        "device": "cuda",
    }
    for key, expected in expected_protocol.items():
        if artifact["config"].get(key) != expected:
            raise RuntimeError(
                f"{path} has config.{key}={artifact['config'].get(key)!r}, "
                f"expected {expected!r}"
            )

    slices = artifact["slices"]
    cuts = [row["cut"] for row in slices]
    if cuts != expected_cuts:
        raise RuntimeError(f"{path} has cuts {cuts}")
    for slice_result in slices:
        if slice_result.get("pca_fit_count") != 10000:
            raise RuntimeError(f"{path} has the wrong PCA-basis fit count")
        if slice_result.get("moment_fit_count") != 50000:
            raise RuntimeError(
                f"{path} did not estimate moments from the full training bank"
            )
        pca_counts = slice_result.get("pca_fit_class_counts", {})
        moment_counts = slice_result.get("moment_fit_class_counts", {})
        if (
            sum(pca_counts.values()) != 10000
            or sum(moment_counts.values()) != 50000
            or set(pca_counts) != set(moment_counts)
        ):
            raise RuntimeError(f"{path} has invalid class-count provenance")
        reference_records = slice_result["reference_records"]
        require_exact_keys(
            path,
            reference_records,
            ("draw", "train_distribution", "eval_distribution", "relax_epoch"),
            {
                (draw, "true", "true", relax_epoch)
                for draw in range(expected_draws)
                for relax_epoch in range(11)
            },
            f"cut-{slice_result['cut']} real-reference",
        )

        rank_results = slice_result["rank_results"]
        if [row["pca_rank"] for row in rank_results] != expected_ranks:
            raise RuntimeError(f"{path} has the wrong PCA ranks")
        for rank_result in rank_results:
            expected_ceiling = min(moment_counts.values()) - 1
            if (
                rank_result.get("empirical_class_covariance_rank_ceiling")
                != expected_ceiling
                or rank_result.get("covariance_shrinkage") != 0.05
            ):
                raise RuntimeError(
                    f"{path} has invalid covariance-fit metadata"
                )
            coverage = rank_result["held_out_coverage"]
            for name in (
                "total_variance_fraction",
                "within_class_variance_fraction",
                "between_class_mean_variance_fraction",
            ):
                value = coverage[name]
                if not 0 <= value <= 1.0001:
                    raise RuntimeError(f"invalid held-out {name}={value} in {path}")
            records = rank_result["records"]
            expected_distributions = {
                "projected_true",
                "gaussian",
                *{f"mean_r{radius:g}" for radius in expected_radii},
            }
            require_exact_keys(
                path,
                records,
                (
                    "draw",
                    "train_distribution",
                    "eval_distribution",
                    "relax_epoch",
                ),
                {
                    (draw, distribution, "true", relax_epoch)
                    for draw in range(expected_draws)
                    for distribution in expected_distributions
                    for relax_epoch in range(11)
                },
                (
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']} relaxation"
                ),
            )
            diagnostics = rank_result["moment_diagnostics"]
            diagnostic_distributions = {
                "gaussian",
                *{f"mean_r{radius:g}" for radius in expected_radii},
            }
            require_exact_keys(
                path,
                diagnostics,
                ("draw", "distribution"),
                {
                    (draw, distribution)
                    for draw in range(expected_draws)
                    for distribution in diagnostic_distributions
                },
                (
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']} moment-diagnostic"
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--pca-rank", type=int, default=512)
    parser.add_argument("--draw-count", type=int, default=3)
    args = parser.parse_args()

    training_path = args.training_root / "training.json"
    training = load(training_path)
    require_measured(
        training_path, training, "lw_post_cnn_checkpoint_trajectory"
    )
    expected_training_config = {
        "fake_data": False,
        "data_backend": "torchvision",
        "train_size": 50000,
        "test_size": 10000,
        "epochs": 30,
        "checkpoint_epochs": [0, 1, 5, 10, 20, 30],
        "widths": [32, 64, 128, 128],
        "batch_size": 256,
        "learning_rate": 0.05,
        "weight_decay": 0.0005,
        "seed": 0,
        "device": "cuda",
    }
    for key, expected in expected_training_config.items():
        if training["config"].get(key) != expected:
            raise RuntimeError(
                f"{training_path} has config.{key}="
                f"{training['config'].get(key)!r}, expected {expected!r}"
            )
    expected_epochs = [0, 1, 5, 10, 20, 30]
    dataset_fingerprints = training.get("dataset")
    if (
        not isinstance(dataset_fingerprints, dict)
        or dataset_fingerprints.get("backend") != "torchvision"
    ):
        raise RuntimeError(
            f"{training_path} lacks canonical torchvision dataset fingerprints"
        )
    provenance = training.get("provenance", {})
    revision_value = provenance.get("source_revision")
    archive_value = provenance.get("source_archive_sha256")
    source_revision = (
        revision_value
        if isinstance(revision_value, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision_value)
        else None
    )
    source_archive_sha256 = (
        archive_value
        if isinstance(archive_value, str)
        and re.fullmatch(r"[0-9a-f]{64}", archive_value)
        else None
    )
    if source_revision is None and source_archive_sha256 is None:
        raise RuntimeError(
            f"{training_path} lacks a clean source revision or source archive hash"
        )
    observed_epochs = [row["epoch"] for row in training["checkpoints"]]
    if observed_epochs != expected_epochs:
        raise RuntimeError(
            f"checkpoint epochs {observed_epochs} != expected {expected_epochs}"
        )
    hashes: dict[int, str] = {}
    for row in training["checkpoints"]:
        checkpoint_path = args.training_root / row["path"]
        if row["path"] != f"checkpoint_epoch{row['epoch']}.pt":
            raise RuntimeError(f"{training_path} has a noncanonical checkpoint path")
        observed_hash = sha256(checkpoint_path)
        if observed_hash != row["sha256"]:
            raise RuntimeError(f"{checkpoint_path} does not match training.json")
        hashes[row["epoch"]] = observed_hash
    require_distinct_checkpoint_hashes(training_path, hashes)

    for epoch in expected_epochs:
        path = args.result_root / f"base_epoch{epoch}" / "post_statistics.json"
        artifact = load(path)
        require_complete_statistics(
            path,
            artifact,
            expected_epoch=epoch,
            expected_cuts=[1, 2, 3, 4],
            expected_ranks=[args.pca_rank],
            expected_radii=[1.0],
            expected_draws=args.draw_count,
            checkpoint_hash=hashes[epoch],
            dataset_fingerprints=dataset_fingerprints,
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
        )

    pca_path = (
        args.result_root / "pca_ablation_epoch30_cut3" / "post_statistics.json"
    )
    require_complete_statistics(
        pca_path,
        load(pca_path),
        expected_epoch=30,
        expected_cuts=[3],
        expected_ranks=[128, 512, 1024],
        expected_radii=[1.0],
        expected_draws=3,
        checkpoint_hash=hashes[30],
        dataset_fingerprints=dataset_fingerprints,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
    )
    noise_path = (
        args.result_root / "noise_ablation_epoch30_cut3" / "post_statistics.json"
    )
    noise_artifact = load(noise_path)
    require_complete_statistics(
        noise_path,
        noise_artifact,
        expected_epoch=30,
        expected_cuts=[3],
        expected_ranks=[args.pca_rank],
        expected_radii=[0.0, 0.1, 1.0, 2.0],
        expected_draws=3,
        checkpoint_hash=hashes[30],
        dataset_fingerprints=dataset_fingerprints,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
    )

    print("LW post artifacts verified.")


if __name__ == "__main__":
    main()
