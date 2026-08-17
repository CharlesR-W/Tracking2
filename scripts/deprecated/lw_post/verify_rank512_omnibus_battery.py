from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tracking2.cnn_checkpoints import sha256


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    artifact = json.loads(path.read_text())
    if not isinstance(artifact, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return artifact


def require_finite_number(
    value: object,
    *,
    path: Path,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"{path} has non-finite or non-numeric {field}={value!r}")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise RuntimeError(f"{path} has {field}={numeric}, below {minimum}")
    if maximum is not None and numeric > maximum:
        raise RuntimeError(f"{path} has {field}={numeric}, above {maximum}")
    return numeric


def require_valid_relaxation_records(
    path: Path, records: list[dict], label: str
) -> None:
    trajectory_initials: dict[tuple[object, object, object], tuple[float, float]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} has a non-object row in {label}")
        require_finite_number(
            row.get("accuracy"),
            path=path,
            field=f"{label}.accuracy",
            minimum=0,
            maximum=1,
        )
        require_finite_number(
            row.get("loss"),
            path=path,
            field=f"{label}.loss",
            minimum=0,
        )
        initial_loss = require_finite_number(
            row.get("initial_training_loss"),
            path=path,
            field=f"{label}.initial_training_loss",
            minimum=0,
        )
        gradient_norm = require_finite_number(
            row.get("initial_gradient_norm"),
            path=path,
            field=f"{label}.initial_gradient_norm",
            minimum=0,
        )
        require_finite_number(
            row.get("initial_gradient_rms"),
            path=path,
            field=f"{label}.initial_gradient_rms",
            minimum=0,
        )
        require_finite_number(
            row.get("initial_suffix_weight_norm"),
            path=path,
            field=f"{label}.initial_suffix_weight_norm",
            minimum=0,
        )
        require_finite_number(
            row.get("first_step_update_norm"),
            path=path,
            field=f"{label}.first_step_update_norm",
            minimum=0,
        )
        require_finite_number(
            row.get("first_step_update_to_weight_ratio"),
            path=path,
            field=f"{label}.first_step_update_to_weight_ratio",
            minimum=0,
        )
        batch_hash = row.get("initial_batch_index_sha256")
        if (
            not isinstance(batch_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", batch_hash) is None
        ):
            raise RuntimeError(f"{path} has invalid {label} first-batch hash")
        key = (
            row.get("draw"),
            row.get("train_distribution"),
            row.get("eval_distribution"),
        )
        initial = trajectory_initials.setdefault(
            key, (initial_loss, gradient_norm)
        )
        if initial != (initial_loss, gradient_norm):
            raise RuntimeError(
                f"{path} changes initial diagnostics within {label} trajectory {key}"
            )


def require_common_relaxation_endpoint(
    path: Path,
    reference_records: list[dict],
    candidate_records: list[dict],
    *,
    label: str,
) -> None:
    reference_by_draw = {
        row["draw"]: (float(row["loss"]), float(row["accuracy"]))
        for row in reference_records
        if row["relax_epoch"] == 0
    }
    observed_by_eval: dict[
        tuple[object, object], tuple[float, float]
    ] = {}
    for row in candidate_records:
        if row["relax_epoch"] != 0:
            continue
        observed = (float(row["loss"]), float(row["accuracy"]))
        eval_key = (row["draw"], row["eval_distribution"])
        expected = observed_by_eval.setdefault(eval_key, observed)
        if observed != expected:
            raise RuntimeError(
                f"{path} does not share the unrelaxed endpoint for {label}, "
                f"draw {row['draw']}, eval {row['eval_distribution']}"
            )
        if row["eval_distribution"] == "true" and observed != (
            reference_by_draw[row["draw"]]
        ):
            raise RuntimeError(
                f"{path} does not match the compatibility true endpoint for "
                f"{label}, draw {row['draw']}"
            )


def require_valid_moment_diagnostics(
    path: Path, diagnostics: list[dict], label: str
) -> None:
    for row in diagnostics:
        if row.get("diagnostic_space") != "fitted PCA subspace":
            raise RuntimeError(f"{path} uses the wrong diagnostic space for {label}")
        for field in (
            "class_mean_relative_error",
            "class_covariance_relative_error",
        ):
            require_finite_number(
                row.get(field),
                path=path,
                field=f"{label}.{field}",
                minimum=0,
            )


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
    expected_shrinkages: list[float],
    expected_draws: int,
    expected_true_eval_only: bool,
    expected_relax_epochs: int,
    checkpoint_hash: str,
    dataset_fingerprints: dict,
    source_revision: str | None,
    source_archive_sha256: str | None,
    expected_seed: int,
) -> None:
    require_measured(path, artifact, "lw_post_cnn_suffix_statistics")
    if artifact.get("schema_version") != 2:
        raise RuntimeError(f"{path} has the wrong schema version")
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
    if (
        artifact["config"].get("gaussian_covariance_shrinkages")
        != expected_shrinkages
    ):
        raise RuntimeError(f"{path} has the wrong covariance shrinkage grid")
    if artifact["config"].get("true_eval_only") is not expected_true_eval_only:
        raise RuntimeError(f"{path} has the wrong evaluation protocol")
    if artifact["config"]["relax_epochs"] != expected_relax_epochs:
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
        "seed": expected_seed,
        "device": "cuda",
        "suffix_initialization": "warm",
        "learning_rate_regime": "fixed_lr",
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
                for relax_epoch in range(expected_relax_epochs + 1)
            },
            f"cut-{slice_result['cut']} real-reference",
        )
        require_valid_relaxation_records(
            path, reference_records, f"cut-{slice_result['cut']} real-reference"
        )

        rank_results = slice_result["rank_results"]
        if [row["pca_rank"] for row in rank_results] != expected_ranks:
            raise RuntimeError(f"{path} has the wrong PCA ranks")
        for rank_result in rank_results:
            expected_ceiling = min(moment_counts.values()) - 1
            if (
                rank_result.get("empirical_class_covariance_rank_ceiling")
                != expected_ceiling
                or rank_result.get(
                    "rank_exceeds_empirical_class_covariance_ceiling"
                )
                is not False
                or rank_result["pca_rank"] > expected_ceiling
            ):
                raise RuntimeError(
                    f"{path} has invalid covariance-fit metadata"
                )
            gaussian_distributions = [
                (
                    "gaussian_empirical"
                    if shrinkage == 0
                    else f"gaussian_shrunk_s{shrinkage:g}"
                )
                for shrinkage in expected_shrinkages
            ]
            estimators = rank_result.get("gaussian_covariance_estimators")
            if (
                not isinstance(estimators, list)
                or [row.get("distribution") for row in estimators]
                != gaussian_distributions
            ):
                raise RuntimeError(
                    f"{path} has the wrong covariance estimator provenance grid"
                )
            for shrinkage, estimator in zip(expected_shrinkages, estimators):
                if (
                    estimator.get("requested_covariance_shrinkage")
                    != shrinkage
                    or estimator.get("maximum_factor_jitter") != 0.0
                ):
                    raise RuntimeError(
                        f"{path} has invalid covariance estimator provenance"
                    )
                if shrinkage == 0 and (
                    estimator.get("exact_empirical_covariance") is not True
                    or estimator.get(
                        "total_clipped_negative_eigenvalue_count"
                    )
                    != 0
                ):
                    raise RuntimeError(
                        f"{path} did not realize the exact empirical covariance"
                    )
            pooled_variance = require_finite_number(
                rank_result.get(
                    "pooled_within_class_variance_per_pca_coordinate"
                ),
                path=path,
                field="pooled_within_class_variance_per_pca_coordinate",
                minimum=0,
            )
            trace = require_finite_number(
                rank_result.get("trace_matched_isotropic_covariance_trace"),
                path=path,
                field="trace_matched_isotropic_covariance_trace",
                minimum=0,
            )
            if not math.isclose(
                trace,
                rank_result["pca_rank"] * pooled_variance,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"{path} has inconsistent trace matching")
            coverage = rank_result["held_out_coverage"]
            for name in (
                "total_variance_fraction",
                "within_class_variance_fraction",
                "between_class_mean_variance_fraction",
            ):
                value = coverage[name]
                require_finite_number(
                    value,
                    path=path,
                    field=f"held_out_coverage.{name}",
                    minimum=0,
                    maximum=1.0001,
                )
            records = rank_result["records"]
            expected_distributions = {
                "true",
                "projected_true",
                *gaussian_distributions,
                *{f"mean_r{radius:g}" for radius in expected_radii},
            }
            expected_eval_distributions = (
                {"true"}
                if expected_true_eval_only
                else expected_distributions
            )
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
                    (draw, distribution, eval_distribution, relax_epoch)
                    for draw in range(expected_draws)
                    for distribution in expected_distributions
                    for eval_distribution in expected_eval_distributions
                    for relax_epoch in range(expected_relax_epochs + 1)
                },
                (
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']} relaxation"
                ),
            )
            require_valid_relaxation_records(
                path,
                records,
                (
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']} relaxation"
                ),
            )
            require_common_relaxation_endpoint(
                path,
                reference_records,
                records,
                label=(
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']}"
                ),
            )
            diagnostics = rank_result["moment_diagnostics"]
            diagnostic_distributions = {
                *gaussian_distributions,
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
            require_valid_moment_diagnostics(
                path,
                diagnostics,
                (
                    f"cut-{slice_result['cut']} rank-"
                    f"{rank_result['pca_rank']} moment-diagnostic"
                ),
            )
            step_zero = rank_result.get("step_zero_true_vs_projected", {})
            for field in (
                "true_loss",
                "projected_true_loss",
                "true_to_projected_predictive_kl",
            ):
                require_finite_number(
                    step_zero.get(field),
                    path=path,
                    field=f"step_zero_true_vs_projected.{field}",
                    minimum=0,
                )
            for field in ("true_accuracy", "projected_true_accuracy"):
                require_finite_number(
                    step_zero.get(field),
                    path=path,
                    field=f"step_zero_true_vs_projected.{field}",
                    minimum=0,
                    maximum=1,
                )
        for coverage_name in (
            "total_variance_fraction",
            "within_class_variance_fraction",
            "between_class_mean_variance_fraction",
        ):
            values = [
                row["held_out_coverage"][coverage_name]
                for row in rank_results
            ]
            if any(
                right + 1e-8 < left
                for left, right in zip(values, values[1:])
            ):
                raise RuntimeError(
                    f"{path} has non-monotone held-out {coverage_name}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--pca-rank", type=int, default=512)
    parser.add_argument("--draw-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
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
        "seed": args.seed,
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
            expected_shrinkages=[0.0],
            expected_draws=args.draw_count,
            expected_true_eval_only=False,
            expected_relax_epochs=10,
            checkpoint_hash=hashes[epoch],
            dataset_fingerprints=dataset_fingerprints,
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
            expected_seed=args.seed,
        )

    for cut, ranks in ((1, [128, 512, 1024]), (4, [128, 512, 1024, 2048])):
        pca_path = (
            args.result_root
            / f"pca_ablation_epoch30_cut{cut}"
            / "post_statistics.json"
        )
        require_complete_statistics(
            pca_path,
            load(pca_path),
            expected_epoch=30,
            expected_cuts=[cut],
            expected_ranks=ranks,
            expected_radii=[1.0],
            expected_shrinkages=[0.0],
            expected_draws=3,
            expected_true_eval_only=False,
            expected_relax_epochs=10,
            checkpoint_hash=hashes[30],
            dataset_fingerprints=dataset_fingerprints,
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
            expected_seed=args.seed,
        )
    noise_path = (
        args.result_root
        / "noise_ablation_epoch30_cuts1_4"
        / "post_statistics.json"
    )
    noise_artifact = load(noise_path)
    require_complete_statistics(
        noise_path,
        noise_artifact,
        expected_epoch=30,
        expected_cuts=[1, 4],
        expected_ranks=[args.pca_rank],
        expected_radii=[0.0, 0.1, 0.5, 1.0, 2.0],
        expected_shrinkages=[0.0],
        expected_draws=3,
        expected_true_eval_only=True,
        expected_relax_epochs=10,
        checkpoint_hash=hashes[30],
        dataset_fingerprints=dataset_fingerprints,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        expected_seed=args.seed,
    )
    covariance_path = (
        args.result_root
        / "covariance_ablation_epoch30_cuts1_4"
        / "post_statistics.json"
    )
    require_complete_statistics(
        covariance_path,
        load(covariance_path),
        expected_epoch=30,
        expected_cuts=[1, 4],
        expected_ranks=[args.pca_rank],
        expected_radii=[1.0],
        expected_shrinkages=[0.0, 0.01, 0.05],
        expected_draws=3,
        expected_true_eval_only=True,
        expected_relax_epochs=10,
        checkpoint_hash=hashes[30],
        dataset_fingerprints=dataset_fingerprints,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        expected_seed=args.seed,
    )

    print("Deprecated rank-512 omnibus LW artifacts verified.")


if __name__ == "__main__":
    main()
