from __future__ import annotations

import argparse
import json
import math
import re
from itertools import product
from pathlib import Path

from tracking2.cnn_checkpoints import sha256


RANKS = (128, 256, 512)
CUTS = (3, 7)
SOURCE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
BLOCK_NAMES = [
    "stage1.resblk1",
    "stage1.resblk2",
    "stage2.resblk1",
    "stage2.resblk2",
    "stage3.resblk1",
    "stage3.resblk2",
    "stage4.resblk1",
    "stage4.resblk2",
]


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def source_identity_is_recorded(provenance: object) -> bool:
    if not isinstance(provenance, dict):
        return False
    revision = provenance.get("source_revision")
    archive_hash = provenance.get("source_archive_sha256")
    return (
        isinstance(revision, str)
        and SOURCE_REVISION.fullmatch(revision) is not None
    ) or (
        isinstance(archive_hash, str)
        and SHA256.fullmatch(archive_hash) is not None
    )


def require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{context} is not a SHA-256 digest")
    return value


def verify_distribution_grid(
    records: object,
    *,
    distributions: set[str],
    draws: set[int],
    epochs: set[int],
    context: str,
) -> None:
    if not isinstance(records, list):
        raise RuntimeError(f"{context} records must be an array")
    actual = [
        (
            row.get("draw"),
            row.get("train_distribution"),
            row.get("eval_distribution"),
            row.get("relax_epoch"),
        )
        for row in records
    ]
    expected = set(product(draws, distributions, {"true"}, epochs))
    if len(actual) != len(set(actual)):
        raise RuntimeError(f"{context} contains duplicate trajectory cells")
    if set(actual) != expected:
        missing = expected - set(actual)
        unexpected = set(actual) - expected
        raise RuntimeError(
            f"{context} trajectory grid mismatch; missing={list(missing)!r}, "
            f"unexpected={list(unexpected)!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the bounded nested ResNet PCA controls for the LW note."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    checkpoint_hash = sha256(args.checkpoint)
    training_manifest = load(args.training_manifest)
    artifact_path = args.output / "resnet_suffix_statistics.json"
    artifact = load(artifact_path)

    expected_architecture = {
        "name": "InstrumentedResNet18V2",
        "width": 64,
        "block_names": BLOCK_NAMES,
    }
    if training_manifest.get("status") != "MEASURED":
        raise RuntimeError("training manifest is not measured")
    if training_manifest.get("experiment") != "resnet18_training_checkpoints":
        raise RuntimeError("training manifest has the wrong experiment")
    if training_manifest.get("architecture") != expected_architecture:
        raise RuntimeError("training manifest architecture mismatch")
    if not source_identity_is_recorded(training_manifest.get("provenance")):
        raise RuntimeError("training manifest does not identify its source")
    training_config = training_manifest.get("config", {})
    if (
        training_config.get("seed") != args.seed
        or training_config.get("fake_data") is not False
        or training_config.get("data_backend") != "torchvision"
        or training_config.get("width") != 64
        or training_config.get("learning_rate") != 0.05
    ):
        raise RuntimeError(
            "training manifest seed/backend/architecture/learning-rate mismatch"
        )
    training_checkpoint = [
        row
        for row in training_manifest.get("checkpoints", [])
        if row.get("epoch") == 100
    ]
    if len(training_checkpoint) != 1:
        raise RuntimeError("training manifest lacks one epoch-100 checkpoint")
    if (
        training_checkpoint[0].get("path") != args.checkpoint.name
        or training_checkpoint[0].get("sha256") != checkpoint_hash
        or (args.training_manifest.parent / training_checkpoint[0]["path"]).resolve()
        != args.checkpoint.resolve()
    ):
        raise RuntimeError("training checkpoint path/hash lineage mismatch")

    if artifact.get("status") != "MEASURED":
        raise RuntimeError(f"{artifact_path} is not measured")
    if artifact.get("experiment") != "resnet18_suffix_statistics_sweep":
        raise RuntimeError(f"{artifact_path} has the wrong experiment")
    if artifact.get("schema_version") != 3:
        raise RuntimeError(f"{artifact_path} has the wrong schema")
    if artifact.get("architecture") != expected_architecture:
        raise RuntimeError(f"{artifact_path} architecture mismatch")
    if not source_identity_is_recorded(artifact.get("provenance")):
        raise RuntimeError(f"{artifact_path} does not identify its analysis source")

    expected_config = {
        "checkpoint_epoch": 100,
        "data_backend": "torchvision",
        "fake_data": False,
        "train_size": 10000,
        "test_size": 2000,
        "batch_size": 128,
        "width": 64,
        "cuts": list(CUTS),
        "pca_fit_size": 5000,
        "pca_ranks": list(RANKS),
        "surrogate_draws": 3,
        "mean_noise_radii": [1.0],
        "include_projected_true": True,
        "true_eval_only": True,
        "relax_epochs": 5,
        "relax_learning_rate": 0.01,
        "seed": args.seed,
    }
    config = artifact.get("config", {})
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{artifact_path} has config.{key}={config.get(key)!r}, "
                f"expected {expected!r}"
            )
    if Path(config.get("training_manifest", "")).resolve() != args.training_manifest.resolve():
        raise RuntimeError("analysis config points at the wrong training manifest")

    lineage = artifact.get("lineage", {})
    if lineage.get("model_seed") != args.seed:
        raise RuntimeError("analysis lineage model seed mismatch")
    if lineage.get("architecture") != expected_architecture:
        raise RuntimeError("analysis lineage architecture mismatch")
    lineage_checkpoint = lineage.get("checkpoint", {})
    if (
        lineage_checkpoint.get("epoch") != 100
        or lineage_checkpoint.get("sha256") != checkpoint_hash
        or Path(lineage_checkpoint.get("path", "")).resolve()
        != args.checkpoint.resolve()
    ):
        raise RuntimeError("analysis checkpoint lineage mismatch")
    lineage_manifest = lineage.get("training_manifest", {})
    if (
        Path(lineage_manifest.get("path", "")).resolve()
        != args.training_manifest.resolve()
        or lineage_manifest.get("sha256") != sha256(args.training_manifest)
        or lineage_manifest.get("status") != "MEASURED"
        or lineage_manifest.get("experiment") != "resnet18_training_checkpoints"
    ):
        raise RuntimeError("analysis training-manifest lineage mismatch")

    dataset = artifact.get("dataset", {})
    training_dataset = training_manifest.get("dataset", {})
    if dataset.get("backend") != "torchvision":
        raise RuntimeError("analysis dataset backend is not torchvision")
    for split in ("train", "test"):
        source = dataset.get(split, {})
        training_source = training_dataset.get(split, {})
        if (
            source.get("source_count") != training_source.get("source_count")
            or source.get("source_sha256") != training_source.get("source_sha256")
        ):
            raise RuntimeError(f"analysis {split} dataset lineage mismatch")
        require_sha256(source.get("source_sha256"), f"dataset.{split}.source_sha256")

    banks = artifact.get("analysis_banks", {})
    for split, seed, count in (
        ("train", args.seed + 2_000_000, 10000),
        ("test", args.seed + 2_000_001, 2000),
    ):
        bank = banks.get(split, {})
        if bank.get("seed") != seed or bank.get("count") != count:
            raise RuntimeError(f"analysis {split} bank seed/count mismatch")
        require_sha256(bank.get("sha256"), f"analysis_banks.{split}.sha256")

    slices = artifact.get("slices", [])
    if [row.get("cut") for row in slices] != list(CUTS):
        raise RuntimeError(f"{artifact_path} does not contain the two declared cuts")
    draws = {0, 1, 2}
    epochs = set(range(6))
    for slice_result in slices:
        context = f"cut {slice_result.get('cut')}"
        if slice_result.get("pca_basis_fit_count") != 5000:
            raise RuntimeError(f"{context} has the wrong PCA fit count")
        if slice_result.get("moment_fit_count") != 10000:
            raise RuntimeError(f"{context} does not use the full moment-fit bank")
        if sum(slice_result.get("pca_basis_fit_class_counts", {}).values()) != 5000:
            raise RuntimeError(f"{context} PCA class counts do not sum to 5000")
        moment_counts = slice_result.get("moment_fit_class_counts", {})
        if sum(moment_counts.values()) != 10000:
            raise RuntimeError(f"{context} moment class counts do not sum to 10000")
        if slice_result.get("maximal_pca_rank") != max(RANKS):
            raise RuntimeError(f"{context} has the wrong maximal PCA rank")
        maximal_basis = require_sha256(
            slice_result.get("maximal_pca_basis_sha256"),
            f"{context} maximal PCA basis",
        )
        verify_distribution_grid(
            slice_result.get("reference_records"),
            distributions={"true"},
            draws=draws,
            epochs=epochs,
            context=f"{context} true reference",
        )

        rank_results = slice_result.get("rank_results", [])
        if [row.get("pca_rank") for row in rank_results] != list(RANKS):
            raise RuntimeError(f"{context} does not contain nested ranks {RANKS}")
        for rank_result in rank_results:
            rank = rank_result["pca_rank"]
            rank_context = f"{context}, rank {rank}"
            if rank_result.get("maximal_pca_basis_sha256") != maximal_basis:
                raise RuntimeError(f"{rank_context} does not share the maximal basis")
            require_sha256(
                rank_result.get("pca_basis_prefix_sha256"),
                f"{rank_context} PCA prefix",
            )
            if rank_result.get("pca_basis_fit_count") != 5000:
                raise RuntimeError(f"{rank_context} PCA fit count mismatch")
            if rank_result.get("moment_fit_count") != 10000:
                raise RuntimeError(f"{rank_context} moment fit count mismatch")
            rank_moment_counts = rank_result.get("moment_fit_class_counts", {})
            if rank_moment_counts != moment_counts:
                raise RuntimeError(f"{rank_context} moment class counts mismatch")
            ceiling = min(moment_counts.values()) - 1
            if (
                rank_result.get("empirical_class_covariance_rank_ceiling") != ceiling
                or rank_result.get(
                    "rank_exceeds_empirical_class_covariance_ceiling"
                )
                is not False
                or rank > ceiling
            ):
                raise RuntimeError(f"{rank_context} violates covariance rank ceiling")
            if rank_result.get("covariance_shrinkage") != 0.05:
                raise RuntimeError(f"{rank_context} covariance shrinkage mismatch")
            pooled = rank_result.get(
                "pooled_within_class_variance_per_pca_coordinate"
            )
            trace = rank_result.get("trace_matched_isotropic_covariance_trace")
            if (
                not isinstance(pooled, (int, float))
                or isinstance(pooled, bool)
                or not math.isfinite(pooled)
                or pooled <= 0
                or not math.isclose(trace, rank * pooled, rel_tol=1e-9)
            ):
                raise RuntimeError(f"{rank_context} pooled variance/trace mismatch")
            coverage = rank_result.get("held_out_coverage", {})
            for name in (
                "total_variance_fraction",
                "within_class_variance_fraction",
                "between_class_mean_variance_fraction",
            ):
                value = coverage.get(name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= value <= 1.0001
                ):
                    raise RuntimeError(
                        f"{rank_context} has invalid held-out {name}={value!r}"
                    )
            verify_distribution_grid(
                rank_result.get("records"),
                distributions={"projected_true", "gaussian", "mean"},
                draws=draws,
                epochs=epochs,
                context=rank_context,
            )

    print("LW post nested ResNet controls verified.")


if __name__ == "__main__":
    main()
