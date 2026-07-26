"""Build the measured-only interactive data appendix for the LW research note.

The builder deliberately does not discover artifacts.  Every input must appear in
an explicit manifest with a SHA-256 digest and expected checkpoint metadata.  This
keeps a partially copied experiment directory from silently changing the public
report.

Manifest schema (version 1)::

    {
      "schema_version": 1,
      "title": "Free-Body Diagrams for Neural Networks — Interactive Data Appendix (WIP)",
      "status": "MEASURED",
      "source_commit": "0123456789abcdef",
      "inputs": [
        {
          "id": "cnn-epoch-1",
          "kind": "cnn",
          "format": "post_statistics",
          "label": "Four-block CNN, epoch 1",
          "path": "data/cnn-epoch-1.json",
          "sha256": "...",
          "checkpoint_epoch": 1,
          "model_seed": 0
        }
      ]
    }

Relative input paths are resolved from the manifest's directory.  The canonical
report requires at least one measured CNN input and one measured ResNet-18 input.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


TITLE = "Free-Body Diagrams for Neural Networks — Interactive Data Appendix (WIP)"
DEFAULT_OUTPUT = "free-body-diagrams-for-neural-networks.html"
MANIFEST_SCHEMA_VERSION = 1

REAL = "#222222"
PROJECTED_REAL = "#737373"
GAUSSIAN = "#0072B2"
MEAN = "#D55E00"

_EXPECTED_EXPERIMENTS = {
    "cnn": ("lw_post_cnn_suffix_statistics", {1, 2}),
    "resnet": ("resnet18_suffix_statistics_sweep", {1, 2, 3, 4}),
}
_EXPECTED_FORMATS = {
    ("cnn", "post_statistics"),
    ("cnn", "legacy_cnn_suffix_statistics"),
    ("resnet", "resnet_suffix_statistics"),
}
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_GAUSSIAN_SHRINKAGE_RE = re.compile(
    r"^gaussian_shrunk_s(?P<amount>(?:\d+(?:\.\d*)?|\.\d+))$"
)


class ReportInputError(ValueError):
    """Raised when an input cannot safely support the public report."""


@dataclass(frozen=True)
class LoadedArtifact:
    id: str
    kind: str
    format: str
    label: str
    relative_path: str
    sha256: str
    checkpoint_epoch: int
    model_seed: int
    payload: dict[str, Any]


def _bad_constant(value: str) -> None:
    raise ReportInputError(f"JSON contains non-finite constant {value!r}")


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReportInputError(f"{context} does not exist: {path}")
    try:
        value = json.loads(path.read_text(), parse_constant=_bad_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportInputError(f"Could not read {context} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportInputError(f"{context} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportInputError(f"{context}.{key} must be a non-empty string")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportInputError(f"{context}.{key} must be an integer")
    return value


def _require_digest(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_string(mapping, key, context).lower()
    if not _DIGEST_RE.fullmatch(value):
        raise ReportInputError(f"{context}.{key} must contain 64 hex characters")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportInputError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportInputError(f"{context} must be finite")
    return result


def _integral_number(value: Any, context: str) -> int:
    number = _number(value, context)
    if not number.is_integer():
        raise ReportInputError(f"{context} must be an integer-valued number")
    return int(number)


def _objects(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ReportInputError(f"{context} must be a non-empty array")
    if not all(isinstance(item, dict) for item in value):
        raise ReportInputError(f"{context} must contain only objects")
    return value


def _artifact_epoch(
    payload: Mapping[str, Any], kind: str, artifact_format: str, context: str
) -> int:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ReportInputError(f"{context}.config must be an object")
    epoch = _require_int(config, "checkpoint_epoch", f"{context}.config")
    has_structured_checkpoint = (
        kind == "cnn" and artifact_format == "post_statistics"
    ) or (kind == "resnet" and payload.get("schema_version") in {3, 4})
    if has_structured_checkpoint:
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ReportInputError(f"{context}.checkpoint must be an object")
        checkpoint_epoch = _require_int(checkpoint, "epoch", f"{context}.checkpoint")
        _require_string(checkpoint, "path", f"{context}.checkpoint")
        _require_digest(checkpoint, "sha256", f"{context}.checkpoint")
        if checkpoint_epoch != epoch:
            raise ReportInputError(
                f"{context} disagrees internally about checkpoint epoch: "
                f"{checkpoint_epoch} != {epoch}"
            )
    return epoch


def _source_identity(
    provenance: Any, context: str, *, required: bool
) -> tuple[str, str] | None:
    if not isinstance(provenance, dict):
        if required:
            raise ReportInputError(f"{context} must be an object")
        return None
    revision = provenance.get("source_revision")
    if isinstance(revision, str) and _SOURCE_REVISION_RE.fullmatch(revision):
        return "source revision", revision.lower()
    archive = provenance.get("source_archive_sha256")
    if isinstance(archive, str) and _DIGEST_RE.fullmatch(archive):
        return "source archive", archive.lower()
    if required:
        raise ReportInputError(
            f"{context} must record a full clean source revision or source archive SHA-256"
        )
    return None


def _validate_resnet_canonical(
    payload: Mapping[str, Any],
    *,
    context: str,
    epoch: int,
    seed: int,
) -> None:
    """Validate the lineage and controls promised by canonical ResNet artifacts."""

    config = payload["config"]
    schema = payload.get("schema_version")
    expected_flags = {
        "data_backend": "torchvision",
        "include_projected_true": True,
        "true_eval_only": schema == 3,
    }
    for key, expected in expected_flags.items():
        if config.get(key) != expected:
            raise ReportInputError(
                f"{context}.config.{key} must be {expected!r} for canonical "
                "ResNet controls"
            )
    draws = _require_int(config, "surrogate_draws", f"{context}.config")
    if draws < 1:
        raise ReportInputError(f"{context}.config.surrogate_draws must be positive")
    relax_epochs = _require_int(config, "relax_epochs", f"{context}.config")
    if relax_epochs < 0:
        raise ReportInputError(
            f"{context}.config.relax_epochs must be non-negative"
        )
    ranks = config.get("pca_ranks")
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in ranks)
    ):
        raise ReportInputError(
            f"{context}.config.pca_ranks must contain positive integer ranks"
        )
    expected_ranks = sorted(set(ranks))
    if ranks != expected_ranks:
        raise ReportInputError(
            f"{context}.config.pca_ranks must be sorted and unique"
        )
    radii = config.get("mean_noise_radii")
    if not isinstance(radii, list) or not radii:
        raise ReportInputError(
            f"{context}.config.mean_noise_radii must be a non-empty array"
        )
    for index, radius in enumerate(radii):
        if _number(radius, f"{context}.config.mean_noise_radii[{index}]") < 0:
            raise ReportInputError(
                f"{context}.config.mean_noise_radii[{index}] must be non-negative"
            )
    shrinkages = config.get("gaussian_covariance_shrinkages")
    if schema == 4:
        if not isinstance(shrinkages, list) or not shrinkages:
            raise ReportInputError(
                f"{context}.config.gaussian_covariance_shrinkages must be "
                "a non-empty array"
            )
        for index, shrinkage in enumerate(shrinkages):
            value = _number(
                shrinkage,
                f"{context}.config.gaussian_covariance_shrinkages[{index}]",
            )
            if not 0 <= value <= 1:
                raise ReportInputError(
                    f"{context}.config.gaussian_covariance_shrinkages[{index}] "
                    "must be in [0, 1]"
                )

    _require_string(payload, "mean_noise_definition", context)
    _require_string(payload, "pca_protocol", context)
    _source_identity(payload.get("provenance"), f"{context}.provenance", required=True)

    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ReportInputError(f"{context}.architecture must be an object")
    if architecture.get("name") != "InstrumentedResNet18V2":
        raise ReportInputError(
            f"{context}.architecture.name must be 'InstrumentedResNet18V2'"
        )
    width = _require_int(architecture, "width", f"{context}.architecture")
    if width != _require_int(config, "width", f"{context}.config"):
        raise ReportInputError(f"{context} architecture width disagrees with config")
    block_names = architecture.get("block_names")
    if (
        not isinstance(block_names, list)
        or not block_names
        or not all(isinstance(name, str) and name for name in block_names)
    ):
        raise ReportInputError(
            f"{context}.architecture.block_names must be a non-empty string array"
        )

    checkpoint = payload["checkpoint"]
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ReportInputError(f"{context}.lineage must be an object")
    if lineage.get("checkpoint") != checkpoint:
        raise ReportInputError(f"{context}.lineage.checkpoint must match checkpoint")
    if lineage.get("model_seed") != seed:
        raise ReportInputError(f"{context}.lineage.model_seed must match config.seed")
    if lineage.get("architecture") != architecture:
        raise ReportInputError(
            f"{context}.lineage.architecture must match architecture"
        )
    _source_identity(
        lineage.get("training_source"),
        f"{context}.lineage.training_source",
        required=True,
    )
    training_manifest = lineage.get("training_manifest")
    if not isinstance(training_manifest, dict):
        raise ReportInputError(
            f"{context}.lineage.training_manifest must be an object"
        )
    _require_string(
        training_manifest, "path", f"{context}.lineage.training_manifest"
    )
    _require_digest(
        training_manifest, "sha256", f"{context}.lineage.training_manifest"
    )
    if (
        training_manifest.get("status") != "MEASURED"
        or training_manifest.get("experiment") != "resnet18_training_checkpoints"
    ):
        raise ReportInputError(
            f"{context}.lineage.training_manifest must identify measured "
            "ResNet training checkpoints"
        )

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("backend") != "torchvision":
        raise ReportInputError(
            f"{context}.dataset must identify the torchvision backend"
        )
    for split in ("train", "test"):
        split_data = dataset.get(split)
        if not isinstance(split_data, dict):
            raise ReportInputError(f"{context}.dataset.{split} must be an object")
        _require_digest(split_data, "source_sha256", f"{context}.dataset.{split}")
    expected_dataset_source = {
        "backend": dataset["backend"],
        "train": dataset["train"],
        "test": dataset["test"],
    }
    if lineage.get("dataset_source") != expected_dataset_source:
        raise ReportInputError(
            f"{context}.lineage.dataset_source must match dataset fingerprints"
        )

    banks = payload.get("analysis_banks")
    if not isinstance(banks, dict):
        raise ReportInputError(f"{context}.analysis_banks must be an object")
    _require_string(banks, "definition", f"{context}.analysis_banks")
    train_size = _require_int(config, "train_size", f"{context}.config")
    test_size = _require_int(config, "test_size", f"{context}.config")
    for split, expected_count, expected_seed in (
        ("train", train_size, seed + 2_000_000),
        ("test", test_size, seed + 2_000_001),
    ):
        bank = banks.get(split)
        if not isinstance(bank, dict):
            raise ReportInputError(
                f"{context}.analysis_banks.{split} must be an object"
            )
        if (
            _require_int(bank, "count", f"{context}.analysis_banks.{split}")
            != expected_count
        ):
            raise ReportInputError(
                f"{context}.analysis_banks.{split}.count must match config"
            )
        if (
            _require_int(bank, "seed", f"{context}.analysis_banks.{split}")
            != expected_seed
        ):
            raise ReportInputError(
                f"{context}.analysis_banks.{split}.seed does not match protocol"
            )
        _require_digest(bank, "sha256", f"{context}.analysis_banks.{split}")

    if checkpoint.get("epoch") != epoch:
        raise ReportInputError(f"{context}.checkpoint.epoch must match config")
    if payload.get("module_names") != block_names:
        raise ReportInputError(
            f"{context}.module_names must match architecture.block_names"
        )

    configured_cuts = config.get("cuts")
    if (
        not isinstance(configured_cuts, list)
        or not configured_cuts
        or any(
            isinstance(cut, bool) or not isinstance(cut, int)
            for cut in configured_cuts
        )
    ):
        raise ReportInputError(
            f"{context}.config.cuts must be a non-empty integer array"
        )
    slices = _objects(payload.get("slices"), f"{context}.slices")
    if [slice_.get("cut") for slice_ in slices] != configured_cuts:
        raise ReportInputError(f"{context}.slices must match config.cuts in order")
    pca_fit_size = _require_int(config, "pca_fit_size", f"{context}.config")
    expected_fit_count = min(pca_fit_size, train_size)

    def count_total(value: Any, count_context: str) -> int:
        if not isinstance(value, dict) or not value:
            raise ReportInputError(f"{count_context} must be a non-empty object")
        total = 0
        for class_id, count in value.items():
            if not isinstance(class_id, str) or not class_id:
                raise ReportInputError(
                    f"{count_context} must use non-empty string class ids"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ReportInputError(
                    f"{count_context}[{class_id!r}] must be a non-negative integer"
                )
            total += count
        return total

    for slice_index, slice_ in enumerate(slices):
        slice_context = f"{context}.slices[{slice_index}]"
        if slice_.get("condition") not in (None, "native"):
            raise ReportInputError(
                f"{slice_context}.condition must be 'native' when present"
            )
        _require_string(slice_, "module", slice_context)
        maximal_rank = _require_int(
            slice_, "maximal_pca_rank", slice_context
        )
        if maximal_rank != expected_ranks[-1]:
            raise ReportInputError(
                f"{slice_context}.maximal_pca_rank must match config.pca_ranks"
            )
        maximal_basis = _require_digest(
            slice_, "maximal_pca_basis_sha256", slice_context
        )
        if (
            _require_int(slice_, "pca_basis_fit_count", slice_context)
            != expected_fit_count
        ):
            raise ReportInputError(
                f"{slice_context}.pca_basis_fit_count must match the fit bank"
            )
        if _require_int(slice_, "moment_fit_count", slice_context) != train_size:
            raise ReportInputError(
                f"{slice_context}.moment_fit_count must match the train bank"
            )
        if (
            count_total(
                slice_.get("pca_basis_fit_class_counts"),
                f"{slice_context}.pca_basis_fit_class_counts",
            )
            != expected_fit_count
        ):
            raise ReportInputError(
                f"{slice_context} PCA-fit class counts do not sum to fit count"
            )
        moment_counts = slice_.get("moment_fit_class_counts")
        if (
            count_total(
                moment_counts, f"{slice_context}.moment_fit_class_counts"
            )
            != train_size
        ):
            raise ReportInputError(
                f"{slice_context} moment class counts do not sum to train size"
            )
        rank_results = _objects(
            slice_.get("rank_results"), f"{slice_context}.rank_results"
        )
        if [result.get("pca_rank") for result in rank_results] != expected_ranks:
            raise ReportInputError(
                f"{slice_context}.rank_results must match config.pca_ranks"
            )
        for rank_index, result in enumerate(rank_results):
            rank_context = f"{slice_context}.rank_results[{rank_index}]"
            rank = _require_int(result, "pca_rank", rank_context)
            if (
                _require_digest(
                    result, "maximal_pca_basis_sha256", rank_context
                )
                != maximal_basis
            ):
                raise ReportInputError(
                    f"{rank_context} does not share the maximal PCA basis"
                )
            _require_digest(result, "pca_basis_prefix_sha256", rank_context)
            if (
                _require_int(result, "pca_basis_fit_count", rank_context)
                != expected_fit_count
                or _require_int(result, "moment_fit_count", rank_context)
                != train_size
            ):
                raise ReportInputError(
                    f"{rank_context} fit counts do not match the declared banks"
                )
            if result.get("moment_fit_class_counts") != moment_counts:
                raise ReportInputError(
                    f"{rank_context}.moment_fit_class_counts must match its slice"
                )
            ceiling = _require_int(
                result, "empirical_class_covariance_rank_ceiling", rank_context
            )
            if result.get(
                "rank_exceeds_empirical_class_covariance_ceiling"
            ) is not (rank > ceiling):
                raise ReportInputError(
                    f"{rank_context} covariance-rank flag is inconsistent"
                )
            if schema == 3:
                shrinkage = _number(
                    result.get("covariance_shrinkage"),
                    f"{rank_context}.covariance_shrinkage",
                )
                if not 0 <= shrinkage <= 1:
                    raise ReportInputError(
                        f"{rank_context}.covariance_shrinkage must be in [0, 1]"
                    )
            else:
                estimators = _objects(
                    result.get("gaussian_covariance_estimators"),
                    f"{rank_context}.gaussian_covariance_estimators",
                )
                names = [
                    _require_string(
                        estimator,
                        "distribution",
                        f"{rank_context}.gaussian_covariance_estimators[{index}]",
                    )
                    for index, estimator in enumerate(estimators)
                ]
                if any(_distribution_family(name) != "gaussian" for name in names):
                    raise ReportInputError(
                        f"{rank_context}.gaussian_covariance_estimators must "
                        "contain only registered Gaussian distributions"
                    )
            pooled_variance = _number(
                result.get(
                    "pooled_within_class_variance_per_pca_coordinate"
                ),
                f"{rank_context}.pooled_within_class_variance_per_pca_coordinate",
            )
            isotropic_trace = _number(
                result.get("trace_matched_isotropic_covariance_trace"),
                f"{rank_context}.trace_matched_isotropic_covariance_trace",
            )
            if pooled_variance <= 0 or not math.isclose(
                isotropic_trace, rank * pooled_variance, rel_tol=1e-9
            ):
                raise ReportInputError(
                    f"{rank_context} isotropic covariance trace is inconsistent"
                )


def load_manifest(manifest_path: str | Path) -> tuple[dict[str, Any], list[LoadedArtifact]]:
    """Load and validate a complete, measured-only input manifest."""

    path = Path(manifest_path)
    manifest = _load_json(path, context="manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReportInputError(
            f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("title") != TITLE:
        raise ReportInputError(f"manifest.title must be exactly {TITLE!r}")
    if manifest.get("status") != "MEASURED":
        raise ReportInputError("manifest.status must be exactly 'MEASURED'")
    source_commit = _require_string(manifest, "source_commit", "manifest")
    if not _COMMIT_RE.fullmatch(source_commit):
        raise ReportInputError(
            "manifest.source_commit must be a 7–64 character hexadecimal commit id"
        )

    entries = _objects(manifest.get("inputs"), "manifest.inputs")
    seen_ids: set[str] = set()
    loaded: list[LoadedArtifact] = []
    for index, entry in enumerate(entries):
        context = f"manifest.inputs[{index}]"
        input_id = _require_string(entry, "id", context)
        if input_id in seen_ids:
            raise ReportInputError(f"Duplicate manifest input id: {input_id!r}")
        seen_ids.add(input_id)

        kind = _require_string(entry, "kind", context)
        if kind not in _EXPECTED_EXPERIMENTS:
            raise ReportInputError(
                f"{context}.kind must be one of {sorted(_EXPECTED_EXPERIMENTS)}"
            )
        artifact_format = _require_string(entry, "format", context)
        if (kind, artifact_format) not in _EXPECTED_FORMATS:
            valid_formats = sorted(
                format_name
                for format_kind, format_name in _EXPECTED_FORMATS
                if format_kind == kind
            )
            raise ReportInputError(
                f"{context}.format is invalid for kind {kind!r}; "
                f"expected one of {valid_formats}"
            )
        label = _require_string(entry, "label", context)
        relative_path = _require_string(entry, "path", context)
        expected_digest = _require_string(entry, "sha256", context).lower()
        if not _DIGEST_RE.fullmatch(expected_digest):
            raise ReportInputError(f"{context}.sha256 must contain 64 hex characters")
        expected_epoch = _require_int(entry, "checkpoint_epoch", context)
        expected_seed = _require_int(entry, "model_seed", context)

        artifact_path = Path(relative_path)
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise ReportInputError(
                f"Manifest input {input_id!r} is missing: {artifact_path}"
            )
        actual_digest = _sha256(artifact_path)
        if actual_digest != expected_digest:
            raise ReportInputError(
                f"SHA-256 mismatch for {input_id!r}: expected {expected_digest}, "
                f"found {actual_digest}"
            )

        payload = _load_json(artifact_path, context=f"artifact {input_id!r}")
        if artifact_format == "legacy_cnn_suffix_statistics":
            if payload.get("experiment") is not None or payload.get("schema_version") is not None:
                raise ReportInputError(
                    f"Artifact {input_id!r} does not match the explicitly selected "
                    "legacy CNN format"
                )
        else:
            experiment, supported_versions = _EXPECTED_EXPERIMENTS[kind]
            if payload.get("experiment") != experiment:
                raise ReportInputError(
                    f"Artifact {input_id!r} is not a {kind} artifact: expected "
                    f"experiment {experiment!r}, found {payload.get('experiment')!r}"
                )
            if payload.get("schema_version") not in supported_versions:
                raise ReportInputError(
                    f"Artifact {input_id!r} has unsupported schema_version "
                    f"{payload.get('schema_version')!r}"
                )
        if payload.get("status") != "MEASURED":
            raise ReportInputError(
                f"Artifact {input_id!r} status must be exactly 'MEASURED'"
            )
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ReportInputError(f"Artifact {input_id!r}.config must be an object")
        if config.get("fake_data") is not False:
            raise ReportInputError(
                f"Artifact {input_id!r} must explicitly record config.fake_data=false"
            )
        actual_epoch = _artifact_epoch(
            payload, kind, artifact_format, f"artifact {input_id!r}"
        )
        actual_seed = _require_int(config, "seed", f"artifact {input_id!r}.config")
        if actual_epoch != expected_epoch:
            raise ReportInputError(
                f"Checkpoint epoch mismatch for {input_id!r}: manifest says "
                f"{expected_epoch}, artifact says {actual_epoch}"
            )
        if actual_seed != expected_seed:
            raise ReportInputError(
                f"Model seed mismatch for {input_id!r}: manifest says "
                f"{expected_seed}, artifact says {actual_seed}"
            )
        if kind == "resnet" and payload.get("schema_version") in {3, 4}:
            _validate_resnet_canonical(
                payload,
                context=f"artifact {input_id!r}",
                epoch=actual_epoch,
                seed=actual_seed,
            )
        if kind == "cnn" and payload.get("schema_version") == 2:
            _source_identity(
                payload.get("provenance"),
                f"artifact {input_id!r}.provenance",
                required=True,
            )

        loaded.append(
            LoadedArtifact(
                id=input_id,
                kind=kind,
                format=artifact_format,
                label=label,
                relative_path=relative_path,
                sha256=actual_digest,
                checkpoint_epoch=actual_epoch,
                model_seed=actual_seed,
                payload=payload,
            )
        )

    if not loaded:
        raise ReportInputError("manifest.inputs must contain measured evidence")
    return manifest, loaded


def _mean_radius(distribution: str) -> float | None:
    if distribution == "mean":
        return 1.0
    if not distribution.startswith("mean_r"):
        return None
    try:
        radius = float(distribution.removeprefix("mean_r"))
    except ValueError as error:
        raise ReportInputError(
            f"Malformed mean-noise distribution name {distribution!r}"
        ) from error
    if not math.isfinite(radius) or radius < 0:
        raise ReportInputError(
            f"Mean-noise radius in {distribution!r} must be finite and non-negative"
        )
    return radius


def _gaussian_shrinkage(distribution: str) -> float | None:
    """Return the declared spherical-shrinkage fraction for a Gaussian variant."""

    if distribution == "gaussian":
        return None
    if distribution == "gaussian_empirical":
        return 0.0
    match = _GAUSSIAN_SHRINKAGE_RE.fullmatch(distribution)
    if match is None:
        return None
    amount = float(match.group("amount"))
    # ``s05`` is the concise artifact spelling for 5%, while decimal spellings
    # such as ``s0.05`` remain unambiguous.
    if "." not in match.group("amount") and amount > 1:
        amount /= 100
    if not 0 <= amount <= 1:
        raise ReportInputError(
            f"Gaussian shrinkage in {distribution!r} must lie in [0, 1]"
        )
    return amount


def _distribution_family(distribution: str) -> str | None:
    if distribution == "true":
        return "true"
    if distribution == "projected_true":
        return "projected_true"
    if distribution in {"gaussian", "gaussian_empirical"}:
        return "gaussian"
    if _GAUSSIAN_SHRINKAGE_RE.fullmatch(distribution):
        _gaussian_shrinkage(distribution)
        return "gaussian"
    if distribution == "mean" or _mean_radius(distribution) is not None:
        return "mean"
    return None


def _validate_distribution(distribution: Any, context: str) -> str:
    if not isinstance(distribution, str):
        raise ReportInputError(f"{context}.train_distribution must be a string")
    if _distribution_family(distribution) is not None:
        return distribution
    raise ReportInputError(
        f"{context} has unsupported train_distribution {distribution!r}"
    )


def _coverage_details(source: Mapping[str, Any], context: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "total": None,
        "within_class": None,
        "between_class": None,
        "basis": "not recorded",
    }
    direct = source.get("held_out_explained_variance_fraction")
    if direct is not None:
        value = _number(direct, f"{context}.held_out_explained_variance_fraction")
        if not 0 <= value <= 1:
            raise ReportInputError(f"{context} held-out PCA coverage must be in [0, 1]")
        details["total"] = value
        details["basis"] = "held-out activations"

    held_out = source.get("held_out_coverage")
    if held_out is not None:
        if not isinstance(held_out, dict):
            raise ReportInputError(f"{context}.held_out_coverage must be an object")
        for source_key, target_key in (
            ("total_variance_fraction", "total"),
            ("explained_variance_fraction", "total"),
            ("within_class_variance_fraction", "within_class"),
            ("between_class_mean_variance_fraction", "between_class"),
        ):
            if held_out.get(source_key) is None:
                continue
            value = _number(
                held_out[source_key], f"{context}.held_out_coverage.{source_key}"
            )
            if not 0 <= value <= 1.0001:
                raise ReportInputError(
                    f"{context} held-out PCA coverage must be in [0, 1]"
                )
            if details[target_key] is None:
                details[target_key] = min(value, 1.0)
        details["basis"] = "held-out activations"
        return details

    if details["total"] is not None:
        return details

    legacy = source.get("explained_variance_fraction")
    if legacy is not None:
        value = _number(legacy, f"{context}.explained_variance_fraction")
        if not 0 <= value <= 1:
            raise ReportInputError(f"{context} PCA coverage must be in [0, 1]")
        details["total"] = value
        details["basis"] = "legacy reported value; evaluation population unspecified"
    return details


def _coverage(source: Mapping[str, Any], context: str) -> tuple[float | None, str]:
    """Backward-compatible total-coverage accessor."""

    details = _coverage_details(source, context)
    return details["total"], str(details["basis"])


def _normalise_records(
    records: Any,
    *,
    context: str,
    forced_distribution: str | None = None,
    default_draw: int | None = None,
    reject_nontrue_evaluation: bool = False,
) -> list[dict[str, Any]]:
    rows = _objects(records, context)
    normalised: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        evaluation = _validate_distribution(
            row.get("eval_distribution"), f"{row_context}.eval_distribution"
        )
        if evaluation != "true" and reject_nontrue_evaluation:
            raise ReportInputError(
                f"{row_context}.eval_distribution must be 'true'"
            )
        if forced_distribution is not None:
            if row.get("train_distribution") != forced_distribution:
                raise ReportInputError(
                    f"{row_context}.train_distribution must be "
                    f"{forced_distribution!r}"
                )
            distribution = forced_distribution
        else:
            distribution = _validate_distribution(
                row.get("train_distribution"), row_context
            )
        if row.get("draw") is None and default_draw is not None:
            draw = default_draw
        else:
            draw = _integral_number(row.get("draw"), f"{row_context}.draw")
        relax_epoch = _integral_number(
            row.get("relax_epoch"), f"{row_context}.relax_epoch"
        )
        if draw < 0 or relax_epoch < 0:
            raise ReportInputError(f"{row_context} draw and relax_epoch must be >= 0")
        accuracy = _number(row.get("accuracy"), f"{row_context}.accuracy")
        if not 0 <= accuracy <= 1:
            raise ReportInputError(f"{row_context}.accuracy must be in [0, 1]")
        loss = row.get("loss")
        if loss is not None:
            loss = _number(loss, f"{row_context}.loss")
            if loss < 0:
                raise ReportInputError(f"{row_context}.loss must be non-negative")
        initial_training_loss = row.get("initial_training_loss")
        if initial_training_loss is not None:
            initial_training_loss = _number(
                initial_training_loss, f"{row_context}.initial_training_loss"
            )
            if initial_training_loss < 0:
                raise ReportInputError(
                    f"{row_context}.initial_training_loss must be non-negative"
                )
        initial_gradient_norm = row.get("initial_gradient_norm")
        if initial_gradient_norm is not None:
            initial_gradient_norm = _number(
                initial_gradient_norm, f"{row_context}.initial_gradient_norm"
            )
            if initial_gradient_norm < 0:
                raise ReportInputError(
                    f"{row_context}.initial_gradient_norm must be non-negative"
                )
        initial_gradient_rms = row.get("initial_gradient_rms")
        if initial_gradient_rms is not None:
            initial_gradient_rms = _number(
                initial_gradient_rms, f"{row_context}.initial_gradient_rms"
            )
            if initial_gradient_rms < 0:
                raise ReportInputError(
                    f"{row_context}.initial_gradient_rms must be non-negative"
                )
        suffix_parameter_count = row.get("initial_suffix_parameter_count")
        if suffix_parameter_count is not None:
            suffix_parameter_count = _integral_number(
                suffix_parameter_count,
                f"{row_context}.initial_suffix_parameter_count",
            )
            if suffix_parameter_count < 1:
                raise ReportInputError(
                    f"{row_context}.initial_suffix_parameter_count must be positive"
                )
        suffix_weight_norm = row.get(
            "initial_suffix_weight_norm", row.get("initial_weight_norm")
        )
        if suffix_weight_norm is not None:
            suffix_weight_norm = _number(
                suffix_weight_norm, f"{row_context}.initial_suffix_weight_norm"
            )
            if suffix_weight_norm < 0:
                raise ReportInputError(
                    f"{row_context}.initial_suffix_weight_norm must be non-negative"
                )
        update_to_weight_ratio = row.get(
            "first_step_update_to_weight_ratio",
            row.get("initial_update_to_weight_ratio"),
        )
        if update_to_weight_ratio is not None:
            update_to_weight_ratio = _number(
                update_to_weight_ratio,
                f"{row_context}.initial_update_to_weight_ratio",
            )
            if update_to_weight_ratio < 0:
                raise ReportInputError(
                    f"{row_context}.initial_update_to_weight_ratio must be non-negative"
                )
        lr_regime = next(
            (
                row[key]
                for key in (
                    "learning_rate_regime",
                    "lr_regime",
                    "gradient_scale_regime",
                )
                if isinstance(row.get(key), str) and row[key].strip()
            ),
            "fixed_lr",
        )
        suffix_initialization = row.get("suffix_initialization", "warm")
        if suffix_initialization not in {"warm", "reinitialized"}:
            raise ReportInputError(
                f"{row_context}.suffix_initialization must be "
                "'warm' or 'reinitialized'"
            )
        if suffix_initialization == "reinitialized":
            lr_regime = f"reinitialized_{lr_regime}"
        key = (distribution, evaluation, draw, relax_epoch, lr_regime)
        if key in seen:
            raise ReportInputError(
                f"{context} contains duplicate evaluation record {key}"
            )
        seen.add(key)
        normalised.append(
            {
                "distribution": distribution,
                "eval_distribution": evaluation,
                "draw": draw,
                "relax_epoch": relax_epoch,
                "accuracy": accuracy,
                "loss": loss,
                "initial_training_loss": initial_training_loss,
                "initial_gradient_norm": initial_gradient_norm,
                "initial_gradient_rms": initial_gradient_rms,
                "initial_suffix_parameter_count": suffix_parameter_count,
                "initial_suffix_weight_norm": suffix_weight_norm,
                "initial_update_to_weight_ratio": update_to_weight_ratio,
                "lr_regime": lr_regime,
            }
        )
    if not normalised:
        raise ReportInputError(f"{context} has no evaluation records")
    return normalised


def _resnet_rank(payload: Mapping[str, Any], slice_: Mapping[str, Any]) -> int:
    if slice_.get("pca_rank") is not None:
        return int(_number(slice_["pca_rank"], "ResNet slice pca_rank"))
    config = payload["config"]
    requested = _require_int(config, "pca_components", "ResNet config")
    fit_count = _require_int(config, "pca_fit_size", "ResNet config")
    shape = slice_.get("representation_shape")
    if not isinstance(shape, list) or not shape:
        raise ReportInputError("ResNet slice.representation_shape must be non-empty")
    native_dimension = 1
    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ReportInputError(
                f"ResNet representation_shape[{index}] must be a positive integer"
            )
        native_dimension *= dimension
    return min(requested, fit_count - 1, native_dimension)


def _legacy_cnn_cell_rows(
    artifact: LoadedArtifact,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = artifact.payload
    config = payload["config"]
    cut = _require_int(config, "cut", f"artifact {artifact.id!r}.config")
    shape = payload.get("representation_shape")
    if not isinstance(shape, list) or not shape:
        raise ReportInputError(
            f"Artifact {artifact.id!r}.representation_shape must be non-empty"
        )
    native_dimension = 1
    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ReportInputError(
                f"Artifact {artifact.id!r}.representation_shape[{index}] "
                "must be a positive integer"
            )
        native_dimension *= dimension
    requested_rank = _require_int(
        config, "pca_components", f"artifact {artifact.id!r}.config"
    )
    fit_count = _require_int(
        config, "pca_fit_size", f"artifact {artifact.id!r}.config"
    )
    rank = min(requested_rank, fit_count - 1, native_dimension)
    coverage, coverage_basis = _coverage(payload, f"artifact {artifact.id!r}")
    records = _normalise_records(
        payload.get("records"),
        context=f"artifact {artifact.id!r}.records",
        default_draw=0,
    )
    _validate_cell_series(
        records,
        f"artifact {artifact.id!r}, cut {cut}",
        require_projected=False,
    )
    module = f"residual block {cut}"
    slice_like = {
        "representation_shape": shape,
        "native_dimension": native_dimension,
        "pca_fit_count": fit_count,
    }
    cell = _cell_metadata(
        artifact,
        slice_like,
        payload,
        cut=cut,
        module=module,
        rank=rank,
        coverage=coverage,
        coverage_basis=coverage_basis,
    )
    observations = _decorate_records(
        artifact,
        records,
        cut=cut,
        module=module,
        rank=rank,
        coverage=coverage,
        coverage_basis=coverage_basis,
    )
    return observations, [cell]


def _cell_rows(
    artifact: LoadedArtifact,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if artifact.format == "legacy_cnn_suffix_statistics":
        return _legacy_cnn_cell_rows(artifact)

    payload = artifact.payload
    slices = _objects(payload.get("slices"), f"artifact {artifact.id!r}.slices")
    observations: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    seen_cells: set[tuple[int, int]] = set()

    for slice_index, slice_ in enumerate(slices):
        slice_context = f"artifact {artifact.id!r}.slices[{slice_index}]"
        cut = _require_int(slice_, "cut", slice_context)
        nested_rank_schema = artifact.kind == "cnn" or (
            artifact.kind == "resnet" and payload.get("schema_version") in {3, 4}
        )
        if nested_rank_schema:
            if artifact.kind == "cnn":
                module = f"residual block {cut}"
            else:
                if slice_.get("condition") not in (None, "native"):
                    raise ReportInputError(
                        f"{slice_context}.condition must be 'native' when present"
                    )
                module = _require_string(slice_, "module", slice_context)
            reference = _normalise_records(
                slice_.get("reference_records"),
                context=f"{slice_context}.reference_records",
                forced_distribution="true",
                reject_nontrue_evaluation=(
                    artifact.kind == "resnet"
                    and payload.get("schema_version") == 3
                ),
            )
            rank_results = _objects(
                slice_.get("rank_results"), f"{slice_context}.rank_results"
            )
            for rank_index, result in enumerate(rank_results):
                rank_context = f"{slice_context}.rank_results[{rank_index}]"
                rank = _require_int(result, "pca_rank", rank_context)
                key = (cut, rank)
                if key in seen_cells:
                    raise ReportInputError(
                        f"Artifact {artifact.id!r} has duplicate cut/rank cell {key}"
                    )
                seen_cells.add(key)
                coverage, coverage_basis = _coverage(result, rank_context)
                result_records = _normalise_records(
                    result.get("records"),
                    context=f"{rank_context}.records",
                    reject_nontrue_evaluation=(
                        artifact.kind == "resnet"
                        and payload.get("schema_version") == 3
                    ),
                )
                schema = payload.get("schema_version")
                rank_local_reference = (
                    artifact.kind == "resnet" and schema == 4
                ) or (
                    artifact.kind == "cnn" and schema == 2
                )
                if not rank_local_reference and any(
                    row["distribution"] == "true" for row in result_records
                ):
                    raise ReportInputError(
                        f"{rank_context}.records must not repeat the shared true "
                        "reference"
                    )
                records = result_records if rank_local_reference else [
                    *reference,
                    *result_records,
                ]
                if rank_local_reference and any(
                    row.get("loss") is None for row in records
                ):
                    raise ReportInputError(
                        f"{rank_context}.records must record cross-entropy loss "
                        "for every canonical evaluation cell"
                    )
                _validate_cell_series(
                    records, f"artifact {artifact.id!r}, cut {cut}, PCA rank {rank}",
                    require_projected=True,
                    expected_draws=(
                        _require_int(
                            payload["config"],
                            "surrogate_draws",
                            f"artifact {artifact.id!r}.config",
                        )
                        if (
                            artifact.kind == "resnet"
                            and payload.get("schema_version") in {3, 4}
                        )
                        or (artifact.kind == "cnn" and schema == 2)
                        else None
                    ),
                    expected_relax_epochs=(
                        _require_int(
                            payload["config"],
                            "relax_epochs",
                            f"artifact {artifact.id!r}.config",
                        )
                        if (
                            artifact.kind == "resnet"
                            and payload.get("schema_version") in {3, 4}
                        )
                        or (artifact.kind == "cnn" and schema == 2)
                        else None
                    ),
                )
                if (
                    artifact.kind == "resnet" and schema == 4
                ) or (
                    artifact.kind == "cnn"
                    and schema == 2
                    and payload["config"].get("true_eval_only") is not True
                ):
                    _validate_full_train_eval_matrix(
                        records,
                        f"artifact {artifact.id!r}, cut {cut}, PCA rank {rank}",
                    )
                cells.append(
                    _cell_metadata(
                        artifact,
                        slice_,
                        result,
                        cut=cut,
                        module=module,
                        rank=rank,
                        coverage=coverage,
                        coverage_basis=coverage_basis,
                    )
                )
                observations.extend(
                    _decorate_records(
                        artifact,
                        records,
                        cut=cut,
                        module=module,
                        rank=rank,
                        coverage=coverage,
                        coverage_basis=coverage_basis,
                    )
                )
        else:
            if slice_.get("condition") not in (None, "native"):
                raise ReportInputError(
                    f"{slice_context}.condition must be 'native' when present"
                )
            module = _require_string(slice_, "module", slice_context)
            rank = _resnet_rank(payload, slice_)
            key = (cut, rank)
            if key in seen_cells:
                raise ReportInputError(
                    f"Artifact {artifact.id!r} has duplicate cut/rank cell {key}"
                )
            seen_cells.add(key)
            coverage, coverage_basis = _coverage(slice_, slice_context)
            records = _normalise_records(
                slice_.get("records"), context=f"{slice_context}.records"
            )
            _validate_cell_series(
                records, f"artifact {artifact.id!r}, cut {cut}", require_projected=False
            )
            cells.append(
                _cell_metadata(
                    artifact,
                    slice_,
                    slice_,
                    cut=cut,
                    module=module,
                    rank=rank,
                    coverage=coverage,
                    coverage_basis=coverage_basis,
                )
            )
            observations.extend(
                _decorate_records(
                    artifact,
                    records,
                    cut=cut,
                    module=module,
                    rank=rank,
                    coverage=coverage,
                    coverage_basis=coverage_basis,
                )
            )
    return observations, cells


def _validate_cell_series(
    records: Sequence[Mapping[str, Any]],
    context: str,
    *,
    require_projected: bool,
    expected_draws: int | None = None,
    expected_relax_epochs: int | None = None,
) -> None:
    by_distribution: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row["eval_distribution"] != "true":
            continue
        by_distribution[str(row["distribution"])].append(row)
    required = {"true"}
    if require_projected:
        required.add("projected_true")
    missing = required - by_distribution.keys()
    if missing:
        raise ReportInputError(
            f"{context} is missing required distributions: {', '.join(sorted(missing))}"
        )
    if not any(
        _distribution_family(distribution) == "gaussian"
        for distribution in by_distribution
    ):
        raise ReportInputError(f"{context} is missing a Gaussian distribution")
    if not any(_mean_radius(name) is not None for name in by_distribution):
        raise ReportInputError(f"{context} has no mean/noise condition")

    coordinate_sets = {
        distribution: {
            (
                int(row["draw"]),
                int(row["relax_epoch"]),
                str(row["lr_regime"]),
            )
            for row in rows
        }
        for distribution, rows in by_distribution.items()
    }
    expected_coordinates = next(iter(coordinate_sets.values()))
    for distribution, coordinates in coordinate_sets.items():
        if coordinates != expected_coordinates:
            raise ReportInputError(
                f"{context} has mismatched draw/relaxation grids for "
                f"{distribution!r}"
            )
    if expected_draws is not None and expected_relax_epochs is not None:
        declared_coordinates = {
            (draw, epoch, "fixed_lr")
            for draw in range(expected_draws)
            for epoch in range(expected_relax_epochs + 1)
        }
        if expected_coordinates != declared_coordinates:
            raise ReportInputError(
                f"{context} does not match the declared draw/relaxation grid"
            )


def _validate_full_train_eval_matrix(
    records: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    """Require a complete train × evaluation grid for every draw and epoch."""

    train_distributions = {str(row["distribution"]) for row in records}
    eval_distributions = {str(row["eval_distribution"]) for row in records}
    if train_distributions != eval_distributions:
        raise ReportInputError(
            f"{context} train/evaluation distribution registries do not match"
        )
    coordinates = {
        (
            int(row["draw"]),
            int(row["relax_epoch"]),
            str(row["lr_regime"]),
        )
        for row in records
    }
    observed = {
        (
            int(row["draw"]),
            int(row["relax_epoch"]),
            str(row["lr_regime"]),
            str(row["distribution"]),
            str(row["eval_distribution"]),
        )
        for row in records
    }
    expected = {
        (*coordinate, train_distribution, eval_distribution)
        for coordinate in coordinates
        for train_distribution in train_distributions
        for eval_distribution in eval_distributions
    }
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise ReportInputError(
            f"{context} has an incomplete train/evaluation matrix "
            f"({missing} missing, {extra} unexpected cells)"
        )


def _cell_metadata(
    artifact: LoadedArtifact,
    slice_: Mapping[str, Any],
    rank_source: Mapping[str, Any],
    *,
    cut: int,
    module: str,
    rank: int,
    coverage: float | None,
    coverage_basis: str,
) -> dict[str, Any]:
    coverage_details = _coverage_details(
        rank_source,
        f"artifact {artifact.id!r}, cut {cut}, PCA rank {rank}",
    )
    diagnostics: list[dict[str, Any]] = []
    raw_diagnostics = rank_source.get("moment_diagnostics")
    if isinstance(raw_diagnostics, list):
        for index, diagnostic in enumerate(raw_diagnostics):
            if not isinstance(diagnostic, dict):
                raise ReportInputError(
                    f"artifact {artifact.id!r} moment_diagnostics[{index}] "
                    "must be an object"
                )
            item: dict[str, Any] = {
                "draw": diagnostic.get("draw"),
                "distribution": diagnostic.get("distribution"),
                "diagnostic_space": diagnostic.get("diagnostic_space"),
            }
            for key in (
                "class_mean_relative_error",
                "class_covariance_relative_error",
                "support_outlier_fraction",
                "range_violation_fraction",
            ):
                value = diagnostic.get(key)
                if value is None:
                    item[key] = None
                    continue
                number = _number(
                    value,
                    f"artifact {artifact.id!r}.moment_diagnostics[{index}].{key}",
                )
                if number < 0:
                    raise ReportInputError(
                        f"artifact {artifact.id!r}.moment_diagnostics[{index}]."
                        f"{key} must be non-negative"
                    )
                item[key] = number
            diagnostics.append(item)
    step_zero_projection = None
    raw_step_zero = rank_source.get("step_zero_true_vs_projected")
    if isinstance(raw_step_zero, dict):
        step_zero_projection = {}
        for key in (
            "true_loss",
            "projected_true_loss",
            "true_accuracy",
            "projected_true_accuracy",
            "true_to_projected_predictive_kl",
        ):
            value = raw_step_zero.get(key)
            if value is None:
                continue
            number = _number(
                value,
                f"artifact {artifact.id!r}.step_zero_true_vs_projected.{key}",
            )
            if number < 0:
                raise ReportInputError(
                    f"artifact {artifact.id!r}.step_zero_true_vs_projected."
                    f"{key} must be non-negative"
                )
            if "accuracy" in key and number > 1:
                raise ReportInputError(
                    f"artifact {artifact.id!r}.step_zero_true_vs_projected."
                    f"{key} must be in [0, 1]"
                )
            step_zero_projection[key] = number
    covariance_estimators = None
    raw_estimators = rank_source.get("gaussian_covariance_estimators")
    if isinstance(raw_estimators, list):
        covariance_estimators = []
        for index, estimator in enumerate(raw_estimators):
            if not isinstance(estimator, dict):
                raise ReportInputError(
                    f"artifact {artifact.id!r}.gaussian_covariance_estimators"
                    f"[{index}] must be an object"
                )
            distribution = _validate_distribution(
                estimator.get("distribution"),
                f"artifact {artifact.id!r}.gaussian_covariance_estimators[{index}]",
            )
            if _distribution_family(distribution) != "gaussian":
                raise ReportInputError(
                    f"artifact {artifact.id!r}.gaussian_covariance_estimators"
                    f"[{index}] must name a Gaussian distribution"
                )
            covariance_estimators.append(dict(estimator))
    radii = sorted(
        {
            radius
            for row in rank_source.get("records", [])
            if isinstance(row, dict)
            for radius in [_mean_radius(str(row.get("train_distribution", "")))]
            if radius is not None
        }
    )
    pca_fit_count = slice_.get("pca_fit_count")
    if pca_fit_count is None:
        pca_fit_count = slice_.get("pca_basis_fit_count")
    if pca_fit_count is None:
        pca_fit_count = rank_source.get("pca_basis_fit_count")
    if pca_fit_count is None:
        pca_fit_count = artifact.payload["config"].get("pca_fit_size")
    moment_fit_count = rank_source.get(
        "moment_fit_count", slice_.get("moment_fit_count")
    )
    nested_surrogate_grid = (
        (
            artifact.kind == "resnet"
            and artifact.payload.get("schema_version") in {3, 4}
        )
        or (
            artifact.kind == "cnn"
            and artifact.payload.get("schema_version") == 2
        )
    )
    configured_ranks = artifact.payload["config"].get("pca_ranks")
    return {
        "input_id": artifact.id,
        "kind": artifact.kind,
        "label": artifact.label,
        "checkpoint_epoch": artifact.checkpoint_epoch,
        "model_seed": artifact.model_seed,
        "cut": cut,
        "module": module,
        "pca_rank": rank,
        "coverage": coverage,
        "coverage_basis": coverage_basis,
        "coverage_total": coverage_details["total"],
        "coverage_within_class": coverage_details["within_class"],
        "coverage_between_class": coverage_details["between_class"],
        "representation_shape": slice_.get("representation_shape"),
        "native_dimension": slice_.get("native_dimension"),
        "pca_fit_count": pca_fit_count,
        "moment_fit_count": moment_fit_count,
        "mean_noise_radii": radii,
        "pooled_within_class_variance_per_pca_coordinate": rank_source.get(
            "pooled_within_class_variance_per_pca_coordinate"
        ),
        "trace_matched_isotropic_covariance_trace": rank_source.get(
            "trace_matched_isotropic_covariance_trace"
        ),
        "covariance_shrinkage": rank_source.get("covariance_shrinkage"),
        "gaussian_covariance_estimators": covariance_estimators,
        "empirical_class_covariance_rank_ceiling": rank_source.get(
            "empirical_class_covariance_rank_ceiling"
        ),
        "rank_exceeds_empirical_class_covariance_ceiling": rank_source.get(
            "rank_exceeds_empirical_class_covariance_ceiling"
        ),
        "maximal_pca_basis_sha256": rank_source.get(
            "maximal_pca_basis_sha256"
        ),
        "pca_basis_prefix_sha256": rank_source.get("pca_basis_prefix_sha256"),
        "paired_nested_noise": nested_surrogate_grid
        and isinstance(configured_ranks, list)
        and len(configured_ranks) > 1,
        "fixed_analysis_bank": (
            artifact.kind == "resnet"
            and artifact.payload.get("schema_version") in {3, 4}
        )
        and isinstance(artifact.payload.get("analysis_banks"), dict),
        "moment_diagnostics": diagnostics,
        "step_zero_true_vs_projected": step_zero_projection,
        "artifact_format": artifact.format,
        "artifact_schema": artifact.payload.get("schema_version"),
    }


def _decorate_records(
    artifact: LoadedArtifact,
    records: Iterable[Mapping[str, Any]],
    *,
    cut: int,
    module: str,
    rank: int,
    coverage: float | None,
    coverage_basis: str,
) -> list[dict[str, Any]]:
    return [
        {
            "input_id": artifact.id,
            "kind": artifact.kind,
            "label": artifact.label,
            "checkpoint_epoch": artifact.checkpoint_epoch,
            "model_seed": artifact.model_seed,
            "cut": cut,
            "module": module,
            "pca_rank": rank,
            "coverage": coverage,
            "coverage_basis": coverage_basis,
            "artifact_format": artifact.format,
            "artifact_schema": artifact.payload.get("schema_version"),
            **row,
        }
        for row in records
    ]


def normalise_artifacts(
    artifacts: Sequence[LoadedArtifact],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_observations, artifact_cells = _cell_rows(artifact)
        observations.extend(artifact_observations)
        cells.extend(artifact_cells)
    return observations, cells


_SUMMARY_KEYS = (
    "input_id",
    "kind",
    "label",
    "checkpoint_epoch",
    "model_seed",
    "cut",
    "module",
    "pca_rank",
    "coverage",
    "coverage_basis",
    "distribution",
    "eval_distribution",
    "relax_epoch",
    "lr_regime",
)


def summarise_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[tuple(row[key] for key in _SUMMARY_KEYS)].append(row)
    result: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        summary = dict(zip(_SUMMARY_KEYS, key))
        accuracy_values = [float(row["accuracy"]) for row in rows]
        loss_values = [
            float(row["loss"]) for row in rows if row.get("loss") is not None
        ]
        initial_training_losses = [
            float(row["initial_training_loss"])
            for row in rows
            if row.get("initial_training_loss") is not None
        ]
        initial_gradient_norms = [
            float(row["initial_gradient_norm"])
            for row in rows
            if row.get("initial_gradient_norm") is not None
        ]
        summary.update(
            {
                "accuracy_mean": fmean(accuracy_values),
                "accuracy_min": min(accuracy_values),
                "accuracy_max": max(accuracy_values),
                "loss_mean": fmean(loss_values) if loss_values else None,
                "loss_min": min(loss_values) if loss_values else None,
                "loss_max": max(loss_values) if loss_values else None,
                "initial_training_loss_mean": (
                    fmean(initial_training_losses)
                    if initial_training_losses
                    else None
                ),
                "initial_gradient_norm_mean": (
                    fmean(initial_gradient_norms)
                    if initial_gradient_norms
                    else None
                ),
                "draw_count": len(rows),
            }
        )
        result.append(summary)
    return sorted(
        result,
        key=lambda row: (
            0 if row["kind"] == "cnn" else 1,
            row["checkpoint_epoch"],
            row["cut"],
            row["pca_rank"],
            _distribution_order(str(row["distribution"])),
            _distribution_order(str(row["eval_distribution"])),
            row["relax_epoch"],
            row["lr_regime"],
        ),
    )


_PAIR_KEYS = (
    "input_id",
    "kind",
    "label",
    "checkpoint_epoch",
    "model_seed",
    "cut",
    "module",
    "pca_rank",
    "lr_regime",
    "draw",
    "relax_epoch",
)


def paired_true_eval_contrasts(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the primary true-evaluation contrast before aggregating draws."""

    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    starts: dict[tuple[Any, ...], float] = {}
    for row in observations:
        if row["eval_distribution"] != "true":
            continue
        key = tuple(row[field] for field in _PAIR_KEYS)
        grouped[key][str(row["distribution"])] = row
        if int(row["relax_epoch"]) == 0:
            start_key = tuple(row[field] for field in _PAIR_KEYS[:-1]) + (
                str(row["distribution"]),
            )
            starts[start_key] = float(row["accuracy"])

    contrasts: list[dict[str, Any]] = []
    for key, by_distribution in grouped.items():
        reference = by_distribution.get("true")
        if reference is None:
            continue
        for distribution, row in by_distribution.items():
            if distribution == "true":
                continue
            loss_excess = None
            if row.get("loss") is not None and reference.get("loss") is not None:
                loss_excess = float(row["loss"]) - float(reference["loss"])
            start_key = key[:-1] + (distribution,)
            start_accuracy = starts.get(start_key)
            contrast = {field: value for field, value in zip(_PAIR_KEYS, key)}
            contrast.update(
                {
                    "distribution": distribution,
                    "loss": row.get("loss"),
                    "reference_loss": reference.get("loss"),
                    "loss_excess": loss_excess,
                    "accuracy": float(row["accuracy"]),
                    "reference_accuracy": float(reference["accuracy"]),
                    "accuracy_shortfall": (
                        float(reference["accuracy"]) - float(row["accuracy"])
                    ),
                    "accuracy_change": (
                        None
                        if start_accuracy is None
                        else float(row["accuracy"]) - start_accuracy
                    ),
                }
            )
            contrasts.append(contrast)
    return contrasts


def summarise_paired_contrasts(
    contrasts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = _PAIR_KEYS[:-2] + ("relax_epoch", "distribution")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in contrasts:
        grouped[tuple(row[key] for key in keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        summary = dict(zip(keys, key))
        for metric in ("loss_excess", "accuracy_shortfall", "accuracy_change"):
            values = [
                float(row[metric]) for row in rows if row.get(metric) is not None
            ]
            summary[f"{metric}_mean"] = fmean(values) if values else None
            summary[f"{metric}_min"] = min(values) if values else None
            summary[f"{metric}_max"] = max(values) if values else None
        summary["draw_count"] = len(rows)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            0 if row["kind"] == "cnn" else 1,
            row["checkpoint_epoch"],
            row["cut"],
            row["pca_rank"],
            row["relax_epoch"],
            _distribution_order(str(row["distribution"])),
        ),
    )


def _distribution_order(distribution: str) -> tuple[int, float, str]:
    if distribution == "true":
        return (0, 0.0, distribution)
    if distribution == "projected_true":
        return (1, 0.0, distribution)
    if _distribution_family(distribution) == "gaussian":
        shrinkage = _gaussian_shrinkage(distribution)
        return (2, shrinkage if shrinkage is not None else 0.5, distribution)
    radius = _mean_radius(distribution)
    return (3, radius if radius is not None else 0.0, distribution)


def _distribution_style(distribution: str) -> tuple[str, str, str]:
    if distribution == "true":
        return "Real data", REAL, ""
    if distribution == "projected_true":
        return "Projected real", PROJECTED_REAL, "7 5"
    if distribution == "gaussian":
        return "Gaussian", GAUSSIAN, ""
    if distribution == "gaussian_empirical":
        return "Gaussian · empirical covariance", GAUSSIAN, ""
    shrinkage = _gaussian_shrinkage(distribution)
    if shrinkage is not None:
        return (
            f"Gaussian · {100 * shrinkage:g}% spherical shrinkage",
            GAUSSIAN,
            "9 4",
        )
    radius = _mean_radius(distribution)
    if distribution == "mean":
        return "Mean + isotropic noise", MEAN, ""
    if radius == 0:
        return "Class mean (r = 0)", MEAN, "2 4"
    if radius == 1:
        return "Mean + isotropic noise (r = 1)", MEAN, ""
    return f"Mean + isotropic noise (r = {radius:g})", MEAN, "10 4 2 4"


def _primary_distributions(distributions: Iterable[str]) -> list[str]:
    available = set(distributions)
    result = [name for name in ("true", "projected_true") if name in available]
    result.extend(
        sorted(
            (
                name
                for name in available
                if _distribution_family(name) == "gaussian"
            ),
            key=_distribution_order,
        )
    )
    for preferred in ("mean", "mean_r1"):
        if preferred in available:
            result.append(preferred)
            break
    else:
        means = sorted(
            (name for name in available if _mean_radius(name) is not None),
            key=_distribution_order,
        )
        if means:
            result.append(means[0])
    return result


def _fmt_percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _network_inset(kind: str, cut: int) -> str:
    count = 4 if kind == "cnn" else 8
    selected = cut if kind == "cnn" else cut + 1
    box_width = 25 if count == 4 else 14
    gap = 4
    start_x = 535
    y = 16
    blocks = []
    for index in range(count):
        x = start_x + index * (box_width + gap)
        fill = "#dedbd4" if index < selected else "#dceaf3"
        blocks.append(
            f'<rect x="{x}" y="{y}" width="{box_width}" height="18" rx="3" '
            f'fill="{fill}" stroke="#777" stroke-width="0.8"/>'
        )
    cut_x = start_x + selected * (box_width + gap) - gap / 2
    label = "4-block CNN" if kind == "cnn" else "ResNet-18 blocks"
    return (
        f'<g aria-label="{_esc(label)} cut after block {selected}">'
        f'<text x="{start_x}" y="11" class="inset-label">{_esc(label)}</text>'
        + "".join(blocks)
        + f'<line x1="{cut_x}" y1="12" x2="{cut_x}" y2="39" '
        'stroke="#7a2b25" stroke-width="2"/>'
        f'<text x="{cut_x + 3}" y="49" class="cut-label">cut</text></g>'
    )


def _line_chart_svg(
    rows: Sequence[Mapping[str, Any]], *, kind: str, cut: int, checkpoint_epoch: int
) -> str:
    distributions = _primary_distributions(
        str(row["distribution"]) for row in rows
    )
    selected = [row for row in rows if row["distribution"] in distributions]
    epochs = sorted({int(row["relax_epoch"]) for row in selected})
    if not epochs:
        raise ReportInputError("Cannot render a trajectory without relaxation epochs")

    width, height = 760, 410
    left, right, top, bottom = 64, 728, 58, 310
    x_min, x_max = min(epochs), max(epochs)

    def x_position(epoch: int) -> float:
        if x_max == x_min:
            return (left + right) / 2
        return left + (epoch - x_min) * (right - left) / (x_max - x_min)

    def y_position(accuracy: float) -> float:
        return bottom - accuracy * (bottom - top)

    parts = [
        f'<svg class="trajectory" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Suffix relaxation accuracy trajectories">',
        '<rect width="760" height="410" fill="#fffdf8"/>',
        f'<text x="{left}" y="25" class="chart-title">'
        f'Prefix from checkpoint at epoch: {checkpoint_epoch}</text>',
        _network_inset(kind, cut),
    ]
    for percent in (0, 25, 50, 75, 100):
        y = y_position(percent / 100)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                'stroke="#e7e2d8" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'class="tick">{percent}</text>',
            ]
        )
    shown_epochs = epochs
    if len(epochs) > 7:
        stride = math.ceil(len(epochs) / 7)
        shown_epochs = epochs[::stride]
        if epochs[-1] not in shown_epochs:
            shown_epochs.append(epochs[-1])
    for epoch in shown_epochs:
        x = x_position(epoch)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" '
                f'y2="{bottom + 5}" stroke="#555"/>',
                f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" '
                f'class="tick">{epoch}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
            'stroke="#555" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            'stroke="#555" stroke-width="1.2"/>',
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 45}" '
            'text-anchor="middle" class="axis-label">Suffix relaxation epoch</text>',
            f'<text x="17" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            'transform="rotate(-90 17 184)" class="axis-label">'
            'Held-out real-data accuracy (%)</text>',
        ]
    )

    by_distribution: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_distribution[str(row["distribution"])].append(row)
    for distribution in distributions:
        series = sorted(
            by_distribution[distribution], key=lambda row: int(row["relax_epoch"])
        )
        label, colour, dash = _distribution_style(distribution)
        for row in series:
            if int(row["draw_count"]) > 1:
                x = x_position(int(row["relax_epoch"]))
                low = y_position(float(row["accuracy_min"]))
                high = y_position(float(row["accuracy_max"]))
                parts.append(
                    f'<line x1="{x:.2f}" y1="{low:.2f}" x2="{x:.2f}" '
                    f'y2="{high:.2f}" stroke="{colour}" stroke-opacity="0.35" '
                    'stroke-width="5"/>'
                )
        points = " ".join(
            f'{x_position(int(row["relax_epoch"])):.2f},'
            f'{y_position(float(row["accuracy_mean"])):.2f}'
            for row in series
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="3"{dash_attribute}/>'
        )
        for row in series:
            x = x_position(int(row["relax_epoch"]))
            y = y_position(float(row["accuracy_mean"]))
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" '
                f'fill="#fffdf8" stroke="{colour}" stroke-width="2"/>'
            )

    legend_y = 377
    for index, distribution in enumerate(distributions):
        label, colour, dash = _distribution_style(distribution)
        x = 65 + (index % 2) * 335
        y = legend_y + (index // 2) * 21
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 27}" y2="{y}" '
                f'stroke="{colour}" stroke-width="3"{dash_attribute}/>',
                f'<text x="{x + 35}" y="{y + 4}" class="legend-label">'
                f'{_esc(label)}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _endpoint_bars(rows: Sequence[Mapping[str, Any]]) -> str:
    distributions = _primary_distributions(str(row["distribution"]) for row in rows)
    endpoints = []
    for distribution in distributions:
        series = [row for row in rows if row["distribution"] == distribution]
        endpoint = max(series, key=lambda row: int(row["relax_epoch"]))
        endpoints.append(endpoint)
    endpoints.sort(key=lambda row: float(row["accuracy_mean"]), reverse=True)
    bars = []
    for row in endpoints:
        distribution = str(row["distribution"])
        label, colour, _dash = _distribution_style(distribution)
        value = 100 * float(row["accuracy_mean"])
        uncertainty = ""
        if int(row["draw_count"]) > 1:
            uncertainty = (
                f' <span class="range">('
                f'{100 * float(row["accuracy_min"]):.1f}–'
                f'{100 * float(row["accuracy_max"]):.1f})</span>'
            )
        bars.append(
            '<div class="endpoint-row">'
            f'<div class="endpoint-label">{_esc(label)}</div>'
            '<div class="bar-track">'
            f'<span class="bar-fill" style="width:{value:.3f}%;'
            f'background:{colour}"></span></div>'
            f'<div class="endpoint-value">{value:.1f}%{uncertainty}</div>'
            "</div>"
        )
    return (
        '<div class="endpoint-extraction">'
        '<div class="arrow" aria-hidden="true">→</div>'
        '<div class="endpoint-panel"><h5>Final relaxation epoch</h5>'
        '<p class="micro">Independent bars, sorted by measured accuracy.</p>'
        + "".join(bars)
        + "</div></div>"
    )


def _architecture_label(kind: str) -> str:
    return "Four-block CNN" if kind == "cnn" else "ResNet-18"


def _render_model_section(
    kind: str,
    artifacts: Sequence[LoadedArtifact],
    summaries: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    section_id: str | None = None,
    eyebrow: str = "Measured experiment",
    heading_prefix: str = "",
) -> str:
    architecture = _architecture_label(kind)
    run_html = []
    kind_artifacts = sorted(
        (artifact for artifact in artifacts if artifact.kind == kind),
        key=lambda artifact: artifact.checkpoint_epoch,
    )
    if not kind_artifacts:
        return ""
    legacy_groups: dict[tuple[int, int], list[LoadedArtifact]] = defaultdict(list)
    run_groups: list[list[LoadedArtifact]] = []
    for artifact in kind_artifacts:
        if artifact.format == "legacy_cnn_suffix_statistics":
            legacy_groups[(artifact.checkpoint_epoch, artifact.model_seed)].append(
                artifact
            )
        else:
            run_groups.append([artifact])
    run_groups.extend(legacy_groups.values())
    run_groups.sort(
        key=lambda group: (
            group[0].checkpoint_epoch,
            group[0].model_seed,
            group[0].id,
        )
    )
    initially_open = next(
        (
            index
            for index, group in enumerate(run_groups)
            if group[0].checkpoint_epoch == 1
        ),
        0,
    )

    for run_index, run_artifacts in enumerate(run_groups):
        artifact_ids = {artifact.id for artifact in run_artifacts}
        checkpoint_epoch = run_artifacts[0].checkpoint_epoch
        model_seed = run_artifacts[0].model_seed
        run_rows = [row for row in summaries if row["input_id"] in artifact_ids]
        run_cells = [cell for cell in cells if cell["input_id"] in artifact_ids]
        cuts = sorted({int(cell["cut"]) for cell in run_cells})
        ranks = sorted({int(cell["pca_rank"]) for cell in run_cells})
        draw_count = max(int(row["draw_count"]) for row in run_rows)
        if len(run_artifacts) > 1:
            cell_keys = [
                (int(cell["cut"]), int(cell["pca_rank"])) for cell in run_cells
            ]
            if len(cell_keys) != len(set(cell_keys)):
                raise ReportInputError(
                    f"Legacy CNN epoch {checkpoint_epoch}, seed {model_seed} "
                    "has duplicate cut/rank cells"
                )
        rank_note = (
            f"The charts use the largest completed PCA rank ({max(ranks)} here) "
            "for each cut. Smaller completed ranks and all noise radii remain in "
            "the robustness table and embedded data."
            if len(ranks) > 1
            else f"The charts use the recorded PCA rank ({ranks[0]} here). "
            "All exact trajectory values remain in the table and embedded data."
        )
        chart_cards = []
        for cut in cuts:
            cut_cells = [cell for cell in run_cells if int(cell["cut"]) == cut]
            primary_rank = max(int(cell["pca_rank"]) for cell in cut_cells)
            cell = next(
                cell
                for cell in cut_cells
                if int(cell["pca_rank"]) == primary_rank
            )
            chart_rows = [
                row
                for row in run_rows
                if int(row["cut"]) == cut and int(row["pca_rank"]) == primary_rank
                and row["eval_distribution"] == "true"
                and row["lr_regime"] == "fixed_lr"
            ]
            coverage_note = (
                f'PCA rank {primary_rank}; coverage '
                f'{_fmt_percent(cell["coverage"])} '
                f'({_esc(cell["coverage_basis"])}).'
            )
            chart_cards.append(
                '<figure class="evidence-card">'
                f'<figcaption><strong>{_esc(architecture)} · '
                f'{_esc(cell["module"])}</strong><br>'
                f'<span>{coverage_note}</span></figcaption>'
                '<div class="evidence-pair">'
                + _line_chart_svg(
                    chart_rows,
                    kind=kind,
                    cut=cut,
                    checkpoint_epoch=checkpoint_epoch,
                )
                + _endpoint_bars(chart_rows)
                + "</div></figure>"
            )
        open_attribute = " open" if run_index == initially_open else ""
        run_label = (
            run_artifacts[0].label
            if len(run_artifacts) == 1
            else f"{architecture} · nominal epoch {checkpoint_epoch} · legacy cut grid"
        )
        run_html.append(
            f'<details class="run"{open_attribute}>'
            f'<summary><span>{_esc(run_label)}</span>'
            f'<span class="summary-meta">epoch {checkpoint_epoch} · '
            f'seed {model_seed} · {len(cuts)} cuts · '
            f'{draw_count} draw{"s" if draw_count != 1 else ""}</span></summary>'
            '<div class="run-intro">'
            f"<p>{_esc(rank_note)}</p>"
            "</div>"
            + "".join(chart_cards)
            + "</details>"
        )
    has_legacy_cnn_grid = kind == "cnn" and any(
        artifact.format == "legacy_cnn_suffix_statistics"
        for artifact in kind_artifacts
    )
    has_legacy_cnn_schema = kind == "cnn" and any(
        artifact.format == "post_statistics"
        and artifact.payload.get("schema_version") != 2
        for artifact in kind_artifacts
    )
    has_canonical_resnet = kind == "resnet" and any(
        artifact.payload.get("schema_version") == 4
        for artifact in kind_artifacts
    )
    has_legacy_resnet = kind == "resnet" and any(
        artifact.payload.get("schema_version") in {1, 2, 3}
        for artifact in kind_artifacts
    )
    intro = (
        "True-data relaxation is the reference. New-format cells include "
        "projected-real replay as an empirical in-subspace control at the PCA boundary; "
        "Gaussian and mean-based replay test which retained distributional "
        "structure helps the suffix relearn."
        if kind == "cnn"
        else (
            "The canonical ResNet control uses fingerprinted fixed analysis banks, "
            "nested PCA ranks, paired noise, projected-real replay, and held-out "
            "real evaluation. Older flat sweeps remain visible as legacy context."
            if has_canonical_resnet
            else
            "These legacy ResNet sweeps use the same true-evaluation target. Their "
            "reported PCA coverage is labelled conservatively unless the artifact "
            "contains an explicit held-out coverage field."
        )
    )
    legacy_notes = []
    if has_legacy_cnn_grid:
        legacy_notes.append(
        '<aside class="evidence-note"><strong>Legacy CNN grid.</strong> The prefixes '
        "at different nominal epochs came from separately trained and scheduled "
        "models, with different scheduler horizons and checkpoint hashes. Each "
        "epoch × cut suffix-relaxation cell was also run separately. Adjacent "
        "nominal epochs are therefore neither one prefix-training trajectory nor "
        "one continued suffix-training trajectory. PCA coverage in these files is "
        "a legacy fit-bank value, not held-out coverage.</aside>"
        )
    if has_legacy_cnn_schema:
        legacy_notes.append(
            '<aside class="evidence-note"><strong>Earlier CNN schema.</strong> '
            "This artifact predates the full train × evaluation loss matrix and "
            "paired optimizer diagnostics required by the canonical schema."
            "</aside>"
        )
    if has_legacy_resnet:
        legacy_notes.append(
            '<aside class="evidence-note"><strong>Legacy ResNet sweeps.</strong> '
            "Schema-1/2 inputs do not record the stronger bank and lineage "
            "controls. Schema 3 adds a Fixed analysis bank, Paired rank comparison, "
            "Trace-matched isotropic noise, and Full-bank class moments, but it "
            "still records true-only accuracy rather than the primary loss matrix. "
            "The canonical ResNet control uses fingerprinted fixed analysis banks "
            "only when those controls accompany the schema-4 loss matrix."
            "</aside>"
        )
    return (
        f'<section id="{_esc(section_id or kind)}" class="section">'
        f'<div class="section-heading"><p class="eyebrow">{_esc(eyebrow)}</p>'
        f'<h2>{_esc(heading_prefix + architecture)}</h2><p>{_esc(intro)}</p></div>'
        + "".join(legacy_notes)
        + "".join(run_html)
        + "</section>"
    )


def _is_legacy_artifact(artifact: LoadedArtifact) -> bool:
    return (
        artifact.format == "legacy_cnn_suffix_statistics"
        or (
            artifact.kind == "cnn"
            and artifact.payload.get("schema_version") != 2
        )
        or (
            artifact.kind == "resnet"
            and artifact.payload.get("schema_version") != 4
        )
    )


def _endpoint_paired_rows(
    paired_summaries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_summaries:
        key = (
            row["input_id"],
            row["cut"],
            row["pca_rank"],
            row["lr_regime"],
            row["distribution"],
        )
        groups[key].append(row)
    return [
        max(rows, key=lambda row: int(row["relax_epoch"]))
        for rows in groups.values()
    ]


def _metric_with_range(row: Mapping[str, Any], metric: str, digits: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    if mean is None:
        return "—"
    text = f"{float(mean):.{digits}f}"
    if int(row.get("draw_count", 1)) > 1:
        low = float(row[f"{metric}_min"])
        high = float(row[f"{metric}_max"])
        text += f" ({low:.{digits}f}–{high:.{digits}f})"
    return text


def _percentage_point_with_range(
    row: Mapping[str, Any],
    metric: str,
) -> str:
    mean = row.get(f"{metric}_mean")
    if mean is None:
        return "—"
    text = f"{100 * float(mean):.2f}"
    if int(row.get("draw_count", 1)) > 1:
        low = 100 * float(row[f"{metric}_min"])
        high = 100 * float(row[f"{metric}_max"])
        text += f" ({low:.2f}–{high:.2f})"
    return text + " pp"


def _primary_estimand_table(
    paired_summaries: Sequence[Mapping[str, Any]],
) -> str:
    endpoints = [
        row
        for row in _endpoint_paired_rows(paired_summaries)
        if row.get("loss_excess_mean") is not None
    ]
    if not endpoints:
        return (
            '<aside class="status-panel diagnostic"><strong>PRIMARY LOSS '
            "ESTIMAND NOT RECORDED.</strong> These inputs can show accuracy "
            "trajectories, but cannot support the declared paired cross-entropy "
            "contrast.</aside>"
        )
    rows = []
    for row in sorted(
        endpoints,
        key=lambda item: (
            0 if item["kind"] == "cnn" else 1,
            item["checkpoint_epoch"],
            item["cut"],
            item["pca_rank"],
            _distribution_order(str(item["distribution"])),
        ),
    ):
        label, _colour, _dash = _distribution_style(str(row["distribution"]))
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(row["kind"])))}</td>'
            f'<td>{row["checkpoint_epoch"]}</td>'
            f'<td>{row["model_seed"]}</td>'
            f'<td>{_esc(row["module"])}</td>'
            f'<td>{row["pca_rank"]}</td>'
            f"<td>{_esc(label)}</td>"
            f'<td class="number">{_metric_with_range(row, "loss_excess")}</td>'
            f'<td class="number">{_percentage_point_with_range(row, "accuracy_shortfall")}</td>'
            f'<td>{row["draw_count"]} paired draw'
            f'{"s" if int(row["draw_count"]) != 1 else ""}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Model</th><th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Replay distribution</th><th>Excess loss (nats/example)</th>"
        "<th>Accuracy shortfall (percentage points)</th><th>Uncertainty unit</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _matrix_cards(observations: Sequence[Mapping[str, Any]]) -> str:
    cell_fields = (
        "input_id",
        "kind",
        "label",
        "checkpoint_epoch",
        "model_seed",
        "cut",
        "module",
        "pca_rank",
        "lr_regime",
    )
    max_rank: dict[tuple[str, int], int] = defaultdict(int)
    for row in observations:
        max_rank[(str(row["input_id"]), int(row["cut"]))] = max(
            max_rank[(str(row["input_id"]), int(row["cut"]))],
            int(row["pca_rank"]),
        )
    cells: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("loss") is None:
            continue
        if int(row["pca_rank"]) != max_rank[
            (str(row["input_id"]), int(row["cut"]))
        ]:
            continue
        cells[tuple(row[field] for field in cell_fields)].append(row)

    cards: list[str] = []
    for key, rows in sorted(
        cells.items(),
        key=lambda item: (
            0 if item[0][1] == "cnn" else 1,
            item[0][3],
            item[0][5],
            item[0][7],
        ),
    ):
        endpoint_rows: dict[tuple[int, str, str], Mapping[str, Any]] = {}
        for row in rows:
            endpoint_key = (
                int(row["draw"]),
                str(row["distribution"]),
                str(row["eval_distribution"]),
            )
            previous = endpoint_rows.get(endpoint_key)
            if previous is None or int(row["relax_epoch"]) > int(
                previous["relax_epoch"]
            ):
                endpoint_rows[endpoint_key] = row
        train_distributions = sorted(
            {item[1] for item in endpoint_rows}, key=_distribution_order
        )
        eval_distributions = sorted(
            {item[2] for item in endpoint_rows}, key=_distribution_order
        )
        if len(train_distributions) < 2 or len(eval_distributions) < 2:
            continue
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (_draw, train_distribution, eval_distribution), row in endpoint_rows.items():
            values[(train_distribution, eval_distribution)].append(float(row["loss"]))
        required = {
            (train_distribution, eval_distribution)
            for train_distribution in train_distributions
            for eval_distribution in eval_distributions
        }
        if not required.issubset(values):
            continue
        header = "".join(
            f"<th>{_esc(_distribution_style(name)[0])}</th>"
            for name in train_distributions
        )
        body_rows = []
        for eval_distribution in eval_distributions:
            diagonal = values.get((eval_distribution, eval_distribution))
            diagonal_mean = fmean(diagonal) if diagonal else None
            cells_html = []
            for train_distribution in train_distributions:
                mean_loss = fmean(values[(train_distribution, eval_distribution)])
                delta = (
                    ""
                    if diagonal_mean is None
                    else f'<span class="matrix-delta">Δ {mean_loss - diagonal_mean:+.4f}</span>'
                )
                cells_html.append(
                    f'<td class="number"><strong>{mean_loss:.4f}</strong>{delta}</td>'
                )
            body_rows.append(
                "<tr>"
                f"<th>{_esc(_distribution_style(eval_distribution)[0])}</th>"
                + "".join(cells_html)
                + "</tr>"
            )
        metadata = dict(zip(cell_fields, key))
        cards.append(
            '<details class="matrix-card">'
            "<summary><span>"
            f'{_esc(_architecture_label(str(metadata["kind"])))} · '
            f'epoch {metadata["checkpoint_epoch"]} · seed {metadata["model_seed"]} · '
            f'{_esc(metadata["module"])}'
            "</span><span class=\"summary-meta\">"
            f'rank {metadata["pca_rank"]} · {_esc(metadata["lr_regime"])}</span>'
            "</summary><div class=\"matrix-body\">"
            '<p class="micro">Columns are relaxation distributions; rows are '
            "evaluation distributions. Each cell is endpoint cross-entropy. "
            "Δ subtracts the matched diagonal for that evaluation distribution.</p>"
            '<div class="table-scroll"><table class="matrix-table"><thead><tr>'
            "<th>Evaluate ↓ / relax →</th>"
            + header
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table></div></div></details>"
        )
    if not cards:
        return (
            '<aside class="status-panel diagnostic"><strong>FULL MATRIX NOT '
            "RECORDED.</strong> This build contains true-evaluation-only or "
            "incomplete legacy rows. No cross-distribution conclusion is shown.</aside>"
        )
    cards[0] = cards[0].replace('<details class="matrix-card">', '<details class="matrix-card" open>', 1)
    return "".join(cards)


def _rank_outcome_table(
    paired_summaries: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> str:
    endpoints = _endpoint_paired_rows(paired_summaries)
    rank_groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    for cell in cells:
        rank_groups[(str(cell["input_id"]), int(cell["cut"]))].add(
            int(cell["pca_rank"])
        )
    multi = {key for key, ranks in rank_groups.items() if len(ranks) > 1}
    if not multi:
        return '<p class="empty-control">No measured multi-rank outcome sweep in this build.</p>'
    cell_lookup = {
        (str(cell["input_id"]), int(cell["cut"]), int(cell["pca_rank"])): cell
        for cell in cells
    }
    rows = []
    for row in sorted(
        (
            row
            for row in endpoints
            if (str(row["input_id"]), int(row["cut"])) in multi
            and _distribution_family(str(row["distribution"]))
            in {"projected_true", "gaussian"}
        ),
        key=lambda item: (
            item["input_id"],
            item["cut"],
            item["pca_rank"],
            _distribution_order(str(item["distribution"])),
        ),
    ):
        cell = cell_lookup[
            (str(row["input_id"]), int(row["cut"]), int(row["pca_rank"]))
        ]
        label, _colour, _dash = _distribution_style(str(row["distribution"]))
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(row["kind"])))}</td>'
            f'<td>{row["checkpoint_epoch"]}</td><td>{row["model_seed"]}</td>'
            f'<td>{_esc(row["module"])}</td>'
            f'<td>{row["pca_rank"]}</td>'
            f'<td>{_fmt_percent(cell.get("coverage_within_class"))}</td>'
            f"<td>{_esc(label)}</td>"
            f'<td class="number">{_metric_with_range(row, "loss_excess")}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Model</th>'
        "<th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Within-class coverage</th><th>Replay</th><th>Excess loss</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _noise_outcome_table(
    paired_summaries: Sequence[Mapping[str, Any]],
) -> str:
    endpoints = _endpoint_paired_rows(paired_summaries)
    groups: dict[tuple[str, int, int], set[float]] = defaultdict(set)
    for row in endpoints:
        radius = _mean_radius(str(row["distribution"]))
        if radius is not None:
            groups[
                (str(row["input_id"]), int(row["cut"]), int(row["pca_rank"]))
            ].add(radius)
    multi = {key for key, radii in groups.items() if len(radii) > 1}
    if not multi:
        return '<p class="empty-control">No measured multi-radius outcome sweep in this build.</p>'
    rows = []
    for row in sorted(
        (
            row
            for row in endpoints
            if (
                str(row["input_id"]),
                int(row["cut"]),
                int(row["pca_rank"]),
            )
            in multi
            and _mean_radius(str(row["distribution"])) is not None
        ),
        key=lambda item: (
            item["input_id"],
            item["cut"],
            item["pca_rank"],
            float(_mean_radius(str(item["distribution"])) or 0),
        ),
    ):
        radius = _mean_radius(str(row["distribution"]))
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(row["kind"])))}</td>'
            f'<td>{row["checkpoint_epoch"]}</td><td>{row["model_seed"]}</td>'
            f'<td>{_esc(row["module"])}</td>'
            f'<td>{row["pca_rank"]}</td><td>{radius:g}</td>'
            f'<td class="number">{_metric_with_range(row, "loss_excess")}</td>'
            f'<td class="number">{_percentage_point_with_range(row, "accuracy_shortfall")}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Model</th>'
        "<th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Noise radius</th>"
        "<th>Excess loss</th><th>Accuracy shortfall (percentage points)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _moment_diagnostic_table(cells: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for cell in cells:
        for diagnostic in cell.get("moment_diagnostics", []):
            distribution = diagnostic.get("distribution")
            if not isinstance(distribution, str):
                continue
            label, _colour, _dash = _distribution_style(distribution)
            mean_error = diagnostic.get("class_mean_relative_error")
            covariance_error = diagnostic.get("class_covariance_relative_error")
            rows.append(
                "<tr>"
                f'<td>{_esc(_architecture_label(str(cell["kind"])))}</td>'
                f'<td>{cell["checkpoint_epoch"]}</td><td>{cell["model_seed"]}</td>'
                f'<td>{_esc(cell["module"])}</td>'
                f'<td>{cell["pca_rank"]}</td><td>{_esc(label)}</td>'
                f'<td>{_esc(diagnostic.get("draw", "—"))}</td>'
                f'<td class="number">{("—" if mean_error is None else f"{float(mean_error):.4f}")}</td>'
                f'<td class="number">{("—" if covariance_error is None else f"{float(covariance_error):.4f}")}</td>'
                f'<td>{_esc(diagnostic.get("diagnostic_space") or "not recorded")}</td>'
                "</tr>"
            )
    if not rows:
        return '<p class="empty-control">No normalized moment-fidelity diagnostics in this build.</p>'
    return (
        '<div class="table-scroll"><table><thead><tr><th>Model</th>'
        "<th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Surrogate</th>"
        "<th>Draw</th><th>Class-mean relative error</th>"
        "<th>Class-covariance relative error</th><th>Diagnostic space</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _gradient_diagnostic_table(
    observations: Sequence[Mapping[str, Any]],
) -> str:
    groups: dict[
        tuple[Any, ...], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in observations:
        if (
            row["eval_distribution"] != "true"
            or int(row["relax_epoch"]) != 0
            or row.get("initial_gradient_norm") is None
        ):
            continue
        key = (
            row["input_id"],
            row["kind"],
            row["checkpoint_epoch"],
            row["model_seed"],
            row["cut"],
            row["module"],
            row["pca_rank"],
            row["lr_regime"],
            row["draw"],
        )
        groups[key][str(row["distribution"])] = row
    ratios: dict[
        tuple[Any, ...], list[tuple[float, float | None, float | None]]
    ] = defaultdict(list)
    for key, values in groups.items():
        reference = values.get("true")
        if reference is None or float(reference["initial_gradient_norm"]) <= 0:
            continue
        for distribution, row in values.items():
            if distribution == "true":
                continue
            total_ratio = float(row["initial_gradient_norm"]) / float(
                reference["initial_gradient_norm"]
            )
            rms_ratio = None
            if (
                row.get("initial_gradient_rms") is not None
                and reference.get("initial_gradient_rms") not in {None, 0}
            ):
                rms_ratio = float(row["initial_gradient_rms"]) / float(
                    reference["initial_gradient_rms"]
                )
            update_ratio = None
            if (
                row.get("initial_update_to_weight_ratio") is not None
                and reference.get("initial_update_to_weight_ratio") not in {None, 0}
            ):
                update_ratio = float(
                    row["initial_update_to_weight_ratio"]
                ) / float(reference["initial_update_to_weight_ratio"])
            ratios[key[:-1] + (distribution,)].append(
                (total_ratio, rms_ratio, update_ratio)
            )
    if not ratios:
        return '<p class="empty-control">No initial suffix-gradient norms in this build.</p>'
    rows = []
    for key, values in sorted(
        ratios.items(),
        key=lambda item: (
            0 if item[0][1] == "cnn" else 1,
            item[0][2],
            item[0][4],
            item[0][6],
            _distribution_order(str(item[0][-1])),
        ),
    ):
        (
            _input_id,
            kind,
            epoch,
            model_seed,
            _cut,
            module,
            rank,
            regime,
            distribution,
        ) = key
        label, _colour, _dash = _distribution_style(str(distribution))
        total_values = [value[0] for value in values]
        rms_values = [value[1] for value in values if value[1] is not None]
        update_values = [value[2] for value in values if value[2] is not None]
        row_html = (
            "<tr>"
            f'<td>{_esc(_architecture_label(str(kind)))}</td><td>{epoch}</td>'
            f"<td>{model_seed}</td>"
            f"<td>{_esc(module)}</td><td>{rank}</td><td>{_esc(label)}</td>"
            f'<td class="number">{fmean(total_values):.3f}×</td>'
        )
        row_html += (
            f'<td class="number">{fmean(rms_values):.3f}×</td>'
            if rms_values
            else '<td class="number">—</td>'
        )
        row_html += (
            f'<td class="number">{fmean(update_values):.3f}×</td>'
            if update_values
            else '<td class="number">—</td>'
        )
        row_html += (
            f"<td>{len(values)} paired draw"
            f'{"s" if len(values) != 1 else ""}</td><td>{_esc(regime)}</td>'
            "</tr>"
        )
        rows.append(row_html)
    return (
        '<div class="table-scroll"><table><thead><tr><th>Model</th>'
        "<th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Replay</th>"
        "<th>Total gradient / true</th><th>RMS gradient / true</th>"
        "<th>Update-to-weight / true</th><th>Uncertainty unit</th><th>LR regime</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _projection_diagnostic_table(
    cells: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for cell in cells:
        diagnostic = cell.get("step_zero_true_vs_projected")
        if not isinstance(diagnostic, dict):
            continue
        true_loss = diagnostic.get("true_loss")
        projected_loss = diagnostic.get("projected_true_loss")
        true_accuracy = diagnostic.get("true_accuracy")
        projected_accuracy = diagnostic.get("projected_true_accuracy")
        loss_delta = (
            float(projected_loss) - float(true_loss)
            if true_loss is not None and projected_loss is not None
            else None
        )
        accuracy_delta = (
            100 * (float(projected_accuracy) - float(true_accuracy))
            if true_accuracy is not None and projected_accuracy is not None
            else None
        )
        predictive_kl = diagnostic.get("true_to_projected_predictive_kl")
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(cell["kind"])))}</td>'
            f'<td>{cell["checkpoint_epoch"]}</td>'
            f'<td>{cell["model_seed"]}</td>'
            f'<td>{_esc(cell["module"])}</td><td>{cell["pca_rank"]}</td>'
            f'<td class="number">{("—" if loss_delta is None else f"{loss_delta:+.5f}")}</td>'
            f'<td class="number">{("—" if accuracy_delta is None else f"{accuracy_delta:+.2f} pp")}</td>'
            f'<td class="number">{("—" if predictive_kl is None else f"{float(predictive_kl):.5f}")}</td>'
            "</tr>"
        )
    if not rows:
        return (
            '<p class="empty-control">No step-zero true-versus-projected '
            "functional diagnostic in this build.</p>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Model</th>'
        "<th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Projected − true loss</th><th>Projected − true accuracy</th>"
        "<th>Predictive KL(true ∥ projected)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _completed_controls(
    summaries: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    controls: list[tuple[str, str]] = []
    distributions = {str(row["distribution"]) for row in summaries}
    if "projected_true" in distributions:
        controls.append(
            (
                "Projected-real replay",
                "Provides an empirical in-subspace baseline before the Gaussian approximation.",
            )
        )
    ranks_by_cut: dict[tuple[str, int], set[int]] = defaultdict(set)
    for cell in cells:
        ranks_by_cut[(str(cell["input_id"]), int(cell["cut"]))].add(
            int(cell["pca_rank"])
        )
    if any(len(ranks) > 1 for ranks in ranks_by_cut.values()):
        controls.append(
            (
                "PCA-rank sweep",
                "Uses nested leading components from one fitted maximal basis.",
            )
        )
    if any(cell.get("paired_nested_noise") is True for cell in cells):
        controls.append(
            (
                "Paired rank comparison",
                "Nested ranks reuse leading coordinates of the same sampled noise banks.",
            )
        )
    radii = {
        radius
        for distribution in distributions
        for radius in [_mean_radius(distribution)]
        if radius is not None
    }
    if len(radii) > 1:
        controls.append(
            (
                "Mean-noise radius sweep",
                "Compares exact centroids with trace-scaled isotropic noise.",
            )
        )
    if any(
        cell.get("pooled_within_class_variance_per_pca_coordinate") is not None
        and cell.get("trace_matched_isotropic_covariance_trace") is not None
        for cell in cells
    ):
        controls.append(
            (
                "Trace-matched isotropic noise",
                "Radius one matches the pooled within-class covariance trace in PCA space.",
            )
        )
    if any(
        isinstance(cell.get("moment_fit_count"), int)
        and isinstance(cell.get("pca_fit_count"), int)
        and cell["moment_fit_count"] > cell["pca_fit_count"]
        for cell in cells
    ):
        controls.append(
            (
                "Full-bank class moments",
                "PCA is fitted on its declared subset; class means and covariances use the full analysis bank.",
            )
        )
    if any(cell.get("fixed_analysis_bank") is True for cell in cells):
        controls.append(
            (
                "Fixed analysis bank",
                "Every ResNet cut and rank reuses the same fingerprinted input realizations.",
            )
        )
    if any(cell["coverage_basis"] == "held-out activations" for cell in cells):
        controls.append(
            (
                "Held-out PCA coverage",
                "Measures retained variance on evaluation activations rather than PCA-fit rows.",
            )
        )
    return controls


def _coverage_table(cells: Sequence[Mapping[str, Any]]) -> str:
    if not cells:
        return (
            '<p class="empty-control">No canonical PCA coverage metadata in '
            "this build.</p>"
        )
    rows = []
    for cell in sorted(
        cells,
        key=lambda row: (
            0 if row["kind"] == "cnn" else 1,
            row["checkpoint_epoch"],
            row["cut"],
            row["pca_rank"],
        ),
    ):
        radii = ", ".join(f"{radius:g}" for radius in cell["mean_noise_radii"]) or "1"
        pca_fit_count = cell.get("pca_fit_count")
        moment_fit_count = cell.get("moment_fit_count")
        shrinkage = cell.get("covariance_shrinkage")
        estimators = cell.get("gaussian_covariance_estimators")
        estimator_text = "—"
        if isinstance(estimators, list):
            estimator_text = ", ".join(
                _distribution_style(str(item["distribution"]))[0]
                for item in estimators
                if isinstance(item, dict)
                and isinstance(item.get("distribution"), str)
            ) or "—"
        isotropic_trace = cell.get("trace_matched_isotropic_covariance_trace")
        shrinkage_text = (
            _fmt_percent(float(shrinkage)) if shrinkage is not None else "—"
        )
        isotropic_trace_text = (
            f"{float(isotropic_trace):.4g}"
            if isotropic_trace is not None
            else "—"
        )
        rank_flag = ""
        if cell.get("rank_exceeds_empirical_class_covariance_ceiling") is True:
            ceiling = cell.get("empirical_class_covariance_rank_ceiling")
            rank_flag = (
                f'<span class="caution" title="Empirical per-class covariance rank '
                f'is at most {ceiling}."> †</span>'
            )
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(cell["kind"])))}</td>'
            f'<td>{cell["checkpoint_epoch"]}</td>'
            f'<td>{cell["model_seed"]}</td>'
            f'<td>{_esc(cell["module"])}</td>'
            f'<td>{cell["pca_rank"]}{rank_flag}</td>'
            f'<td>{_fmt_percent(cell.get("coverage_total"))}</td>'
            f'<td>{_fmt_percent(cell.get("coverage_within_class"))}</td>'
            f'<td>{_fmt_percent(cell.get("coverage_between_class"))}</td>'
            f'<td>{_esc(cell["coverage_basis"])}</td>'
            f'<td>{_esc(cell.get("native_dimension") or "—")}</td>'
            f'<td>{_esc(pca_fit_count if pca_fit_count is not None else "—")}</td>'
            f'<td>{_esc(moment_fit_count if moment_fit_count is not None else "—")}</td>'
            f"<td>{shrinkage_text}</td>"
            f"<td>{_esc(estimator_text)}</td>"
            f"<td>{isotropic_trace_text}</td>"
            f'<td>{_esc(radii)}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Model</th><th>Checkpoint epoch</th><th>Model seed</th>"
        "<th>Cut</th><th>PCA rank</th>"
        "<th>Total variance</th><th>Within-class variance</th>"
        "<th>Between-class mean variance</th><th>Coverage population</th>"
        "<th>Native coordinates</th>"
        "<th>PCA-fit n</th><th>Moment-fit n</th><th>Legacy cov. shrinkage</th>"
        "<th>Gaussian covariance estimators</th>"
        "<th>Isotropic trace</th><th>Mean-noise r</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _exact_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in summaries:
        label, _colour, _dash = _distribution_style(str(row["distribution"]))
        eval_label, _eval_colour, _eval_dash = _distribution_style(
            str(row["eval_distribution"])
        )
        loss_text = (
            f'{float(row["loss_mean"]):.5f}'
            if row.get("loss_mean") is not None
            else "—"
        )
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(row["kind"])))}</td>'
            f'<td>{row["checkpoint_epoch"]}</td>'
            f'<td>{row["model_seed"]}</td>'
            f'<td>{_esc(row["module"])}</td>'
            f'<td>{row["pca_rank"]}</td>'
            f'<td>{_esc(label)}</td>'
            f'<td>{_esc(eval_label)}</td>'
            f'<td>{row["relax_epoch"]}</td>'
            f"<td>{loss_text}</td>"
            f'<td>{100 * float(row["accuracy_mean"]):.3f}%</td>'
            f'<td>{100 * float(row["accuracy_min"]):.3f}%–'
            f'{100 * float(row["accuracy_max"]):.3f}%</td>'
            f'<td>{row["draw_count"]}</td>'
            f'<td>{_esc(row["lr_regime"])}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll exact-table"><table><thead><tr>'
        "<th>Model</th><th>Checkpoint</th><th>Model seed</th><th>Cut</th><th>Rank</th>"
        "<th>Training distribution</th><th>Evaluation distribution</th>"
        "<th>Relax epoch</th><th>Mean loss</th>"
        "<th>Mean accuracy</th><th>Draw range</th><th>n</th><th>LR regime</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _manifest_cards(artifacts: Sequence[LoadedArtifact]) -> str:
    cards = []
    for kind in ("cnn", "resnet"):
        group = [artifact for artifact in artifacts if artifact.kind == kind]
        if not group:
            cards.append(
                '<article class="input-card absent">'
                f'<p class="eyebrow">{_esc(_architecture_label(kind))}</p>'
                "<h3>Not loaded</h3><p>No measured artifact for this "
                "architecture is present in the manifest.</p></article>"
            )
            continue
        epochs = sorted({artifact.checkpoint_epoch for artifact in group})
        seeds = sorted({artifact.model_seed for artifact in group})
        draw_counts = sorted(
            {
                int(artifact.payload["config"].get("surrogate_draws", 1))
                for artifact in group
            }
        )
        draw_text = (
            str(draw_counts[0])
            if len(draw_counts) == 1
            else f"{draw_counts[0]}–{draw_counts[-1]}"
        )
        epoch_text = ", ".join(str(epoch) for epoch in epochs)
        seed_text = ", ".join(str(seed) for seed in seeds)
        cards.append(
            '<article class="input-card">'
            f'<p class="eyebrow">{_esc(_architecture_label(kind))}</p>'
            f'<h3>{len(group)} hashed input{"s" if len(group) != 1 else ""}</h3>'
            f'<dl><div><dt>Nominal epochs</dt><dd>{_esc(epoch_text)}</dd></div>'
            f'<div><dt>Model seeds</dt><dd>{_esc(seed_text)}</dd></div>'
            f'<div><dt>Draws per cell</dt><dd>{_esc(draw_text)}</dd></div>'
            f'<div><dt>Formats</dt><dd>{len({artifact.format for artifact in group})}'
            "</dd></div></dl></article>"
        )
    return "".join(cards)


def _provenance_details(artifact: LoadedArtifact) -> str:
    payload = artifact.payload
    details: list[tuple[str, str]] = [
        ("Artifact SHA-256", artifact.sha256),
        ("Artifact schema", str(payload.get("schema_version", "legacy"))),
        ("Device", str(payload.get("device", "not recorded"))),
    ]
    source = _source_identity(
        payload.get("provenance"),
        f"artifact {artifact.id!r}.provenance",
        required=False,
    )
    if source is not None:
        details.append((source[0].title(), source[1]))
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("sha256"), str):
        details.append(
            (
                "Checkpoint",
                f"epoch {checkpoint.get('epoch')} · sha256:{checkpoint['sha256']}",
            )
        )
    lineage = payload.get("lineage")
    if isinstance(lineage, dict):
        training_source = _source_identity(
            lineage.get("training_source"),
            f"artifact {artifact.id!r}.lineage.training_source",
            required=False,
        )
        if training_source is not None:
            details.append(
                (
                    f"Training {training_source[0]}",
                    training_source[1],
                )
            )
        training_manifest = lineage.get("training_manifest")
        if isinstance(training_manifest, dict) and isinstance(
            training_manifest.get("sha256"), str
        ):
            details.append(
                ("Training manifest SHA-256", training_manifest["sha256"])
            )
    dataset = payload.get("dataset")
    if isinstance(dataset, dict):
        details.append(("Dataset backend", str(dataset.get("backend", "not recorded"))))
    banks = payload.get("analysis_banks")
    if isinstance(banks, dict):
        for split in ("train", "test"):
            bank = banks.get(split)
            if isinstance(bank, dict) and isinstance(bank.get("sha256"), str):
                details.append(
                    (
                        f"{split.title()} analysis bank",
                        f"n={bank.get('count')} · seed={bank.get('seed')} · "
                        f"sha256:{bank['sha256']}",
                    )
                )
    return (
        "<dl>"
        + "".join(
            f"<div><dt>{_esc(label)}</dt><dd><code>{_esc(value)}</code></dd></div>"
            for label, value in details
        )
        + "</dl>"
    )


def _method_diagram() -> str:
    return """
<svg class="method-diagram" viewBox="0 0 980 240" role="img"
     aria-label="A class-conditioned input distribution is pushed through a network prefix,
     fitted in PCA space, replayed under four distributions, and used to train the suffix.">
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4"
            orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#777"/></marker>
  </defs>
  <g class="diagram-box">
    <rect x="15" y="50" width="130" height="130" rx="12"/>
    <text x="80" y="74" text-anchor="middle" class="diagram-title">One class</text>
    <text x="80" y="139" text-anchor="middle" class="digit">3</text>
    <text x="80" y="165" text-anchor="middle">x ∼ p(x | y = 3)</text>
  </g>
  <line x1="150" y1="115" x2="215" y2="115" class="diagram-arrow"/>
  <g class="diagram-box">
    <rect x="220" y="50" width="150" height="130" rx="12"/>
    <text x="295" y="76" text-anchor="middle" class="diagram-title">Network prefix</text>
    <rect x="242" y="101" width="25" height="30" rx="3"/>
    <rect x="273" y="101" width="25" height="30" rx="3"/>
    <rect x="304" y="101" width="25" height="30" rx="3"/>
    <line x1="336" y1="91" x2="336" y2="142" stroke="#7a2b25" stroke-width="3"/>
    <text x="336" y="156" text-anchor="middle">cut</text>
  </g>
  <line x1="375" y1="115" x2="440" y2="115" class="diagram-arrow"/>
  <g class="diagram-box">
    <rect x="445" y="30" width="185" height="170" rx="12"/>
    <text x="537" y="56" text-anchor="middle" class="diagram-title">Fit on activations</text>
    <circle cx="493" cy="102" r="5" fill="#333"/><circle cx="518" cy="84" r="5" fill="#333"/>
    <circle cx="545" cy="115" r="5" fill="#333"/><circle cx="566" cy="92" r="5" fill="#333"/>
    <circle cx="586" cy="126" r="5" fill="#333"/>
    <line x1="480" y1="145" x2="595" y2="73" stroke="#777" stroke-dasharray="6 4"/>
    <text x="537" y="177" text-anchor="middle">PCA + class moments</text>
  </g>
  <line x1="635" y1="115" x2="700" y2="115" class="diagram-arrow"/>
  <g class="diagram-box">
    <rect x="705" y="15" width="150" height="200" rx="12"/>
    <text x="780" y="42" text-anchor="middle" class="diagram-title">Replay</text>
    <line x1="728" y1="70" x2="760" y2="70" stroke="#222" stroke-width="4"/>
    <text x="769" y="74">real</text>
    <line x1="728" y1="104" x2="760" y2="104" stroke="#737373" stroke-width="4"
          stroke-dasharray="7 5"/><text x="769" y="108">projected</text>
    <line x1="728" y1="138" x2="760" y2="138" stroke="#0072B2" stroke-width="4"/>
    <text x="769" y="142">Gaussian</text>
    <line x1="728" y1="172" x2="760" y2="172" stroke="#D55E00" stroke-width="4"/>
    <text x="769" y="176">mean + noise</text>
  </g>
  <line x1="860" y1="115" x2="910" y2="115" class="diagram-arrow"/>
  <g class="diagram-box">
    <rect x="915" y="65" width="50" height="100" rx="10"/>
    <text x="940" y="98" text-anchor="middle" class="diagram-title">Train</text>
    <text x="940" y="122" text-anchor="middle">the</text>
    <text x="940" y="145" text-anchor="middle">suffix</text>
  </g>
</svg>
"""


def _style_sheet() -> str:
    return f"""
:root {{
  --paper: #f4f0e7;
  --card: #fffdf8;
  --ink: #222222;
  --muted: #655f56;
  --rule: #d9d2c5;
  --accent: #7a2b25;
  --real: {REAL};
  --projected: {PROJECTED_REAL};
  --gaussian: {GAUSSIAN};
  --mean: {MEAN};
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
               "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--paper);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; line-height: 1.55; background: var(--paper); }}
a {{ color: #155d85; text-underline-offset: 3px; }}
nav {{
  position: sticky; top: 0; z-index: 20; display: flex; gap: .25rem;
  justify-content: center; flex-wrap: wrap; padding: .55rem 1rem;
  background: rgba(255,253,248,.96); border-bottom: 1px solid var(--rule);
}}
nav a {{ color: var(--ink); text-decoration: none; padding: .4rem .7rem; border-radius: 999px; }}
nav a:hover, nav a:focus {{ background: #ece5d9; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 0 1.25rem 5rem; }}
.hero {{ padding: 5rem 0 3.5rem; max-width: 900px; }}
.eyebrow {{
  color: var(--accent); font-size: .75rem; font-weight: 750; letter-spacing: .12em;
  text-transform: uppercase; margin: 0 0 .45rem;
}}
h1 {{ font-family: Georgia, serif; font-size: clamp(2.35rem, 6vw, 5rem); line-height: 1.02;
      letter-spacing: -.035em; margin: 0 0 1.2rem; }}
h2 {{ font-family: Georgia, serif; font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 1.08; margin: 0 0 .7rem; }}
h3 {{ margin: .1rem 0 .65rem; }}
.lede {{ font-size: clamp(1.08rem, 2vw, 1.32rem); max-width: 760px; color: #413d37; }}
.status-line {{ display: flex; gap: .65rem; align-items: center; flex-wrap: wrap; margin-top: 1.5rem; }}
.badge {{ display: inline-block; border: 1px solid #7d786e; border-radius: 999px;
          padding: .25rem .65rem; font-size: .78rem; font-weight: 700; }}
.wip {{ background: #f6e6bd; border-color: #c49932; }}
.measured {{ background: #e3eee4; border-color: #719176; }}
.section {{ padding: 4rem 0 1rem; scroll-margin-top: 3.6rem; }}
.section-heading {{ max-width: 780px; margin-bottom: 1.8rem; }}
.question {{
  font-family: Georgia, serif; font-size: 1.23rem; line-height: 1.4;
  color: #37322c; margin: .8rem 0 0;
}}
.evidence-note {{
  background: #f7edda; border: 1px solid #caa969; border-left: 4px solid #9d6e19;
  border-radius: 8px; padding: .85rem 1rem; margin: 0 0 1rem;
}}
.status-panel {{
  background: #f8e9d6; border: 1px solid #c59d62; border-left: 5px solid var(--accent);
  border-radius: 10px; padding: 1rem 1.15rem; margin: 1rem 0 1.4rem;
}}
.status-panel.measured-result {{ background: #e9f1e8; border-color: #7b9b7e; }}
.status-panel.diagnostic {{ background: #f8e9d6; border-color: #c59d62; }}
.status-panel p {{ margin: .35rem 0 0; }}
.input-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 1rem; }}
.input-card, .method-card, .control-card, .provenance-card, .estimand-card {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  padding: 1.15rem; box-shadow: 0 2px 14px rgba(45,37,28,.04);
}}
.input-card.absent {{ color: var(--muted); background: #eeeae2; }}
dl {{ margin: .8rem 0 0; }}
dl div {{ display: flex; justify-content: space-between; gap: 1rem;
          padding: .35rem 0; border-top: 1px solid #ebe6dc; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; text-align: right; }}
.method-card {{ padding: 1.4rem; }}
.method-diagram {{ width: 100%; height: auto; margin: 1rem 0; }}
.diagram-box rect:first-child {{ fill: #fffdf8; stroke: #aaa297; }}
.diagram-box text {{ font-size: 13px; fill: #3f3b35; }}
.diagram-box .diagram-title {{ font-size: 13px; font-weight: 700; }}
.diagram-box .digit {{ font-family: Georgia, serif; font-size: 57px; fill: #222; }}
.diagram-arrow {{ stroke: #777; stroke-width: 2; marker-end: url(#arrowhead); }}
.method-notes {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(230px,1fr));
                 gap: 1rem; margin-top: 1rem; }}
.method-notes h3 {{ font-size: 1rem; }}
.method-notes p {{ color: var(--muted); margin: .3rem 0 0; }}
.equation-box {{
  margin: 1.2rem 0; padding: 1rem 1.2rem; border: 1px solid #bdb4a6;
  border-radius: 9px; background: #f2eee6; overflow-x: auto;
}}
.equation {{
  font-family: "STIX Two Text", Georgia, serif; font-size: 1.08rem;
  white-space: nowrap; font-variant-numeric: tabular-nums;
}}
.equation-box p {{ margin: .4rem 0 0; color: var(--muted); font-size: .88rem; }}
.criteria-list {{ margin: .4rem 0 0; padding-left: 1.1rem; color: var(--muted); }}
.criteria-list li {{ margin: .28rem 0; }}
.legend-strip {{ display: flex; flex-wrap: wrap; gap: .75rem 1.2rem; margin: 1rem 0; }}
.legend-item {{ display: flex; align-items: center; gap: .45rem; font-size: .9rem; }}
.legend-swatch {{ width: 30px; height: 4px; display: inline-block; }}
.legend-swatch.projected {{
  height: 3px; background: repeating-linear-gradient(90deg,var(--projected) 0 7px,
          transparent 7px 12px);
}}
details.run, details.data-details {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  margin: 1rem 0; overflow: clip;
}}
details > summary {{
  cursor: pointer; padding: 1rem 1.2rem; display: flex; justify-content: space-between;
  align-items: baseline; gap: 1rem; font-weight: 700;
}}
details > summary::marker {{ color: var(--accent); }}
.summary-meta {{ color: var(--muted); font-size: .83rem; font-weight: 500; text-align: right; }}
.run-intro {{ padding: 0 1.2rem; color: var(--muted); }}
.evidence-card {{ border-top: 1px solid var(--rule); margin: 0; padding: 1.2rem; }}
.evidence-card figcaption {{ margin-bottom: .75rem; }}
.evidence-card figcaption span {{ color: var(--muted); font-size: .88rem; }}
.evidence-pair {{ display: grid; grid-template-columns: minmax(0,2fr) minmax(250px,1fr);
                  gap: 1rem; align-items: center; }}
.trajectory {{ width: 100%; height: auto; border: 1px solid #ebe5da; border-radius: 8px; }}
.trajectory text {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: #3d3933; }}
.trajectory .chart-title {{ font-size: 15px; font-weight: 720; }}
.trajectory .tick {{ font-size: 11px; }}
.trajectory .axis-label {{ font-size: 12px; font-weight: 650; }}
.trajectory .legend-label {{ font-size: 11px; }}
.trajectory .inset-label, .trajectory .cut-label {{ font-size: 9px; }}
.endpoint-extraction {{ display: flex; align-items: center; gap: .8rem; }}
.arrow {{ color: var(--accent); font-size: 2rem; font-weight: 300; }}
.endpoint-panel {{ width: 100%; }}
.endpoint-panel h5 {{ margin: 0; font-size: .95rem; }}
.micro {{ margin: .15rem 0 .7rem; color: var(--muted); font-size: .78rem; }}
.endpoint-row {{ display: grid; grid-template-columns: 1fr; gap: .2rem; margin: .65rem 0; }}
.endpoint-label {{ font-size: .78rem; }}
.bar-track {{ height: 8px; background: #ebe6dc; border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; display: block; border-radius: 999px; }}
.endpoint-value {{ font-variant-numeric: tabular-nums; font-size: .78rem; font-weight: 700; }}
.range {{ color: var(--muted); font-weight: 400; }}
.control-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(230px,1fr));
                 gap: 1rem; margin: 1rem 0 2rem; }}
.control-card h3 {{ font-size: 1rem; }}
.control-card p {{ margin: .3rem 0 0; color: var(--muted); }}
.control-stack {{ display: grid; gap: 2rem; }}
.control-block {{ border-top: 1px solid var(--rule); padding-top: 1.2rem; }}
.control-block h3 {{ font-family: Georgia, serif; font-size: 1.35rem; }}
.control-block > p {{ color: var(--muted); max-width: 800px; }}
.empty-control {{
  color: var(--muted); border: 1px dashed #bdb4a6; border-radius: 8px;
  padding: .8rem 1rem; background: rgba(255,253,248,.5);
}}
.matrix-card {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  margin: 1rem 0; overflow: clip;
}}
.matrix-body {{ padding: 0 1.1rem 1.1rem; }}
.matrix-table th:first-child {{ background: #e7dfd3; }}
.matrix-table td {{ vertical-align: top; }}
.matrix-delta {{ display: block; color: var(--muted); font-size: .72rem; margin-top: .15rem; }}
.number {{ font-variant-numeric: tabular-nums; text-align: right; }}
.result-note {{
  color: var(--muted); max-width: 820px; border-left: 3px solid var(--rule);
  padding-left: .85rem; margin: 1rem 0;
}}
.legacy-divider {{
  border-top: 4px double #9c9489; margin-top: 4rem; padding-top: 1rem;
}}
.legacy-divider > .section-heading {{ margin-bottom: .8rem; }}
.table-scroll {{ overflow-x: auto; background: var(--card); border: 1px solid var(--rule);
                 border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: .82rem; }}
th, td {{ text-align: left; padding: .55rem .65rem; border-bottom: 1px solid #e8e2d8;
          white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #eee8dd; font-weight: 750; }}
tbody tr:hover {{ background: #faf6ee; }}
.caution {{ color: var(--accent); font-weight: 800; }}
.data-actions {{ display: flex; flex-wrap: wrap; gap: .7rem; margin: 1rem 0; }}
button {{
  appearance: none; border: 1px solid #58534b; background: var(--card); color: var(--ink);
  padding: .58rem .8rem; border-radius: 7px; cursor: pointer; font: inherit; font-weight: 650;
}}
button:hover, button:focus {{ background: #eee8dd; }}
.data-details > div {{ padding: 0 1rem 1rem; }}
.provenance-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
                    gap: 1rem; }}
code {{ font-size: .82em; overflow-wrap: anywhere; }}
.footer-note {{ color: var(--muted); font-size: .86rem; margin-top: 2rem; }}
@media (max-width: 820px) {{
  .evidence-pair {{ grid-template-columns: 1fr; }}
  .endpoint-extraction {{ padding: 0 .5rem .8rem; }}
  .arrow {{ transform: rotate(90deg); }}
  .method-diagram {{ min-width: 760px; }}
  .method-card {{ overflow-x: auto; }}
}}
@media (max-width: 700px) {{
  main {{ padding-inline: .8rem; }}
  .hero {{ padding-top: 3rem; }}
  nav {{ justify-content: flex-start; overflow-x: auto; flex-wrap: nowrap; }}
  nav a {{ white-space: nowrap; }}
  details > summary {{ display: block; }}
  .summary-meta {{ display: block; text-align: left; margin-top: .25rem; }}
}}
"""


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_report(
    manifest: Mapping[str, Any],
    artifacts: Sequence[LoadedArtifact],
) -> str:
    """Render validated artifacts as one self-contained HTML document."""

    observations, cells = normalise_artifacts(artifacts)
    summaries = summarise_observations(observations)
    paired_draws = paired_true_eval_contrasts(observations)
    paired_summaries = summarise_paired_contrasts(paired_draws)

    canonical_artifacts = [
        artifact for artifact in artifacts if not _is_legacy_artifact(artifact)
    ]
    legacy_artifacts = [
        artifact for artifact in artifacts if _is_legacy_artifact(artifact)
    ]
    canonical_ids = {artifact.id for artifact in canonical_artifacts}
    legacy_ids = {artifact.id for artifact in legacy_artifacts}
    canonical_observations = [
        row for row in observations if row["input_id"] in canonical_ids
    ]
    canonical_cells = [row for row in cells if row["input_id"] in canonical_ids]
    canonical_summaries = [
        row for row in summaries if row["input_id"] in canonical_ids
    ]
    canonical_paired = [
        row for row in paired_summaries if row["input_id"] in canonical_ids
    ]
    legacy_summaries = [row for row in summaries if row["input_id"] in legacy_ids]
    legacy_cells = [row for row in cells if row["input_id"] in legacy_ids]

    controls = _completed_controls(canonical_summaries, canonical_cells)
    control_html = "".join(
        f'<article class="control-card"><h3>{_esc(name)}</h3>'
        f'<p>{_esc(description)}</p></article>'
        for name, description in controls
    ) or (
        '<article class="control-card"><h3>No canonical control suite loaded</h3>'
        "<p>The measured legacy evidence remains below, but it is not promoted "
        "to a completed canonical control.</p></article>"
    )
    has_primary = any(
        row.get("loss_excess_mean") is not None for row in canonical_paired
    )
    status_html = (
        '<aside class="status-panel measured-result"><strong>PRIMARY ESTIMAND '
        "AVAILABLE.</strong><p>The canonical artifacts record paired, draw-level "
        "true-evaluation cross-entropy contrasts. Accuracy remains a secondary "
        "readout.</p></aside>"
        if has_primary
        else
        '<aside class="status-panel diagnostic"><strong>DIAGNOSTIC BUILD.</strong>'
        "<p>The loaded canonical artifacts do not yet record the paired "
        "cross-entropy estimand. Accuracy-only trajectories are descriptive and "
        "cannot settle the main claim.</p></aside>"
    )
    embedded = {
        "manifest": manifest,
        "inputs": [
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "format": artifact.format,
                "label": artifact.label,
                "path": artifact.relative_path,
                "sha256": artifact.sha256,
                "payload": artifact.payload,
            }
            for artifact in artifacts
        ],
        "normalised_evaluation_records": observations,
        "normalised_true_evaluation_records": [
            row for row in observations if row["eval_distribution"] == "true"
        ],
        "exact_summary_rows": summaries,
        "paired_true_evaluation_draw_contrasts": paired_draws,
        "paired_true_evaluation_summaries": paired_summaries,
        "cell_metadata": cells,
    }
    provenance_cards = "".join(
        '<article class="provenance-card">'
        f'<p class="eyebrow">{_esc(artifact.id)}</p>'
        f'<p><strong>{_esc(artifact.relative_path)}</strong></p>'
        f"{_provenance_details(artifact)}"
        "</article>"
        for artifact in artifacts
    )
    canonical_sections = "".join(
        _render_model_section(
            kind,
            canonical_artifacts,
            canonical_summaries,
            canonical_cells,
            section_id=kind,
            eyebrow="Canonical measured evidence",
        )
        for kind in ("cnn", "resnet")
    )
    legacy_sections = "".join(
        _render_model_section(
            kind,
            legacy_artifacts,
            legacy_summaries,
            legacy_cells,
            section_id=f"legacy-{kind}",
            eyebrow="Legacy measured appendix",
            heading_prefix="Legacy: ",
        )
        for kind in ("cnn", "resnet")
    )
    legacy_html = ""
    if legacy_artifacts:
        legacy_html = (
            '<section id="legacy" class="section legacy-divider">'
            '<div class="section-heading"><p class="eyebrow">Legacy appendix</p>'
            "<h2>Earlier measured evidence, kept in its lane</h2>"
            "<p>These hashed artifacts remain available for audit and historical "
            "context. They are excluded from the primary estimand and canonical "
            "control claims because they lack one or more required measurement or "
            "pairing fields.</p></div></section>"
            + legacy_sections
        )
    canonical_nav = "".join(
        f'<a href="#{kind}">{_esc(_architecture_label(kind))}</a>'
        for kind in ("cnn", "resnet")
        if any(artifact.kind == kind for artifact in canonical_artifacts)
    )
    legacy_nav = '<a href="#legacy">Legacy appendix</a>' if legacy_artifacts else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(TITLE)}</title>
  <style>{_style_sheet()}</style>
</head>
<body>
<nav aria-label="Report sections">
  <a href="#overview">Overview</a>
  <a href="#method">Methodology</a>
  <a href="#estimand">Primary estimand</a>
  <a href="#matrix">Full matrix</a>
  <a href="#controls">Controls</a>
  {canonical_nav}
  {legacy_nav}
  <a href="#data">Data &amp; provenance</a>
</nav>
<main>
  <header id="overview" class="hero">
    <p class="eyebrow">Research note · measured dashboard</p>
    <h1>{_esc(TITLE)}</h1>
    <p class="lede">How much of a trained suffix's relearning behaviour is explained
      by low-order, class-conditional activation statistics—and what fails when
      projection, covariance, or optimization is changed?</p>
    <div class="status-line">
      <span class="badge wip">Work in progress</span>
      <span class="badge measured">Measured inputs only</span>
      <span>{len(canonical_artifacts)} canonical · {len(legacy_artifacts)} legacy ·
        {len(artifacts)} hashed artifacts · dashboard source base
        <code>{_esc(manifest["source_commit"])}</code></span>
    </div>
  </header>

  <section class="section" aria-labelledby="evidence-heading">
    <div class="section-heading">
      <p class="eyebrow">Evidence in this build</p>
      <h2 id="evidence-heading">A narrow, auditable view</h2>
      <p class="question">The claim is credible only if the paired true-evaluation
        loss contrast survives the projection, rank, moment, noise, and optimizer
        controls below.</p>
    </div>
    {status_html}
    <div class="input-grid">{_manifest_cards(artifacts)}</div>
  </section>

  <section id="method" class="section">
    <div class="section-heading">
      <p class="eyebrow">Methodology and decision rule</p>
      <h2>Cut, fit, replay, relax—then test the right contrast</h2>
      <p>A checkpoint supplies one prefix and suffix. We push a labelled input
        distribution through the prefix, fit PCA and class-conditional moments at
        the cut, replay controlled distributions there, and train only the suffix.</p>
    </div>
    <div class="method-card">
      {_method_diagram()}
      <div class="legend-strip" aria-label="Distribution colour key">
        <span class="legend-item"><i class="legend-swatch" style="background:{REAL}"></i>Real data</span>
        <span class="legend-item"><i class="legend-swatch projected"></i>Projected real</span>
        <span class="legend-item"><i class="legend-swatch" style="background:{GAUSSIAN}"></i>Gaussian</span>
        <span class="legend-item"><i class="legend-swatch" style="background:{MEAN}"></i>Mean + isotropic noise</span>
      </div>
      <div class="method-notes">
        <div><h3>Experimental unit</h3><p>Checkpoint/model seed × cut × nested PCA
          rank × surrogate draw. Draws estimate surrogate-sampling variation; they
          are not independent model seeds.</p></div>
        <div><h3>Banks and splits</h3><p>PCA and class moments are fitted only on the
          declared analysis bank. Coverage and outcome evaluation must identify
          their held-out population and bank fingerprints.</p></div>
        <div><h3>Projected-real control</h3><p>Projects empirical activations into
          the fitted PCA subspace, giving an in-subspace empirical baseline before
          any Gaussian approximation. Because projection changes both replay and
          deployment inputs, it is not a pure reconstruction-loss measurement.</p></div>
        <div><h3>Gaussian registry</h3><p><code>gaussian_empirical</code> means the
          empirical PCA-space covariance. <code>gaussian_shrunk_s05</code> means
          5% spherical shrinkage. Plain <code>gaussian</code> remains a legacy
          label whose shrinkage must be read from metadata.</p></div>
        <div><h3>PCA is an approximation</h3><p>The dashboard records native
          coordinate count, fitted rank, total/within/between-class coverage, and
          the empirical per-class covariance ceiling. “All components” means all
          estimable components, at most min(native dimension, PCA-fit n − 1).
          Native coordinates are activation-map entries (channels × height ×
          width), not raw pixels or model parameters, so an early cut can exceed
          ten thousand coordinates.</p></div>
        <div><h3>Mean + isotropic noise</h3><p>This is not epsilon jitter. At
          r = 1 its covariance trace matches the pooled average within-class
          covariance trace; radius r scales trace by r².</p></div>
      </div>
      <div class="equation-box">
        <div class="equation">Δ<sub>true|r</sub>(t, ℓ, u) =
          L<sub>true eval</sub>(suffix trained on r) −
          L<sub>true eval</sub>(suffix trained on true)</div>
        <p>The primary estimand is this paired cross-entropy excess loss at matched
          checkpoint t, cut ℓ, PCA rank, draw u, minibatch order, and learning-rate
          regime. Summaries average paired draw-level contrasts, not unpaired bars.</p>
      </div>
      <div class="method-notes">
        <div><h3>Optimizer control</h3><p>The fixed-learning-rate regime is the
          primary comparison. Initial training loss and suffix-gradient norm expose
          scale mismatches; any gradient-normalized sensitivity regime is reported
          separately, never silently pooled.</p></div>
        <div><h3>Failure criteria</h3><ul class="criteria-list">
          <li>Large projected-real gap: PCA truncation, not moment failure.</li>
          <li>Rank-sensitive conclusion or poor held-out within-class coverage.</li>
          <li>Surrogate moment errors outside declared tolerances.</li>
          <li>Step-zero or initial-gradient mismatch that changes under the LR control.</li>
          <li>Missing cells in the train × evaluation distribution loss matrix.</li>
        </ul></div>
      </div>
    </div>
  </section>

  <section id="estimand" class="section">
    <div class="section-heading">
      <p class="eyebrow">Primary result</p>
      <h2>Paired true-evaluation excess loss</h2>
      <p class="question">At the relaxation endpoint, how much cross-entropy is
        added by training on each surrogate instead of matched real activations?</p>
      <p>Read zero as parity with the paired real-replay baseline. Positive values
        mean worse held-out real loss. Parentheses are the min–max range across
        surrogate draws; model-seed uncertainty requires additional checkpoints.</p>
    </div>
    {_primary_estimand_table(canonical_paired)}
  </section>

  <section id="matrix" class="section">
    <div class="section-heading">
      <p class="eyebrow">Distribution shift diagnostic</p>
      <h2>Full train × evaluation loss matrix</h2>
      <p class="question">Does a replay distribution merely fit its own support, or
        does it transfer to real and alternative activation distributions?</p>
      <p>Rows change the held-out evaluation distribution; columns change the
        relaxation distribution. The matrix is shown only when every declared cell
        is measured at the endpoint.</p>
    </div>
    {_matrix_cards(canonical_observations)}
  </section>

  <section id="controls" class="section">
    <div class="section-heading">
      <p class="eyebrow">Ablations and diagnostics</p>
      <h2>What would invalidate the simple interpretation?</h2>
      <p>Control settings are useful only when connected to outcomes. The panels
        below therefore show measured effect estimates where available and state
        plainly when the relevant diagnostic was not recorded.</p>
    </div>
    <div class="control-grid">{control_html}</div>
    <div class="control-stack">
      <div class="control-block"><h3>PCA-rank sensitivity</h3>
        <p>Compare projected-real and Gaussian excess loss across nested ranks.
          Within-class coverage is the relevant companion statistic.</p>
        {_rank_outcome_table(canonical_paired, canonical_cells)}</div>
      <div class="control-block"><h3>Mean-noise radius sensitivity</h3>
        <p>Tests whether an apparent centroid result depends on the chosen
          trace-scaled isotropic radius.</p>
        {_noise_outcome_table(canonical_paired)}</div>
      <div class="control-block"><h3>Step-zero PCA functional change</h3>
        <p>Before relaxation, compare the unchanged suffix on true and projected
          held-out activations. A large gap means rank truncation has already
          changed the function presented to the suffix.</p>
        {_projection_diagnostic_table(canonical_cells)}</div>
      <div class="control-block"><h3>Initial suffix-gradient scale</h3>
        <p>Ratios are paired to real replay at the same cell and draw. A large
          mismatch motivates a separately labelled optimizer-sensitivity regime.</p>
        {_gradient_diagnostic_table(canonical_observations)}</div>
      <div class="control-block"><h3>Surrogate moment fidelity</h3>
        <p>Generated-bank class means and covariances should match the target
          moments in the declared diagnostic space.</p>
        {_moment_diagnostic_table(canonical_cells)}</div>
      <div class="control-block"><h3>PCA coverage and sampling metadata</h3>
        <p>Total variance can be dominated by between-class mean separation, so
          total, within-class, and between-class coverage are kept separate.</p>
        {_coverage_table(canonical_cells)}</div>
    </div>
  </section>

  {canonical_sections}
  {legacy_html}

  <section id="data" class="section">
    <div class="section-heading">
      <p class="eyebrow">Data and provenance</p>
      <h2>Exact values and hashed inputs</h2>
      <p>The full source payloads, normalized evaluation records, paired
        draw-level contrasts, aggregate rows, and metadata are embedded in this
        self-contained file.</p>
    </div>
    <details class="data-details">
      <summary><span>Exact trajectory values</span>
        <span class="summary-meta">{len(summaries)} aggregated rows</span></summary>
      <div>{_exact_table(summaries)}</div>
    </details>
    <div class="data-actions">
      <button id="download-all" type="button">Download embedded data (JSON)</button>
      <button id="download-summary" type="button">Download exact summary (JSON)</button>
    </div>
    <details class="data-details">
      <summary><span>Provenance and input hashes</span>
        <span class="summary-meta">{len(artifacts)} verified inputs</span></summary>
      <div class="provenance-grid">{provenance_cards}</div>
    </details>
    <p class="footer-note">Manifest schema {MANIFEST_SCHEMA_VERSION}. This report
      rejects missing files, digest mismatches, non-measured statuses, fake-data
      configurations, unsupported experiments, and checkpoint metadata mismatches.
      The manifest commit identifies the dashboard branch base, not experiment
      lineage. Where legacy artifacts do not record their generating code revision,
      their SHA-256 digests identify the exact evidence files used here.</p>
  </section>
</main>
<script id="embedded-data" type="application/json">{_safe_script_json(embedded)}</script>
<script>
(() => {{
  const payload = JSON.parse(document.getElementById("embedded-data").textContent);
  function download(value, filename) {{
    const blob = new Blob([JSON.stringify(value, null, 2)], {{type: "application/json"}});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }}
  document.getElementById("download-all").addEventListener("click", () =>
    download(payload, "free-body-diagrams-data.json"));
  document.getElementById("download-summary").addEventListener("click", () =>
    download(payload.exact_summary_rows, "free-body-diagrams-exact-values.json"));
}})();
</script>
</body>
</html>
"""


def build_report(
    manifest_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Validate the manifest and write the canonical self-contained report."""

    manifest, artifacts = load_manifest(manifest_path)
    rendered = render_report(manifest, artifacts)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = "\n".join(line.rstrip() for line in rendered.splitlines())
    output.write_text(cleaned + "\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the measured-only interactive appendix for the LW post."
    )
    parser.add_argument("manifest", help="Path to the version-1 input manifest")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_report(args.manifest, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
