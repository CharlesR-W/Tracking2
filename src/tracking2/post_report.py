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
    "cnn": ("lw_post_cnn_suffix_statistics", {1}),
    "resnet": ("resnet18_suffix_statistics_sweep", {1, 2}),
}
_EXPECTED_FORMATS = {
    ("cnn", "post_statistics"),
    ("cnn", "legacy_cnn_suffix_statistics"),
    ("resnet", "resnet_suffix_statistics"),
}
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")


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
    if kind == "cnn" and artifact_format == "post_statistics":
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ReportInputError(f"{context}.checkpoint must be an object")
        checkpoint_epoch = _require_int(checkpoint, "epoch", f"{context}.checkpoint")
        if checkpoint_epoch != epoch:
            raise ReportInputError(
                f"{context} disagrees internally about checkpoint epoch: "
                f"{checkpoint_epoch} != {epoch}"
            )
    return epoch


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

    kinds = {item.kind for item in loaded}
    missing_kinds = {"cnn", "resnet"} - kinds
    if missing_kinds:
        raise ReportInputError(
            "Canonical report requires measured inputs for both CNN and ResNet; "
            f"missing {', '.join(sorted(missing_kinds))}"
        )
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


def _validate_distribution(distribution: Any, context: str) -> str:
    if not isinstance(distribution, str):
        raise ReportInputError(f"{context}.train_distribution must be a string")
    if distribution in {"true", "projected_true", "gaussian"}:
        return distribution
    if _mean_radius(distribution) is not None:
        return distribution
    raise ReportInputError(
        f"{context} has unsupported train_distribution {distribution!r}"
    )


def _coverage(source: Mapping[str, Any], context: str) -> tuple[float | None, str]:
    direct = source.get("held_out_explained_variance_fraction")
    if direct is not None:
        value = _number(direct, f"{context}.held_out_explained_variance_fraction")
        if not 0 <= value <= 1:
            raise ReportInputError(f"{context} held-out PCA coverage must be in [0, 1]")
        return value, "held-out activations"

    held_out = source.get("held_out_coverage")
    if held_out is not None:
        if not isinstance(held_out, dict):
            raise ReportInputError(f"{context}.held_out_coverage must be an object")
        for key in ("total_variance_fraction", "explained_variance_fraction"):
            if held_out.get(key) is not None:
                value = _number(held_out[key], f"{context}.held_out_coverage.{key}")
                if not 0 <= value <= 1:
                    raise ReportInputError(
                        f"{context} held-out PCA coverage must be in [0, 1]"
                    )
                return value, "held-out activations"

    legacy = source.get("explained_variance_fraction")
    if legacy is not None:
        value = _number(legacy, f"{context}.explained_variance_fraction")
        if not 0 <= value <= 1:
            raise ReportInputError(f"{context} PCA coverage must be in [0, 1]")
        return value, "legacy reported value; evaluation population unspecified"
    return None, "not recorded"


def _normalise_records(
    records: Any,
    *,
    context: str,
    forced_distribution: str | None = None,
    default_draw: int | None = None,
) -> list[dict[str, Any]]:
    rows = _objects(records, context)
    normalised: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        evaluation = row.get("eval_distribution")
        if evaluation != "true":
            continue
        distribution = (
            forced_distribution
            if forced_distribution is not None
            else _validate_distribution(row.get("train_distribution"), row_context)
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
        key = (distribution, draw, relax_epoch)
        if key in seen:
            raise ReportInputError(
                f"{context} contains duplicate true-evaluation record {key}"
            )
        seen.add(key)
        normalised.append(
            {
                "distribution": distribution,
                "draw": draw,
                "relax_epoch": relax_epoch,
                "accuracy": accuracy,
            }
        )
    if not normalised:
        raise ReportInputError(f"{context} has no true-evaluation records")
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
        if artifact.kind == "cnn":
            module = f"residual block {cut}"
            reference = _normalise_records(
                slice_.get("reference_records"),
                context=f"{slice_context}.reference_records",
                forced_distribution="true",
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
                    result.get("records"), context=f"{rank_context}.records"
                )
                records = [*reference, *result_records]
                _validate_cell_series(
                    records, f"artifact {artifact.id!r}, cut {cut}, PCA rank {rank}",
                    require_projected=True,
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
    records: Sequence[Mapping[str, Any]], context: str, *, require_projected: bool
) -> None:
    by_distribution: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_distribution[str(row["distribution"])].append(row)
    required = {"true", "gaussian"}
    if require_projected:
        required.add("projected_true")
    missing = required - by_distribution.keys()
    if missing:
        raise ReportInputError(
            f"{context} is missing required distributions: {', '.join(sorted(missing))}"
        )
    if not any(_mean_radius(name) is not None for name in by_distribution):
        raise ReportInputError(f"{context} has no mean/noise condition")

    epoch_sets = {
        distribution: {int(row["relax_epoch"]) for row in rows}
        for distribution, rows in by_distribution.items()
    }
    expected_epochs = next(iter(epoch_sets.values()))
    for distribution, epochs in epoch_sets.items():
        if epochs != expected_epochs:
            raise ReportInputError(
                f"{context} has mismatched relaxation epochs for {distribution!r}"
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
    radii = sorted(
        {
            radius
            for row in rank_source.get("records", [])
            if isinstance(row, dict)
            for radius in [_mean_radius(str(row.get("train_distribution", "")))]
            if radius is not None
        }
    )
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
        "representation_shape": slice_.get("representation_shape"),
        "native_dimension": slice_.get("native_dimension"),
        "pca_fit_count": slice_.get(
            "pca_fit_count", artifact.payload["config"].get("pca_fit_size")
        ),
        "mean_noise_radii": radii,
        "pooled_within_class_variance_per_pca_coordinate": rank_source.get(
            "pooled_within_class_variance_per_pca_coordinate"
        ),
        "trace_matched_isotropic_covariance_trace": rank_source.get(
            "trace_matched_isotropic_covariance_trace"
        ),
        "covariance_shrinkage": rank_source.get("covariance_shrinkage"),
        "empirical_class_covariance_rank_ceiling": rank_source.get(
            "empirical_class_covariance_rank_ceiling"
        ),
        "rank_exceeds_empirical_class_covariance_ceiling": rank_source.get(
            "rank_exceeds_empirical_class_covariance_ceiling"
        ),
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
    "relax_epoch",
)


def summarise_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in observations:
        grouped[tuple(row[key] for key in _SUMMARY_KEYS)].append(float(row["accuracy"]))
    result: list[dict[str, Any]] = []
    for key, values in grouped.items():
        summary = dict(zip(_SUMMARY_KEYS, key))
        summary.update(
            {
                "accuracy_mean": fmean(values),
                "accuracy_min": min(values),
                "accuracy_max": max(values),
                "draw_count": len(values),
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
            row["relax_epoch"],
        ),
    )


def _distribution_order(distribution: str) -> tuple[int, float, str]:
    if distribution == "true":
        return (0, 0.0, distribution)
    if distribution == "projected_true":
        return (1, 0.0, distribution)
    if distribution == "gaussian":
        return (2, 0.0, distribution)
    radius = _mean_radius(distribution)
    return (3, radius if radius is not None else 0.0, distribution)


def _distribution_style(distribution: str) -> tuple[str, str, str]:
    if distribution == "true":
        return "Real data", REAL, ""
    if distribution == "projected_true":
        return "Projected real", PROJECTED_REAL, "7 5"
    if distribution == "gaussian":
        return "Gaussian", GAUSSIAN, ""
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
    result = [
        name
        for name in ("true", "projected_true", "gaussian")
        if name in available
    ]
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
) -> str:
    architecture = _architecture_label(kind)
    run_html = []
    kind_artifacts = sorted(
        (artifact for artifact in artifacts if artifact.kind == kind),
        key=lambda artifact: artifact.checkpoint_epoch,
    )
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
    has_legacy_cnn = kind == "cnn" and any(
        artifact.format == "legacy_cnn_suffix_statistics"
        for artifact in kind_artifacts
    )
    intro = (
        "True-data relaxation is the reference. New-format cells include "
        "projected-real replay to isolate information lost at the PCA boundary; "
        "Gaussian and mean-based replay test which retained distributional "
        "structure helps the suffix relearn."
        if kind == "cnn"
        else
        "These legacy ResNet sweeps use the same true-evaluation target. Their "
        "reported PCA coverage is labelled conservatively unless the artifact "
        "contains an explicit held-out coverage field."
    )
    legacy_note = (
        '<aside class="evidence-note"><strong>Legacy CNN grid.</strong> The prefixes '
        "at different nominal epochs came from separately trained and scheduled "
        "models, with different scheduler horizons and checkpoint hashes. Each "
        "epoch × cut suffix-relaxation cell was also run separately. Adjacent "
        "nominal epochs are therefore neither one prefix-training trajectory nor "
        "one continued suffix-training trajectory. PCA coverage in these files is "
        "a legacy fit-bank value, not held-out coverage.</aside>"
        if has_legacy_cnn
        else ""
    )
    return (
        f'<section id="{kind}" class="section">'
        f'<div class="section-heading"><p class="eyebrow">Measured experiment</p>'
        f'<h2>{_esc(architecture)}</h2><p>{_esc(intro)}</p></div>'
        + legacy_note
        + "".join(run_html)
        + "</section>"
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
                "Separates PCA projection loss from the Gaussian approximation.",
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
    if any(cell["coverage_basis"] == "held-out activations" for cell in cells):
        controls.append(
            (
                "Held-out PCA coverage",
                "Measures retained variance on evaluation activations rather than PCA-fit rows.",
            )
        )
    return controls


def _coverage_table(cells: Sequence[Mapping[str, Any]]) -> str:
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
            f'<td>{_esc(cell["module"])}</td>'
            f'<td>{cell["pca_rank"]}{rank_flag}</td>'
            f'<td>{_fmt_percent(cell["coverage"])}</td>'
            f'<td>{_esc(cell["coverage_basis"])}</td>'
            f'<td>{_esc(radii)}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Model</th><th>Checkpoint epoch</th><th>Cut</th><th>PCA rank</th>"
        "<th>Variance retained</th><th>Coverage population</th><th>Mean-noise r</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _exact_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in summaries:
        label, _colour, _dash = _distribution_style(str(row["distribution"]))
        rows.append(
            "<tr>"
            f'<td>{_esc(_architecture_label(str(row["kind"])))}</td>'
            f'<td>{row["checkpoint_epoch"]}</td>'
            f'<td>{_esc(row["module"])}</td>'
            f'<td>{row["pca_rank"]}</td>'
            f'<td>{_esc(label)}</td>'
            f'<td>{row["relax_epoch"]}</td>'
            f'<td>{100 * float(row["accuracy_mean"]):.3f}%</td>'
            f'<td>{100 * float(row["accuracy_min"]):.3f}%–'
            f'{100 * float(row["accuracy_max"]):.3f}%</td>'
            f'<td>{row["draw_count"]}</td>'
            "</tr>"
        )
    return (
        '<div class="table-scroll exact-table"><table><thead><tr>'
        "<th>Model</th><th>Checkpoint</th><th>Cut</th><th>Rank</th>"
        "<th>Training distribution</th><th>Relax epoch</th>"
        "<th>Mean accuracy</th><th>Draw range</th><th>n</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _manifest_cards(artifacts: Sequence[LoadedArtifact]) -> str:
    cards = []
    for kind in ("cnn", "resnet"):
        group = [artifact for artifact in artifacts if artifact.kind == kind]
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
.evidence-note {{
  background: #f7edda; border: 1px solid #caa969; border-left: 4px solid #9d6e19;
  border-radius: 8px; padding: .85rem 1rem; margin: 0 0 1rem;
}}
.input-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 1rem; }}
.input-card, .method-card, .control-card, .provenance-card {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  padding: 1.15rem; box-shadow: 0 2px 14px rgba(45,37,28,.04);
}}
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
    controls = _completed_controls(summaries, cells)
    control_html = "".join(
        f'<article class="control-card"><h3>{_esc(name)}</h3>'
        f'<p>{_esc(description)}</p></article>'
        for name, description in controls
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
        "normalised_true_evaluation_records": observations,
        "exact_summary_rows": summaries,
        "cell_metadata": cells,
    }
    provenance_cards = "".join(
        '<article class="provenance-card">'
        f'<p class="eyebrow">{_esc(artifact.id)}</p>'
        f'<p><strong>{_esc(artifact.relative_path)}</strong></p>'
        f'<p><code>sha256:{artifact.sha256}</code></p>'
        "</article>"
        for artifact in artifacts
    )

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
  <a href="#method">Method</a>
  <a href="#cnn">Four-block CNN</a>
  <a href="#resnet">ResNet-18</a>
  <a href="#data">Robustness &amp; data</a>
</nav>
<main>
  <header id="overview" class="hero">
    <p class="eyebrow">Research note · data appendix</p>
    <h1>{_esc(TITLE)}</h1>
    <p class="lede">This appendix follows class-conditioned activation distributions
      through trained network prefixes, then asks how quickly the remaining suffix
      relearns from real, projected-real, Gaussian, or mean-based replay.</p>
    <div class="status-line">
      <span class="badge wip">Work in progress</span>
      <span class="badge measured">Measured inputs only</span>
      <span>{len(artifacts)} hashed artifacts · dashboard source base
        <code>{_esc(manifest["source_commit"])}</code></span>
    </div>
  </header>

  <section class="section" aria-labelledby="evidence-heading">
    <div class="section-heading">
      <p class="eyebrow">Evidence in this build</p>
      <h2 id="evidence-heading">A narrow, auditable view</h2>
      <p>Each card names the experimental unit. Model seeds and surrogate draws are
        not interchangeable; both are shown explicitly throughout.</p>
    </div>
    <div class="input-grid">{_manifest_cards(artifacts)}</div>
  </section>

  <section id="method" class="section">
    <div class="section-heading">
      <p class="eyebrow">Method</p>
      <h2>Cut, fit, replay, relax</h2>
      <p>A checkpoint supplies one prefix and suffix. We push a labelled input
        distribution through the prefix, fit PCA and class-conditional moments at
        the cut, replay controlled distributions there, and train only the suffix.
        Every curve is evaluated on held-out real activations.</p>
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
        <div><h3>Projected real</h3><p>Projects empirical activations into the fitted
          PCA subspace, isolating reconstruction loss before distributional fitting.</p></div>
        <div><h3>Gaussian</h3><p>Preserves fitted class means and covariance in PCA
          space, with the shrinkage recorded in the artifact.</p></div>
        <div><h3>Mean + isotropic noise</h3><p>This is not epsilon jitter. At
          r = 1 its covariance trace matches the pooled average within-class
          covariance trace; radius r scales trace by r².</p></div>
      </div>
    </div>
  </section>

  {_render_model_section("cnn", artifacts, summaries, cells)}
  {_render_model_section("resnet", artifacts, summaries, cells)}

  <section id="data" class="section">
    <div class="section-heading">
      <p class="eyebrow">Robustness &amp; data</p>
      <h2>Completed controls and exact values</h2>
      <p>Only controls present in the hashed measured artifacts appear here. The
        full source payloads and the normalized true-evaluation table are embedded
        in this HTML file.</p>
    </div>
    <div class="control-grid">{control_html}</div>
    <h3>PCA coverage and noise settings</h3>
    {_coverage_table(cells)}
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
      lineage. The legacy artifacts did not record their generating code revision;
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
