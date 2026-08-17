"""Verify the checked-in July 26 LessWrong publication package on CPU.

This verifier reads committed JSON/figure/appendix artifacts. It never trains a
model, uses a GPU, or accesses the network. The default check rebuilds the
self-contained appendix into a temporary directory and requires byte identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "lw_post"
MEASURED_ROOT = ARTIFACT_ROOT / "measured"
DEFAULT_MANIFEST = ARTIFACT_ROOT / "dashboard_manifest.json"
DEFAULT_APPENDIX = (
    PROJECT_ROOT
    / "LW post"
    / "free-body-diagrams-for-neural-networks.html"
)
DIGEST_LENGTH = 64
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    """The checked-in publication package violates its declared contract."""


@dataclass(frozen=True)
class InputSpec:
    input_id: str
    kind: str
    artifact_format: str
    sha256: str
    seed: int
    schema_version: int
    experiment: str
    config: Mapping[str, object]


COMMON_CNN_CONFIG: dict[str, object] = {
    "checkpoint_epoch": 30,
    "data_backend": "torchvision",
    "fake_data": False,
    "train_size": 50000,
    "test_size": 10000,
    "widths": [32, 64, 128, 128],
    "batch_size": 256,
    "pca_fit_size": 10000,
    "surrogate_draws": 1,
    "device": "cuda",
    "relax_learning_rate": 0.01,
}


def cnn_config(
    *,
    seed: int,
    cuts: list[int],
    ranks: list[int],
    shrinkages: list[float],
    radii: list[float],
    true_eval_only: bool,
    initialization: str,
    learning_rate_regime: str,
    relax_epochs: int,
) -> dict[str, object]:
    return {
        **COMMON_CNN_CONFIG,
        "seed": seed,
        "cuts": cuts,
        "pca_ranks": ranks,
        "gaussian_covariance_shrinkages": shrinkages,
        "mean_noise_radii": radii,
        "true_eval_only": true_eval_only,
        "suffix_initialization": initialization,
        "learning_rate_regime": learning_rate_regime,
        "relax_epochs": relax_epochs,
    }


INPUT_SPECS: dict[str, InputSpec] = {
    "cnn_fixed_gate_seed0/cut1_r2048_true_eval_sensitivity.json": InputSpec(
        "cnn-extra-0-s0-e30",
        "cnn",
        "post_statistics",
        "f8b357a70230b1cd838937f24f72328823c78145c091d6b8b05e5bbefcd43668",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[1],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=True,
            initialization="warm",
            learning_rate_regime="fixed_lr",
            relax_epochs=5,
        ),
    ),
    "cnn_fixed_gate_seed0/cut4_r2048_true_eval_sensitivity.json": InputSpec(
        "cnn-extra-1-s0-e30",
        "cnn",
        "post_statistics",
        "c076ff2119ff398a9f33922efcd5410e5d4070e389d7c2d788f5589fc2f3bbf2",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[4],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=True,
            initialization="warm",
            learning_rate_regime="fixed_lr",
            relax_epochs=5,
        ),
    ),
    "cnn_matched_seed0/cuts1_4_covariance_noise_ablation.json": InputSpec(
        "cnn-extra-2-s0-e30",
        "cnn",
        "post_statistics",
        "60ecc9c36eac0578b1c03e93daa4be5c785750aaeb3d3869203e7af2f78dc1e1",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[1, 4],
            ranks=[2048],
            shrinkages=[0.0, 0.05],
            radii=[0.0, 0.5, 1.0, 2.0],
            true_eval_only=True,
            initialization="warm",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=5,
        ),
    ),
    "cnn_matched_seed0/cut1_r4096_true_eval.json": InputSpec(
        "cnn-extra-3-s0-e30",
        "cnn",
        "post_statistics",
        "8e7b29695728b99553283c721d9969f9b79f03bf341c980ef805c86fc28cccd0",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[1],
            ranks=[4096],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=True,
            initialization="warm",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=5,
        ),
    ),
    "cnn_matched_seed0/cuts1_4_reinitialized_true_eval.json": InputSpec(
        "cnn-extra-4-s0-e30",
        "cnn",
        "post_statistics",
        "e166fdd28460ee43b82e888377e4b6b4c30f5702ebc53a6878fb99976928c0c5",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[1, 4],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=True,
            initialization="reinitialized",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=5,
        ),
    ),
    "cnn_matched_seed0/cuts1_4_horizon20_true_eval.json": InputSpec(
        "cnn-extra-5-s0-e30",
        "cnn",
        "post_statistics",
        "20b8ea86add8849dc4e7a9b16d319b0c1d5a163d407d01a51b3c5bd051f48a49",
        0,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=0,
            cuts=[1, 4],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=True,
            initialization="warm",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=20,
        ),
    ),
    "cnn_matched_seed1/cuts1_4_r2048_full_matrix.json": InputSpec(
        "cnn-extra-6-s1-e30",
        "cnn",
        "post_statistics",
        "59ef4ef44dac951cb5fb3003b0e1f709cae5d35b1bc07d27dc185d4066a420ba",
        1,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=1,
            cuts=[1, 4],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=False,
            initialization="warm",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=5,
        ),
    ),
    "cnn_matched_seed2/cuts1_4_r2048_full_matrix.json": InputSpec(
        "cnn-extra-7-s2-e30",
        "cnn",
        "post_statistics",
        "428366b4927770a07dba3426b89d7745236da59608f8f2b7af3e1496e3d4a357",
        2,
        2,
        "lw_post_cnn_suffix_statistics",
        cnn_config(
            seed=2,
            cuts=[1, 4],
            ranks=[2048],
            shrinkages=[0.0],
            radii=[1.0],
            true_eval_only=False,
            initialization="warm",
            learning_rate_regime="match_true_initial_update",
            relax_epochs=5,
        ),
    ),
    "cnn_projection_seed0/cut1_pca_adequacy.json": InputSpec(
        "cnn-projection-0-s0-e30",
        "cnn_projection",
        "cnn_projection_adequacy",
        "5930d82d9527144c8ba8029cf2459d0bed59422e1d1d4091774475ec67f2bfaa",
        0,
        1,
        "lw_post_cnn_projection_adequacy",
        {
            "checkpoint_epoch": 30,
            "data_backend": "torchvision",
            "fake_data": False,
            "train_size": 50000,
            "test_size": 10000,
            "widths": [32, 64, 128, 128],
            "batch_size": 256,
            "pca_fit_size": 10000,
            "cuts": [1],
            "pca_ranks": [2048, 3072, 4096],
            "seed": 0,
            "device": "cuda",
        },
    ),
}


TRAINING_MANIFEST_DIGESTS = {
    0: "063867dc7750b4c27331c3644b9e99b87cf5471d3cef4371d7064555dcf81bcc",
    1: "5281f89a014db4523cfb2138a9513cb7bd5d850c137c8f8885af3467531027ab",
    2: "9bb502eaae11100675353f14d3aaf342c58ad3d22da34177c8ed60714c1184cd",
}

PRIMARY_PATHS = {
    0: "cnn_matched_seed0/cuts1_4_covariance_noise_ablation.json",
    1: "cnn_matched_seed1/cuts1_4_r2048_full_matrix.json",
    2: "cnn_matched_seed2/cuts1_4_r2048_full_matrix.json",
}

EXPECTED_PRIMARY_ANALYSIS: dict[str, object] = {
    "status": "MEASURED",
    "model_kind": "cnn",
    "checkpoint_epoch": 30,
    "cuts": [1, 4],
    "pca_rank": 2048,
    "projection_scope": "retained_pca_subspace_only",
    "projection_gate_input_id": "cnn-projection-0-s0-e30",
    "suffix_initialization": "warm",
    "learning_rate_regime": "match_true_initial_update",
    "relax_epochs": 5,
    "model_seeds": [0, 1, 2],
    "input_ids_by_model_seed": {
        "0": "cnn-extra-2-s0-e30",
        "1": "cnn-extra-6-s1-e30",
        "2": "cnn-extra-7-s2-e30",
    },
    "surrogate_draws_per_model": 1,
    "contrasts": [
        {
            "id": "gaussian_minus_projected_true",
            "minuend_distribution": "gaussian_empirical",
            "subtrahend_distribution": "projected_true",
        },
        {
            "id": "mean_r1_minus_gaussian",
            "minuend_distribution": "mean_r1",
            "subtrahend_distribution": "gaussian_empirical",
        },
    ],
    "uncertainty": {
        "statistic": "sample_standard_deviation",
        "independent_unit": "trained_model_seed",
        "n": 3,
        "within_model_surrogate_redraw_uncertainty": "not_measured",
    },
    "first_update_matching": {
        "relative_tolerance": 0.01,
        "learning_rate_multiplier_bounds": [0.001, 1000.0],
        "clipped_status": "approximately_matched_clipped",
        "outside_tolerance_policy": "reject_primary_display_secondary",
    },
    "headline_assertion_absolute_tolerance": 0.0000005,
    "headline_assertions": [
        {
            "cut": 1,
            "contrast_id": "gaussian_minus_projected_true",
            "mean": 0.182913,
            "sample_standard_deviation": 0.055069,
        },
        {
            "cut": 1,
            "contrast_id": "mean_r1_minus_gaussian",
            "mean": 0.144051,
            "sample_standard_deviation": 0.019072,
        },
        {
            "cut": 4,
            "contrast_id": "gaussian_minus_projected_true",
            "mean": -0.012495,
            "sample_standard_deviation": 0.001530,
        },
        {
            "cut": 4,
            "contrast_id": "mean_r1_minus_gaussian",
            "mean": 0.022739,
            "sample_standard_deviation": 0.010517,
        },
    ],
}

EXPECTED_HEADLINES = {
    (1, "gaussian_minus_projected_true"): (
        0.18291344839731857,
        0.05506913979079249,
    ),
    (1, "mean_r1_minus_gaussian"): (
        0.1440512680689494,
        0.019072251608814983,
    ),
    (4, "gaussian_minus_projected_true"): (
        -0.012494543361663854,
        0.0015303445936978583,
    ),
    (4, "mean_r1_minus_gaussian"): (
        0.0227389348347982,
        0.01051742948798972,
    ),
}

FIGURES = {
    "cnn_measured_controls.png": {
        "sha256": (
            "1cde8775af500597a9b2751fe727fd8a4c35f0a5dd2a3d0f21afdda7d920d8f5"
        ),
        "inputs": {
            "cnn_fixed_gate_seed0/cut1_r2048_true_eval_sensitivity.json",
            *PRIMARY_PATHS.values(),
        },
        "training_seeds": {0, 1, 2},
    },
    "cnn_measured_horizon.png": {
        "sha256": (
            "b5d59e1b99b97cb28d9c05f3442f488af715ea10886531516bc393deb2c643f4"
        ),
        "inputs": {
            "cnn_matched_seed0/cuts1_4_horizon20_true_eval.json",
        },
        "training_seeds": {0},
    },
}

EXPECTED_CORRECTIONS = {
    "2078fea640cfd578586a833f72ad73ef1a012d28": {
        "intended_commit": "2078fea64937ff7fdd89d11010d2d2ed24b7bfc8",
        "prefix": "2078fea",
        "source_archive_sha256": (
            "eeee12d5976d1cfd644b8f3758298b259917174798b3be279f34f2c6d37f588f"
        ),
        "source_archive_status": "digest_recorded_archive_not_retained_locally",
        "affected_locations": {
            (
                "measured/cnn_fixed_gate_seed0/"
                "cut1_r2048_true_eval_sensitivity.json",
                "/provenance/source_revision",
            )
        },
    },
    "5873b1bc8736e623effd57f83d976366458dedca": {
        "intended_commit": "5873b1bfe329606d6848cdf01d40db3e4ac4f44a",
        "prefix": "5873b1b",
        "source_archive_sha256": (
            "b31c0c452dd999b7dccce8e3cd571eb5a859ba4095d6f5666853a2d01a38280b"
        ),
        "source_archive_status": "digest_recorded_archive_not_retained_locally",
        "affected_locations": {
            (
                "measured/training_manifests/cnn_seed0_training.json",
                "/provenance/source_revision",
            ),
            (
                "measured/cnn_projection_seed0/cut1_pca_adequacy.json",
                "/lineage/training_manifest/source_provenance/source_revision",
            ),
        },
    },
    "13aff88d1d4b09ec84f93e9bc3d37a47cd20962d": {
        "intended_commit": "13aff88de2775df9073dda15c6174bcd5a7ed309",
        "prefix": "13aff88",
        "source_archive_sha256": None,
        "source_archive_status": "not_applicable_manifest_builder_revision",
        "affected_locations": {
            ("dashboard_manifest.json", "/source_commit"),
        },
    },
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise VerificationError(f"missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"missing JSON file: {path}")

    def reject_constant(value: str) -> None:
        raise VerificationError(f"{path} contains non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise VerificationError(f"{path} must contain a JSON object")
    require_finite_tree(payload, str(path))
    return payload


def require_finite_tree(value: object, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"{context} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_tree(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_tree(child, f"{context}[{index}]")


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected or (
        isinstance(expected, bool) and type(actual) is not bool
    ):
        raise VerificationError(
            f"{context}={actual!r}, expected exactly {expected!r}"
        )


def require_config(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
    context: str,
) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise VerificationError(f"{context}.config must be an object")
    for key, value in expected.items():
        require_equal(config.get(key), value, f"{context}.config.{key}")


def verify_manifest_inputs(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    measured_root: Path,
    locked_snapshot: bool,
) -> dict[str, dict[str, Any]]:
    require_equal(manifest.get("schema_version"), 2, "manifest.schema_version")
    require_equal(manifest.get("status"), "MEASURED", "manifest.status")
    entries = manifest.get("inputs")
    if not isinstance(entries, list) or len(entries) != len(INPUT_SPECS):
        raise VerificationError(
            f"manifest must list exactly {len(INPUT_SPECS)} publication inputs"
        )
    observed_paths: set[str] = set()
    observed_ids: set[str] = set()
    payloads: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        context = f"manifest.inputs[{index}]"
        if not isinstance(entry, dict):
            raise VerificationError(f"{context} must be an object")
        relative_entry_path = entry.get("path")
        if not isinstance(relative_entry_path, str):
            raise VerificationError(f"{context}.path must be a string")
        expected_prefix = "../lw_post/measured/"
        if not relative_entry_path.startswith(expected_prefix):
            raise VerificationError(
                f"{context}.path is outside the measured publication package"
            )
        relative = relative_entry_path.removeprefix(expected_prefix)
        if relative in observed_paths:
            raise VerificationError(f"duplicate manifest input path: {relative}")
        observed_paths.add(relative)
        spec = INPUT_SPECS.get(relative)
        if spec is None:
            raise VerificationError(f"unexpected manifest input: {relative}")
        input_id = entry.get("id")
        if not isinstance(input_id, str) or input_id in observed_ids:
            raise VerificationError(f"invalid or duplicate {context}.id")
        observed_ids.add(input_id)
        require_equal(input_id, spec.input_id, f"{context}.id")
        require_equal(entry.get("kind"), spec.kind, f"{context}.kind")
        require_equal(
            entry.get("format"), spec.artifact_format, f"{context}.format"
        )
        require_equal(entry.get("checkpoint_epoch"), 30, f"{context}.checkpoint_epoch")
        require_equal(entry.get("model_seed"), spec.seed, f"{context}.model_seed")
        declared_digest = entry.get("sha256")
        if not isinstance(declared_digest, str) or DIGEST_RE.fullmatch(
            declared_digest
        ) is None:
            raise VerificationError(f"{context}.sha256 is not a SHA-256 digest")
        if locked_snapshot:
            require_equal(declared_digest, spec.sha256, f"{context}.sha256")
        artifact_path = (manifest_path.parent / relative_entry_path).resolve()
        expected_path = (measured_root / relative).resolve()
        if artifact_path != expected_path:
            raise VerificationError(f"{context}.path resolves to an unexpected file")
        observed_digest = sha256(artifact_path)
        require_equal(
            observed_digest, declared_digest, f"SHA-256 for {artifact_path}"
        )
        payload = load_json(artifact_path)
        require_equal(payload.get("status"), "MEASURED", f"{relative}.status")
        require_equal(
            payload.get("schema_version"),
            spec.schema_version,
            f"{relative}.schema_version",
        )
        require_equal(
            payload.get("experiment"), spec.experiment, f"{relative}.experiment"
        )
        require_config(payload, spec.config, relative)
        payloads[relative] = payload
    if observed_paths != set(INPUT_SPECS):
        raise VerificationError("manifest input path set is incomplete")
    return payloads


def verify_training_manifests(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    measured_root: Path,
    locked_snapshot: bool,
    checkpoint_root: Path | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    manifests: dict[int, dict[str, Any]] = {}
    manifest_digests: dict[int, str] = {}
    shared_dataset: object | None = None
    common_config = {
        "data_backend": "torchvision",
        "fake_data": False,
        "train_size": 50000,
        "test_size": 10000,
        "epochs": 30,
        "widths": [32, 64, 128, 128],
        "batch_size": 256,
        "learning_rate": 0.05,
        "weight_decay": 0.0005,
        "device": "cuda",
    }
    for seed in (0, 1, 2):
        path = (
            measured_root
            / "training_manifests"
            / f"cnn_seed{seed}_training.json"
        )
        manifest_digest = sha256(path)
        if locked_snapshot:
            require_equal(
                manifest_digest,
                TRAINING_MANIFEST_DIGESTS[seed],
                f"SHA-256 for seed-{seed} training manifest",
            )
        manifest_digests[seed] = manifest_digest
        payload = load_json(path)
        require_equal(payload.get("status"), "MEASURED", f"{path}.status")
        require_equal(payload.get("schema_version"), 1, f"{path}.schema_version")
        require_equal(
            payload.get("experiment"),
            "lw_post_cnn_checkpoint_trajectory",
            f"{path}.experiment",
        )
        expected_config = {
            **common_config,
            "seed": seed,
            "checkpoint_epochs": [0, 1, 5, 10, 20, 30]
            if seed == 0
            else [30],
        }
        require_config(payload, expected_config, str(path))
        if "one uninterrupted end-to-end training run" not in str(
            payload.get("trajectory_semantics", "")
        ):
            raise VerificationError(f"{path} lacks uninterrupted-run semantics")
        dataset = payload.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("backend") != "torchvision":
            raise VerificationError(f"{path} lacks ordered torchvision fingerprints")
        if shared_dataset is None:
            shared_dataset = dataset
        elif dataset != shared_dataset:
            raise VerificationError("training manifests use different ordered datasets")
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise VerificationError(f"{path}.checkpoints must be a list")
        epoch30 = [
            row
            for row in checkpoints
            if isinstance(row, dict) and row.get("epoch") == 30
        ]
        if len(epoch30) != 1:
            raise VerificationError(f"{path} must identify one epoch-30 checkpoint")
        checkpoint_hash = epoch30[0].get("sha256")
        if (
            not isinstance(checkpoint_hash, str)
            or DIGEST_RE.fullmatch(checkpoint_hash) is None
        ):
            raise VerificationError(f"{path} has an invalid checkpoint digest")
        if checkpoint_root is not None:
            checkpoint_path = (
                checkpoint_root
                / f"cnn_seed{seed}"
                / "checkpoint_epoch30.pt"
            )
            require_equal(
                sha256(checkpoint_path),
                checkpoint_hash,
                f"seed-{seed} staged epoch-30 checkpoint SHA-256",
            )
        for relative, artifact in payloads.items():
            if INPUT_SPECS[relative].seed != seed:
                continue
            if artifact.get("dataset") != dataset:
                raise VerificationError(
                    f"{relative} does not match seed-{seed} ordered dataset lineage"
                )
            if INPUT_SPECS[relative].kind == "cnn":
                checkpoint = artifact.get("checkpoint")
            else:
                lineage = artifact.get("lineage", {})
                checkpoint = lineage.get("checkpoint") if isinstance(lineage, dict) else None
            if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != checkpoint_hash:
                raise VerificationError(
                    f"{relative} does not match seed-{seed} epoch-30 checkpoint"
                )
        manifests[seed] = payload
    projection = payloads["cnn_projection_seed0/cut1_pca_adequacy.json"]
    projection_lineage = projection.get("lineage")
    embedded_training = (
        projection_lineage.get("training_manifest")
        if isinstance(projection_lineage, dict)
        else None
    )
    if not isinstance(embedded_training, dict):
        raise VerificationError("projection gate lacks embedded training lineage")
    require_equal(
        embedded_training.get("sha256"),
        manifest_digests[0],
        "projection gate training-manifest SHA-256",
    )
    require_equal(
        embedded_training.get("status"),
        "MEASURED",
        "projection gate training-manifest status",
    )
    require_equal(
        embedded_training.get("experiment"),
        "lw_post_cnn_checkpoint_trajectory",
        "projection gate training-manifest experiment",
    )
    return manifests, manifest_digests


def json_pointer_value(payload: object, pointer: str, context: str) -> object:
    if not pointer.startswith("/"):
        raise VerificationError(f"{context} has an invalid JSON pointer")
    current = payload
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise VerificationError(f"{context} points to a missing value")
    return current


def git_commit_for_prefix(prefix: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{prefix}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VerificationError(
            f"correction prefix {prefix!r} does not resolve uniquely to a commit"
        )
    return result.stdout.strip()


def git_object_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def verify_correction_sidecar(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> None:
    reference = manifest.get("provenance_corrections")
    if not isinstance(reference, dict):
        raise VerificationError("manifest lacks provenance_corrections reference")
    require_equal(
        reference.get("path"),
        "provenance_corrections.json",
        "manifest.provenance_corrections.path",
    )
    sidecar_path = (manifest_path.parent / str(reference["path"])).resolve()
    if sidecar_path != (ARTIFACT_ROOT / "provenance_corrections.json").resolve():
        raise VerificationError("correction sidecar resolves outside canonical path")
    observed_digest = sha256(sidecar_path)
    require_equal(
        reference.get("sha256"),
        observed_digest,
        "manifest.provenance_corrections.sha256",
    )
    sidecar = load_json(sidecar_path)
    require_equal(sidecar.get("schema_version"), 1, "corrections.schema_version")
    require_equal(sidecar.get("status"), "AUDITED", "corrections.status")
    rows = sidecar.get("corrections")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CORRECTIONS):
        raise VerificationError("correction sidecar must contain exactly three records")
    by_recorded: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise VerificationError(f"corrections[{index}] must be an object")
        recorded = row.get("recorded_value")
        if not isinstance(recorded, str) or recorded in by_recorded:
            raise VerificationError(f"corrections[{index}] has invalid recorded_value")
        by_recorded[recorded] = row
    if set(by_recorded) != set(EXPECTED_CORRECTIONS):
        raise VerificationError("correction sidecar records the wrong typo set")
    payload_cache: dict[str, Mapping[str, object]] = {
        "dashboard_manifest.json": manifest
    }
    for recorded, expected in EXPECTED_CORRECTIONS.items():
        row = by_recorded[recorded]
        intended = expected["intended_commit"]
        require_equal(row.get("intended_commit"), intended, f"correction {recorded}")
        require_equal(
            row.get("source_archive_sha256"),
            expected["source_archive_sha256"],
            f"correction {recorded}.source_archive_sha256",
        )
        require_equal(
            row.get("source_archive_status"),
            expected["source_archive_status"],
            f"correction {recorded}.source_archive_status",
        )
        evidence = row.get("unique_prefix_evidence")
        if not isinstance(evidence, dict):
            raise VerificationError(f"correction {recorded} lacks prefix evidence")
        prefix = str(expected["prefix"])
        require_equal(evidence.get("prefix"), prefix, f"correction {recorded}.prefix")
        require_equal(
            evidence.get("resolved_commit"),
            intended,
            f"correction {recorded}.resolved_commit",
        )
        if not recorded.startswith(prefix) or not str(intended).startswith(prefix):
            raise VerificationError(f"correction {recorded} uses unrelated prefix")
        require_equal(
            git_commit_for_prefix(prefix),
            intended,
            f"Git resolution for correction {recorded}",
        )
        if not git_object_exists(str(intended)):
            raise VerificationError(f"intended commit {intended} is not a Git object")
        if git_object_exists(recorded):
            raise VerificationError(
                f"recorded value {recorded} is already a commit; correction is stale"
            )
        locations = row.get("affected_locations")
        if not isinstance(locations, list):
            raise VerificationError(f"correction {recorded} lacks affected locations")
        observed_locations: set[tuple[str, str]] = set()
        for location in locations:
            if not isinstance(location, dict):
                raise VerificationError(f"correction {recorded} has invalid location")
            relative = location.get("path")
            pointer = location.get("json_pointer")
            if not isinstance(relative, str) or not isinstance(pointer, str):
                raise VerificationError(f"correction {recorded} has invalid location")
            observed_locations.add((relative, pointer))
            if relative not in payload_cache:
                path = (ARTIFACT_ROOT / relative).resolve()
                try:
                    path.relative_to(ARTIFACT_ROOT.resolve())
                except ValueError as error:
                    raise VerificationError(
                        f"correction {recorded} points outside artifact root"
                    ) from error
                payload_cache[relative] = load_json(path)
            raw_value = json_pointer_value(
                payload_cache[relative], pointer, f"{relative}{pointer}"
            )
            require_equal(raw_value, recorded, f"raw value at {relative}{pointer}")
        require_equal(
            observed_locations,
            expected["affected_locations"],
            f"affected locations for correction {recorded}",
        )


def _source_identity(payload: Mapping[str, object], context: str) -> tuple[str, str]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise VerificationError(f"{context} lacks source provenance")
    revision = provenance.get("source_revision")
    archive = provenance.get("source_archive_sha256")
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise VerificationError(f"{context} lacks a full source commit")
    if not isinstance(archive, str) or DIGEST_RE.fullmatch(archive) is None:
        raise VerificationError(f"{context} lacks a source-archive SHA-256")
    return revision, archive


def git_archive_commit(path: Path) -> str:
    if not path.is_file():
        raise VerificationError(f"missing staged source archive: {path}")
    with path.open("rb") as archive:
        result = subprocess.run(
            ["git", "get-tar-commit-id"],
            cwd=PROJECT_ROOT,
            stdin=archive,
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0 or COMMIT_RE.fullmatch(result.stdout.strip()) is None:
        raise VerificationError(
            "staged source archive is not an uncompressed `git archive` tar "
            "with an embedded commit ID"
        )
    return result.stdout.strip()


def verify_staged_provenance(
    manifest: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, Any]],
    training: Mapping[int, Mapping[str, Any]],
    source_archive: Path,
) -> None:
    if "provenance_corrections" in manifest:
        raise VerificationError(
            "a clean staged rerun must not inherit the July 26 correction sidecar"
        )
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise VerificationError("staged manifest.source_commit must be a full commit")
    if not git_object_exists(source_commit):
        raise VerificationError(
            f"staged source commit {source_commit} is not available in local Git"
        )
    archive_digest = sha256(source_archive)
    require_equal(
        git_archive_commit(source_archive),
        source_commit,
        "commit embedded in staged source archive",
    )
    for seed, training_manifest in training.items():
        require_equal(
            _source_identity(training_manifest, f"seed-{seed} training manifest"),
            (source_commit, archive_digest),
            f"seed-{seed} training source identity",
        )
    for relative, artifact in payloads.items():
        require_equal(
            _source_identity(artifact, relative),
            (source_commit, archive_digest),
            f"{relative} source identity",
        )
    projection = payloads["cnn_projection_seed0/cut1_pca_adequacy.json"]
    lineage = projection.get("lineage")
    embedded = (
        lineage.get("training_manifest")
        if isinstance(lineage, dict)
        else None
    )
    embedded_source = (
        embedded.get("source_provenance")
        if isinstance(embedded, dict)
        else None
    )
    if not isinstance(embedded_source, dict):
        raise VerificationError("projection gate lacks embedded training source")
    require_equal(
        _source_identity(
            {"provenance": embedded_source},
            "projection embedded training manifest",
        ),
        (source_commit, archive_digest),
        "projection embedded training source identity",
    )


def rank_result(
    artifact: Mapping[str, object],
    *,
    cut: int,
    rank: int,
    context: str,
) -> Mapping[str, object]:
    slices = artifact.get("slices")
    if not isinstance(slices, list):
        raise VerificationError(f"{context}.slices must be a list")
    matches = [
        row for row in slices if isinstance(row, dict) and row.get("cut") == cut
    ]
    if len(matches) != 1:
        raise VerificationError(f"{context} must contain exactly one cut-{cut} slice")
    ranks = matches[0].get("rank_results")
    if not isinstance(ranks, list):
        raise VerificationError(f"{context} cut-{cut} lacks rank results")
    selected = [
        row for row in ranks if isinstance(row, dict) and row.get("pca_rank") == rank
    ]
    if len(selected) != 1:
        raise VerificationError(
            f"{context} must contain exactly one cut-{cut}, rank-{rank} cell"
        )
    return selected[0]


def record(
    rows: object,
    *,
    distribution: str,
    relax_epoch: int,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(rows, list):
        raise VerificationError(f"{context} records must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("draw") == 0
        and row.get("train_distribution") == distribution
        and row.get("eval_distribution") == "true"
        and row.get("relax_epoch") == relax_epoch
    ]
    if len(matches) != 1:
        raise VerificationError(
            f"{context} has {len(matches)} endpoints for {distribution}, "
            f"expected one"
        )
    return matches[0]


def finite_float(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise VerificationError(f"{context} must be a finite number")
    return float(value)


def verify_primary_analysis(
    manifest: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    locked_snapshot: bool,
) -> dict[tuple[int, str], tuple[float, float]]:
    contract = manifest.get("primary_analysis")
    if not isinstance(contract, dict):
        raise VerificationError("manifest.primary_analysis must be an object")
    if locked_snapshot:
        require_equal(
            contract,
            EXPECTED_PRIMARY_ANALYSIS,
            "manifest.primary_analysis",
        )
    else:
        staged_structure = {
            key: value
            for key, value in contract.items()
            if key != "headline_assertions"
        }
        expected_structure = {
            key: value
            for key, value in EXPECTED_PRIMARY_ANALYSIS.items()
            if key != "headline_assertions"
        }
        require_equal(
            staged_structure,
            expected_structure,
            "staged manifest.primary_analysis structure",
        )
    match = contract["first_update_matching"]
    assert isinstance(match, dict)
    tolerance = float(match["relative_tolerance"])
    lower, upper = match["learning_rate_multiplier_bounds"]
    contrasts: dict[tuple[int, str], list[float]] = {
        (cut, contrast): []
        for cut in (1, 4)
        for contrast in (
            "gaussian_minus_projected_true",
            "mean_r1_minus_gaussian",
        )
    }
    for seed, relative in PRIMARY_PATHS.items():
        artifact = payloads[relative]
        for cut in (1, 4):
            cell = rank_result(
                artifact, cut=cut, rank=2048, context=f"primary seed {seed}"
            )
            rows = cell.get("records")
            endpoints: dict[str, float] = {}
            for distribution in (
                "projected_true",
                "gaussian_empirical",
                "mean_r1",
            ):
                endpoint = record(
                    rows,
                    distribution=distribution,
                    relax_epoch=5,
                    context=f"primary seed {seed} cut {cut}",
                )
                endpoints[distribution] = finite_float(
                    endpoint.get("loss"),
                    f"primary seed {seed} cut {cut} {distribution} loss",
                )
            contrasts[(cut, "gaussian_minus_projected_true")].append(
                endpoints["gaussian_empirical"] - endpoints["projected_true"]
            )
            contrasts[(cut, "mean_r1_minus_gaussian")].append(
                endpoints["mean_r1"] - endpoints["gaussian_empirical"]
            )
            slices = artifact["slices"]
            slice_row = next(row for row in slices if row["cut"] == cut)
            reference = record(
                slice_row.get("reference_records"),
                distribution="true",
                relax_epoch=0,
                context=f"primary seed {seed} cut {cut} true reference",
            )
            true_update = finite_float(
                reference.get("matched_first_step_update_norm"),
                f"primary seed {seed} cut {cut} true update",
            )
            for distribution in (
                "projected_true",
                "gaussian_empirical",
                "mean_r1",
            ):
                initial = record(
                    rows,
                    distribution=distribution,
                    relax_epoch=0,
                    context=f"primary seed {seed} cut {cut}",
                )
                matched_update = finite_float(
                    initial.get("matched_first_step_update_norm"),
                    f"primary seed {seed} cut {cut} {distribution} update",
                )
                relative_error = abs(matched_update - true_update) / true_update
                if relative_error > tolerance:
                    raise VerificationError(
                        f"primary seed {seed} cut {cut} {distribution} update "
                        f"mismatch {relative_error:.3%} exceeds {tolerance:.3%}"
                    )
                multiplier = finite_float(
                    initial.get("learning_rate_multiplier"),
                    f"primary seed {seed} cut {cut} {distribution} multiplier",
                )
                if not float(lower) < multiplier < float(upper):
                    raise VerificationError(
                        f"primary seed {seed} cut {cut} {distribution} is clipped"
                    )
    summaries: dict[tuple[int, str], tuple[float, float]] = {}
    for key, values in contrasts.items():
        summary = (statistics.mean(values), statistics.stdev(values))
        if locked_snapshot:
            expected = EXPECTED_HEADLINES[key]
            if not all(
                math.isclose(actual, target, rel_tol=0, abs_tol=1e-12)
                for actual, target in zip(summary, expected)
            ):
                raise VerificationError(
                    f"headline {key} recomputed as {summary}, expected {expected}"
                )
        summaries[key] = summary
    assertion_tolerance = float(contract["headline_assertion_absolute_tolerance"])
    assertions = contract.get("headline_assertions")
    if not isinstance(assertions, list) or len(assertions) != len(summaries):
        raise VerificationError("primary headline assertions must have four rows")
    observed_assertion_keys: set[tuple[int, str]] = set()
    for row in assertions:
        if not isinstance(row, dict) or set(row) != {
            "cut",
            "contrast_id",
            "mean",
            "sample_standard_deviation",
        }:
            raise VerificationError("primary headline assertion has wrong fields")
        key = (int(row["cut"]), str(row["contrast_id"]))
        if key in observed_assertion_keys or key not in summaries:
            raise VerificationError(f"invalid or duplicate headline assertion {key}")
        observed_assertion_keys.add(key)
        mean, standard_deviation = summaries[key]
        if abs(float(row["mean"]) - mean) > assertion_tolerance:
            raise VerificationError(f"manifest headline mean fails for {key}")
        if (
            abs(float(row["sample_standard_deviation"]) - standard_deviation)
            > assertion_tolerance
        ):
            raise VerificationError(f"manifest headline SD fails for {key}")
    if observed_assertion_keys != set(summaries):
        raise VerificationError("primary headline assertion grid is incomplete")
    verify_radius_zero_control(
        payloads[PRIMARY_PATHS[0]],
        tolerance,
        lower=float(lower),
        upper=float(upper),
        locked_snapshot=locked_snapshot,
    )
    return summaries


def verify_radius_zero_control(
    seed_zero_artifact: Mapping[str, object],
    tolerance: float,
    *,
    lower: float,
    upper: float,
    locked_snapshot: bool,
) -> None:
    cell = rank_result(
        seed_zero_artifact,
        cut=1,
        rank=2048,
        context="seed-0 clipped radius-zero control",
    )
    radius_zero = record(
        cell.get("records"),
        distribution="mean_r0",
        relax_epoch=0,
        context="seed-0 clipped radius-zero control",
    )
    slice_row = next(row for row in seed_zero_artifact["slices"] if row["cut"] == 1)
    reference = record(
        slice_row.get("reference_records"),
        distribution="true",
        relax_epoch=0,
        context="seed-0 true reference",
    )
    multiplier = finite_float(
        radius_zero.get("learning_rate_multiplier"),
        "radius-zero learning-rate multiplier",
    )
    observed = finite_float(
        radius_zero.get("matched_first_step_update_norm"),
        "radius-zero matched update",
    )
    target = finite_float(
        reference.get("matched_first_step_update_norm"),
        "radius-zero target update",
    )
    relative_error = abs(observed - target) / target
    clipped = multiplier in {lower, upper}
    if locked_snapshot:
        if (
            multiplier != lower
            or relative_error <= tolerance
            or not math.isclose(
                relative_error, 0.094631954, rel_tol=0, abs_tol=0.000001
            )
        ):
            raise VerificationError(
                "radius-zero control no longer has the audited clipped 9.463% mismatch"
            )
    elif not clipped and relative_error > tolerance:
        raise VerificationError(
            "staged radius-zero control misses update tolerance without clipping"
        )


def verify_projection_gate(
    payloads: Mapping[str, Mapping[str, Any]], *, locked_snapshot: bool
) -> None:
    relative = "cnn_projection_seed0/cut1_pca_adequacy.json"
    artifact = payloads[relative]
    cell = rank_result(artifact, cut=1, rank=2048, context=relative)
    coverage = cell.get("held_out_coverage")
    if not isinstance(coverage, dict):
        raise VerificationError("projection gate lacks held-out coverage")
    retained = finite_float(
        coverage.get("total_variance_fraction"),
        "rank-2048 projection retained variance",
    )
    if retained <= 0 or retained > 1:
        raise VerificationError("rank-2048 projection coverage is outside (0, 1]")
    if locked_snapshot and not math.isclose(
        retained, 0.8802306941035515, rel_tol=0, abs_tol=1e-15
    ):
        raise VerificationError("rank-2048 projection coverage changed")


def verify_figure_inputs(
    payloads: Mapping[str, Mapping[str, Any]],
    training: Mapping[int, Mapping[str, Any]],
    *,
    figure_dir: Path,
    measured_root: Path,
    locked_snapshot: bool,
) -> dict[str, str]:
    manifest_paths = set(payloads)
    digests: dict[str, str] = {}
    for name, spec in FIGURES.items():
        inputs = spec["inputs"]
        seeds = spec["training_seeds"]
        assert isinstance(inputs, set) and isinstance(seeds, set)
        if not inputs <= manifest_paths:
            raise VerificationError(f"{name} depends on an unmanifested input")
        if not seeds <= set(training):
            raise VerificationError(f"{name} lacks a training manifest")
        path = figure_dir / name
        observed_digest = sha256(path)
        if locked_snapshot:
            require_equal(observed_digest, spec["sha256"], f"SHA-256 for {name}")
        digests[name] = observed_digest
    if not locked_snapshot:
        verify_staged_figure_rebuild(measured_root, figure_dir)
    return digests


def verify_staged_figure_rebuild(measured_root: Path, figure_dir: Path) -> None:
    generator_path = PROJECT_ROOT / "LW post" / "notebooks" / "lw_post_figures.py"
    module_name = "_tracking2_staged_lw_figures"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise VerificationError("cannot load the publication figure generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        control_data = module.load_measured_cnn_control_data(measured_root)
        horizon_data = module.load_measured_cnn_horizon(measured_root)
        with tempfile.TemporaryDirectory(
            prefix="tracking2-lw-figures-"
        ) as temporary:
            temporary_root = Path(temporary)
            rebuilt = {
                "cnn_measured_controls.png": temporary_root
                / "cnn_measured_controls.png",
                "cnn_measured_horizon.png": temporary_root
                / "cnn_measured_horizon.png",
            }
            module.save_figure(
                module.measured_cnn_controls_figure(control_data),
                [rebuilt["cnn_measured_controls.png"]],
                dpi=200,
            )
            module.save_figure(
                module.measured_cnn_horizon_figure(horizon_data),
                [rebuilt["cnn_measured_horizon.png"]],
                dpi=200,
            )
            for name, rebuilt_path in rebuilt.items():
                staged_path = figure_dir / name
                if rebuilt_path.read_bytes() != staged_path.read_bytes():
                    raise VerificationError(
                        f"staged figure rebuild is not byte-identical: {name}; "
                        f"staged={sha256(staged_path)}, rebuilt={sha256(rebuilt_path)}"
                    )
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"staged figure rebuild failed: {error}") from error
    finally:
        sys.modules.pop(module_name, None)


def verify_appendix_rebuild(manifest_path: Path, appendix_path: Path) -> str:
    committed_digest = sha256(appendix_path)
    with tempfile.TemporaryDirectory(prefix="tracking2-lw-verify-") as temporary:
        rebuilt = Path(temporary) / "appendix.html"
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        source_path = str(PROJECT_ROOT / "src")
        prior_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not prior_python_path
            else os.pathsep.join((source_path, prior_python_path))
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tracking2.post_report",
                str(manifest_path),
                "--output",
                str(rebuilt),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise VerificationError(f"appendix rebuild failed: {detail}")
        rebuilt_digest = sha256(rebuilt)
        if rebuilt.read_bytes() != appendix_path.read_bytes():
            raise VerificationError(
                "appendix rebuild is not byte-identical: "
                f"committed={committed_digest}, rebuilt={rebuilt_digest}"
            )
    return committed_digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify either the locked July 26 LW package or an isolated clean "
            "rerun staging layout, including deterministic figures and appendix."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("committed", "staged"),
        default="committed",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        help="Isolated canonical staging root; required by --profile staged.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--appendix", type=Path)
    parser.add_argument(
        "--skip-appendix-rebuild",
        action="store_true",
        help="Skip only the byte-identical appendix rebuild (for diagnostics).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        locked_snapshot = args.profile == "committed"
        if locked_snapshot:
            if args.stage_root is not None:
                raise VerificationError(
                    "--stage-root is valid only with --profile staged"
                )
            manifest_path = (args.manifest or DEFAULT_MANIFEST).resolve()
            appendix_path = (args.appendix or DEFAULT_APPENDIX).resolve()
            if manifest_path != DEFAULT_MANIFEST.resolve():
                raise VerificationError(
                    "the committed profile accepts only the canonical manifest"
                )
            if appendix_path != DEFAULT_APPENDIX.resolve():
                raise VerificationError(
                    "the committed profile accepts only the canonical appendix"
                )
            measured_root = MEASURED_ROOT.resolve()
            figure_dir = (PROJECT_ROOT / "LW post" / "figures").resolve()
            checkpoint_root = None
            source_archive = None
        else:
            if args.stage_root is None:
                raise VerificationError(
                    "--profile staged requires an isolated --stage-root"
                )
            if args.manifest is not None or args.appendix is not None:
                raise VerificationError(
                    "the staged profile derives paths from --stage-root; do not "
                    "mix explicit committed paths into the promotion gate"
                )
            stage_root = args.stage_root.resolve()
            manifest_path = (
                stage_root / "artifacts" / "lw_post" / "dashboard_manifest.json"
            )
            measured_root = manifest_path.parent / "measured"
            appendix_path = (
                stage_root
                / "LW post"
                / "free-body-diagrams-for-neural-networks.html"
            )
            figure_dir = stage_root / "LW post" / "figures"
            checkpoint_root = stage_root / "checkpoints"
            source_archive = stage_root / "source" / "source.tar"
        manifest = load_json(manifest_path)
        payloads = verify_manifest_inputs(
            manifest_path,
            manifest,
            measured_root=measured_root,
            locked_snapshot=locked_snapshot,
        )
        training, _training_digests = verify_training_manifests(
            payloads,
            measured_root=measured_root,
            locked_snapshot=locked_snapshot,
            checkpoint_root=checkpoint_root,
        )
        if locked_snapshot:
            verify_correction_sidecar(manifest_path, manifest)
        else:
            assert source_archive is not None
            verify_staged_provenance(
                manifest, payloads, training, source_archive
            )
        headlines = verify_primary_analysis(
            manifest,
            payloads,
            locked_snapshot=locked_snapshot,
        )
        verify_projection_gate(payloads, locked_snapshot=locked_snapshot)
        figure_digests = verify_figure_inputs(
            payloads,
            training,
            figure_dir=figure_dir,
            measured_root=measured_root,
            locked_snapshot=locked_snapshot,
        )
        appendix_digest = (
            sha256(appendix_path)
            if args.skip_appendix_rebuild
            else verify_appendix_rebuild(manifest_path, appendix_path)
        )
    except VerificationError as error:
        print(f"LW publication verification failed: {error}", file=sys.stderr)
        return 1
    profile_label = (
        "committed July 26 package"
        if locked_snapshot
        else "isolated clean-rerun staging package"
    )
    print(f"Verified the {profile_label}:")
    print(f"  measured manifest inputs: {len(payloads)}")
    print(f"  measured training manifests: {len(training)}")
    if locked_snapshot:
        print(f"  audited provenance corrections: {len(EXPECTED_CORRECTIONS)}")
    else:
        print("  source archive and epoch-30 checkpoints: digest-verified")
    for (cut, contrast), (mean, standard_deviation) in sorted(headlines.items()):
        print(
            f"  cut {cut} {contrast}: "
            f"{mean:.6f} ± {standard_deviation:.6f}"
        )
    print(f"  measured figures: {len(figure_digests)}")
    print(f"  appendix SHA-256: {appendix_digest}")
    if args.skip_appendix_rebuild:
        print("  appendix rebuild: skipped by request")
    else:
        print("  appendix rebuild: byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
