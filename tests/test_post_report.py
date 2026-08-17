from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.make_lw_post_manifest import (
    CANONICAL_RESNET_PATH,
    build_manifest,
    canonical_resnet_entries,
    explicit_post_cnn_entries,
    projection_adequacy_entries,
)
from tracking2.post_report import (
    DEFAULT_OUTPUT,
    GAUSSIAN,
    MEAN,
    PROJECTED_REAL,
    REAL,
    TITLE,
    ReportInputError,
    annotate_first_update_matching,
    build_report,
    load_manifest,
    main,
    normalise_artifacts,
    paired_true_eval_contrasts,
    primary_analysis_summary,
    summarise_paired_contrasts,
)


GIT_HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
CANONICAL_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "lw_post"
    / "dashboard_manifest.json"
)


def _records(distribution: str, start: float, end: float) -> list[dict]:
    return [
        {
            "draw": 0,
            "train_distribution": distribution,
            "eval_distribution": "true",
            "relax_epoch": 0,
            "loss": 1.0,
            "accuracy": start,
        },
        {
            "draw": 0,
            "train_distribution": distribution,
            "eval_distribution": "true",
            "relax_epoch": 1,
            "loss": 0.8,
            "accuracy": end,
        },
    ]


def _cnn_payload() -> dict:
    return {
        "schema_version": 1,
        "experiment": "lw_post_cnn_suffix_statistics",
        "status": "MEASURED",
        "architecture": "four-block residual CNN with GroupNorm",
        "checkpoint": {
            "epoch": 1,
            "path": "checkpoint_epoch1.pt",
            "sha256": "1" * 64,
        },
        "mean_noise_definition": "Trace matched at radius one.",
        "pca_protocol": "Held-out coverage.",
        "config": {
            "checkpoint_epoch": 1,
            "fake_data": False,
            "seed": 0,
            "surrogate_draws": 1,
            "pca_fit_size": 8,
            "pca_ranks": [2],
            "mean_noise_radii": [0.0, 1.0],
        },
        "slices": [
            {
                "cut": 1,
                "representation_shape": [4, 2, 2],
                "native_dimension": 16,
                "pca_fit_count": 8,
                "pca_fit_class_counts": {"0": 4, "1": 4},
                "reference_records": _records("true", 0.50, 0.80),
                "rank_results": [
                    {
                        "pca_rank": 2,
                        "held_out_explained_variance_fraction": 0.80,
                        "held_out_coverage": {
                            "total_variance_fraction": 0.80,
                            "within_class_variance_fraction": 0.75,
                            "between_class_mean_variance_fraction": 0.90,
                        },
                        "pooled_within_class_variance_per_pca_coordinate": 0.25,
                        "trace_matched_isotropic_covariance_trace": 0.50,
                        "covariance_shrinkage": 0.05,
                        "empirical_class_covariance_rank_ceiling": 3,
                        "rank_exceeds_empirical_class_covariance_ceiling": False,
                        "records": [
                            *_records("projected_true", 0.50, 0.75),
                            *_records("gaussian", 0.50, 0.70),
                            *_records("mean_r0", 0.50, 0.55),
                            *_records("mean_r1", 0.50, 0.60),
                        ],
                        "moment_diagnostics": [],
                    }
                ],
            }
        ],
        "runtime_seconds": 1.0,
    }


def _resnet_payload(*, held_out: bool = False) -> dict:
    slice_payload = {
        "cut": 3,
        "module": "stage2.resblk2",
        "condition": "native",
        "representation_shape": [4, 2, 2],
        "explained_variance_fraction": 0.70,
        "moment_diagnostics": [],
        "records": [
            *_records("true", 0.45, 0.78),
            *_records("gaussian", 0.45, 0.68),
            *_records("mean", 0.45, 0.58),
            {
                "draw": 0,
                "train_distribution": "true",
                "eval_distribution": "gaussian",
                "relax_epoch": 0,
                "loss": 1.0,
                "accuracy": 0.99,
            },
        ],
    }
    if held_out:
        slice_payload.update(
            {
                "held_out_explained_variance_fraction": 0.82,
                "held_out_coverage": {
                    "total_variance_fraction": 0.82,
                    "within_class_variance_fraction": 0.79,
                    "between_class_mean_variance_fraction": 0.91,
                },
                "coverage_evaluation": "held-out CIFAR-10 test activations",
            }
        )
    return {
        "schema_version": 2 if held_out else 1,
        "experiment": "resnet18_suffix_statistics_sweep",
        "status": "MEASURED",
        "config": {
            "checkpoint_epoch": 1,
            "fake_data": False,
            "seed": 0,
            "surrogate_draws": 1,
            "pca_fit_size": 8,
            "pca_components": 3,
        },
        "device": "cpu",
        "module_names": ["stage2.resblk2"],
        "slices": [slice_payload],
        "runtime_seconds": 1.0,
    }


def _cnn_v2_matrix_payload(
    learning_rate_regime: str = "fixed_lr",
    suffix_initialization: str = "warm",
) -> dict:
    payload = _cnn_payload()
    payload["schema_version"] = 2
    payload["provenance"] = {"source_revision": GIT_HEAD}
    payload["config"].update(
        {
            "relax_epochs": 1,
            "true_eval_only": False,
            "gaussian_covariance_shrinkages": [0.0, 0.05],
            "learning_rate_regime": learning_rate_regime,
            "suffix_initialization": suffix_initialization,
        }
    )
    distributions = [
        "true",
        "projected_true",
        "gaussian_empirical",
        "gaussian_shrunk_s05",
        "mean_r1",
    ]
    train_penalty = {
        "true": 0.0,
        "projected_true": 0.03,
        "gaussian_empirical": 0.10,
        "gaussian_shrunk_s05": 0.15,
        "mean_r1": 0.25,
    }
    records = []
    for train_distribution in distributions:
        lr_multiplier = (
            1.0
            if learning_rate_regime == "fixed_lr"
            else 1.0 / (1.0 + train_penalty[train_distribution])
        )
        for eval_distribution in distributions:
            for relax_epoch in (0, 1):
                records.append(
                    {
                        "draw": 0,
                        "train_distribution": train_distribution,
                        "eval_distribution": eval_distribution,
                        "relax_epoch": relax_epoch,
                        "loss": (
                            0.8
                            - 0.2 * relax_epoch
                            + train_penalty[train_distribution]
                            + 0.02
                            * (eval_distribution != train_distribution)
                        ),
                        "accuracy": (
                            0.5
                            + 0.2 * relax_epoch
                            - train_penalty[train_distribution] / 2
                        ),
                        "initial_training_loss": 0.9
                        + train_penalty[train_distribution],
                        "initial_gradient_norm": 1.0
                        + train_penalty[train_distribution],
                        "initial_gradient_rms": 0.1
                        + train_penalty[train_distribution] / 10,
                        "initial_suffix_parameter_count": 100,
                        "initial_suffix_weight_norm": 2.0,
                        "first_step_update_to_weight_ratio": 0.01
                        + train_penalty[train_distribution] / 100,
                        "learning_rate_regime": learning_rate_regime,
                        "suffix_initialization": suffix_initialization,
                        "base_relax_learning_rate": 0.01,
                        "learning_rate_multiplier": lr_multiplier,
                        "effective_relax_learning_rate": 0.01
                        * lr_multiplier,
                        "matched_first_step_update_norm": 0.1,
                    }
                )
    rank_result = payload["slices"][0]["rank_results"][0]
    rank_result["records"] = records
    rank_result["gaussian_covariance_estimators"] = [
        {
            "distribution": "gaussian_empirical",
            "requested_covariance_shrinkage": 0.0,
        },
        {
            "distribution": "gaussian_shrunk_s05",
            "requested_covariance_shrinkage": 0.05,
        },
    ]
    rank_result["moment_diagnostics"] = [
        {
            "draw": 0,
            "distribution": "gaussian_empirical",
            "class_mean_relative_error": 0.01,
            "class_covariance_relative_error": 0.04,
            "diagnostic_space": "fitted PCA subspace",
        }
    ]
    rank_result["step_zero_true_vs_projected"] = {
        "true_loss": 0.8,
        "projected_true_loss": 0.83,
        "true_accuracy": 0.50,
        "projected_true_accuracy": 0.49,
        "true_to_projected_predictive_kl": 0.02,
    }
    payload["slices"][0]["reference_records"] = [
        row
        for row in records
        if row["train_distribution"] == "true"
        and row["eval_distribution"] == "true"
    ]
    return payload


def _cnn_v2_noise_reversal_payload() -> dict:
    payload = _cnn_v2_matrix_payload()
    payload["config"]["true_eval_only"] = True
    rank_result = payload["slices"][0]["rank_results"][0]
    records = [
        row
        for row in rank_result["records"]
        if row["eval_distribution"] == "true"
    ]
    for row in records:
        if (
            row["train_distribution"] == "mean_r1"
            and row["relax_epoch"] == 1
        ):
            row["loss"] = 0.65
    template = next(
        row
        for row in records
        if row["train_distribution"] == "mean_r1"
        and row["relax_epoch"] == 0
    )
    for relax_epoch, loss in ((0, 0.9), (1, 0.8)):
        mean_zero = copy.deepcopy(template)
        mean_zero.update(
            {
                "train_distribution": "mean_r0",
                "relax_epoch": relax_epoch,
                "loss": loss,
                "accuracy": 0.5 + 0.05 * relax_epoch,
            }
        )
        records.append(mean_zero)
    rank_result["records"] = records
    payload["slices"][0]["reference_records"] = [
        row
        for row in records
        if row["train_distribution"] == "true"
    ]
    return payload


def _resnet_v3_payload() -> dict:
    block_names = [
        "stage1.resblk1",
        "stage1.resblk2",
        "stage2.resblk1",
        "stage2.resblk2",
        "stage3.resblk1",
        "stage3.resblk2",
        "stage4.resblk1",
        "stage4.resblk2",
    ]
    architecture = {
        "name": "InstrumentedResNet18V2",
        "width": 4,
        "block_names": block_names,
    }
    checkpoint = {
        "epoch": 1,
        "path": "checkpoint_epoch1.pt",
        "sha256": "2" * 64,
    }
    dataset = {
        "backend": "torchvision",
        "train": {"source_count": 50000, "source_sha256": "3" * 64},
        "test": {"source_count": 10000, "source_sha256": "4" * 64},
    }
    rank_results = []
    for rank, coverage in ((2, 0.70), (3, 0.82)):
        rank_results.append(
            {
                "pca_rank": rank,
                "maximal_pca_basis_sha256": "5" * 64,
                "pca_basis_prefix_sha256": str(rank) * 64,
                "pca_basis_fit_count": 6,
                "pca_basis_fit_class_counts": {"0": 3, "1": 3},
                "moment_fit_count": 8,
                "moment_fit_class_counts": {"0": 4, "1": 4},
                "empirical_class_covariance_rank_ceiling": 3,
                "rank_exceeds_empirical_class_covariance_ceiling": False,
                "covariance_shrinkage": 0.05,
                "pooled_within_class_variance_per_pca_coordinate": 0.25,
                "trace_matched_isotropic_covariance_trace": rank * 0.25,
                "held_out_explained_variance_fraction": coverage,
                "held_out_coverage": {
                    "total_variance_fraction": coverage,
                    "within_class_variance_fraction": coverage - 0.05,
                    "between_class_mean_variance_fraction": 0.90,
                },
                "coverage_evaluation": "held-out CIFAR-10 test activation bank",
                "records": [
                    *_records("projected_true", 0.45, 0.74),
                    *_records("gaussian", 0.45, 0.68),
                    *_records("mean_r0", 0.45, 0.54),
                    *_records("mean_r1", 0.45, 0.58),
                ],
                "moment_diagnostics": [],
            }
        )
    return {
        "schema_version": 3,
        "experiment": "resnet18_suffix_statistics_sweep",
        "status": "MEASURED",
        "mean_noise_definition": "Trace matched at radius one.",
        "pca_protocol": "Nested basis and paired maximal noise banks.",
        "checkpoint": checkpoint,
        "lineage": {
            "training_manifest": {
                "path": "resnet_training.json",
                "sha256": "6" * 64,
                "status": "MEASURED",
                "experiment": "resnet18_training_checkpoints",
            },
            "checkpoint": checkpoint,
            "model_seed": 0,
            "architecture": architecture,
            "training_source": {"source_revision": GIT_HEAD},
            "dataset_source": {
                "backend": dataset["backend"],
                "train": dataset["train"],
                "test": dataset["test"],
            },
        },
        "dataset": dataset,
        "analysis_banks": {
            "definition": "Fixed input banks reused across cuts and ranks.",
            "train": {"seed": 2_000_000, "count": 8, "sha256": "7" * 64},
            "test": {"seed": 2_000_001, "count": 4, "sha256": "8" * 64},
        },
        "config": {
            "checkpoint_epoch": 1,
            "data_backend": "torchvision",
            "fake_data": False,
            "train_size": 8,
            "test_size": 4,
            "width": 4,
            "cuts": [3],
            "seed": 0,
            "surrogate_draws": 1,
            "pca_fit_size": 6,
            "pca_ranks": [2, 3],
            "mean_noise_radii": [0.0, 1.0],
            "include_projected_true": True,
            "true_eval_only": True,
            "relax_epochs": 1,
        },
        "device": "cpu",
        "provenance": {
            "source_revision": GIT_HEAD,
            "source_archive_sha256": "not recorded",
        },
        "architecture": architecture,
        "module_names": block_names,
        "slices": [
            {
                "cut": 3,
                "module": "stage2.resblk2",
                "condition": "native",
                "representation_shape": [4, 2, 2],
                "native_dimension": 16,
                "maximal_pca_rank": 3,
                "maximal_pca_basis_sha256": "5" * 64,
                "pca_basis_fit_count": 6,
                "pca_basis_fit_class_counts": {"0": 3, "1": 3},
                "moment_fit_count": 8,
                "moment_fit_class_counts": {"0": 4, "1": 4},
                "reference_records": _records("true", 0.45, 0.78),
                "rank_results": rank_results,
            }
        ],
        "runtime_seconds": 1.0,
    }


def _resnet_v4_payload() -> dict:
    payload = _resnet_v3_payload()
    payload["schema_version"] = 4
    payload["config"].update(
        {
            "true_eval_only": False,
            "gaussian_covariance_shrinkages": [0.0, 0.05],
        }
    )
    distributions = [
        "true",
        "projected_true",
        "gaussian_empirical",
        "gaussian_shrunk_s0.05",
        "mean_r0",
        "mean_r1",
    ]
    first_reference = None
    for rank_result in payload["slices"][0]["rank_results"]:
        records = []
        for train_index, train_distribution in enumerate(distributions):
            for eval_index, eval_distribution in enumerate(distributions):
                for relax_epoch in (0, 1):
                    records.append(
                        {
                            "draw": 0,
                            "train_distribution": train_distribution,
                            "eval_distribution": eval_distribution,
                            "relax_epoch": relax_epoch,
                            "loss": (
                                0.9
                                - 0.2 * relax_epoch
                                + 0.03 * train_index
                                + 0.01 * eval_index
                            ),
                            "accuracy": (
                                0.45
                                + 0.2 * relax_epoch
                                - 0.01 * train_index
                            ),
                            "initial_training_loss": 1.0 + 0.03 * train_index,
                            "initial_gradient_norm": 2.0 + 0.03 * train_index,
                        }
                    )
        rank_result.pop("covariance_shrinkage")
        rank_result["gaussian_covariance_estimators"] = [
            {"distribution": "gaussian_empirical"},
            {"distribution": "gaussian_shrunk_s0.05"},
        ]
        rank_result["records"] = records
        if first_reference is None:
            first_reference = [
                row
                for row in records
                if row["train_distribution"] == "true"
                and row["eval_distribution"] == "true"
            ]
    payload["slices"][0]["reference_records"] = first_reference
    return payload


def _projection_adequacy_payload() -> dict:
    dataset = {
        "backend": "torchvision",
        "splits": {
            "train": {
                "count": 50000,
                "ordered_images_sha256": "1" * 64,
                "ordered_labels_sha256": "2" * 64,
            },
            "test": {
                "count": 10000,
                "ordered_images_sha256": "3" * 64,
                "ordered_labels_sha256": "4" * 64,
            },
        },
    }
    return {
        "schema_version": 1,
        "experiment": "lw_post_cnn_projection_adequacy",
        "status": "MEASURED",
        "config": {
            "checkpoint_epoch": 30,
            "fake_data": False,
            "seed": 0,
            "train_size": 50000,
            "test_size": 10000,
            "pca_fit_size": 5000,
            "cuts": [1],
            "pca_ranks": [2048, 4096],
            "widths": [32, 64, 128, 128],
        },
        "architecture": {
            "name": "four-block residual CNN with GroupNorm",
            "widths": [32, 64, 128, 128],
        },
        "lineage": {
            "checkpoint": {
                "epoch": 30,
                "path": "checkpoint_epoch30.pt",
                "sha256": "5" * 64,
            },
            "training_manifest": {
                "path": "training.json",
                "sha256": "6" * 64,
                "status": "MEASURED",
                "experiment": "lw_post_cnn_checkpoint_trajectory",
                "source_provenance": {"source_revision": GIT_HEAD},
                "dataset": dataset,
            },
        },
        "dataset": dataset,
        "pca_fit_prefix": {
            "count": 5000,
            "ordered_images_sha256": "7" * 64,
            "ordered_labels_sha256": "8" * 64,
            "selection": "range(0, 5000)",
        },
        "provenance": {"source_revision": GIT_HEAD},
        "device": "cpu",
        "slices": [
            {
                "cut": 1,
                "representation_shape": [64, 32, 32],
                "activation_flattened_dimension": 65536,
                "pca_fit_count": 5000,
                "pca_fit_class_counts": {
                    str(class_id): 500 for class_id in range(10)
                },
                "sample_rank_ceiling": 4999,
                "activation_dimension_rank_ceiling": 65536,
                "maximal_feasible_rank": 4999,
                "maximal_requested_rank": 4096,
                "maximal_pca_basis_sha256": "d" * 64,
                "rank_results": [
                    {
                        "pca_rank": 2048,
                        "pca_basis_prefix_sha256": "c" * 64,
                        "held_out_coverage": {
                            "total_variance_fraction": 0.94,
                            "within_class_variance_fraction": 0.91,
                            "between_class_mean_variance_fraction": 0.99,
                        },
                        "step_zero_true_vs_projected": {
                            "true_loss": 0.5,
                            "true_accuracy": 0.82,
                            "projected_true_loss": 0.53,
                            "projected_true_accuracy": 0.81,
                            "true_to_projected_predictive_kl": 0.025,
                        },
                    },
                    {
                        "pca_rank": 4096,
                        "pca_basis_prefix_sha256": "d" * 64,
                        "held_out_coverage": {
                            "total_variance_fraction": 0.98,
                            "within_class_variance_fraction": 0.97,
                            "between_class_mean_variance_fraction": 0.995,
                        },
                        "step_zero_true_vs_projected": {
                            "true_loss": 0.5,
                            "true_accuracy": 0.82,
                            "projected_true_loss": 0.52,
                            "projected_true_accuracy": 0.815,
                            "true_to_projected_predictive_kl": 0.015,
                        },
                    },
                ],
            }
        ],
    }


def _canonical_resnet_entry_payload() -> dict:
    payload = copy.deepcopy(_resnet_v3_payload())
    checkpoint = {
        "epoch": 100,
        "path": "checkpoint_epoch100.pt",
        "sha256": "2" * 64,
    }
    architecture = {
        **payload["architecture"],
        "width": 64,
    }
    payload["checkpoint"] = checkpoint
    payload["lineage"]["checkpoint"] = checkpoint
    payload["lineage"]["architecture"] = architecture
    payload["architecture"] = architecture
    payload["config"].update(
        {
            "checkpoint": "checkpoint_epoch100.pt",
            "training_manifest": "resnet_training.json",
            "checkpoint_epoch": 100,
            "train_size": 10000,
            "test_size": 2000,
            "batch_size": 128,
            "width": 64,
            "cuts": [3, 7],
            "pca_fit_size": 5000,
            "pca_ranks": [128, 256, 512],
            "surrogate_draws": 3,
            "mean_noise_radii": [1.0],
            "include_projected_true": True,
            "true_eval_only": True,
            "relax_epochs": 5,
            "relax_learning_rate": 0.01,
        }
    )
    return payload


def _legacy_cnn_payload() -> dict:
    records = []
    for distribution, start, end in (
        ("true", 0.50, 0.80),
        ("gaussian", 0.50, 0.70),
        ("mean", 0.50, 0.60),
    ):
        for row in _records(distribution, start, end):
            legacy_row = dict(row)
            legacy_row.pop("draw")
            legacy_row["relax_epoch"] = float(legacy_row["relax_epoch"])
            records.append(legacy_row)
        records.append(
            {
                "train_distribution": distribution,
                "eval_distribution": "gaussian",
                "relax_epoch": 0.0,
                "relax_batch": 0,
                "loss": 0.5,
                "accuracy": 0.9,
            }
        )
    return {
        "status": "MEASURED",
        "config": {
            "checkpoint_epoch": 1,
            "checkpoint_batches": None,
            "cut": 3,
            "widths": [32, 64, 128, 128],
            "fake_data": False,
            "seed": 0,
            "pca_fit_size": 10000,
            "pca_components": 256,
        },
        "device": "cuda",
        "representation_shape": [128, 8, 8],
        "explained_variance_fraction": 0.62,
        "moment_diagnostics": {},
        "records": records,
        "runtime_seconds": 1.0,
    }


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest(
    tmp_path: Path,
    *,
    cnn: dict | None = None,
    resnet: dict | None = None,
    cnn_format: str = "post_statistics",
) -> Path:
    cnn = _cnn_payload() if cnn is None else cnn
    resnet = _resnet_payload() if resnet is None else resnet
    cnn_path = tmp_path / "cnn.json"
    resnet_path = tmp_path / "resnet.json"
    cnn_digest = _write_json(cnn_path, cnn)
    resnet_digest = _write_json(resnet_path, resnet)
    manifest = {
        "schema_version": 1,
        "title": TITLE,
        "status": "MEASURED",
        "source_commit": GIT_HEAD,
        "inputs": [
            {
                "id": "cnn-epoch-1",
                "kind": "cnn",
                "format": cnn_format,
                "label": "Four-block CNN, epoch 1",
                "path": cnn_path.name,
                "sha256": cnn_digest,
                "checkpoint_epoch": 1,
                "model_seed": 0,
            },
            {
                "id": "resnet-epoch-1",
                "kind": "resnet",
                "format": "resnet_suffix_statistics",
                "label": "ResNet-18, epoch 1",
                "path": resnet_path.name,
                "sha256": resnet_digest,
                "checkpoint_epoch": 1,
                "model_seed": 0,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _append_manifest_input(
    manifest_path: Path,
    *,
    payload: dict,
    input_id: str,
    kind: str,
    artifact_format: str,
    label: str,
    epoch: int,
    seed: int,
) -> Path:
    artifact_path = manifest_path.parent / f"{input_id}.json"
    artifact_digest = _write_json(artifact_path, payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"].append(
        {
            "id": input_id,
            "kind": kind,
            "format": artifact_format,
            "label": label,
            "path": artifact_path.name,
            "sha256": artifact_digest,
            "checkpoint_epoch": epoch,
            "model_seed": seed,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest_path


def test_builds_measured_only_self_contained_report(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    output = build_report(manifest_path, tmp_path / DEFAULT_OUTPUT)
    rendered = output.read_text()

    assert f"<title>{TITLE}</title>" in rendered
    assert rendered.count(TITLE) >= 2
    assert "Four-block CNN" in rendered
    assert "ResNet-18" in rendered
    assert "Prefix from checkpoint at epoch: 1" in rendered
    assert "Suffix relaxation epoch" in rendered
    assert "Held-out real-data accuracy (%)" in rendered
    assert "Download embedded data (JSON)" in rendered
    assert 'id="embedded-data"' in rendered
    assert "not epsilon jitter" in rendered
    assert "legacy reported value; evaluation population unspecified" in rendered
    assert "80.0%" in rendered
    for colour in (REAL, PROJECTED_REAL, GAUSSIAN, MEAN):
        assert colour in rendered
    assert 'stroke-dasharray="7 5"' in rendered
    for forbidden in ("MOCKUP", "PLANNED", "UNRUN"):
        assert forbidden not in rendered

    endpoint = rendered.index('<div class="endpoint-panel">')
    next_panel = rendered.index("</figure>", endpoint)
    panel = rendered[endpoint:next_panel]
    assert panel.index("Real data") < panel.index("Projected real")
    assert panel.index("Projected real") < panel.index("Gaussian")
    assert panel.index("Gaussian") < panel.index("Mean + isotropic noise")


def test_manifest_fails_loudly_for_missing_file_or_digest_mismatch(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    manifest["inputs"][0]["path"] = "missing.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReportInputError, match="is missing"):
        load_manifest(manifest_path)

    manifest_path = _make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReportInputError, match="SHA-256 mismatch"):
        load_manifest(manifest_path)


def test_rejects_non_measured_or_mismatched_inputs(tmp_path):
    cnn = _cnn_payload()
    cnn["status"] = "MOCKUP / PIPELINE SMOKE TEST"
    manifest_path = _make_manifest(tmp_path, cnn=cnn)
    with pytest.raises(ReportInputError, match="status must be exactly"):
        load_manifest(manifest_path)

    manifest_path = _make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"][1]["kind"] = "cnn"
    manifest["inputs"][1]["format"] = "post_statistics"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReportInputError, match="is not a cnn artifact"):
        load_manifest(manifest_path)

    manifest_path = _make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"][0]["checkpoint_epoch"] = 5
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReportInputError, match="Checkpoint epoch mismatch"):
        load_manifest(manifest_path)


def test_new_resnet_held_out_coverage_overrides_legacy_field(tmp_path):
    manifest_path = _make_manifest(tmp_path, resnet=_resnet_payload(held_out=True))
    _manifest, artifacts = load_manifest(manifest_path)
    _observations, cells = normalise_artifacts(artifacts)
    resnet_cell = next(cell for cell in cells if cell["kind"] == "resnet")
    assert resnet_cell["coverage"] == pytest.approx(0.82)
    assert resnet_cell["coverage_basis"] == "held-out activations"

    rendered = build_report(manifest_path, tmp_path / "report.html").read_text()
    assert "82.0%" in rendered


def test_tiny_floating_point_coverage_overshoot_is_clamped(tmp_path):
    cnn = _cnn_v2_matrix_payload()
    rank_result = cnn["slices"][0]["rank_results"][0]
    rank_result["held_out_explained_variance_fraction"] = 1.00000006
    rank_result["held_out_coverage"][
        "total_variance_fraction"
    ] = 1.00000006
    manifest_path = _make_manifest(tmp_path, cnn=cnn)
    _manifest, artifacts = load_manifest(manifest_path)
    _observations, cells = normalise_artifacts(artifacts)
    cnn_cell = next(cell for cell in cells if cell["kind"] == "cnn")

    assert cnn_cell["coverage_total"] == 1.0


def test_schema_v3_resnet_loads_normalises_and_renders(tmp_path):
    manifest_path = _make_manifest(tmp_path, resnet=_resnet_v3_payload())
    _manifest, artifacts = load_manifest(manifest_path)
    observations, cells = normalise_artifacts(artifacts)

    resnet_cells = [cell for cell in cells if cell["kind"] == "resnet"]
    assert [cell["pca_rank"] for cell in resnet_cells] == [2, 3]
    assert {cell["coverage_basis"] for cell in resnet_cells} == {
        "held-out activations"
    }
    assert all(cell["pca_fit_count"] == 6 for cell in resnet_cells)
    assert all(cell["moment_fit_count"] == 8 for cell in resnet_cells)
    assert all(cell["paired_nested_noise"] is True for cell in resnet_cells)
    assert all(cell["fixed_analysis_bank"] is True for cell in resnet_cells)
    assert len(
        [row for row in observations if row["kind"] == "resnet"]
    ) == 20

    rendered = build_report(manifest_path, tmp_path / "schema-v3.html").read_text()
    assert "The canonical ResNet control uses fingerprinted fixed analysis banks" in rendered
    assert "Paired rank comparison" in rendered
    assert "Trace-matched isotropic noise" in rendered
    assert "Fixed analysis bank" in rendered
    assert "Full-bank class moments" in rendered
    assert "Source Revision" in rendered
    assert "Checkpoint" in rendered


def test_schema_v3_resnet_rejects_missing_provenance_and_invalid_grid(tmp_path):
    missing_provenance = _resnet_v3_payload()
    missing_provenance["provenance"] = {
        "source_revision": "not recorded",
        "source_archive_sha256": "not recorded",
    }
    manifest_path = _make_manifest(tmp_path, resnet=missing_provenance)
    with pytest.raises(ReportInputError, match="must record a full clean source"):
        load_manifest(manifest_path)

    invalid_grid = _resnet_v3_payload()
    invalid_grid["slices"][0]["rank_results"][0]["records"].pop()
    manifest_path = _make_manifest(tmp_path, resnet=invalid_grid)
    _manifest, artifacts = load_manifest(manifest_path)
    with pytest.raises(ReportInputError, match="mismatched draw/relaxation grids"):
        normalise_artifacts(artifacts)


def test_schema_v2_cnn_retains_full_matrix_and_primary_loss_estimand(tmp_path):
    manifest_path = _make_manifest(tmp_path, cnn=_cnn_v2_matrix_payload())
    _manifest, artifacts = load_manifest(manifest_path)
    observations, _cells = normalise_artifacts(artifacts)
    cnn_observations = [
        row for row in observations if row["kind"] == "cnn"
    ]

    assert len(cnn_observations) == 50
    assert {row["eval_distribution"] for row in cnn_observations} == {
        "true",
        "projected_true",
        "gaussian_empirical",
        "gaussian_shrunk_s05",
        "mean_r1",
    }
    assert all(
        row["initial_gradient_rms"] is not None
        and row["initial_suffix_parameter_count"] == 100
        and row["initial_update_to_weight_ratio"] is not None
        for row in cnn_observations
    )
    paired = summarise_paired_contrasts(
        paired_true_eval_contrasts(cnn_observations)
    )
    empirical_endpoint = next(
        row
        for row in paired
        if row["distribution"] == "gaussian_empirical"
        and row["relax_epoch"] == 1
    )
    assert empirical_endpoint["loss_excess_mean"] == pytest.approx(0.12)

    rendered = build_report(manifest_path, tmp_path / "schema-v2.html").read_text()
    assert "DIAGNOSTIC BUILD" in rendered
    assert "version-1 compatibility manifest" in rendered
    assert "Full train × evaluation loss matrix" in rendered
    assert "Gaussian · empirical covariance" in rendered
    assert "Gaussian · 5% spherical shrinkage" in rendered
    assert "Initial suffix-gradient scale" in rendered
    assert "Class-covariance relative error" in rendered
    assert "Step-zero PCA functional change" in rendered
    assert "Predictive KL(true ∥ projected)" in rendered


def test_schema_v4_resnet_is_canonical_and_requires_complete_matrix(tmp_path):
    manifest_path = _make_manifest(tmp_path, resnet=_resnet_v4_payload())
    _manifest, artifacts = load_manifest(manifest_path)
    observations, _cells = normalise_artifacts(artifacts)
    resnet_observations = [
        row for row in observations if row["kind"] == "resnet"
    ]
    assert len(resnet_observations) == 2 * 6 * 6 * 2

    rendered = build_report(manifest_path, tmp_path / "schema-v4.html").read_text()
    assert "DIAGNOSTIC BUILD" in rendered
    assert "Canonical measured evidence" in rendered
    assert "Legacy: ResNet-18" not in rendered
    assert "Gaussian · 5% spherical shrinkage" in rendered

    incomplete = _resnet_v4_payload()
    incomplete["slices"][0]["rank_results"][0]["records"].pop()
    manifest_path = _make_manifest(tmp_path, resnet=incomplete)
    _manifest, artifacts = load_manifest(manifest_path)
    with pytest.raises(ReportInputError, match="incomplete train/evaluation matrix"):
        normalise_artifacts(artifacts)


def test_projection_adequacy_artifact_renders_artifact_driven_rank_gates(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    _append_manifest_input(
        manifest_path,
        payload=_projection_adequacy_payload(),
        input_id="cnn-projection-s0-e30",
        kind="cnn_projection",
        artifact_format="cnn_projection_adequacy",
        label="CNN seed 0 epoch 30 projection adequacy",
        epoch=30,
        seed=0,
    )
    _manifest, artifacts = load_manifest(manifest_path)
    assert any(artifact.kind == "cnn_projection" for artifact in artifacts)

    rendered = build_report(
        manifest_path, tmp_path / "projection-adequacy.html"
    ).read_text()
    assert "Does the PCA rank preserve the suffix function?" in rendered
    assert "65,536" in rendered
    assert "SHALLOW r = 2,048 FAILS THE KL GATE" in rendered
    assert "only a projected-subspace estimand" in rendered
    assert "SHALLOW r = 4,096 PASSES BOTH DECLARED GATES" in rendered
    assert "For seed 0" in rendered
    assert "0.02500" in rendered
    assert "0.01500" in rendered
    assert "pca_projection_adequacy_rows" in rendered


def test_projection_adequacy_rejects_bad_lineage_grid_and_numerics(tmp_path):
    bad_lineage = _projection_adequacy_payload()
    bad_lineage["lineage"]["training_manifest"]["source_provenance"] = {
        "source_revision": "not recorded"
    }
    lineage_dir = tmp_path / "lineage"
    lineage_dir.mkdir()
    manifest_path = _make_manifest(lineage_dir)
    _append_manifest_input(
        manifest_path,
        payload=bad_lineage,
        input_id="bad-projection-lineage",
        kind="cnn_projection",
        artifact_format="cnn_projection_adequacy",
        label="bad lineage",
        epoch=30,
        seed=0,
    )
    with pytest.raises(ReportInputError, match="full clean source"):
        load_manifest(manifest_path)

    bad_grid = _projection_adequacy_payload()
    bad_grid["slices"][0]["rank_results"].pop()
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    manifest_path = _make_manifest(grid_dir)
    _append_manifest_input(
        manifest_path,
        payload=bad_grid,
        input_id="bad-projection-grid",
        kind="cnn_projection",
        artifact_format="cnn_projection_adequacy",
        label="bad grid",
        epoch=30,
        seed=0,
    )
    with pytest.raises(ReportInputError, match="must match config.pca_ranks"):
        load_manifest(manifest_path)

    bad_numeric = _projection_adequacy_payload()
    bad_numeric["slices"][0]["rank_results"][0]["held_out_coverage"][
        "within_class_variance_fraction"
    ] = 1.5
    numeric_dir = tmp_path / "numeric"
    numeric_dir.mkdir()
    manifest_path = _make_manifest(numeric_dir)
    _append_manifest_input(
        manifest_path,
        payload=bad_numeric,
        input_id="bad-projection-numeric",
        kind="cnn_projection",
        artifact_format="cnn_projection_adequacy",
        label="bad numeric",
        epoch=30,
        seed=0,
    )
    with pytest.raises(ReportInputError, match="must be in \\[0, 1\\]"):
        load_manifest(manifest_path)


def test_matched_update_regime_is_separate_optimizer_sensitivity(tmp_path):
    manifest_path = _make_manifest(
        tmp_path, cnn=_cnn_v2_matrix_payload("fixed_lr")
    )
    _append_manifest_input(
        manifest_path,
        payload=_cnn_v2_matrix_payload("match_true_initial_update"),
        input_id="cnn-matched-update-s0-e1",
        kind="cnn",
        artifact_format="post_statistics",
        label="CNN matched-update scale control",
        epoch=1,
        seed=0,
    )
    _manifest, artifacts = load_manifest(manifest_path)
    observations, _cells = normalise_artifacts(artifacts)
    matched = [
        row
        for row in observations
        if row["lr_regime"] == "match_true_initial_update"
    ]
    assert matched
    assert all(
        row["base_learning_rate"] == pytest.approx(0.01)
        and row["learning_rate_multiplier"] is not None
        and row["effective_learning_rate"]
        == pytest.approx(
            row["base_learning_rate"] * row["learning_rate_multiplier"]
        )
        and row["matched_first_step_update_norm"] == pytest.approx(0.1)
        for row in matched
    )

    rendered = build_report(
        manifest_path, tmp_path / "optimizer-sensitivity.html"
    ).read_text()
    assert "Matched first updates are primary" in rendered
    assert "fixed LR is the optimizer-shock sensitivity" in rendered
    assert "Matched-first-update" in rendered
    assert "scale-control sensitivity" in rendered
    assert "magnitude claims optimizer-regime-dependent" in rendered
    assert "Base LR" in rendered
    assert "LR multiplier" in rendered
    assert "Effective LR" in rendered
    assert "Matched first-update norm" in rendered
    assert "Matched-first-update primary regime" in rendered


def test_noise_radius_reversal_is_artifact_driven_and_prominent(tmp_path):
    manifest_path = _make_manifest(
        tmp_path, cnn=_cnn_v2_noise_reversal_payload()
    )
    rendered = build_report(
        manifest_path, tmp_path / "noise-reversal.html"
    ).read_text()

    assert "GAUSSIAN–MEAN ORDERING REVERSAL" in rendered
    assert "radius-dependent for that stratum" in rendered
    assert "Gaussian − mean r=0" in rendered
    assert "Gaussian − mean r=1" in rendered
    assert "<td>YES</td>" in rendered


def test_covariance_sensitivity_reports_delta_without_pass_threshold(tmp_path):
    manifest_path = _make_manifest(
        tmp_path, cnn=_cnn_v2_matrix_payload()
    )
    rendered = build_report(
        manifest_path, tmp_path / "covariance-sensitivity.html"
    ).read_text()

    assert "Covariance-shrinkage sensitivity" in rendered
    assert "Shrunk − exact" in rendered
    assert "No pass threshold was preregistered" in rendered
    assert "assigns no pass/fail label" in rendered


def test_reinitialized_suffix_is_never_merged_with_warm_replicates(tmp_path):
    manifest_path = _make_manifest(
        tmp_path, cnn=_cnn_v2_matrix_payload(suffix_initialization="warm")
    )
    _append_manifest_input(
        manifest_path,
        payload=_cnn_v2_matrix_payload(
            suffix_initialization="reinitialized"
        ),
        input_id="cnn-reinitialized-s0-e1",
        kind="cnn",
        artifact_format="post_statistics",
        label="CNN reinitialized suffix control",
        epoch=1,
        seed=0,
    )
    _manifest, artifacts = load_manifest(manifest_path)
    observations, _cells = normalise_artifacts(artifacts)
    paired = paired_true_eval_contrasts(observations)
    summaries = summarise_paired_contrasts(paired)

    assert {row["suffix_initialization"] for row in paired} == {
        "warm",
        "reinitialized",
    }
    empirical_endpoints = [
        row
        for row in summaries
        if row["distribution"] == "gaussian_empirical"
        and row["relax_epoch"] == 1
    ]
    assert len(empirical_endpoints) == 2
    assert {row["suffix_initialization"] for row in empirical_endpoints} == {
        "warm",
        "reinitialized",
    }
    assert all(row["draw_count"] == 1 for row in empirical_endpoints)

    rendered = build_report(
        manifest_path, tmp_path / "reinitialized.html"
        ).read_text()
    assert "Reinitialized suffixes are a separate control stratum" in rendered
    assert "later cuts still leave a different amount of the network" in rendered
    assert "warm suffix" in rendered
    assert "reinitialized suffix" in rendered
    optimizer_start = rendered.index(
        "<h3>Optimizer-regime outcome sensitivity</h3>"
    )
    optimizer_end = rendered.index("<h3>Surrogate moment fidelity</h3>")
    optimizer_block = rendered[optimizer_start:optimizer_end]
    assert "<td>reinitialized</td>" not in optimizer_block


def test_canonical_resnet_entries_accepts_canonical_profile_at_fixed_path(tmp_path):
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "lw_post" / "dashboard_manifest.json"
    artifact_path = artifact_root / CANONICAL_RESNET_PATH
    artifact_path.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_digest = _write_json(
        artifact_path, _canonical_resnet_entry_payload()
    )

    entries = canonical_resnet_entries(artifact_root, output)

    assert entries == [
        {
            "id": "resnet-e100-pca-ablation-cuts4-8",
            "kind": "resnet",
            "format": "resnet_suffix_statistics",
            "label": (
                "ResNet-18 · epoch 100 · nested PCA controls · cuts 4 and 8"
            ),
            "path": f"../{CANONICAL_RESNET_PATH}",
            "sha256": artifact_digest,
            "checkpoint_epoch": 100,
            "model_seed": 0,
        }
    ]


def test_manifest_accepts_explicit_projection_and_optimizer_sensitivity_paths(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "lw_post" / "dashboard_manifest.json"
    output.parent.mkdir(parents=True)
    projection_path = (
        artifact_root / "lw_post" / "projection" / "cnn_projection_adequacy.json"
    )
    projection_path.parent.mkdir(parents=True)
    projection_digest = _write_json(
        projection_path, _projection_adequacy_payload()
    )
    matched_path = (
        artifact_root / "lw_post" / "matched" / "post_statistics.json"
    )
    matched_path.parent.mkdir(parents=True)
    matched_digest = _write_json(
        matched_path,
        _cnn_v2_matrix_payload("match_true_initial_update"),
    )

    projection_entries = projection_adequacy_entries(
        output, [projection_path]
    )
    sensitivity_entries = explicit_post_cnn_entries(
        output, [matched_path]
    )
    assert projection_entries == [
        {
            "id": "cnn-projection-0-s0-e30",
            "kind": "cnn_projection",
            "format": "cnn_projection_adequacy",
            "label": (
                "Four-block CNN · seed 0 · epoch 30 · "
                "PCA-only adequacy gate"
            ),
            "path": "../lw_post/projection/cnn_projection_adequacy.json",
            "sha256": projection_digest,
            "checkpoint_epoch": 30,
            "model_seed": 0,
        }
    ]
    assert sensitivity_entries[0]["kind"] == "cnn"
    assert sensitivity_entries[0]["format"] == "post_statistics"
    assert sensitivity_entries[0]["sha256"] == matched_digest
    assert "match_true_initial_update" in sensitivity_entries[0]["label"]


def test_manifest_can_use_only_explicit_measured_inputs(tmp_path):
    output = tmp_path / "artifacts" / "lw_post" / "dashboard_manifest.json"
    output.parent.mkdir(parents=True)
    matched_path = output.parent / "matched.json"
    _write_json(
        matched_path,
        _cnn_v2_matrix_payload("match_true_initial_update"),
    )
    manifest = build_manifest(
        tmp_path / "unused",
        output,
        GIT_HEAD,
        cnn_source="none",
        extra_cnn_paths=[matched_path],
    )
    assert len(manifest["inputs"]) == 1
    assert manifest["inputs"][0]["format"] == "post_statistics"


def test_publication_builder_rejects_archive_only_discovery_modes(tmp_path):
    output = tmp_path / "dashboard_manifest.json"
    with pytest.raises(RuntimeError, match="legacy CNN discovery mode"):
        build_manifest(tmp_path, output, GIT_HEAD, cnn_source="legacy")
    with pytest.raises(RuntimeError, match="legacy ResNet discovery mode"):
        build_manifest(
            tmp_path,
            output,
            GIT_HEAD,
            cnn_source="none",
            resnet_source="legacy",
        )


def test_unresolvable_git_hash_requires_exact_audited_correction(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    replacement = "0" if GIT_HEAD[20] != "0" else "1"
    recorded = GIT_HEAD[:20] + replacement + GIT_HEAD[21:]
    manifest["source_commit"] = recorded
    _write_json(manifest_path, manifest)

    with pytest.raises(ReportInputError, match="Unresolvable Git revision"):
        load_manifest(manifest_path)

    sidecar_path = tmp_path / "provenance_corrections.json"
    sidecar_digest = _write_json(
        sidecar_path,
        {
            "schema_version": 1,
            "status": "AUDITED",
            "corrections": [
                {
                    "recorded_value": recorded,
                    "intended_commit": GIT_HEAD,
                    "unique_prefix_evidence": {
                        "prefix": GIT_HEAD[:12],
                        "resolved_commit": GIT_HEAD,
                    },
                    "source_archive_sha256": None,
                    "source_archive_status": (
                        "not_applicable_manifest_builder_revision"
                    ),
                    "affected_locations": [
                        {
                            "path": manifest_path.name,
                            "json_pointer": "/source_commit",
                        }
                    ],
                }
            ],
        },
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance_corrections"] = {
        "path": sidecar_path.name,
        "sha256": sidecar_digest,
    }
    _write_json(manifest_path, manifest)

    loaded_manifest, _artifacts = load_manifest(manifest_path)
    audit = loaded_manifest["provenance_correction_audit"]
    assert audit["corrections"][0]["recorded_value"] == recorded
    assert audit["corrections"][0]["intended_commit"] == GIT_HEAD


def test_publication_contract_recomputes_headlines_without_pooling(tmp_path):
    manifest, artifacts = load_manifest(CANONICAL_MANIFEST)
    observations, _cells = normalise_artifacts(artifacts)
    summary = primary_analysis_summary(manifest, observations)
    aggregates = {
        (row["cut"], row["contrast_id"]): row
        for row in summary["aggregate_contrasts"]
    }
    assert aggregates[(1, "gaussian_minus_projected_true")]["mean"] == pytest.approx(
        0.18291344839731857
    )
    assert aggregates[(1, "gaussian_minus_projected_true")][
        "sample_standard_deviation"
    ] == pytest.approx(0.05506913979079249)
    assert aggregates[(1, "mean_r1_minus_gaussian")]["mean"] == pytest.approx(
        0.1440512680689494
    )
    assert aggregates[(1, "mean_r1_minus_gaussian")][
        "sample_standard_deviation"
    ] == pytest.approx(0.019072251608814983)
    assert aggregates[(4, "gaussian_minus_projected_true")]["mean"] == pytest.approx(
        -0.012494543361663854
    )
    assert aggregates[(4, "mean_r1_minus_gaussian")]["mean"] == pytest.approx(
        0.0227389348347982
    )
    assert {
        row["input_id"] for row in summary["seed_contrasts"]
    } == {
        "cnn-extra-2-s0-e30",
        "cnn-extra-6-s1-e30",
        "cnn-extra-7-s2-e30",
    }
    seed_zero_shallow = next(
        row
        for row in summary["seed_contrasts"]
        if row["model_seed"] == 0
        and row["cut"] == 1
        and row["contrast_id"] == "gaussian_minus_projected_true"
    )
    assert seed_zero_shallow["value"] == pytest.approx(0.11965266590118406)
    assert summary["projection_scope"] == "retained_pca_subspace_only"

    annotated = annotate_first_update_matching(
        observations, manifest["primary_analysis"]["first_update_matching"]
    )
    radius_zero = next(
        row
        for row in annotated
        if row["input_id"] == "cnn-extra-2-s0-e30"
        and row["cut"] == 1
        and row["distribution"] == "mean_r0"
        and row["eval_distribution"] == "true"
        and row["relax_epoch"] == 0
    )
    assert radius_zero["first_update_match_status"] == (
        "approximately_matched_clipped"
    )
    assert radius_zero["learning_rate_multiplier_clipped"] is True
    assert radius_zero["first_update_relative_error"] == pytest.approx(
        0.09463195693378723
    )

    first = build_report(CANONICAL_MANIFEST, tmp_path / "first.html")
    second = build_report(CANONICAL_MANIFEST, tmp_path / "second.html")
    assert first.read_bytes() == second.read_bytes()
    rendered = first.read_text()
    for value in (
        "+0.182913 ± 0.055069",
        "+0.144051 ± 0.019072",
        "-0.012495 ± 0.001530",
        "+0.022739 ± 0.010517",
    ):
        assert value in rendered
    assert "approximately matched / clipped" in rendered
    assert "9.463%" in rendered
    assert "Fixed-LR optimizer-shock sensitivity" in rendered
    assert "retained PCA" in rendered
    assert "AUDITED PROVENANCE CORRECTIONS APPLIED" in rendered


def test_publication_headline_assertions_fail_on_drift():
    manifest, artifacts = load_manifest(CANONICAL_MANIFEST)
    observations, _cells = normalise_artifacts(artifacts)
    altered = copy.deepcopy(manifest)
    altered["primary_analysis"]["headline_assertions"][0]["mean"] += 0.01
    with pytest.raises(ReportInputError, match="headline assertion failed"):
        primary_analysis_summary(altered, observations)


def test_explicit_legacy_cnn_format_builds_and_discloses_limitations(tmp_path):
    manifest_path = _make_manifest(
        tmp_path,
        cnn=_legacy_cnn_payload(),
        cnn_format="legacy_cnn_suffix_statistics",
    )
    rendered = build_report(manifest_path, tmp_path / "legacy.html").read_text()
    assert "Legacy CNN grid." in rendered
    assert "separately trained and scheduled models" in rendered
    assert "neither one prefix-training trajectory nor" in rendered
    assert "legacy fit-bank value, not held-out coverage" in rendered
    assert "62.0%" in rendered


def test_cli_uses_canonical_default_filename(tmp_path, monkeypatch, capsys):
    manifest_path = _make_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main([str(manifest_path)]) == 0
    assert (tmp_path / DEFAULT_OUTPUT).is_file()
    assert capsys.readouterr().out.strip() == DEFAULT_OUTPUT
