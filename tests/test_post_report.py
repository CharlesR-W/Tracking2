from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.make_lw_post_manifest import (
    CANONICAL_RESNET_PATH,
    canonical_resnet_entries,
)
from tracking2.post_report import (
    DEFAULT_OUTPUT,
    GAUSSIAN,
    MEAN,
    PROJECTED_REAL,
    REAL,
    TITLE,
    ReportInputError,
    build_report,
    load_manifest,
    main,
    normalise_artifacts,
    paired_true_eval_contrasts,
    summarise_paired_contrasts,
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


def _cnn_v2_matrix_payload() -> dict:
    payload = _cnn_payload()
    payload["schema_version"] = 2
    payload["provenance"] = {"source_revision": "a" * 40}
    payload["config"].update(
        {
            "relax_epochs": 1,
            "true_eval_only": False,
            "gaussian_covariance_shrinkages": [0.0, 0.05],
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
            "training_source": {"source_revision": "b" * 40},
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
            "source_revision": "a" * 40,
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
        "source_commit": "deadbeef",
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
    assert "PRIMARY ESTIMAND AVAILABLE" in rendered
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
    assert "PRIMARY ESTIMAND AVAILABLE" in rendered
    assert "Canonical measured evidence" in rendered
    assert "Legacy: ResNet-18" not in rendered
    assert "Gaussian · 5% spherical shrinkage" in rendered

    incomplete = _resnet_v4_payload()
    incomplete["slices"][0]["rank_results"][0]["records"].pop()
    manifest_path = _make_manifest(tmp_path, resnet=incomplete)
    _manifest, artifacts = load_manifest(manifest_path)
    with pytest.raises(ReportInputError, match="incomplete train/evaluation matrix"):
        normalise_artifacts(artifacts)


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
