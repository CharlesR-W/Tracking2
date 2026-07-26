from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
