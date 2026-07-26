from __future__ import annotations

import json

import numpy as np
import pytest

from tracking2.cnn_checkpoints import (
    CNNCheckpointConfig,
    sha256,
    run as train_cnn,
)
from tracking2.cnn_projection_adequacy import (
    CNNProjectionAdequacyConfig,
    run as run_projection_adequacy,
)


def _smoke_checkpoint(tmp_path):
    training_path = train_cnn(
        CNNCheckpointConfig(
            output=str(tmp_path / "training"),
            fake_data=True,
            train_size=10,
            test_size=4,
            epochs=1,
            checkpoint_epochs=(1,),
            widths=(4, 4),
            batch_size=4,
            seed=0,
            device="cpu",
        )
    )
    training = json.loads(training_path.read_text())
    checkpoint = training_path.parent / training["checkpoints"][0]["path"]
    return training_path, checkpoint


def test_projection_adequacy_smoke_has_exact_nested_rank_grid(
    tmp_path, monkeypatch
):
    training_path, checkpoint = _smoke_checkpoint(tmp_path)

    def reject_covariance(*args, **kwargs):
        raise AssertionError("projection-only gate must not fit covariance")

    monkeypatch.setattr(np, "cov", reject_covariance)
    path = run_projection_adequacy(
        CNNProjectionAdequacyConfig(
            output=str(tmp_path / "projection"),
            checkpoint=str(checkpoint),
            training_manifest=str(training_path),
            checkpoint_epoch=1,
            fake_data=True,
            train_size=10,
            pca_fit_size=8,
            test_size=4,
            cuts=(1, 2),
            widths=(4, 4),
            batch_size=2,
            pca_ranks=(2, 3),
            seed=0,
            device="cpu",
        )
    )
    artifact = json.loads(path.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["experiment"] == "lw_post_cnn_projection_adequacy"
    assert artifact["status"] == "MOCKUP / PIPELINE SMOKE TEST"
    assert artifact["dataset"]["backend"] == "fake_data"
    assert artifact["dataset"]["splits"]["train"]["count"] == 10
    assert artifact["dataset"]["splits"]["test"]["count"] == 4
    assert artifact["pca_fit_prefix"]["count"] == 8
    assert artifact["pca_fit_prefix"]["selection"] == "range(0, 8)"
    assert len(artifact["pca_fit_prefix"]["ordered_images_sha256"]) == 64
    assert len(artifact["pca_fit_prefix"]["ordered_labels_sha256"]) == 64
    assert (
        artifact["pca_fit_prefix"]["ordered_images_sha256"]
        != artifact["dataset"]["splits"]["train"][
            "ordered_images_sha256"
        ]
    )
    assert artifact["lineage"]["checkpoint"]["sha256"] == sha256(checkpoint)
    assert artifact["lineage"]["training_manifest"]["sha256"] == sha256(
        training_path
    )
    assert [row["cut"] for row in artifact["slices"]] == [1, 2]
    for slice_result in artifact["slices"]:
        assert slice_result["pca_fit_count"] == 8
        assert sum(slice_result["pca_fit_class_counts"].values()) == 8
        assert slice_result["sample_rank_ceiling"] == 7
        assert slice_result["maximal_requested_rank"] == 3
        assert slice_result["maximal_feasible_rank"] == min(
            7, slice_result["activation_flattened_dimension"]
        )
        assert len(slice_result["maximal_pca_basis_sha256"]) == 64
        assert [
            row["pca_rank"] for row in slice_result["rank_results"]
        ] == [2, 3]
        assert (
            slice_result["rank_results"][-1][
                "pca_basis_prefix_sha256"
            ]
            == slice_result["maximal_pca_basis_sha256"]
        )
        for rank_result in slice_result["rank_results"]:
            assert len(rank_result["pca_basis_prefix_sha256"]) == 64
            assert set(rank_result["held_out_coverage"]) == {
                "total_variance_fraction",
                "within_class_variance_fraction",
                "between_class_mean_variance_fraction",
            }
            assert set(rank_result["step_zero_true_vs_projected"]) == {
                "true_loss",
                "true_accuracy",
                "projected_true_loss",
                "projected_true_accuracy",
                "true_to_projected_predictive_kl",
            }
            functional = rank_result["step_zero_true_vs_projected"]
            assert 0 <= functional["true_accuracy"] <= 1
            assert 0 <= functional["projected_true_accuracy"] <= 1
            assert functional["true_loss"] >= 0
            assert functional["projected_true_loss"] >= 0
            assert functional["true_to_projected_predictive_kl"] >= 0


def test_projection_adequacy_rejects_inexact_infeasible_rank(tmp_path):
    training_path, checkpoint = _smoke_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="exceeds cut-1 ceiling 7"):
        run_projection_adequacy(
            CNNProjectionAdequacyConfig(
                output=str(tmp_path / "projection"),
                checkpoint=str(checkpoint),
                training_manifest=str(training_path),
                checkpoint_epoch=1,
                fake_data=True,
                train_size=10,
                pca_fit_size=8,
                test_size=4,
                cuts=(1,),
                widths=(4, 4),
                batch_size=4,
                pca_ranks=(8,),
                seed=0,
                device="cpu",
            )
        )


def test_projection_adequacy_requires_manifest_for_measured_run(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    with pytest.raises(ValueError, match="training_manifest is required"):
        run_projection_adequacy(
            CNNProjectionAdequacyConfig(
                output=str(tmp_path / "projection"),
                checkpoint=str(checkpoint),
                checkpoint_epoch=1,
                fake_data=False,
                train_size=8,
                pca_fit_size=8,
                test_size=4,
                cuts=(1,),
                widths=(4, 4),
                pca_ranks=(2,),
                device="cpu",
            )
        )


def test_projection_adequacy_rejects_mismatched_full_train_lineage(
    tmp_path,
):
    training_path, checkpoint = _smoke_checkpoint(tmp_path)
    manifest = json.loads(training_path.read_text())
    manifest["dataset"]["splits"]["train"][
        "ordered_images_sha256"
    ] = "f" * 64
    mismatched_path = training_path.parent / "mismatched_training.json"
    mismatched_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full dataset fingerprints"):
        run_projection_adequacy(
            CNNProjectionAdequacyConfig(
                output=str(tmp_path / "projection"),
                checkpoint=str(checkpoint),
                training_manifest=str(mismatched_path),
                checkpoint_epoch=1,
                fake_data=True,
                train_size=10,
                pca_fit_size=8,
                test_size=4,
                cuts=(1,),
                widths=(4, 4),
                batch_size=4,
                pca_ranks=(2,),
                seed=0,
                device="cpu",
            )
        )
