import json

import pytest

from scripts.verify_lw_post_artifacts import (
    require_distinct_checkpoint_hashes,
    require_exact_keys,
)
from tracking2.cnn_checkpoints import CNNCheckpointConfig, run as train_cnn
from tracking2.post_statistics import (
    PostStatisticsConfig,
    condition_shuffle_seed,
    run as run_statistics,
)


def test_condition_shuffle_seed_is_shared_within_comparison():
    assert condition_shuffle_seed(7, cut=3, draw=2) == 203_007


def test_cnn_verifier_rejects_duplicate_cartesian_cells(tmp_path):
    rows = [
        {
            "draw": draw,
            "train_distribution": distribution,
            "eval_distribution": "true",
            "relax_epoch": epoch,
        }
        for draw in (0, 1)
        for distribution in ("true", "gaussian")
        for epoch in (0, 1)
    ]
    fields = (
        "draw",
        "train_distribution",
        "eval_distribution",
        "relax_epoch",
    )
    expected = {
        (draw, distribution, "true", epoch)
        for draw in (0, 1)
        for distribution in ("true", "gaussian")
        for epoch in (0, 1)
    }
    require_exact_keys(tmp_path / "fixture.json", rows, fields, expected, "fixture")
    with pytest.raises(RuntimeError, match="duplicate"):
        require_exact_keys(
            tmp_path / "fixture.json",
            [*rows[:-1], rows[0]],
            fields,
            expected,
            "fixture",
        )


def test_cnn_verifier_rejects_reused_checkpoint_weights(tmp_path):
    manifest = tmp_path / "training.json"
    require_distinct_checkpoint_hashes(
        manifest, {0: "a" * 64, 1: "b" * 64}
    )
    with pytest.raises(RuntimeError, match="reuses identical checkpoint weights"):
        require_distinct_checkpoint_hashes(
            manifest, {0: "a" * 64, 1: "a" * 64}
        )


def test_post_statistics_smoke_records_pca_and_noise_controls(tmp_path):
    training_path = train_cnn(
        CNNCheckpointConfig(
            output=str(tmp_path / "training"),
            fake_data=True,
            train_size=8,
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
    path = run_statistics(
        PostStatisticsConfig(
            output=str(tmp_path / "statistics"),
            checkpoint=str(checkpoint),
            checkpoint_epoch=1,
            fake_data=True,
            train_size=8,
            test_size=4,
            cuts=(1,),
            widths=(4, 4),
            batch_size=4,
            pca_fit_size=8,
            pca_ranks=(2, 3),
            mean_noise_radii=(0.0, 1.0),
            surrogate_draws=1,
            relax_epochs=1,
            seed=0,
            device="cpu",
        )
    )
    artifact = json.loads(path.read_text())
    assert artifact["status"] == "MOCKUP / PIPELINE SMOKE TEST"
    assert artifact["dataset"]["backend"] == "fake_data"
    assert "same total trace" in artifact["mean_noise_definition"]
    slice_result = artifact["slices"][0]
    assert slice_result["pca_fit_count"] == 8
    assert slice_result["moment_fit_count"] == 8
    ranks = slice_result["rank_results"]
    assert [row["pca_rank"] for row in ranks] == [2, 3]
    assert all("held_out_explained_variance_fraction" in row for row in ranks)
    distributions = {
        row["train_distribution"] for row in ranks[0]["records"]
    }
    assert distributions == {
        "projected_true",
        "gaussian",
        "mean_r0",
        "mean_r1",
    }
