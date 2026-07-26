import json

import numpy as np
import pytest
import torch

import tracking2.post_statistics as post_statistics
from scripts.verify_lw_post_artifacts import (
    require_common_relaxation_endpoint,
    require_distinct_checkpoint_hashes,
    require_exact_keys,
    require_valid_moment_diagnostics,
    require_valid_relaxation_records,
)
from tracking2.cnn_checkpoints import CNNCheckpointConfig, run as train_cnn
from tracking2.post_statistics import (
    PostStatisticsConfig,
    _initial_step_diagnostics,
    _relax_suffix,
    _shuffle_indices,
    _shuffled_representation_loader,
    condition_shuffle_seed,
    run as run_statistics,
)
from tracking2.models import ArchitectureSpec, SplitConvNet


def test_condition_shuffle_seed_is_shared_within_comparison():
    assert condition_shuffle_seed(7, cut=3, draw=2) == 203_007


def test_diagnostic_batch_is_exactly_first_optimizer_batch():
    representations = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
    labels = np.arange(10, dtype=np.int64)
    seed = 17
    expected = _shuffle_indices(len(representations), seed)
    data = _shuffled_representation_loader(
        representations, labels, batch_size=4, seed=seed
    )
    first_representation, first_labels = next(iter(data))
    assert first_representation.flatten().numpy().tolist() == (
        expected[:4].astype(float).tolist()
    )
    assert first_labels.numpy().tolist() == expected[:4].tolist()


def test_initial_step_diagnostics_restore_unrelaxed_suffix():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SplitConvNet(ArchitectureSpec("residual", (4, 4)))
    representations = rng.normal(size=(6, 4, 32, 32)).astype(np.float32)
    labels = np.arange(6, dtype=np.int64)
    suffix_parameters = model.suffix_parameters(1)
    optimizer = torch.optim.SGD(suffix_parameters, lr=0.01, momentum=0.9)
    before = [parameter.detach().clone() for parameter in suffix_parameters]
    diagnostics = _initial_step_diagnostics(
        model,
        optimizer,
        1,
        representations,
        labels,
        batch_size=3,
        shuffle_seed=11,
        device=torch.device("cpu"),
    )
    assert all(
        torch.equal(parameter, initial)
        for parameter, initial in zip(suffix_parameters, before)
    )
    assert diagnostics["initial_batch_size"] == 3
    assert diagnostics["initial_gradient_total_norm"] > 0
    assert diagnostics["initial_gradient_rms"] == pytest.approx(
        diagnostics["initial_gradient_total_norm"]
        / diagnostics["initial_suffix_parameter_count"] ** 0.5
    )
    assert diagnostics["initial_suffix_weight_norm"] > 0
    assert diagnostics["first_step_update_norm"] > 0
    assert diagnostics["first_step_update_to_weight_ratio"] == pytest.approx(
        diagnostics["first_step_update_norm"]
        / diagnostics["initial_suffix_weight_norm"]
    )


def test_relaxation_step_zero_is_common_across_train_distributions():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SplitConvNet(ArchitectureSpec("residual", (4, 4)))
    true_train = rng.normal(size=(6, 4, 32, 32)).astype(np.float32)
    shifted_train = true_train + 0.5
    labels = np.arange(6, dtype=np.int64)
    test_labels = np.arange(4, dtype=np.int64)
    evaluation_sets = {
        "true": rng.normal(size=(4, 4, 32, 32)).astype(np.float32),
        "shifted": rng.normal(size=(4, 4, 32, 32)).astype(np.float32),
    }
    common = dict(
        model=model,
        cut=1,
        train_labels=labels,
        evaluation_sets=evaluation_sets,
        test_labels=test_labels,
        draw=0,
        batch_size=3,
        relax_epochs=0,
        relax_learning_rate=0.01,
        shuffle_seed=23,
        device=torch.device("cpu"),
    )
    true_records = _relax_suffix(
        train_rep=true_train, distribution="true", **common
    )
    shifted_records = _relax_suffix(
        train_rep=shifted_train, distribution="shifted", **common
    )
    assert {
        (row["train_distribution"], row["eval_distribution"])
        for row in [*true_records, *shifted_records]
    } == {
        ("true", "true"),
        ("true", "shifted"),
        ("shifted", "true"),
        ("shifted", "shifted"),
    }
    for evaluation_distribution in evaluation_sets:
        true_row = next(
            row
            for row in true_records
            if row["eval_distribution"] == evaluation_distribution
        )
        shifted_row = next(
            row
            for row in shifted_records
            if row["eval_distribution"] == evaluation_distribution
        )
        assert true_row["loss"] == pytest.approx(shifted_row["loss"])
        assert true_row["accuracy"] == shifted_row["accuracy"]
        assert (
            true_row["initial_batch_index_sha256"]
            == shifted_row["initial_batch_index_sha256"]
        )

    matched_true = _relax_suffix(
        train_rep=true_train,
        distribution="true",
        learning_rate_regime="match_true_initial_update",
        **common,
    )
    update_target = matched_true[0]["matched_first_step_update_norm"]
    matched_shifted = _relax_suffix(
        train_rep=shifted_train,
        distribution="shifted",
        learning_rate_regime="match_true_initial_update",
        target_first_step_update_norm=update_target,
        **common,
    )
    assert matched_true[0]["learning_rate_regime"] == (
        "match_true_initial_update"
    )
    assert matched_shifted[0]["matched_first_step_update_norm"] == pytest.approx(
        update_target
    )
    assert matched_shifted[0]["effective_relax_learning_rate"] == pytest.approx(
        0.01 * matched_shifted[0]["learning_rate_multiplier"]
    )

    reinitialized_common = {
        **common,
        "suffix_initialization": "reinitialized",
        "initialization_seed": 991,
    }
    reinitialized_true = _relax_suffix(
        train_rep=true_train,
        distribution="true",
        **reinitialized_common,
    )
    reinitialized_shifted = _relax_suffix(
        train_rep=shifted_train,
        distribution="shifted",
        **reinitialized_common,
    )
    assert {
        row["suffix_initialization"]
        for row in [*reinitialized_true, *reinitialized_shifted]
    } == {"reinitialized"}
    assert reinitialized_true[0]["initial_suffix_weight_norm"] == pytest.approx(
        reinitialized_shifted[0]["initial_suffix_weight_norm"]
    )
    for evaluation_distribution in evaluation_sets:
        true_row = next(
            row
            for row in reinitialized_true
            if row["eval_distribution"] == evaluation_distribution
        )
        shifted_row = next(
            row
            for row in reinitialized_shifted
            if row["eval_distribution"] == evaluation_distribution
        )
        assert true_row["loss"] == pytest.approx(shifted_row["loss"])
        assert true_row["accuracy"] == shifted_row["accuracy"]


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


def test_cnn_verifier_rejects_nonfinite_metrics_and_mismatched_baseline(tmp_path):
    path = tmp_path / "fixture.json"
    reference = [
        {
            "draw": 0,
            "train_distribution": "true",
            "eval_distribution": "true",
            "relax_epoch": epoch,
            "loss": 1.0 - 0.1 * epoch,
            "accuracy": 0.5 + 0.1 * epoch,
            "initial_training_loss": 1.2,
            "initial_gradient_norm": 0.8,
            "initial_gradient_rms": 0.08,
            "initial_suffix_weight_norm": 2.0,
            "first_step_update_norm": 0.008,
            "first_step_update_to_weight_ratio": 0.004,
            "initial_batch_index_sha256": "a" * 64,
        }
        for epoch in (0, 1)
    ]
    require_valid_relaxation_records(path, reference, "reference")
    candidate = [
        {
            **row,
            "train_distribution": "gaussian",
        }
        for row in reference
    ]
    require_common_relaxation_endpoint(
        path, reference, candidate, label="candidate"
    )
    candidate[0]["loss"] = 1.1
    with pytest.raises(RuntimeError, match="endpoint"):
        require_common_relaxation_endpoint(
            path, reference, candidate, label="candidate"
        )
    reference[0]["initial_gradient_norm"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        require_valid_relaxation_records(path, reference, "reference")


def test_cnn_verifier_requires_finite_pca_space_moment_diagnostics(tmp_path):
    path = tmp_path / "fixture.json"
    diagnostic = {
        "diagnostic_space": "fitted PCA subspace",
        "class_mean_relative_error": 0.1,
        "class_covariance_relative_error": 0.2,
    }
    require_valid_moment_diagnostics(path, [diagnostic], "diagnostic")
    with pytest.raises(RuntimeError, match="diagnostic space"):
        require_valid_moment_diagnostics(
            path, [{**diagnostic, "diagnostic_space": "native"}], "diagnostic"
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        require_valid_moment_diagnostics(
            path,
            [{**diagnostic, "class_covariance_relative_error": float("inf")}],
            "diagnostic",
        )


def test_post_statistics_smoke_records_pca_and_noise_controls(
    tmp_path, monkeypatch
):
    truncate_calls = []
    original_truncate = (
        post_statistics.truncate_representation_surrogate
    )

    def record_truncate(model, representations, labels, rank, **kwargs):
        truncate_calls.append((rank, kwargs.get("covariance_shrinkage")))
        return original_truncate(
            model, representations, labels, rank, **kwargs
        )

    monkeypatch.setattr(
        post_statistics,
        "truncate_representation_surrogate",
        record_truncate,
    )
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
            gaussian_covariance_shrinkages=(0.0, 0.05),
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
        "true",
        "projected_true",
        "gaussian_empirical",
        "gaussian_shrunk_s0.05",
        "mean_r0",
        "mean_r1",
    }
    evaluations = {
        row["eval_distribution"] for row in ranks[0]["records"]
    }
    assert evaluations == distributions
    assert len(ranks[0]["records"]) == (
        len(distributions) * len(evaluations) * 2
    )
    assert len(slice_result["reference_records"]) == 2
    assert ranks[0]["evaluation_protocol"] == (
        "full train-by-evaluation distribution matrix"
    )
    estimators = ranks[0]["gaussian_covariance_estimators"]
    assert [
        row["distribution"] for row in estimators
    ] == ["gaussian_empirical", "gaussian_shrunk_s0.05"]
    assert estimators[0]["requested_covariance_shrinkage"] == 0.0
    assert estimators[1]["requested_covariance_shrinkage"] == 0.05
    assert (3, 0.0) not in truncate_calls
    assert truncate_calls == [(2, 0.0), (2, 0.05), (3, 0.05)]
    assert ranks[0]["step_zero_true_vs_projected"].keys() == {
        "true_loss",
        "true_accuracy",
        "projected_true_loss",
        "projected_true_accuracy",
        "true_to_projected_predictive_kl",
    }
    assert all(
        row["initial_gradient_total_norm"] == row["initial_gradient_norm"]
        and row["initial_gradient_rms"] >= 0
        and row["initial_suffix_weight_norm"] > 0
        and row["first_step_update_to_weight_ratio"] >= 0
        for row in ranks[0]["records"]
    )


def test_post_statistics_true_eval_only_is_targeted_shortcut(tmp_path):
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
            pca_ranks=(2,),
            mean_noise_radii=(1.0,),
            surrogate_draws=1,
            true_eval_only=True,
            relax_epochs=0,
            seed=0,
            device="cpu",
        )
    )
    result = json.loads(path.read_text())["slices"][0]["rank_results"][0]
    assert result["evaluation_protocol"] == (
        "true-eval-only targeted sensitivity"
    )
    assert {row["eval_distribution"] for row in result["records"]} == {"true"}
    assert {row["train_distribution"] for row in result["records"]} == {
        "true",
        "projected_true",
        "gaussian_empirical",
        "mean_r1",
    }
