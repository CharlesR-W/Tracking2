import os
import subprocess
import sys
from pathlib import Path

import pytest

from tracking2.report import (
    _statistics_sweep_data,
    experiment_a_figure,
    experiment_a_reverse_figure,
    suffix_accuracy_decomposition_figures,
    sweep_accuracy_change_figure,
    sweep_focal_trajectory_figure,
    vgg_statistics_sweep_figures,
)


def sweep_payload(epoch: int = 0) -> dict:
    records = []
    values = {
        "true": (1.0, 0.80),
        "gaussian": (1.2, 0.75),
        "mean": (1.5, 0.60),
    }
    for train_distribution, (loss, accuracy) in values.items():
        records.append({
            "draw": 0,
            "train_distribution": train_distribution,
            "eval_distribution": "true",
            "relax_epoch": 0,
            "loss": 1.4,
            "accuracy": 0.70,
        })
        records.append({
            "draw": 0,
            "train_distribution": train_distribution,
            "eval_distribution": "true",
            "relax_epoch": 1,
            "loss": loss,
            "accuracy": accuracy,
        })
    return {
        "schema_version": 3,
        "experiment": "vgg_suffix_statistics_sweep",
        "status": "MEASURED",
        "config": {"checkpoint_epoch": epoch, "fake_data": False},
        "slices": [{
            "cut": 7,
            "module": "stage3.conv4",
            "condition": "native",
            "explained_variance_fraction": 0.9,
            "records": records,
            "moment_diagnostics": [],
        }],
    }


def test_architecture_sweep_uses_positive_accuracy_shortfall_for_worse_surrogate():
    data = _statistics_sweep_data([sweep_payload()])
    assert data["gaps"]["gaussian"] == [[pytest.approx(0.2)]]
    assert data["gaps"]["mean"] == [[pytest.approx(0.5)]]
    assert data["accuracy_shortfalls"]["gaussian"] == [[pytest.approx(5.0)]]
    assert data["accuracy_shortfalls"]["mean"] == [[pytest.approx(20.0)]]
    assert data["accuracy_bound"] == pytest.approx(20.0)


def test_part_c_plots_accuracy_with_part_b_trajectory_grammar():
    payload = sweep_payload()
    gap_figure, _coverage_figure, _diagnostics = vgg_statistics_sweep_figures([payload])
    assert gap_figure.data[0].z == ([20.0],)
    assert gap_figure.data[1].z == ([5.0],)
    assert gap_figure.data[1].colorbar.title.text == "shortfall (pp)"
    assert "positive means worse" in gap_figure.layout.title.text
    assert "CE" not in gap_figure.data[0].hovertemplate
    assert "cross-entropy" not in gap_figure.data[0].hovertemplate

    trajectory, _cut, _module = sweep_focal_trajectory_figure([payload])
    assert [list(trace.y) for trace in trajectory.data] == [
        [70.0, 80.0], [70.0, 75.0], [70.0, 60.0],
    ]
    assert trajectory.layout.yaxis.title.text == "held-out true-activation accuracy (%)"

    changes = sweep_accuracy_change_figure([payload])
    assert [list(trace.y) for trace in changes.data] == [[-10.0], [5.0], [10.0]]
    assert changes.layout.yaxis.title.text == "accuracy change from unrelaxed suffix (pp)"
    assert changes.layout.barmode == "relative"


def test_part_a_figures_plot_accuracy_only():
    rows = []
    for train_distribution, accuracy in (("mean", 0.5), ("covariance", 0.65), ("true", 0.8)):
        rows.append({
            "architecture": "residual", "train_distribution": train_distribution,
            "test_distribution": "true", "epoch": 1, "accuracy": accuracy, "loss": 9.0,
        })
    for test_distribution, accuracy in (("mean", 0.9), ("covariance", 0.85)):
        rows.append({
            "architecture": "residual", "train_distribution": "true",
            "test_distribution": test_distribution, "epoch": 1,
            "accuracy": accuracy, "loss": 9.0,
        })

    forward = experiment_a_figure(rows)
    reverse = experiment_a_reverse_figure(rows)
    assert [list(trace.y) for trace in forward.data] == [[50.0], [65.0], [80.0]]
    assert [list(trace.y) for trace in reverse.data] == [[90.0], [85.0], [80.0]]
    assert "accuracy" in forward.layout.yaxis.title.text
    assert "accuracy" in reverse.layout.yaxis.title.text
    assert all("loss" not in trace.hovertemplate for trace in (*forward.data, *reverse.data))


def test_part_b_accuracy_effect_bars_are_signed_stacks():
    records = []
    for distribution, endpoint in (("mean", 0.6), ("gaussian", 0.75), ("true", 0.8)):
        records.append({
            "train_distribution": distribution, "eval_distribution": "true",
            "relax_epoch": 0, "accuracy": 0.7,
        })
        records.append({
            "train_distribution": distribution, "eval_distribution": "true",
            "relax_epoch": 1, "accuracy": endpoint,
        })
    payload = {
        "config": {"cut": 3, "checkpoint_epoch": 1, "checkpoint_batches": None,
                   "train_size": 100, "batch_size": 10},
        "records": records,
    }
    figure_html = suffix_accuracy_decomposition_figures([payload])
    assert '"barmode":"relative"' in figure_html
    assert "cross-entropy" not in figure_html
    assert " CE" not in figure_html


def test_architecture_sweep_rejects_duplicate_epoch_cut_cells():
    with pytest.raises(ValueError, match="Duplicate native suffix-statistics cell"):
        _statistics_sweep_data([sweep_payload(), sweep_payload()])


def test_omnibus_report_cli_requires_explicit_output_path(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "tracking2.report", str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 2
    assert "the following arguments are required: --output" in result.stderr
