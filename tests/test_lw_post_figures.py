import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_figure_notebook():
    path = (
        Path(__file__).resolve().parents[1]
        / "LW post"
        / "notebooks"
        / "lw_post_figures.py"
    )
    spec = importlib.util.spec_from_file_location("tracking2_lw_post_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_measured_control_extraction_uses_endpoint_excess_loss():
    figures = _load_figure_notebook()
    data = figures.load_measured_cnn_control_data()
    np.testing.assert_allclose(
        data.seed_excess_loss["gaussian"],
        [
            [0.11965266590118406, -0.010843602180481],
            [0.2201264202117922, -0.0138656782150269],
            [0.2089612590789796, -0.0127743496894837],
        ],
    )
    np.testing.assert_allclose(
        data.seed_excess_loss["mean_isotropic"],
        [
            [0.2676884841918945, -0.000247576904],
            [0.343427926636, 0.015113926125],
            [0.369777738571, 0.015866825199],
        ],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        data.seed_excess_loss["projected_real"], np.zeros((3, 2))
    )


def test_measured_optimizer_and_noise_controls_plot_saved_losses():
    figures = _load_figure_notebook()
    data = figures.load_measured_cnn_control_data()
    np.testing.assert_allclose(
        data.optimizer_excess_loss["gaussian"],
        [0.8383379648208617, 0.11965266590118406],
    )
    np.testing.assert_allclose(
        data.optimizer_excess_loss["mean_isotropic"],
        [2.3465473260879517, 0.2676884841918945],
    )
    np.testing.assert_allclose(
        data.noise_endpoint_loss,
        [
            0.9911849679946899,
            1.3237548622131348,
            1.1927040658950805,
            1.041233747291565,
        ],
    )
    assert data.exact_gaussian_endpoint_loss == 1.04466824760437
    assert data.projected_endpoint_loss == 0.925015581703186


def test_measured_horizon_extraction_is_complete_through_epoch_20():
    figures = _load_figure_notebook()
    trajectories = figures.load_measured_cnn_horizon()
    assert set(trajectories) == {1, 4}
    for cut in (1, 4):
        assert set(trajectories[cut]) == {
            "real",
            "projected_real",
            "gaussian",
            "mean_isotropic",
        }
        for trajectory in trajectories[cut].values():
            np.testing.assert_array_equal(trajectory.steps, np.arange(21))
            assert np.isfinite(trajectory.mean).all()
    assert trajectories[1]["mean_isotropic"].mean[-1] == 1.6049159830093385
    assert trajectories[4]["gaussian"].mean[-1] == 0.9097770383834839


def test_default_cli_is_measured_only(monkeypatch, capsys):
    figures = _load_figure_notebook()
    calls = []

    def fake_render():
        calls.append("measured")
        return [figures.PUBLICATION_FIGURE_DIR / "cnn_measured_controls.png"]

    monkeypatch.setattr(figures, "render_measured_cnn_blog_figures", fake_render)
    monkeypatch.setattr(sys, "argv", ["lw_post_figures.py"])
    figures.main()

    assert calls == ["measured"]
    assert "Rendered measured CNN figures" in capsys.readouterr().out


def test_explicit_legacy_mode_routes_static_outputs_to_archive(monkeypatch):
    figures = _load_figure_notebook()
    output_dirs = []

    def fake_static(output_dir):
        output_dirs.append(output_dir)
        return []

    monkeypatch.setattr(figures, "render_static_suite", fake_static)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lw_post_figures.py", "--legacy-full-suite", "--static-only"],
    )
    figures.main()

    assert output_dirs == [figures.DEPRECATED_FIGURE_DIR]
    assert figures.OMNIBUS_ARTIFACT_ROOT == (
        figures.PROJECT_ROOT / "deprecated" / "2026-07-omnibus" / "artifacts"
    )


def test_legacy_loaders_read_only_from_physical_archive():
    figures = _load_figure_notebook()
    cnn_changes = figures.load_legacy_cnn_changes()
    resnet_changes = figures.load_resnet_changes()

    assert set(cnn_changes) == set(figures.CNN_CHECKPOINTS)
    assert set(resnet_changes) == set(figures.RESNET_CHECKPOINTS)
    assert not (figures.PROJECT_ROOT / "artifacts" / "suffix_statistics").exists()
    assert not (
        figures.PROJECT_ROOT / "artifacts" / "resnet_suffix_statistics"
    ).exists()


def test_legacy_only_flags_require_explicit_legacy_mode(monkeypatch):
    figures = _load_figure_notebook()
    monkeypatch.setattr(sys, "argv", ["lw_post_figures.py", "--static-only"])
    with pytest.raises(SystemExit, match="2"):
        figures.main()
