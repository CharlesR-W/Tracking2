"""Publication figures for the Tracking2 LessWrong research note.

This is a Jupytext-style notebook: editors can run the ``# %%`` cells, or the
whole suite can be rendered with:

    .venv/bin/python "LW post/notebooks/lw_post_figures.py"

Result plots read saved JSON artifacts only.  In particular, this file never
imports model or training code.  The canonical CNN checkpoint animation is
rendered only when the new one-run checkpoint battery is present and passes
provenance checks.  A legacy, explicitly watermarked fallback can be requested
with ``--allow-legacy-cnn-checkpoints``. The revised measured-only blog panels
are rendered without any legacy fallback via:

    .venv/bin/python "LW post/notebooks/lw_post_figures.py" --measured-cnn-only
"""

# %%
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tracking2-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)
import numpy as np
from PIL import Image


# %% Paths and one shared visual grammar
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "LW post" / "figures"
MEASURED_RESULT_ROOT = PROJECT_ROOT / "artifacts" / "lw_post" / "measured"

INK = "#262626"
MUTED = "#68717a"
LIGHT_MUTED = "#aeb5bc"
PANEL = "#f3f5f7"
GRID = "#ffffff"
ACCENT = "#b24a3a"
FROZEN_FILL = "#dfe3e6"
TRAINABLE_FILL = "#ffffff"
CUT = "#b24a3a"

# The palette is intentionally global.  Do not give a distribution a different
# color in a one-off figure.
REAL = "#262626"
PROJECTED_REAL = "#8a9299"
GAUSSIAN = "#277da1"
MEAN_ISOTROPIC = "#e8871e"

SERIES_STYLE = {
    "real": {
        "label": "real activations",
        "color": REAL,
        "linestyle": "-",
        "marker": "o",
    },
    "projected_real": {
        "label": "PCA-projected real",
        "color": PROJECTED_REAL,
        "linestyle": "--",
        "marker": "s",
    },
    "gaussian": {
        "label": "class Gaussian",
        "color": GAUSSIAN,
        "linestyle": "-",
        "marker": "^",
    },
    "mean_isotropic": {
        "label": "class mean + isotropic noise",
        "color": MEAN_ISOTROPIC,
        "linestyle": "-",
        "marker": "D",
    },
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": 650,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": PANEL,
        "axes.edgecolor": LIGHT_MUTED,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 1.15,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)

CNN_CHECKPOINTS = (0, 1, 5, 10, 20, 30)
RESNET_CHECKPOINTS = (0, 1, 5, 20, 100)
GIF_DURATIONS_RELAXATION = (3000,) + (1000,) * 10 + (6000,)
GIF_DURATIONS_CNN_CHECKPOINTS = (3000,) + (1000,) * 5 + (6000,)
GIF_DURATIONS_RESNET_CHECKPOINTS = (3000,) + (1000,) * 4 + (6000,)


class FigureDataError(RuntimeError):
    """A saved result artifact is absent or unsafe to present as measured."""


@dataclass(frozen=True)
class Trajectory:
    steps: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class MeasuredControlFigureData:
    """Values plotted in the measured three-panel CNN control figure."""

    seeds: np.ndarray
    cuts: np.ndarray
    seed_excess_loss: dict[str, np.ndarray]
    optimizer_regimes: tuple[str, ...]
    optimizer_excess_loss: dict[str, np.ndarray]
    noise_radii: np.ndarray
    noise_endpoint_loss: np.ndarray
    exact_gaussian_endpoint_loss: float
    projected_endpoint_loss: float


# %% Artifact-only data loading
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FigureDataError(f"Missing saved artifact: {path}")
    return json.loads(path.read_text())


def require_measured(artifact: Mapping[str, object], path: Path) -> None:
    status = artifact.get("status")
    if status != "MEASURED":
        raise FigureDataError(
            f"{path} has status {status!r}; publication figures require MEASURED."
        )


def _canonical_series_name(raw_name: str) -> str:
    if raw_name == "true":
        return "real"
    if raw_name == "projected_true":
        return "projected_real"
    if raw_name == "gaussian":
        return "gaussian"
    if raw_name == "mean" or raw_name.startswith("mean_r"):
        return "mean_isotropic"
    raise KeyError(raw_name)


def trajectory_from_records(
    records: Sequence[Mapping[str, object]],
    distribution: str,
) -> Trajectory:
    """Average a saved relaxation trajectory across draws, without interpolation."""
    selected = [
        row
        for row in records
        if row["train_distribution"] == distribution
        and row.get("eval_distribution", "true") == "true"
    ]
    if not selected:
        raise FigureDataError(f"No held-out-real records for {distribution!r}.")
    steps = np.asarray(
        sorted({float(row["relax_epoch"]) for row in selected}), dtype=float
    )
    means: list[float] = []
    stds: list[float] = []
    for step in steps:
        values = np.asarray(
            [
                100.0 * float(row["accuracy"])
                for row in selected
                if float(row["relax_epoch"]) == step
            ],
            dtype=float,
        )
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return Trajectory(steps, np.asarray(means), np.asarray(stds))


def endpoint_change_from_records(
    records: Sequence[Mapping[str, object]],
    distribution: str,
) -> tuple[float, float]:
    """Mean and draw-to-draw SD of final-minus-initial held-out accuracy (pp)."""
    selected = [
        row
        for row in records
        if row["train_distribution"] == distribution
        and row.get("eval_distribution", "true") == "true"
    ]
    if not selected:
        raise FigureDataError(f"No held-out-real records for {distribution!r}.")
    draws = sorted({int(row.get("draw", 0)) for row in selected})
    changes: list[float] = []
    for draw in draws:
        draw_rows = [row for row in selected if int(row.get("draw", 0)) == draw]
        start = min(draw_rows, key=lambda row: float(row["relax_epoch"]))
        finish = max(draw_rows, key=lambda row: float(row["relax_epoch"]))
        changes.append(100.0 * (float(finish["accuracy"]) - float(start["accuracy"])))
    array = np.asarray(changes, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def endpoint_accuracy_from_records(
    records: Sequence[Mapping[str, object]],
    distribution: str,
) -> tuple[float, float]:
    """Mean and draw-to-draw SD of final held-out accuracy (%)."""
    selected = [
        row
        for row in records
        if row["train_distribution"] == distribution
        and row.get("eval_distribution", "true") == "true"
    ]
    if not selected:
        raise FigureDataError(f"No held-out-real records for {distribution!r}.")
    draws = sorted({int(row.get("draw", 0)) for row in selected})
    endpoints = []
    for draw in draws:
        draw_rows = [row for row in selected if int(row.get("draw", 0)) == draw]
        finish = max(draw_rows, key=lambda row: float(row["relax_epoch"]))
        endpoints.append(100.0 * float(finish["accuracy"]))
    array = np.asarray(endpoints, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


# %% Staged measured CNN controls (never fall back to legacy artifacts)
def _require_config_value(
    path: Path,
    config: Mapping[str, object],
    key: str,
    expected: object,
) -> None:
    if config.get(key) != expected:
        raise FigureDataError(
            f"{path}: config.{key}={config.get(key)!r}, expected {expected!r}."
        )


def _load_measured_cnn_artifact(
    path: Path,
    *,
    seed: int,
    cuts: Sequence[int],
    ranks: Sequence[int],
    learning_rate_regime: str,
    relax_epochs: int,
    result_root: Path,
) -> dict:
    artifact = load_json(path)
    require_measured(artifact, path)
    if artifact.get("schema_version") != 2:
        raise FigureDataError(
            f"{path}: expected CNN statistics schema 2, got "
            f"{artifact.get('schema_version')!r}."
        )
    if artifact.get("experiment") != "lw_post_cnn_suffix_statistics":
        raise FigureDataError(f"{path}: wrong experiment type.")
    config = artifact.get("config")
    if not isinstance(config, Mapping):
        raise FigureDataError(f"{path}: missing config mapping.")
    for key, expected in (
        ("seed", seed),
        ("cuts", list(cuts)),
        ("pca_ranks", list(ranks)),
        ("learning_rate_regime", learning_rate_regime),
        ("suffix_initialization", "warm"),
        ("relax_epochs", relax_epochs),
        ("checkpoint_epoch", 30),
        ("fake_data", False),
    ):
        _require_config_value(path, config, key, expected)

    manifest_path = (
        result_root
        / "training_manifests"
        / f"cnn_seed{seed}_training.json"
    )
    manifest = load_json(manifest_path)
    require_measured(manifest, manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment")
        != "lw_post_cnn_checkpoint_trajectory"
    ):
        raise FigureDataError(f"{manifest_path}: wrong training schema.")
    manifest_config = manifest.get("config", {})
    _require_config_value(manifest_path, manifest_config, "seed", seed)
    checkpoint_rows = [
        row
        for row in manifest.get("checkpoints", [])
        if int(row.get("epoch", -1)) == 30
    ]
    if len(checkpoint_rows) != 1:
        raise FigureDataError(
            f"{manifest_path}: expected exactly one epoch-30 checkpoint."
        )
    artifact_checkpoint = artifact.get("checkpoint", {})
    if artifact_checkpoint.get("sha256") != checkpoint_rows[0].get("sha256"):
        raise FigureDataError(
            f"{path}: checkpoint hash does not match seed-{seed} manifest."
        )
    return artifact


def _rank_result(
    artifact: Mapping[str, object],
    path: Path,
    *,
    cut: int,
    rank: int,
) -> Mapping[str, object]:
    slices = [
        row for row in artifact.get("slices", []) if int(row["cut"]) == cut
    ]
    if len(slices) != 1:
        raise FigureDataError(f"{path}: expected exactly one cut-{cut} slice.")
    matches = [
        row
        for row in slices[0].get("rank_results", [])
        if int(row["pca_rank"]) == rank
    ]
    if len(matches) != 1:
        raise FigureDataError(
            f"{path}: expected exactly one cut-{cut}, rank-{rank} result."
        )
    rank_result = matches[0]
    empirical_estimators = [
        row
        for row in rank_result.get("gaussian_covariance_estimators", [])
        if row.get("distribution") == "gaussian_empirical"
    ]
    if len(empirical_estimators) != 1 or not empirical_estimators[0].get(
        "exact_empirical_covariance"
    ):
        raise FigureDataError(
            f"{path}: Gaussian reference is not certified exact empirical "
            "covariance."
        )
    return rank_result


def _held_out_true_loss_series(
    rank_result: Mapping[str, object],
    path: Path,
    distribution: str,
    *,
    expected_final_epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rank_result.get("records", [])
        if row.get("train_distribution") == distribution
        and row.get("eval_distribution") == "true"
        and int(row.get("draw", 0)) == 0
    ]
    if not selected:
        raise FigureDataError(
            f"{path}: missing held-out-true records for {distribution!r}."
        )
    selected.sort(key=lambda row: int(row["relax_epoch"]))
    epochs = np.asarray([int(row["relax_epoch"]) for row in selected])
    expected_epochs = np.arange(expected_final_epoch + 1)
    if not np.array_equal(epochs, expected_epochs):
        raise FigureDataError(
            f"{path}: {distribution!r} epochs are {epochs.tolist()}, expected "
            f"{expected_epochs.tolist()}."
        )
    losses = np.asarray([float(row["loss"]) for row in selected])
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise FigureDataError(
            f"{path}: {distribution!r} contains invalid held-out loss."
        )
    return epochs.astype(float), losses


def _endpoint_loss(
    rank_result: Mapping[str, object],
    path: Path,
    distribution: str,
    *,
    expected_final_epoch: int,
) -> float:
    _, losses = _held_out_true_loss_series(
        rank_result,
        path,
        distribution,
        expected_final_epoch=expected_final_epoch,
    )
    return float(losses[-1])


def load_measured_cnn_control_data(
    result_root: Path | None = None,
) -> MeasuredControlFigureData:
    """Extract the staged, measured values for the publication control figure."""
    root = result_root or MEASURED_RESULT_ROOT
    matched_paths = {
        0: {
            1: root
            / "cnn_matched_seed0"
            / "cuts1_4_covariance_noise_ablation.json",
            4: root
            / "cnn_matched_seed0"
            / "cuts1_4_covariance_noise_ablation.json",
        },
        1: {
            1: root
            / "cnn_matched_seed1"
            / "cuts1_4_r2048_full_matrix.json",
            4: root
            / "cnn_matched_seed1"
            / "cuts1_4_r2048_full_matrix.json",
        },
        2: {
            1: root
            / "cnn_matched_seed2"
            / "cuts1_4_r2048_full_matrix.json",
            4: root
            / "cnn_matched_seed2"
            / "cuts1_4_r2048_full_matrix.json",
        },
    }
    excess = {
        "projected_real": np.zeros((3, 2), dtype=float),
        "gaussian": np.empty((3, 2), dtype=float),
        "mean_isotropic": np.empty((3, 2), dtype=float),
    }
    raw_names = {
        "projected_real": "projected_true",
        "gaussian": "gaussian_empirical",
        "mean_isotropic": "mean_r1",
    }
    for seed in range(3):
        loaded: dict[Path, Mapping[str, object]] = {}
        for cut_index, cut in enumerate((1, 4)):
            path = matched_paths[seed][cut]
            if path not in loaded:
                loaded[path] = _load_measured_cnn_artifact(
                    path,
                    seed=seed,
                    cuts=(1, 4),
                    ranks=(2048,),
                    learning_rate_regime="match_true_initial_update",
                    relax_epochs=5,
                    result_root=root,
                )
            rank_result = _rank_result(
                loaded[path], path, cut=cut, rank=2048
            )
            baseline = _endpoint_loss(
                rank_result,
                path,
                "projected_true",
                expected_final_epoch=5,
            )
            for name, raw_name in raw_names.items():
                endpoint = _endpoint_loss(
                    rank_result,
                    path,
                    raw_name,
                    expected_final_epoch=5,
                )
                excess[name][seed, cut_index] = endpoint - baseline

    fixed_path = (
        root
        / "cnn_fixed_gate_seed0"
        / "cut1_r2048_true_eval_sensitivity.json"
    )
    matched_path = matched_paths[0][1]
    fixed_artifact = _load_measured_cnn_artifact(
        fixed_path,
        seed=0,
        cuts=(1,),
        ranks=(2048,),
        learning_rate_regime="fixed_lr",
        relax_epochs=5,
        result_root=root,
    )
    matched_artifact = _load_measured_cnn_artifact(
        matched_path,
        seed=0,
        cuts=(1, 4),
        ranks=(2048,),
        learning_rate_regime="match_true_initial_update",
        relax_epochs=5,
        result_root=root,
    )
    optimizer_excess = {
        name: np.empty(2, dtype=float)
        for name in ("projected_real", "gaussian", "mean_isotropic")
    }
    for regime_index, (path, artifact) in enumerate(
        ((fixed_path, fixed_artifact), (matched_path, matched_artifact))
    ):
        rank_result = _rank_result(artifact, path, cut=1, rank=2048)
        baseline = _endpoint_loss(
            rank_result,
            path,
            "projected_true",
            expected_final_epoch=5,
        )
        for name, raw_name in raw_names.items():
            endpoint = _endpoint_loss(
                rank_result,
                path,
                raw_name,
                expected_final_epoch=5,
            )
            optimizer_excess[name][regime_index] = endpoint - baseline

    noise_path = (
        root
        / "cnn_matched_seed0"
        / "cuts1_4_covariance_noise_ablation.json"
    )
    noise_artifact = _load_measured_cnn_artifact(
        noise_path,
        seed=0,
        cuts=(1, 4),
        ranks=(2048,),
        learning_rate_regime="match_true_initial_update",
        relax_epochs=5,
        result_root=root,
    )
    noise_rank_result = _rank_result(
        noise_artifact, noise_path, cut=1, rank=2048
    )
    radii = np.asarray([0.0, 0.5, 1.0, 2.0])
    noise_losses = np.asarray(
        [
            _endpoint_loss(
                noise_rank_result,
                noise_path,
                f"mean_r{radius:g}",
                expected_final_epoch=5,
            )
            for radius in radii
        ]
    )
    return MeasuredControlFigureData(
        seeds=np.arange(3),
        cuts=np.asarray([1, 4]),
        seed_excess_loss=excess,
        optimizer_regimes=("fixed learning rate", "matched initial update"),
        optimizer_excess_loss=optimizer_excess,
        noise_radii=radii,
        noise_endpoint_loss=noise_losses,
        exact_gaussian_endpoint_loss=_endpoint_loss(
            noise_rank_result,
            noise_path,
            "gaussian_empirical",
            expected_final_epoch=5,
        ),
        projected_endpoint_loss=_endpoint_loss(
            noise_rank_result,
            noise_path,
            "projected_true",
            expected_final_epoch=5,
        ),
    )


def load_measured_cnn_horizon(
    result_root: Path | None = None,
) -> dict[int, dict[str, Trajectory]]:
    """Load seed-0 matched-update rank-2048 loss curves through epoch 20."""
    root = result_root or MEASURED_RESULT_ROOT
    path = (
        root
        / "cnn_matched_seed0"
        / "cuts1_4_horizon20_true_eval.json"
    )
    artifact = _load_measured_cnn_artifact(
        path,
        seed=0,
        cuts=(1, 4),
        ranks=(2048,),
        learning_rate_regime="match_true_initial_update",
        relax_epochs=20,
        result_root=root,
    )
    raw_names = {
        "real": "true",
        "projected_real": "projected_true",
        "gaussian": "gaussian_empirical",
        "mean_isotropic": "mean_r1",
    }
    result: dict[int, dict[str, Trajectory]] = {}
    for cut in (1, 4):
        rank_result = _rank_result(artifact, path, cut=cut, rank=2048)
        result[cut] = {}
        for name, raw_name in raw_names.items():
            epochs, losses = _held_out_true_loss_series(
                rank_result,
                path,
                raw_name,
                expected_final_epoch=20,
            )
            result[cut][name] = Trajectory(
                steps=epochs,
                mean=losses,
                std=np.zeros_like(losses),
            )
    return result


def legacy_cnn_artifact(epoch: int, cut: int) -> dict:
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "suffix_statistics"
        / f"t{epoch}_cut{cut}_full_seed0"
        / "suffix_statistics.json"
    )
    artifact = load_json(path)
    require_measured(artifact, path)
    return artifact


def load_epoch1_cnn_trajectories() -> dict[str, Trajectory]:
    """Prefer the one-run post artifact; use one legacy checkpoint if unavailable."""
    post_path = (
        PROJECT_ROOT
        / "artifacts"
        / "lw_post"
        / "cnn_statistics_seed0"
        / "base_epoch1"
        / "post_statistics.json"
    )
    if post_path.exists():
        artifact = load_json(post_path)
        require_measured(artifact, post_path)
        slice_result = next(
            row for row in artifact["slices"] if int(row["cut"]) == 3
        )
        rank_result = max(
            slice_result["rank_results"], key=lambda row: int(row["pca_rank"])
        )
        records_by_raw_name = {
            "real": (slice_result["reference_records"], "true"),
            "projected_real": (rank_result["records"], "projected_true"),
            "gaussian": (rank_result["records"], "gaussian"),
        }
        mean_names = sorted(
            {
                str(row["train_distribution"])
                for row in rank_result["records"]
                if str(row["train_distribution"]).startswith("mean_r")
            }
        )
        preferred_mean = "mean_r1" if "mean_r1" in mean_names else mean_names[-1]
        records_by_raw_name["mean_isotropic"] = (
            rank_result["records"],
            preferred_mean,
        )
        return {
            name: trajectory_from_records(records, raw_name)
            for name, (records, raw_name) in records_by_raw_name.items()
        }

    artifact = legacy_cnn_artifact(1, 3)
    return {
        _canonical_series_name(raw_name): trajectory_from_records(
            artifact["records"], raw_name
        )
        for raw_name in ("true", "gaussian", "mean")
    }


def epoch1_cnn_uses_legacy_artifact() -> bool:
    return not (
        PROJECT_ROOT
        / "artifacts"
        / "lw_post"
        / "cnn_statistics_seed0"
        / "base_epoch1"
        / "post_statistics.json"
    ).exists()


def _one_run_cnn_paths(
    result_root: Path | None = None,
) -> tuple[Path, dict[int, Path]]:
    root = result_root or (
        PROJECT_ROOT / "artifacts" / "lw_post" / "cnn_statistics_seed0"
    )
    training_path = root.parent / "cnn_seed0" / "training.json"
    paths = {
        epoch: root / f"base_epoch{epoch}" / "post_statistics.json"
        for epoch in CNN_CHECKPOINTS
    }
    return training_path, paths


def load_one_run_cnn_changes(
    result_root: Path | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Load canonical CNN endpoint changes and verify one-run checkpoint hashes."""
    training_path, paths = _one_run_cnn_paths(result_root)
    missing = [path for path in (training_path, *paths.values()) if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FigureDataError(
            "The canonical CNN checkpoint figure is data-blocked. Run "
            "scripts/run_lw_post_battery.sh; missing:\n  " + joined
        )
    training = load_json(training_path)
    require_measured(training, training_path)
    if "one uninterrupted end-to-end training run" not in str(
        training.get("trajectory_semantics", "")
    ):
        raise FigureDataError(
            f"{training_path} does not certify one uninterrupted training run."
        )
    checkpoint_rows = {
        int(row["epoch"]): row for row in training.get("checkpoints", [])
    }
    if tuple(checkpoint_rows) != CNN_CHECKPOINTS:
        raise FigureDataError(
            f"{training_path} checkpoint epochs are {tuple(checkpoint_rows)}, "
            f"expected {CNN_CHECKPOINTS}."
        )

    result: dict[int, dict[str, np.ndarray]] = {}
    for epoch, path in paths.items():
        artifact = load_json(path)
        require_measured(artifact, path)
        if artifact["checkpoint"]["sha256"] != checkpoint_rows[epoch]["sha256"]:
            raise FigureDataError(f"{path} does not match checkpoint epoch {epoch}.")
        slices = sorted(artifact["slices"], key=lambda row: int(row["cut"]))
        if [int(row["cut"]) for row in slices] != [1, 2, 3, 4]:
            raise FigureDataError(f"{path} is missing one or more CNN cuts.")
        series_values = {
            name: [] for name in ("real", "projected_real", "gaussian", "mean_isotropic")
        }
        series_stds = {name: [] for name in series_values}
        for slice_result in slices:
            rank_result = max(
                slice_result["rank_results"], key=lambda row: int(row["pca_rank"])
            )
            mean_names = sorted(
                {
                    str(row["train_distribution"])
                    for row in rank_result["records"]
                    if str(row["train_distribution"]).startswith("mean_r")
                }
            )
            preferred_mean = "mean_r1" if "mean_r1" in mean_names else mean_names[-1]
            sources = {
                "real": (slice_result["reference_records"], "true"),
                "projected_real": (rank_result["records"], "projected_true"),
                "gaussian": (rank_result["records"], "gaussian"),
                "mean_isotropic": (rank_result["records"], preferred_mean),
            }
            for name, (records, raw_name) in sources.items():
                mean, std = endpoint_change_from_records(records, raw_name)
                series_values[name].append(mean)
                series_stds[name].append(std)
        result[epoch] = {
            **{name: np.asarray(values) for name, values in series_values.items()},
            **{
                f"{name}_std": np.asarray(values)
                for name, values in series_stds.items()
            },
        }
    return result


def load_legacy_cnn_changes() -> dict[int, dict[str, np.ndarray]]:
    """Clearly marked fallback; these epochs are not one uninterrupted run."""
    result: dict[int, dict[str, np.ndarray]] = {}
    for epoch in CNN_CHECKPOINTS:
        values = {"real": [], "gaussian": [], "mean_isotropic": []}
        stds = {name: [] for name in values}
        for cut in range(1, 5):
            artifact = legacy_cnn_artifact(epoch, cut)
            for raw_name in ("true", "gaussian", "mean"):
                name = _canonical_series_name(raw_name)
                mean, std = endpoint_change_from_records(
                    artifact["records"], raw_name
                )
                values[name].append(mean)
                stds[name].append(std)
        result[epoch] = {
            **{name: np.asarray(array) for name, array in values.items()},
            **{
                f"{name}_std": np.asarray(array)
                for name, array in stds.items()
            },
        }
    return result


def load_resnet_changes() -> dict[int, dict[str, np.ndarray]]:
    paths = {
        0: PROJECT_ROOT
        / "artifacts/resnet_suffix_statistics/seed0-epoch0/resnet_suffix_statistics.json",
        1: PROJECT_ROOT
        / "artifacts/resnet_suffix_statistics/seed0-epoch1/resnet_suffix_statistics.json",
        5: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch5.json",
        20: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch20.json",
        100: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch100.json",
    }
    result: dict[int, dict[str, np.ndarray]] = {}
    for epoch, path in paths.items():
        artifact = load_json(path)
        require_measured(artifact, path)
        slices = sorted(artifact["slices"], key=lambda row: int(row["cut"]))
        values = {"real": [], "gaussian": [], "mean_isotropic": []}
        stds = {name: [] for name in values}
        for slice_result in slices:
            for raw_name in ("true", "gaussian", "mean"):
                name = _canonical_series_name(raw_name)
                mean, std = endpoint_change_from_records(
                    slice_result["records"], raw_name
                )
                values[name].append(mean)
                stds[name].append(std)
        result[epoch] = {
            **{name: np.asarray(array) for name, array in values.items()},
            **{
                f"{name}_std": np.asarray(array)
                for name, array in stds.items()
            },
        }
    return result


# %% Rendering and verification helpers
def figure_to_image(fig: plt.Figure, *, dpi: int = 120) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=None)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def save_figure(
    fig: plt.Figure,
    paths: Sequence[Path],
    *,
    dpi: int = 180,
) -> list[Path]:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if path.suffix.lower() == ".svg":
            cleaned = "\n".join(
                line.rstrip() for line in path.read_text().splitlines()
            )
            path.write_text(cleaned + "\n")
    plt.close(fig)
    return list(paths)


def save_gif(
    frames: Sequence[Image.Image],
    durations_ms: Sequence[int],
    output: Path,
) -> None:
    if len(frames) != len(durations_ms):
        raise ValueError("Each GIF frame needs one explicit duration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    adaptive = getattr(Image, "Palette", Image).ADAPTIVE
    paletted = [frame.convert("P", palette=adaptive, colors=192) for frame in frames]
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=list(durations_ms),
        loop=0,
        optimize=False,
        disposal=2,
    )


def verify_gif(
    path: Path,
    expected_durations_ms: Sequence[int],
) -> dict[str, object]:
    with Image.open(path) as image:
        durations = []
        sizes = []
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))
            sizes.append(image.size)
    expected = list(expected_durations_ms)
    if durations != expected:
        raise AssertionError(f"{path}: durations {durations} != {expected}")
    if len(set(sizes)) != 1:
        raise AssertionError(f"{path}: frame sizes changed: {sizes}")
    return {
        "path": str(path),
        "frame_count": len(durations),
        "durations_ms": durations,
        "size": sizes[0],
        "inspected_frame_indices": [0, len(durations) // 2, len(durations) - 1],
    }


def gif_badge(fig: plt.Figure) -> None:
    fig.text(
        0.978,
        0.965,
        "GIF",
        ha="right",
        va="top",
        color="white",
        fontsize=9,
        fontweight=800,
        bbox={
            "boxstyle": "round,pad=.28",
            "facecolor": ACCENT,
            "edgecolor": ACCENT,
        },
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def series_legend(names: Iterable[str]) -> list[Line2D]:
    handles = []
    for name in names:
        style = SERIES_STYLE[name]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.4,
                markersize=6,
                label=style["label"],
            )
        )
    return handles


def _fixed_limits(arrays: Iterable[np.ndarray], *, include_zero: bool = False) -> tuple[float, float]:
    flattened = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    lower = float(np.nanmin(flattened))
    upper = float(np.nanmax(flattened))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = max(upper - lower, 8.0)
    return lower - 0.12 * span, upper + 0.16 * span


# %% Conventional network glyphs used directly inside plots
def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = LIGHT_MUTED,
    linewidth: float = 1.1,
    radius: float = 0.025,
    zorder: float = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_network_glyph(
    ax: plt.Axes,
    *,
    architecture: str,
    cut: int | None,
    show_key: bool = True,
    cut_moves: bool = False,
) -> None:
    """Draw the post-wide plain-block network motif.

    The block is the relevant unit in these figures. Internal convolutions and
    skip paths only made the small insets harder to read, so every architecture
    is represented by one square per measured block plus the moving cut.
    """
    architecture_key = architecture.lower()
    if architecture_key == "cnn":
        blocks = 4
        stage_breaks: set[int] = set()
    elif architecture_key == "resnet":
        blocks = 8
        stage_breaks = {2, 4, 6}
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")
    if cut is not None and not 0 <= cut <= blocks:
        raise ValueError(f"cut must lie in [0, {blocks}]")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(LIGHT_MUTED)
        spine.set_linewidth(0.8)

    position = ax.get_position()
    physical_aspect = (
        ax.figure.get_figwidth() * position.width
        / (ax.figure.get_figheight() * position.height)
    )
    block_h = 0.34
    block_width = block_h / physical_aspect
    gap = 0.055 if blocks == 4 else 0.032
    chain_width = blocks * block_width + (blocks - 1) * gap
    block_left = (1.0 - chain_width) / 2
    block_y = (1.0 - block_h) / 2

    for index in range(blocks):
        x = block_left + index * (block_width + gap)
        frozen = cut is not None and index < cut
        _rounded_box(
            ax,
            (x, block_y),
            block_width,
            block_h,
            facecolor=FROZEN_FILL if frozen else TRAINABLE_FILL,
            edgecolor=MUTED if frozen else GAUSSIAN,
            linewidth=1.0,
            radius=0.014,
        )
        if index < blocks - 1:
            next_x = x + block_width + gap
            ax.plot(
                [x + block_width, next_x],
                [0.51, 0.51],
                color=MUTED,
                linewidth=1.0,
            )
            if index + 1 in stage_breaks:
                ax.plot(
                    [x + block_width + gap / 2] * 2,
                    [0.30, 0.72],
                    color=LIGHT_MUTED,
                    linewidth=0.7,
                )

    last_right = block_left + blocks * block_width + (blocks - 1) * gap

    cut_x = None
    if cut is not None:
        if cut == 0:
            cut_x = block_left - gap / 2
        elif cut == blocks:
            cut_x = last_right + gap / 2
        else:
            previous_right = block_left + cut * block_width + (cut - 1) * gap
            cut_x = previous_right + gap / 2
        ax.plot(
            [cut_x, cut_x],
            [0.20, 0.82],
            color=CUT,
            linewidth=1.8,
            linestyle=(0, (3, 2)),
            zorder=6,
        )
        ax.text(cut_x, 0.86, "cut", color=CUT, ha="center", va="bottom", fontsize=6.6)
    if cut_moves and cut_x is not None:
        ax.annotate(
            "",
            xy=(last_right, 0.13),
            xytext=(block_left, 0.13),
            arrowprops={
                "arrowstyle": "<->",
                "color": CUT,
                "linewidth": 1.0,
            },
        )
        ax.text(
            (block_left + last_right) / 2,
            0.035,
            "move cut",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=CUT,
        )
    if show_key:
        ax.text(
            block_left,
            0.06,
            "frozen prefix",
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=MUTED,
        )
        ax.text(
            last_right,
            0.06,
            "trainable suffix",
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=GAUSSIAN,
        )


def add_cut_inset(
    ax: plt.Axes,
    *,
    architecture: str,
    cut: int,
    bounds: tuple[float, float, float, float] = (0.57, 0.67, 0.39, 0.25),
    cut_moves: bool = False,
) -> plt.Axes:
    inset = ax.inset_axes(bounds)
    draw_network_glyph(
        inset,
        architecture=architecture,
        cut=cut,
        show_key=False,
        cut_moves=cut_moves,
    )
    inset.set_zorder(10)
    return inset


def architecture_overview_figure() -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 3.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.08, hspace=0.58)
    draw_network_glyph(axes[0], architecture="cnn", cut=2, show_key=False, cut_moves=True)
    axes[0].set_title("four-block CNN", loc="left", fontsize=11.5)
    draw_network_glyph(
        axes[1], architecture="resnet", cut=4, show_key=False, cut_moves=True
    )
    axes[1].set_title("ResNet-18", loc="left", fontsize=11.5)
    return fig


def cut_schematic_figure(architecture: str, cut: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 1.55))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.06)
    draw_network_glyph(ax, architecture=architecture, cut=cut, show_key=False)
    return fig


# %% Conceptual and method diagrams
def _panel_box(
    ax: plt.Axes,
    title: str,
    number: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#cfd4d8")
    ax.text(
        0.06,
        0.91,
        str(number),
        ha="center",
        va="center",
        fontsize=11,
        fontweight=800,
        color="white",
        bbox={"boxstyle": "circle,pad=.32", "facecolor": ACCENT, "edgecolor": ACCENT},
    )
    ax.text(0.14, 0.91, title, ha="left", va="center", fontsize=12.5, fontweight=700)


def conceptual_internal_cut_figure() -> plt.Figure:
    """Newtonian particles → local cuts → continuum; network → cuts → dynamics."""
    fig, ax = plt.subplots(figsize=(12, 5.0))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.08)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def arrow(x0: float, x1: float, y: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.6},
        )

    def plain_chain(x0: float, y: float, *, repeated_cuts: bool) -> None:
        widths = (0.037, 0.052, 0.044, 0.034)
        gap = 0.014
        x = x0
        ax.plot([x0 - 0.025, x0], [y, y], color=MUTED, linewidth=1.2)
        boundaries: list[float] = []
        for width in widths:
            _rounded_box(
                ax,
                (x, y - 0.055),
                width,
                0.11,
                facecolor="#eaf3f7",
                edgecolor=GAUSSIAN,
                linewidth=1.1,
                radius=0.009,
            )
            x += width
            boundaries.append(x + gap / 2)
            ax.plot([x, x + gap], [y, y], color=MUTED, linewidth=1.0)
            x += gap
        ax.plot([x - gap, x + 0.02], [y, y], color=MUTED, linewidth=1.2)
        if repeated_cuts:
            for index, cut_x in enumerate(boundaries[:-1]):
                ax.plot(
                    [cut_x, cut_x],
                    [y - 0.10, y + 0.10],
                    color=CUT,
                    linewidth=1.2,
                    linestyle=(0, (3, 2)),
                    alpha=0.35 + 0.25 * index,
                )

    # Newtonian mechanics: particles and local forces.
    material = FancyBboxPatch(
        (0.085, 0.655),
        0.15,
        0.15,
        boxstyle="round,pad=.008,rounding_size=.012",
        facecolor="#edf2f6",
        edgecolor="#6483a0",
        linewidth=1.1,
    )
    ax.add_patch(material)
    for px in np.linspace(0.105, 0.215, 5):
        for py in np.linspace(0.685, 0.775, 4):
            ax.add_patch(Circle((px, py), 0.005, facecolor="#7d9bb5", edgecolor="none"))
    for start, end in (
        ((0.085, 0.73), (0.055, 0.73)),
        ((0.235, 0.73), (0.265, 0.73)),
    ):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": MEAN_ISOTROPIC, "linewidth": 1.7},
        )

    # Repeat a local boundary through the material.
    slab = FancyBboxPatch(
        (0.405, 0.665),
        0.20,
        0.13,
        boxstyle="round,pad=.006,rounding_size=.010",
        facecolor="#edf2f6",
        edgecolor="#6483a0",
        linewidth=1.1,
    )
    ax.add_patch(slab)
    for index, cut_x in enumerate((0.445, 0.505, 0.565)):
        ax.plot(
            [cut_x, cut_x],
            [0.65, 0.81],
            color=MEAN_ISOTROPIC,
            linewidth=1.3,
            linestyle=(0, (3, 2)),
            alpha=0.35 + 0.25 * index,
        )
    ax.annotate(
        "",
        xy=(0.575, 0.835),
        xytext=(0.435, 0.835),
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=-.25",
            "color": GAUSSIAN,
            "linewidth": 1.4,
        },
    )

    # A smooth field is the macroscopic object assembled from local cuts.
    field_x0, field_y0, field_w, field_h = 0.75, 0.645, 0.19, 0.17
    field_colors = ("#d8edf1", "#e4efcf", "#f4efae", "#f7cf93", "#efae83")
    for index, color in enumerate(field_colors):
        ax.add_patch(
            Rectangle(
                (field_x0 + index * field_w / len(field_colors), field_y0),
                field_w / len(field_colors),
                field_h,
                facecolor=color,
                edgecolor="none",
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (field_x0, field_y0),
            field_w,
            field_h,
            boxstyle="round,pad=.003,rounding_size=.010",
            facecolor="none",
            edgecolor="#6483a0",
            linewidth=1.1,
        )
    )
    curve_x = np.linspace(field_x0, field_x0 + field_w, 100)
    for offset in (0.04, 0.085, 0.13):
        curve_y = field_y0 + offset + 0.008 * np.sin(
            (curve_x - field_x0) / field_w * np.pi
        )
        ax.plot(curve_x, curve_y, color="#7895aa", linewidth=0.6, alpha=0.8)

    # The analogous move through a network.
    plain_chain(0.075, 0.285, repeated_cuts=False)
    plain_chain(0.405, 0.285, repeated_cuts=True)
    rng = np.random.default_rng(19)
    centres = ((0.765, 0.285), (0.815, 0.30), (0.865, 0.27), (0.915, 0.29))
    for depth, centre in enumerate(centres):
        spread = np.asarray([0.015 - 0.0015 * depth, 0.035 - 0.004 * depth])
        points = rng.normal(size=(13, 2)) * spread + np.asarray(centre)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=11,
            color=GAUSSIAN,
            alpha=0.35 + 0.16 * depth,
            edgecolors="none",
        )

    arrow(0.285, 0.375, 0.73)
    arrow(0.635, 0.72, 0.73)
    arrow(0.285, 0.375, 0.285)
    arrow(0.635, 0.72, 0.285)
    ax.plot([0.045, 0.955], [0.50, 0.50], color="#d8dde1", linewidth=0.9)
    ax.text(0.16, 0.88, "Newtonian", ha="center", va="center", fontsize=11.5, fontweight=700)
    ax.text(0.845, 0.88, "continuum", ha="center", va="center", fontsize=11.5, fontweight=700)
    ax.text(0.16, 0.43, "network training", ha="center", va="center", fontsize=11.5, fontweight=700)
    ax.text(0.845, 0.43, "layerwise dynamics", ha="center", va="center", fontsize=11.5, fontweight=700)
    return fig


def load_cifar_airplane_images(
    sample_indices: Sequence[int] = (2, 6, 10, 22),
) -> list[Image.Image]:
    """Load recognizable real CIFAR-10 airplanes from the saved dataset."""
    archive_path = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"
    if archive_path.exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember("cifar-10-batches-py/data_batch_1")
            handle = archive.extractfile(member)
            if handle is None:
                raise FigureDataError(f"Could not read {member.name} from {archive_path}.")
            batch = pickle.load(handle, encoding="bytes")
        pixels = np.asarray(batch[b"data"], dtype=np.uint8)
        labels = np.asarray(batch[b"labels"], dtype=int)
        airplanes = pixels[labels == 0]
        if max(sample_indices) >= len(airplanes):
            raise FigureDataError(
                f"Requested airplane {max(sample_indices)}, but only "
                f"{len(airplanes)} exist."
            )
        return [
            Image.fromarray(
                airplanes[sample_index].reshape(3, 32, 32).transpose(1, 2, 0)
            )
            for sample_index in sample_indices
        ]

    parquet_path = PROJECT_ROOT / "data" / "cifar10-train.parquet"
    if not parquet_path.exists():
        raise FigureDataError(
            f"Method diagram needs the saved CIFAR-10 parquet: {parquet_path}"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise FigureDataError(
            "pyarrow is required to read the saved CIFAR-10 image."
        ) from exc
    rows = pq.read_table(parquet_path, columns=["img", "label"]).to_pylist()
    airplanes = [row for row in rows if int(row["label"]) == 0]
    if max(sample_indices) >= len(airplanes):
        raise FigureDataError(
            f"Requested airplane {max(sample_indices)}, but only "
            f"{len(airplanes)} exist."
        )
    return [
        Image.open(io.BytesIO(airplanes[sample_index]["img"]["bytes"])).convert(
            "RGB"
        )
        for sample_index in sample_indices
    ]


def class_conditioned_pushforward_pca_figure() -> plt.Figure:
    """One class is pushed forward, projected, and replaced by two fitted laws."""
    fig, ax = plt.subplots(figsize=(12, 4.3))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.10)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(14)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.6},
        )

    def dog_card(cx: float, cy: float, scale: float, tint: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (cx - scale, cy - scale),
                2 * scale,
                2 * scale,
                boxstyle="round,pad=.004,rounding_size=.010",
                facecolor=tint,
                edgecolor="#ccd3d8",
                linewidth=0.8,
            )
        )
        ear_y = cy + 0.30 * scale
        ax.add_patch(
            Polygon(
                [
                    (cx - 0.58 * scale, ear_y + 0.42 * scale),
                    (cx - 0.23 * scale, ear_y + 0.18 * scale),
                    (cx - 0.48 * scale, ear_y - 0.25 * scale),
                ],
                closed=True,
                facecolor="#9b7557",
                edgecolor="none",
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (cx + 0.58 * scale, ear_y + 0.42 * scale),
                    (cx + 0.23 * scale, ear_y + 0.18 * scale),
                    (cx + 0.48 * scale, ear_y - 0.25 * scale),
                ],
                closed=True,
                facecolor="#9b7557",
                edgecolor="none",
            )
        )
        ax.add_patch(
            Circle(
                (cx, cy),
                0.48 * scale,
                facecolor="#c99a72",
                edgecolor="#7e624d",
                linewidth=0.7,
            )
        )
        ax.add_patch(Circle((cx - 0.16 * scale, cy + 0.08 * scale), 0.045 * scale, color=INK))
        ax.add_patch(Circle((cx + 0.16 * scale, cy + 0.08 * scale), 0.045 * scale, color=INK))
        ax.add_patch(Circle((cx, cy - 0.12 * scale), 0.065 * scale, color=INK))

    for cx, cy, tint in (
        (0.075, 0.63, "#f4efe7"),
        (0.135, 0.66, "#edf3ea"),
        (0.065, 0.43, "#eef1f5"),
        (0.13, 0.42, "#f4ece8"),
    ):
        dog_card(cx, cy, 0.048, tint)
    ax.text(0.102, 0.27, r"$p(x\mid\mathrm{dog})$", ha="center", va="center", fontsize=12)

    # A deliberately plain prefix chain ending exactly at the activation cut.
    chain_y = 0.52
    block_xs = (0.25, 0.305, 0.36)
    block_w = 0.038
    for index, x in enumerate(block_xs):
        _rounded_box(
            ax,
            (x, chain_y - 0.065),
            block_w,
            0.13,
            facecolor=FROZEN_FILL,
            edgecolor=MUTED,
            linewidth=1.0,
            radius=0.009,
        )
        if index < len(block_xs) - 1:
            ax.plot(
                [x + block_w, block_xs[index + 1]],
                [chain_y, chain_y],
                color=MUTED,
                linewidth=1.0,
            )
    cut_x = 0.415
    ax.plot(
        [cut_x, cut_x],
        [0.37, 0.69],
        color=CUT,
        linewidth=1.8,
        linestyle=(0, (3, 2)),
    )
    ax.text(cut_x, 0.72, "cut", color=CUT, fontsize=8.5, ha="center")
    arrow((0.175, chain_y), (0.225, chain_y))
    arrow((0.425, chain_y), (0.555, chain_y))

    # A visibly non-Gaussian projected activation cloud.
    theta = np.linspace(-1.2, 1.25, 44)
    empirical_x = 0.635 + 0.052 * np.cos(theta) + rng.normal(0, 0.006, len(theta))
    empirical_y = 0.52 + 0.16 * np.sin(theta) + rng.normal(0, 0.012, len(theta))
    empirical_x = np.concatenate([empirical_x, rng.normal(0.66, 0.014, 10)])
    empirical_y = np.concatenate([empirical_y, rng.normal(0.45, 0.025, 10)])
    ax.scatter(empirical_x, empirical_y, s=18, color=REAL, alpha=0.46, edgecolors="none")
    ax.plot([0.575, 0.71], [0.52, 0.52], color=LIGHT_MUTED, linewidth=0.7, zorder=0)
    ax.plot([0.64, 0.64], [0.32, 0.72], color=LIGHT_MUTED, linewidth=0.7, zorder=0)

    arrow((0.715, 0.54), (0.79, 0.68))
    arrow((0.715, 0.50), (0.79, 0.34))

    # Draw both fitted laws in local equal-scale display coordinates. This
    # keeps an isotropic covariance physically circular on the wide page axis.
    position = ax.get_position()
    physical_aspect = (
        fig.get_figwidth() * position.width
        / (fig.get_figheight() * position.height)
    )

    def from_display_coordinates(
        mean: np.ndarray,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            (
                mean[0] + coordinates[:, 0] / physical_aspect,
                mean[1] + coordinates[:, 1],
            )
        )

    angle = np.deg2rad(28)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    principal_sd = np.asarray([0.055, 0.025])
    gaussian_cov = rotation @ np.diag(principal_sd**2) @ rotation.T

    # The fitted class Gaussian.
    gaussian_mean = np.asarray([0.885, 0.69])
    gaussian_local = rng.multivariate_normal(
        np.zeros(2), gaussian_cov, size=36
    )
    gaussian_points = from_display_coordinates(gaussian_mean, gaussian_local)
    ax.scatter(
        gaussian_points[:, 0],
        gaussian_points[:, 1],
        s=18,
        color=GAUSSIAN,
        alpha=0.48,
        edgecolors="none",
    )
    ellipse_theta = np.linspace(0, 2 * np.pi, 160)
    gaussian_contour_local = (
        rotation
        @ np.vstack(
            (
                2 * principal_sd[0] * np.cos(ellipse_theta),
                2 * principal_sd[1] * np.sin(ellipse_theta),
            )
        )
    ).T
    gaussian_contour = from_display_coordinates(
        gaussian_mean, gaussian_contour_local
    )
    ax.plot(
        gaussian_contour[:, 0],
        gaussian_contour[:, 1],
        color=GAUSSIAN,
        linewidth=2.0,
    )
    ax.scatter(*gaussian_mean, s=65, marker="x", linewidths=2.0, color=REAL, zorder=5)

    # The mean-only fit uses a pooled-trace isotropic cloud.
    isotropic_mean = np.asarray([0.885, 0.31])
    isotropic_sd = float(np.sqrt(np.trace(gaussian_cov) / 2))
    isotropic_local = rng.normal(size=(36, 2)) * isotropic_sd
    isotropic_points = from_display_coordinates(
        isotropic_mean, isotropic_local
    )
    ax.scatter(
        isotropic_points[:, 0],
        isotropic_points[:, 1],
        s=18,
        color=MEAN_ISOTROPIC,
        alpha=0.50,
        edgecolors="none",
    )
    isotropic_contour_local = np.column_stack(
        (
            2 * isotropic_sd * np.cos(ellipse_theta),
            2 * isotropic_sd * np.sin(ellipse_theta),
        )
    )
    isotropic_contour = from_display_coordinates(
        isotropic_mean, isotropic_contour_local
    )
    ax.plot(
        isotropic_contour[:, 0],
        isotropic_contour[:, 1],
        color=MEAN_ISOTROPIC,
        linewidth=2.0,
    )
    ax.scatter(*isotropic_mean, s=65, marker="x", linewidths=2.0, color=REAL, zorder=5)
    return fig


def training_comparison_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.subplots_adjust(left=0.035, right=0.98, top=0.84, bottom=0.12)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle(
        "Only the suffix-training distribution changes",
        fontsize=17,
        fontweight=720,
    )
    y_positions = [0.79, 0.61, 0.43, 0.25]
    for y, name in zip(y_positions, SERIES_STYLE):
        style = SERIES_STYLE[name]
        ax.plot(
            [0.025, 0.14],
            [y, y],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=5,
            solid_capstyle="round",
        )
        label = (
            "class mean +\nisotropic noise"
            if name == "mean_isotropic"
            else style["label"]
        )
        ax.text(
            0.16,
            y,
            label,
            ha="left",
            va="center",
            fontsize=9.8 if name == "mean_isotropic" else 10.2,
            linespacing=1.05,
        )
        glyph = ax.inset_axes([0.39, y - 0.075, 0.24, 0.15])
        draw_network_glyph(glyph, architecture="cnn", cut=3, show_key=False)
        ax.annotate(
            "",
            xy=(0.39, y),
            xytext=(0.33, y),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.2},
        )
        ax.annotate(
            "",
            xy=(0.75, y),
            xytext=(0.64, y),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.2},
        )
        ax.text(0.695, y + 0.025, "train", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    _rounded_box(
        ax,
        (0.75, 0.17),
        0.22,
        0.70,
        facecolor="#f7f7f7",
        edgecolor=REAL,
        radius=0.02,
    )
    ax.text(
        0.86,
        0.67,
        "same evaluation set",
        ha="center",
        va="center",
        fontsize=11,
        fontweight=700,
    )
    ax.plot([0.79, 0.93], [0.54, 0.54], color=REAL, linewidth=4)
    ax.text(
        0.86,
        0.46,
        "held-out real\nactivations",
        ha="center",
        va="center",
        fontsize=10.2,
    )
    ax.text(
        0.86,
        0.27,
        "compare held-out\naccuracy trajectories",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )
    ax.text(
        0.49,
        0.08,
        "All suffix copies begin from the same checkpoint and receive the same update budget.",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )
    return fig


# %% Endpoint bands: common baseline, never cumulative
def ordered_endpoint_bands(
    ax: plt.Axes,
    x: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    std_values: Mapping[str, np.ndarray] | None = None,
    maximum_width: float = 0.72,
) -> dict[str, list[Rectangle]]:
    """Overlay endpoint bands from zero, ordered so every true value remains visible.

    These are deliberately *not* stacked contributions.  At each x-position,
    positive and negative values are independently sorted by absolute magnitude.
    The largest is drawn widest and first; smaller-magnitude endpoints are drawn
    narrower on top.  Thus vertical position always means the actual endpoint.
    """
    x = np.asarray(x, dtype=float)
    names = list(values)
    widths = np.linspace(maximum_width, maximum_width * 0.42, max(len(names), 1))
    patches: dict[str, list[Rectangle]] = {name: [] for name in names}
    for point_index, centre in enumerate(x):
        point_rows = [
            (name, float(np.asarray(values[name])[point_index])) for name in names
        ]
        for sign_rows in (
            [row for row in point_rows if row[1] >= 0],
            [row for row in point_rows if row[1] < 0],
        ):
            ordered = sorted(sign_rows, key=lambda row: abs(row[1]), reverse=True)
            for rank, (name, height) in enumerate(ordered):
                style = SERIES_STYLE[name]
                width = float(widths[min(rank, len(widths) - 1)])
                bar = ax.bar(
                    [centre],
                    [height],
                    width=width,
                    bottom=0,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.92,
                    zorder=2 + rank,
                )[0]
                patches[name].append(bar)
        if std_values is not None:
            for name, endpoint in point_rows:
                error = float(np.asarray(std_values[name])[point_index])
                ax.errorbar(
                    [centre],
                    [endpoint],
                    yerr=[error],
                    fmt=SERIES_STYLE[name]["marker"],
                    markersize=3.8,
                    color=SERIES_STYLE[name]["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    capsize=2.2,
                    linewidth=0.9,
                    zorder=8,
                )
    return patches


def endpoint_band_legend(names: Iterable[str]) -> list[Patch]:
    return [
        Patch(
            facecolor=SERIES_STYLE[name]["color"],
            edgecolor="white",
            label=SERIES_STYLE[name]["label"],
        )
        for name in names
    ]


# %% Figure 5: a fixed epoch-1 CNN prefix and its endpoint
def cnn_epoch1_relaxation_frame(
    through_epoch: int,
    trajectories: Mapping[str, Trajectory],
    *,
    callout: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.86, bottom=0.15)
    for name, trajectory in trajectories.items():
        style = SERIES_STYLE[name]
        mask = trajectory.steps <= through_epoch
        ax.plot(
            trajectory.steps[mask],
            trajectory.mean[mask],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=2.5,
            label=style["label"],
            zorder=4,
        )
        if np.any(trajectory.std[mask] > 0):
            ax.fill_between(
                trajectory.steps[mask],
                trajectory.mean[mask] - trajectory.std[mask],
                trajectory.mean[mask] + trajectory.std[mask],
                color=style["color"],
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
    y_limits = _fixed_limits(
        [trajectory.mean for trajectory in trajectories.values()]
    )
    # Reserve a genuine open band for the network inset instead of covering the
    # highest real-activation trajectory, and a lower band for the held callout.
    y_span = y_limits[1] - y_limits[0]
    y_limits = (
        y_limits[0] - 0.16 * y_span,
        y_limits[1] + 0.30 * y_span,
    )
    ax.set_xlim(-0.25, 10.25)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_xlabel("Suffix-relaxation epoch")
    ax.set_ylabel("Held-out real-activation accuracy (%)")
    title = "Four-block CNN · Epoch 1 · cut after block 3"
    if epoch1_cnn_uses_legacy_artifact():
        title += "\nIllustrative pilot · one epoch-1 CNN checkpoint"
    ax.set_title(title, loc="left", pad=8)
    style_axis(ax)
    ax.legend(
        handles=series_legend(trajectories),
        loc="lower left",
        ncol=2,
        fontsize=9.4,
    )
    add_cut_inset(
        ax,
        architecture="cnn",
        cut=3,
        bounds=(0.60, 0.72, 0.36, 0.24),
    )
    if not callout:
        ax.text(
            0.985,
            0.02,
            f"Showing relaxation through epoch {through_epoch}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=9,
        )
    gif_badge(fig)
    if callout:
        endpoints = {
            name: (trajectory.steps[-1], trajectory.mean[-1])
            for name, trajectory in trajectories.items()
        }
        target_name = (
            "mean_isotropic"
            if "mean_isotropic" in endpoints
            else next(iter(endpoints))
        )
        x_value, y_value = endpoints[target_name]
        ax.annotate(
            "Each endpoint becomes one common-baseline band\nin the next summary figure.",
            xy=(x_value, y_value),
            xytext=(4.8, y_limits[0] + 0.075 * (y_limits[1] - y_limits[0])),
            ha="left",
            va="center",
            fontsize=11,
            color=ACCENT,
            bbox={
                "boxstyle": "round,pad=.35",
                "facecolor": "white",
                "edgecolor": ACCENT,
                "alpha": 0.95,
            },
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 1.6},
            zorder=12,
        )
    return fig


def render_cnn_epoch1_relaxation() -> tuple[list[Path], dict[str, object]]:
    trajectories = load_epoch1_cnn_trajectories()
    frames = [
        figure_to_image(cnn_epoch1_relaxation_frame(epoch, trajectories))
        for epoch in range(11)
    ]
    frames.append(
        figure_to_image(
            cnn_epoch1_relaxation_frame(10, trajectories, callout=True)
        )
    )
    gif_path = FIGURE_DIR / "cnn_relaxation_time.gif"
    final_path = FIGURE_DIR / "cnn_relaxation_time_final.png"
    save_gif(frames, GIF_DURATIONS_RELAXATION, gif_path)
    frames[-1].save(final_path)
    return [gif_path, final_path], verify_gif(
        gif_path, GIF_DURATIONS_RELAXATION
    )


def _epoch1_endpoint_changes(
    trajectories: Mapping[str, Trajectory],
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([trajectory.mean[-1] - trajectory.mean[0]])
        for name, trajectory in trajectories.items()
    }


def curve_to_endpoint_bridge_figure() -> plt.Figure:
    trajectories = load_epoch1_cnn_trajectories()
    fig = plt.figure(figsize=(13.5, 6.4))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.45, 0.48, 1.2],
        left=0.065,
        right=0.975,
        top=0.84,
        bottom=0.15,
        wspace=0.12,
    )
    curve_ax = fig.add_subplot(grid[0, 0])
    arrow_ax = fig.add_subplot(grid[0, 1])
    band_ax = fig.add_subplot(grid[0, 2])
    fig.suptitle(
        "The endpoint summary keeps each curve’s final change",
        fontsize=18,
        fontweight=720,
    )

    for name, trajectory in trajectories.items():
        style = SERIES_STYLE[name]
        curve_ax.plot(
            trajectory.steps,
            trajectory.mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.8,
            linewidth=2.2,
            label=style["label"],
        )
        curve_ax.scatter(
            [trajectory.steps[-1]],
            [trajectory.mean[-1]],
            s=75,
            facecolor=style["color"],
            edgecolor="white",
            linewidth=1.1,
            zorder=8,
        )
    curve_ax.set_xlim(-0.25, 10.25)
    curve_limits = _fixed_limits(
        [trajectory.mean for trajectory in trajectories.values()]
    )
    curve_span = curve_limits[1] - curve_limits[0]
    curve_ax.set_ylim(
        curve_limits[0],
        curve_limits[1] + 0.30 * curve_span,
    )
    curve_ax.set_xlabel("Suffix-relaxation epoch")
    curve_ax.set_ylabel("Held-out real-activation accuracy (%)")
    bridge_title = "Four-block CNN · Epoch 1 · cut after block 3"
    if epoch1_cnn_uses_legacy_artifact():
        bridge_title += "\nIllustrative pilot · one epoch-1 CNN checkpoint"
    curve_ax.set_title(bridge_title, loc="left", fontsize=12)
    curve_ax.legend(handles=series_legend(trajectories), loc="lower left", fontsize=8.8)
    style_axis(curve_ax)
    add_cut_inset(
        curve_ax,
        architecture="cnn",
        cut=3,
        bounds=(0.57, 0.72, 0.40, 0.24),
    )

    arrow_ax.set_axis_off()
    arrow_ax.annotate(
        "",
        xy=(0.96, 0.53),
        xytext=(0.04, 0.53),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "simple", "color": ACCENT, "alpha": 0.82},
    )
    arrow_ax.text(
        0.50,
        0.64,
        "subtract the\nshared start",
        transform=arrow_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=ACCENT,
        fontweight=650,
    )

    values = _epoch1_endpoint_changes(trajectories)
    ordered_endpoint_bands(band_ax, np.asarray([0.0]), values, maximum_width=0.82)
    band_ax.axhline(0, color=MUTED, linewidth=1.0, zorder=7)
    band_ax.set_xlim(-0.62, 1.02)
    band_ax.set_ylim(*_fixed_limits(values.values(), include_zero=True))
    band_ax.set_xticks([0], ["epoch-1 prefix"])
    band_ax.set_ylabel("Final accuracy change (pp)")
    band_ax.set_title(
        "same endpoints,\ncommon baseline",
        fontsize=13,
    )
    style_axis(band_ax, grid_axis="y")
    display_names = {
        "real": "real",
        "projected_real": "projected real",
        "gaussian": "Gaussian",
        "mean_isotropic": "mean + noise",
    }
    for name, array in values.items():
        value = float(array[0])
        band_ax.text(
            0.44,
            value,
            f"{display_names[name]}  {value:+.1f}",
            color=SERIES_STYLE[name]["color"],
            ha="left",
            va="center",
            fontsize=8.7,
            fontweight=700,
        )
    band_ax.text(
        0.5,
        -0.13,
        "bands overlap; they do not add",
        transform=band_ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    return fig


def cnn_endpoint_bands_figure() -> plt.Figure:
    trajectories_by_epoch = {
        epoch: {
            _canonical_series_name(raw_name): trajectory_from_records(
                legacy_cnn_artifact(epoch, 3)["records"], raw_name
            )
            for raw_name in ("true", "gaussian", "mean")
        }
        for epoch in CNN_CHECKPOINTS
    }
    values = {
        name: np.asarray(
            [
                trajectories_by_epoch[epoch][name].mean[-1]
                - trajectories_by_epoch[epoch][name].mean[0]
                for epoch in CNN_CHECKPOINTS
            ]
        )
        for name in ("real", "gaussian", "mean_isotropic")
    }
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.84, bottom=0.19)
    x = np.arange(len(CNN_CHECKPOINTS), dtype=float)
    ordered_endpoint_bands(ax, x, values)
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=7)
    ax.set_xlim(-0.62, len(x) - 0.38)
    ax.set_ylim(*_fixed_limits(values.values(), include_zero=True))
    ax.set_xticks(x, [str(epoch) for epoch in CNN_CHECKPOINTS])
    ax.set_xlabel("Prefix checkpoint epoch")
    ax.set_ylabel("Final accuracy change (percentage points)")
    ax.set_title(
        "CNN · Cut after block 3 · relaxation endpoints",
        loc="left",
        pad=12,
    )
    style_axis(ax, grid_axis="y")
    ax.legend(
        handles=endpoint_band_legend(values),
        loc="upper right",
        ncol=3,
        fontsize=9.2,
    )
    add_cut_inset(
        ax,
        architecture="cnn",
        cut=3,
        bounds=(0.66, 0.60, 0.31, 0.23),
    )
    ax.text(
        0.99,
        -0.20,
        "Common-baseline bands; each colored endpoint is an alternative suffix copy, not an additive component.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    ax.text(
        0.01,
        -0.13,
        "Illustrative pilot across separately scheduled CNN runs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=ACCENT,
    )
    return fig


# %% Depth × checkpoint animations
def depth_checkpoint_frame(
    *,
    architecture: str,
    epoch: int,
    changes: Mapping[int, Mapping[str, np.ndarray]],
    legacy_fallback: bool = False,
    callout: bool = False,
) -> plt.Figure:
    values = changes[epoch]
    names = [name for name in SERIES_STYLE if name in values]
    block_count = len(np.asarray(values[names[0]]))
    x = np.arange(1, block_count + 1, dtype=float)
    all_arrays = [
        np.asarray(checkpoint_values[name])
        for checkpoint_values in changes.values()
        for name in names
    ]
    y_limits = _fixed_limits(all_arrays, include_zero=True)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.84, bottom=0.17)
    ordered_endpoint_bands(
        ax,
        x,
        {name: np.asarray(values[name]) for name in names},
        std_values={
            name: np.asarray(values[f"{name}_std"])
            for name in names
            if f"{name}_std" in values
        }
        if all(f"{name}_std" in values for name in names)
        else None,
    )
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=7)
    ax.set_xlim(0.35, block_count + 0.65)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x, [str(index) for index in range(1, block_count + 1)])
    ax.set_xlabel("Cut after residual block")
    ax.set_ylabel("Final accuracy change (percentage points)")
    display_architecture = (
        "Four-block CNN" if architecture == "cnn" else "ResNet-18"
    )
    ax.set_title(
        f"{display_architecture} · Epoch {epoch}",
        loc="left",
        pad=12,
    )
    style_axis(ax, grid_axis="y")
    ax.legend(
        handles=endpoint_band_legend(names),
        loc="lower left",
        ncol=2,
        fontsize=9.0,
    )
    example_cut = 3 if architecture == "cnn" else 4
    add_cut_inset(
        ax,
        architecture=architecture,
        cut=example_cut,
        bounds=(0.62, 0.08, 0.35, 0.24),
        cut_moves=True,
    )
    ax.text(
        0.99,
        0.015,
        "Bands share a zero baseline; they are not stacked contributions.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.8,
    )
    if legacy_fallback:
        fig.text(
            0.50,
            0.015,
            "LEGACY FALLBACK · CHECKPOINTS CAME FROM SEPARATE RUNS",
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight=800,
            bbox={
                "boxstyle": "round,pad=.35",
                "facecolor": ACCENT,
                "edgecolor": ACCENT,
            },
        )
    gif_badge(fig)
    if callout:
        final_values = changes[epoch]
        differences = np.abs(
            np.asarray(final_values["real"])
            - np.asarray(final_values["gaussian"])
        )
        target_index = int(np.argmax(differences))
        target_x = float(target_index + 1)
        target_y = float(final_values["gaussian"][target_index])
        ax.annotate(
            "The Gaussian-relaxation change differs most from real-data relaxation\n"
            f"here, at cut {target_index + 1}.",
            xy=(target_x, target_y),
            xytext=(
                max(1.2, target_x - 0.3),
                y_limits[0] + 0.16 * (y_limits[1] - y_limits[0]),
            ),
            ha="left",
            va="center",
            fontsize=10.5,
            color=ACCENT,
            bbox={
                "boxstyle": "round,pad=.35",
                "facecolor": "white",
                "edgecolor": ACCENT,
                "alpha": 0.95,
            },
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 1.5},
            zorder=12,
        )
    return fig


def render_depth_checkpoint_gif(
    *,
    architecture: str,
    changes: Mapping[int, Mapping[str, np.ndarray]],
    epochs: Sequence[int],
    output_stem: str,
    durations: Sequence[int],
    legacy_fallback: bool = False,
) -> tuple[list[Path], dict[str, object]]:
    frames = [
        figure_to_image(
            depth_checkpoint_frame(
                architecture=architecture,
                epoch=epoch,
                changes=changes,
                legacy_fallback=legacy_fallback,
            )
        )
        for epoch in epochs
    ]
    frames.append(
        figure_to_image(
            depth_checkpoint_frame(
                architecture=architecture,
                epoch=epochs[-1],
                changes=changes,
                legacy_fallback=legacy_fallback,
                callout=True,
            )
        )
    )
    gif_path = FIGURE_DIR / f"{output_stem}.gif"
    final_path = FIGURE_DIR / f"{output_stem}_final.png"
    save_gif(frames, durations, gif_path)
    frames[-1].save(final_path)
    return [gif_path, final_path], verify_gif(gif_path, durations)


# %% PCA coverage and isotropic-noise ablation hook
def _slice_by_cut(artifact: Mapping[str, object], cut: int) -> Mapping[str, object]:
    try:
        return next(row for row in artifact["slices"] if int(row["cut"]) == cut)
    except StopIteration as exc:
        raise FigureDataError(f"Ablation artifact has no cut {cut}.") from exc


def ablation_figure(
    pca_artifact_path: Path,
    noise_artifact_path: Path,
) -> plt.Figure:
    """Render measured PCA-coverage and mean-noise ablations from saved artifacts."""
    missing = [
        path for path in (pca_artifact_path, noise_artifact_path) if not path.exists()
    ]
    if missing:
        raise FigureDataError(
            "Ablation figure is data-blocked. Run scripts/run_lw_post_battery.sh; "
            "missing: " + ", ".join(str(path) for path in missing)
        )
    pca_artifact = load_json(pca_artifact_path)
    noise_artifact = load_json(noise_artifact_path)
    require_measured(pca_artifact, pca_artifact_path)
    require_measured(noise_artifact, noise_artifact_path)
    pca_slice = _slice_by_cut(pca_artifact, 3)
    noise_slice = _slice_by_cut(noise_artifact, 3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.17, wspace=0.24)
    fig.suptitle(
        "CNN · Cut after block 3 · PCA coverage and mean-noise ablations",
        fontsize=17,
        fontweight=720,
    )

    pca_rows = sorted(
        pca_slice["rank_results"], key=lambda row: int(row["pca_rank"])
    )
    reference_accuracy, reference_std = endpoint_accuracy_from_records(
        pca_slice["reference_records"], "true"
    )
    coverages = np.asarray(
        [100.0 * float(row["held_out_explained_variance_fraction"]) for row in pca_rows]
    )
    ranks = [int(row["pca_rank"]) for row in pca_rows]
    for name, raw_name in (
        ("projected_real", "projected_true"),
        ("gaussian", "gaussian"),
    ):
        endpoints = [
            endpoint_accuracy_from_records(row["records"], raw_name) for row in pca_rows
        ]
        style = SERIES_STYLE[name]
        axes[0].errorbar(
            coverages,
            [value[0] for value in endpoints],
            yerr=[value[1] for value in endpoints],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            capsize=2.5,
            label=style["label"],
        )
    axes[0].axhline(
        reference_accuracy,
        color=REAL,
        linewidth=2.0,
        label=SERIES_STYLE["real"]["label"],
    )
    if reference_std:
        axes[0].axhspan(
            reference_accuracy - reference_std,
            reference_accuracy + reference_std,
            color=REAL,
            alpha=0.08,
        )
    for coverage, rank in zip(coverages, ranks):
        axes[0].text(
            coverage,
            axes[0].get_ylim()[0],
            f"k={rank}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    axes[0].set_xlabel("Held-out variance retained by global PCA (%)")
    axes[0].set_ylabel("Final held-out real-activation accuracy (%)")
    axes[0].set_title("Vary retained PCA rank", loc="left")
    style_axis(axes[0])

    noise_rank = max(
        noise_slice["rank_results"], key=lambda row: int(row["pca_rank"])
    )
    noise_records = noise_rank["records"]
    mean_names = sorted(
        {
            str(row["train_distribution"])
            for row in noise_records
            if str(row["train_distribution"]).startswith("mean_r")
        },
        key=lambda name: float(name.removeprefix("mean_r")),
    )
    radii = np.asarray(
        [float(name.removeprefix("mean_r")) for name in mean_names], dtype=float
    )
    mean_endpoints = [
        endpoint_accuracy_from_records(noise_records, name) for name in mean_names
    ]
    mean_style = SERIES_STYLE["mean_isotropic"]
    axes[1].errorbar(
        radii,
        [value[0] for value in mean_endpoints],
        yerr=[value[1] for value in mean_endpoints],
        color=mean_style["color"],
        linestyle=mean_style["linestyle"],
        marker=mean_style["marker"],
        linewidth=2.4,
        capsize=2.5,
        label=mean_style["label"],
    )
    noise_reference, noise_reference_std = endpoint_accuracy_from_records(
        noise_slice["reference_records"], "true"
    )
    axes[1].axhline(
        noise_reference,
        color=REAL,
        linewidth=2.0,
        label=SERIES_STYLE["real"]["label"],
    )
    projected_endpoint, _ = endpoint_accuracy_from_records(
        noise_records, "projected_true"
    )
    axes[1].axhline(
        projected_endpoint,
        color=PROJECTED_REAL,
        linestyle="--",
        linewidth=2.0,
        label=SERIES_STYLE["projected_real"]["label"],
    )
    gaussian_endpoint, _ = endpoint_accuracy_from_records(
        noise_records, "gaussian"
    )
    axes[1].axhline(
        gaussian_endpoint,
        color=GAUSSIAN,
        linewidth=2.0,
        label=SERIES_STYLE["gaussian"]["label"],
    )
    if noise_reference_std:
        axes[1].axhspan(
            noise_reference - noise_reference_std,
            noise_reference + noise_reference_std,
            color=REAL,
            alpha=0.08,
        )
    axes[1].set_xlabel(r"Isotropic-noise radius $r$  (covariance trace ratio $=r^2$)")
    axes[1].set_ylabel("Final held-out real-activation accuracy (%)")
    axes[1].set_title("Vary mean-replay noise scale", loc="left")
    style_axis(axes[1])
    axes[1].text(
        0.02,
        0.03,
        r"$r=0$: exact class centroids   ·   $r=1$: pooled within-class trace",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=MUTED,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.015),
        ncol=4,
        fontsize=9.4,
    )
    add_cut_inset(
        axes[1],
        architecture="cnn",
        cut=3,
        bounds=(0.55, 0.70, 0.42, 0.24),
    )
    return fig


# %% Tracking across ordinary-training time
def tracking_resolving_figure() -> plt.Figure:
    """Show an interface moving while the downstream suffix readapts."""
    fig, ax = plt.subplots(figsize=(12, 4.15))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.94, bottom=0.08)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(7)

    def draw_state(
        x0: float,
        *,
        later: bool,
    ) -> None:
        panel = FancyBboxPatch(
            (x0, 0.16),
            0.405,
            0.68,
            boxstyle="round,pad=.008,rounding_size=.018",
            facecolor="white",
            edgecolor="#cfd7de",
            linewidth=1.0,
        )
        ax.add_patch(panel)
        ax.text(
            x0 + 0.025,
            0.78,
            r"$t+\Delta t$" if later else r"$t$",
            ha="left",
            va="center",
            fontsize=15,
            fontweight=700,
        )

        block_y = 0.43
        block_w = 0.038
        block_h = 0.14
        gap = 0.014
        prefix_x = x0 + 0.045
        for index in range(3):
            bx = prefix_x + index * (block_w + gap)
            _rounded_box(
                ax,
                (bx, block_y),
                block_w,
                block_h,
                facecolor="#d8e7ec" if not later else "#cce5d8",
                edgecolor="#6483a0" if not later else "#4d8e70",
                linewidth=1.0,
                radius=0.008,
            )
            if index:
                ax.plot(
                    [bx - gap, bx],
                    [block_y + block_h / 2] * 2,
                    color=MUTED,
                    linewidth=1.0,
                )

        cut_x = x0 + 0.215
        ax.plot(
            [cut_x, cut_x],
            [0.34, 0.66],
            color=CUT,
            linewidth=1.5,
            linestyle=(0, (3, 2)),
        )

        cloud_centre = np.asarray(
            [x0 + (0.258 if not later else 0.276), 0.50 + (0.0 if not later else 0.045)]
        )
        points = (
            rng.normal(size=(34, 2)) * np.asarray([0.018, 0.060])
            + cloud_centre
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=13,
            color=GAUSSIAN,
            alpha=0.60,
            edgecolors="none",
            zorder=4,
        )
        if later:
            ghost = points - np.asarray([0.018, 0.045])
            ax.scatter(
                ghost[:, 0],
                ghost[:, 1],
                s=10,
                facecolors="none",
                edgecolors=LIGHT_MUTED,
                linewidths=0.45,
                alpha=0.45,
                zorder=2,
            )
            ax.annotate(
                "",
                xy=(cloud_centre[0], cloud_centre[1] + 0.085),
                xytext=(cloud_centre[0] - 0.018, cloud_centre[1] + 0.04),
                arrowprops={"arrowstyle": "-|>", "color": GAUSSIAN, "linewidth": 1.3},
            )

        suffix_x = x0 + 0.315
        for index in range(2):
            bx = suffix_x + index * (block_w + gap)
            _rounded_box(
                ax,
                (bx, block_y),
                block_w,
                block_h,
                facecolor="#fff8ef",
                edgecolor=MEAN_ISOTROPIC,
                linewidth=1.0,
                radius=0.008,
            )
            if index:
                ax.plot(
                    [bx - gap, bx],
                    [block_y + block_h / 2] * 2,
                    color=MUTED,
                    linewidth=1.0,
                )
        if later:
            ax.annotate(
                "",
                xy=(suffix_x + 0.075, 0.64),
                xytext=(suffix_x + 0.005, 0.64),
                arrowprops={
                    "arrowstyle": "-|>",
                    "connectionstyle": "arc3,rad=-.35",
                    "color": MEAN_ISOTROPIC,
                    "linewidth": 1.4,
                },
            )

    draw_state(0.055, later=False)
    draw_state(0.54, later=True)
    ax.annotate(
        "",
        xy=(0.525, 0.50),
        xytext=(0.475, 0.50),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.8},
    )
    return fig


# %% Measured CNN controls for the revised post
def measured_cnn_controls_figure(
    data: MeasuredControlFigureData | None = None,
) -> plt.Figure:
    """Three compact measured controls, all from staged schema-2 artifacts."""
    data = data or load_measured_cnn_control_data()
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.2))
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.68,
        bottom=0.17,
        wspace=0.30,
    )
    fig.suptitle(
        "CNN measured controls · epoch-30 prefixes · held-out real evaluation",
        fontsize=18,
        fontweight=720,
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "Warm suffix · PCA rank 2048 · endpoint after 5 relaxation epochs",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    # A: independent trained seeds, with projected replay as the estimand's zero.
    ax = axes[0]
    x = np.arange(len(data.cuts), dtype=float)
    offsets = {
        "projected_real": -0.22,
        "gaussian": 0.0,
        "mean_isotropic": 0.22,
    }
    seed_jitter = np.linspace(-0.045, 0.045, len(data.seeds))
    for name in ("projected_real", "gaussian", "mean_isotropic"):
        values = np.asarray(data.seed_excess_loss[name])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        positions = x + offsets[name]
        for seed_index, jitter in enumerate(seed_jitter):
            ax.scatter(
                positions + jitter,
                values[seed_index],
                s=24,
                color=SERIES_STYLE[name]["color"],
                alpha=0.48,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        ax.errorbar(
            positions,
            mean,
            yerr=std,
            fmt=SERIES_STYLE[name]["marker"],
            markersize=7.2,
            color=SERIES_STYLE[name]["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3.2,
            linewidth=1.5,
            zorder=7,
        )
    ax.axhline(0, color=PROJECTED_REAL, linewidth=1.0, zorder=1)
    ax.set_xticks(x, [f"after block {cut}" for cut in data.cuts])
    ax.set_ylabel(
        "Endpoint excess held-out loss\nvs PCA-projected real"
    )
    ax.set_title(
        "A · Three trained seeds\nmatched initial update · points = seeds",
        loc="left",
        fontsize=12.5,
    )
    style_axis(ax, grid_axis="y")
    ax.text(
        0.02,
        0.03,
        "large marker ± SD",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )

    # B: same seed and checkpoint, changing only the update-scale regime.
    ax = axes[1]
    regime_x = np.arange(len(data.optimizer_regimes), dtype=float)
    short_regime_labels = ["fixed LR", "matched update"]
    for name in ("projected_real", "gaussian", "mean_isotropic"):
        values = np.asarray(data.optimizer_excess_loss[name])
        style = SERIES_STYLE[name]
        ax.plot(
            regime_x,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6.4,
            linewidth=2.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
        )
    ax.axhline(0, color=PROJECTED_REAL, linewidth=1.0, zorder=1)
    ax.set_xticks(regime_x, short_regime_labels)
    ax.set_ylabel(
        "Endpoint excess held-out loss\nvs PCA-projected real"
    )
    ax.set_title(
        "B · Optimizer-scale control\nseed 0 · cut after block 1",
        loc="left",
        fontsize=12.5,
    )
    style_axis(ax, grid_axis="y")
    ax.text(
        0.98,
        0.97,
        "same checkpoint",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )

    # C: absolute endpoint loss as isotropic mean-noise radius changes.
    ax = axes[2]
    ax.plot(
        data.noise_radii,
        data.noise_endpoint_loss,
        color=MEAN_ISOTROPIC,
        marker=SERIES_STYLE["mean_isotropic"]["marker"],
        markersize=6.4,
        linewidth=2.4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="class mean + isotropic noise",
        zorder=5,
    )
    ax.axhline(
        data.exact_gaussian_endpoint_loss,
        color=GAUSSIAN,
        linewidth=2.0,
        linestyle="--",
        label="exact empirical Gaussian",
        zorder=4,
    )
    ax.axhline(
        data.projected_endpoint_loss,
        color=PROJECTED_REAL,
        linewidth=1.6,
        linestyle=":",
        label="PCA-projected real",
        zorder=3,
    )
    ax.set_xticks(data.noise_radii)
    ax.set_xlabel("Isotropic-noise radius $r$")
    ax.set_ylabel("Endpoint held-out loss")
    ax.set_title(
        "C · Mean-noise sensitivity\nseed 0 · matched update · block 1",
        loc="left",
        fontsize=12.5,
    )
    style_axis(ax, grid_axis="y")
    ax.legend(loc="upper right", fontsize=8.4)

    fig.legend(
        handles=series_legend(
            ("projected_real", "gaussian", "mean_isotropic")
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.815),
        ncol=3,
        fontsize=9.2,
    )
    fig.text(
        0.985,
        0.035,
        "Cross-entropy loss; Gaussian covariance is exact empirical "
        "(zero shrinkage, zero jitter).",
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=MUTED,
    )
    return fig


def measured_cnn_horizon_figure(
    trajectories: Mapping[int, Mapping[str, Trajectory]] | None = None,
) -> plt.Figure:
    """Held-out true-loss trajectories for the measured 20-epoch control."""
    trajectories = trajectories or load_measured_cnn_horizon()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.77,
        bottom=0.17,
        wspace=0.12,
    )
    fig.suptitle(
        "CNN measured horizon control · held-out real-activation loss",
        fontsize=18,
        fontweight=720,
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "Seed 0 · epoch-30 prefix · warm suffix · PCA rank 2048 · "
        "matched initial update",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
    for ax, cut in zip(axes, (1, 4)):
        cut_trajectories = trajectories[cut]
        for name in ("real", "projected_real", "gaussian", "mean_isotropic"):
            trajectory = cut_trajectories[name]
            style = SERIES_STYLE[name]
            ax.plot(
                trajectory.steps,
                trajectory.mean,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=4,
                markersize=4.2,
                linewidth=2.25,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=style["label"],
                zorder=4,
            )
        ax.set_xlim(-0.4, 20.4)
        ax.set_xticks(np.arange(0, 21, 5))
        ax.set_xlabel("Suffix-relaxation epoch")
        ax.set_title(
            f"Cut after block {cut}\n"
            + (
                "shallow projected-subspace estimand"
                if cut == 1
                else "late/native-dimensional estimand"
            ),
            loc="left",
            fontsize=12.5,
        )
        style_axis(ax)
    axes[0].set_ylabel("Held-out true cross-entropy loss")
    axes[0].legend(
        handles=series_legend(
            ("real", "projected_real", "gaussian", "mean_isotropic")
        ),
        loc="upper left",
        fontsize=8.8,
    )
    fig.text(
        0.985,
        0.035,
        "Shared y-axis. Curves are alternative suffix-training datasets; "
        "all evaluations use held-out real activations.",
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=MUTED,
    )
    return fig


def render_measured_cnn_blog_figures(
    result_root: Path | None = None,
) -> list[Path]:
    """Render the two publication PNGs from staged measured artifacts only."""
    control_data = load_measured_cnn_control_data(result_root)
    horizon_data = load_measured_cnn_horizon(result_root)
    outputs: list[Path] = []
    outputs += save_figure(
        measured_cnn_controls_figure(control_data),
        [FIGURE_DIR / "cnn_measured_controls.png"],
        dpi=200,
    )
    outputs += save_figure(
        measured_cnn_horizon_figure(horizon_data),
        [FIGURE_DIR / "cnn_measured_horizon.png"],
        dpi=200,
    )
    return outputs


# %% Main suite
def render_static_suite() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs += save_figure(
        conceptual_internal_cut_figure(),
        [FIGURE_DIR / "conceptual_internal_cut.svg"],
    )
    outputs += save_figure(
        class_conditioned_pushforward_pca_figure(),
        [FIGURE_DIR / "class_conditioned_pushforward_pca.svg"],
    )
    outputs += save_figure(
        training_comparison_figure(),
        [FIGURE_DIR / "method_train_compare.svg"],
    )
    outputs += save_figure(
        architecture_overview_figure(),
        [FIGURE_DIR / "cnn_resnet_architectures.svg"],
    )
    for cut in range(1, 5):
        outputs += save_figure(
            cut_schematic_figure("cnn", cut),
            [FIGURE_DIR / f"cut_cnn_block{cut}.svg"],
        )
    for cut in (1, 2, 8):
        outputs += save_figure(
            cut_schematic_figure("resnet", cut),
            [FIGURE_DIR / f"cut_resnet_block{cut}.svg"],
        )
    outputs += save_figure(
        curve_to_endpoint_bridge_figure(),
        [FIGURE_DIR / "curve_to_endpoint_bridge.png"],
    )
    outputs += save_figure(
        cnn_endpoint_bands_figure(),
        [
            FIGURE_DIR / "cnn_relaxation_endpoint_bands.png",
            FIGURE_DIR / "cnn_relaxation_endpoint_stacked.png",
        ],
    )
    outputs += save_figure(
        tracking_resolving_figure(),
        [
            FIGURE_DIR / "tracking_resolving.svg",
            FIGURE_DIR / "tracking_resolving_only.png",
        ],
    )
    return outputs


def render_ablation_if_available(
    result_root: Path | None = None,
) -> tuple[list[Path], str | None]:
    root = result_root or (
        PROJECT_ROOT / "artifacts" / "lw_post" / "cnn_statistics_seed0"
    )
    pca_path = root / "pca_ablation_epoch30_cut3" / "post_statistics.json"
    noise_path = root / "noise_ablation_epoch30_cut3" / "post_statistics.json"
    try:
        figure = ablation_figure(pca_path, noise_path)
    except FigureDataError as exc:
        return [], str(exc)
    outputs = save_figure(
        figure,
        [FIGURE_DIR / "cnn_pca_noise_ablations.png"],
    )
    return outputs, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-legacy-cnn-checkpoints",
        action="store_true",
        help=(
            "Render a watermarked fallback CNN checkpoint animation from separate "
            "legacy runs when the canonical one-run battery is absent."
        ),
    )
    parser.add_argument(
        "--require-ablations",
        action="store_true",
        help="Fail instead of reporting a data-blocked ablation figure.",
    )
    parser.add_argument(
        "--measured-cnn-only",
        action="store_true",
        help=(
            "Render only cnn_measured_controls.png and "
            "cnn_measured_horizon.png from artifacts/lw_post/measured."
        ),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Render the static post figures without loading checkpoint GIF data.",
    )
    args = parser.parse_args()

    if args.measured_cnn_only:
        outputs = render_measured_cnn_blog_figures()
        print("Rendered measured CNN figures:")
        for path in outputs:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
        return

    outputs = render_static_suite()
    if args.static_only:
        print("Rendered static figures:")
        for path in outputs:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
        return

    gif_reports: list[dict[str, object]] = []

    relaxation_outputs, relaxation_report = render_cnn_epoch1_relaxation()
    outputs += relaxation_outputs
    gif_reports.append(relaxation_report)

    resnet_changes = load_resnet_changes()
    resnet_outputs, resnet_report = render_depth_checkpoint_gif(
        architecture="resnet",
        changes=resnet_changes,
        epochs=RESNET_CHECKPOINTS,
        output_stem="resnet_common_baseline_over_training_time_ARCHIVE",
        durations=GIF_DURATIONS_RESNET_CHECKPOINTS,
    )
    outputs += resnet_outputs
    gif_reports.append(resnet_report)

    blocked: list[str] = []
    try:
        cnn_changes = load_one_run_cnn_changes()
    except FigureDataError as exc:
        blocked.append(str(exc))
        if args.allow_legacy_cnn_checkpoints:
            cnn_changes = load_legacy_cnn_changes()
            cnn_outputs, cnn_report = render_depth_checkpoint_gif(
                architecture="cnn",
                changes=cnn_changes,
                epochs=CNN_CHECKPOINTS,
                output_stem="cnn_common_baseline_over_training_time_ARCHIVE",
                durations=GIF_DURATIONS_CNN_CHECKPOINTS,
                legacy_fallback=True,
            )
            outputs += cnn_outputs
            gif_reports.append(cnn_report)
    else:
        cnn_outputs, cnn_report = render_depth_checkpoint_gif(
            architecture="cnn",
            changes=cnn_changes,
            epochs=CNN_CHECKPOINTS,
            output_stem="cnn_common_baseline_over_training_time_ARCHIVE",
            durations=GIF_DURATIONS_CNN_CHECKPOINTS,
        )
        outputs += cnn_outputs
        gif_reports.append(cnn_report)

    ablation_outputs, ablation_blocker = render_ablation_if_available()
    outputs += ablation_outputs
    if ablation_blocker:
        blocked.append(ablation_blocker)
        if args.require_ablations:
            raise FigureDataError(ablation_blocker)

    print("Rendered:")
    for path in outputs:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("Verified GIF metadata:")
    for report in gif_reports:
        print(
            "  "
            f"{Path(str(report['path'])).relative_to(PROJECT_ROOT)}: "
            f"{report['frame_count']} frames, {report['durations_ms']}, "
            f"{report['size']}; inspect frames {report['inspected_frame_indices']}"
        )
    if blocked:
        print("Data-blocked (not rendered):")
        for message in blocked:
            print(f"  {message}")
    print("The main suite intentionally does not generate the old ResNet heatmap.")


if __name__ == "__main__":
    main()
