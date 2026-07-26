"""Publication figures for the Tracking2 LessWrong research note.

This is a Jupytext-style notebook: editors can run the ``# %%`` cells, or the
whole suite can be rendered with:

    .venv/bin/python notebooks/lw_post_figures.py

Result plots read saved JSON artifacts only.  In particular, this file never
imports model or training code.  The canonical CNN checkpoint animation is
rendered only when the new one-run checkpoint battery is present and passes
provenance checks.  A legacy, explicitly watermarked fallback can be requested
with ``--allow-legacy-cnn-checkpoints``.
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
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
)
import numpy as np
from PIL import Image


# %% Paths and one shared visual grammar
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"

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
    """Draw a recognizable residual-network miniature, with state encoded."""
    architecture_key = architecture.lower()
    if architecture_key == "cnn":
        blocks = 4
        stage_breaks = {1, 2, 3}
        labels = ["B1", "B2", "B3", "B4"]
        architecture_label = "four-block CNN"
    elif architecture_key == "resnet":
        blocks = 8
        stage_breaks = {2, 4, 6}
        labels = [f"R{i}" for i in range(1, 9)]
        architecture_label = "ResNet-18"
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

    input_x = 0.025
    block_left = 0.12
    output_x = 0.93
    gap = 0.015
    available = output_x - block_left - 0.035
    block_width = (available - gap * (blocks - 1)) / blocks
    block_y = 0.34
    block_h = 0.34
    ax.add_patch(
        Rectangle(
            (input_x, 0.40),
            0.045,
            0.22,
            facecolor="#d6e4ea",
            edgecolor=MUTED,
            linewidth=1,
        )
    )
    ax.text(input_x + 0.0225, 0.28, "image", ha="center", va="center", fontsize=6.6)
    ax.plot([0.07, block_left], [0.51, 0.51], color=MUTED, linewidth=1.2)

    block_centres: list[float] = []
    for index in range(blocks):
        x = block_left + index * (block_width + gap)
        block_centres.append(x + block_width / 2)
        frozen = cut is not None and index < cut
        _rounded_box(
            ax,
            (x, block_y),
            block_width,
            block_h,
            facecolor=FROZEN_FILL if frozen else TRAINABLE_FILL,
            edgecolor=MUTED if frozen else GAUSSIAN,
            linewidth=1.0,
            radius=0.018,
        )
        # Two convolutional layers plus a skip path make the residual-block
        # semantics visible even at inset scale.
        inner_margin = 0.16 * block_width
        inner_width = 0.21 * block_width
        for inner_index in range(2):
            inner_x = x + inner_margin + inner_index * 0.36 * block_width
            ax.add_patch(
                Rectangle(
                    (inner_x, block_y + 0.115),
                    inner_width,
                    0.105,
                    facecolor="white",
                    edgecolor=MUTED,
                    linewidth=0.65,
                    zorder=3,
                )
            )
        ax.add_patch(
            Arc(
                (x + block_width / 2, block_y + 0.21),
                block_width * 0.72,
                block_h * 0.72,
                theta1=8,
                theta2=172,
                color=MUTED,
                linewidth=0.7,
                zorder=4,
            )
        )
        ax.text(
            x + block_width / 2,
            block_y + 0.055,
            labels[index],
            ha="center",
            va="center",
            fontsize=5.8 if blocks == 8 else 6.6,
            fontweight=650,
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
                    [0.38, 0.64],
                    color=LIGHT_MUTED,
                    linewidth=0.65,
                )

    last_right = block_left + blocks * block_width + (blocks - 1) * gap
    ax.plot([last_right, output_x], [0.51, 0.51], color=MUTED, linewidth=1.2)
    _rounded_box(
        ax,
        (output_x, 0.40),
        0.045,
        0.22,
        facecolor=TRAINABLE_FILL if cut is None or cut < blocks else FROZEN_FILL,
        edgecolor=GAUSSIAN if cut is None or cut < blocks else MUTED,
        radius=0.012,
    )
    ax.text(output_x + 0.0225, 0.28, "head", ha="center", va="center", fontsize=6.6)

    cut_x = None
    if cut is not None:
        if cut == 0:
            cut_x = block_left - gap / 2
        elif cut == blocks:
            cut_x = last_right + (output_x - last_right) / 2
        else:
            previous_right = block_left + cut * block_width + (cut - 1) * gap
            cut_x = previous_right + gap / 2
        ax.plot(
            [cut_x, cut_x],
            [0.16, 0.84],
            color=CUT,
            linewidth=1.8,
            linestyle=(0, (3, 2)),
            zorder=6,
        )
        ax.text(cut_x, 0.88, "cut", color=CUT, ha="center", va="bottom", fontsize=6.8)
    ax.text(0.02, 0.91, architecture_label, ha="left", va="top", fontsize=7.6, fontweight=700)
    if cut_moves and cut_x is not None:
        ax.annotate(
            "x-axis moves this cut",
            xy=(cut_x, 0.14),
            xytext=(0.50, 0.02),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=CUT,
            arrowprops={"arrowstyle": "->", "color": CUT, "linewidth": 0.8},
        )
    if show_key:
        ax.text(
            0.02,
            0.06,
            "frozen prefix",
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=MUTED,
        )
        ax.text(
            0.98,
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
        show_key=True,
        cut_moves=cut_moves,
    )
    inset.set_zorder(10)
    return inset


def architecture_overview_figure() -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.89, bottom=0.11, hspace=0.52)
    fig.suptitle("Architectures and measured internal cuts", fontsize=17, fontweight=700)
    draw_network_glyph(axes[0], architecture="cnn", cut=2, show_key=False, cut_moves=True)
    axes[0].set_title(
        "CNN · residual blocks with GroupNorm · cuts after blocks 1–4",
        loc="left",
        fontsize=12,
    )
    draw_network_glyph(
        axes[1], architecture="resnet", cut=4, show_key=False, cut_moves=True
    )
    axes[1].set_title(
        "ResNet-18 · eight residual blocks in four stages · cuts after blocks 1–8",
        loc="left",
        fontsize=12,
    )
    return fig


def cut_schematic_figure(architecture: str, cut: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 1.55))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.06)
    draw_network_glyph(ax, architecture=architecture, cut=cut, show_key=True)
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
    """Whole-network regularity → internal cut → same intervention."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.7))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.11, wspace=0.17)
    fig.suptitle(
        "Use a whole-network regularity as a probe inside the network",
        fontsize=18,
        fontweight=720,
    )

    _panel_box(axes[0], "Start with a macroscopic regularity", 1)
    axes[0].text(
        0.50,
        0.73,
        "whole dataset  →  whole network  →  learning curve",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    mini = axes[0].inset_axes([0.10, 0.39, 0.80, 0.23])
    draw_network_glyph(mini, architecture="cnn", cut=None, show_key=False)
    axes[0].text(
        0.50,
        0.27,
        "Simple distributional statistics often\nsupport useful learning earlier.",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    axes[0].text(
        0.50,
        0.105,
        "Think of it as the macroscopic rule we already know—\n"
        "roughly the role of  F = ma  before making an internal cut.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )

    _panel_box(axes[1], "Make an internal cut", 2)
    mini = axes[1].inset_axes([0.08, 0.46, 0.84, 0.25])
    draw_network_glyph(mini, architecture="cnn", cut=3, show_key=True)
    axes[1].text(
        0.50,
        0.34,
        r"$z_\ell=f_{\leq \ell}(x)$ is now the suffix's training data.",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1].text(
        0.50,
        0.17,
        "Freeze the prefix so the activation distribution\n"
        "stops moving during the measurement.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    _panel_box(axes[2], "Apply the same statistical intervention", 3)
    labels = [
        ("real", "saved $z_\\ell$"),
        ("projected_real", "PCA projection"),
        ("gaussian", "class $\\mu_y,\\Sigma_y$"),
        ("mean_isotropic", "class $\\mu_y$ + isotropic noise"),
    ]
    y_positions = np.linspace(0.70, 0.36, len(labels))
    for (name, label), y in zip(labels, y_positions):
        style = SERIES_STYLE[name]
        axes[2].plot(
            [0.12, 0.29],
            [y, y],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=4,
            solid_capstyle="round",
        )
        axes[2].text(0.34, y, label, ha="left", va="center", fontsize=10.5)
    axes[2].annotate(
        "",
        xy=(0.86, 0.52),
        xytext=(0.69, 0.52),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.6},
    )
    suffix = axes[2].inset_axes([0.72, 0.41, 0.25, 0.22])
    draw_network_glyph(suffix, architecture="cnn", cut=3, show_key=False)
    axes[2].text(
        0.50,
        0.20,
        "Train identical suffix copies; evaluate every copy\n"
        "on held-out real activations.",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    axes[2].text(
        0.50,
        0.085,
        "The cut turns a macroscopic bias into a layerwise probe.",
        ha="center",
        va="center",
        fontsize=9.8,
        color=ACCENT,
        fontweight=650,
    )

    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        fig.add_artist(
            FancyArrowPatch(
                (left_box.x1 + 0.006, (left_box.y0 + left_box.y1) / 2),
                (right_box.x0 - 0.006, (right_box.y0 + right_box.y1) / 2),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                color=ACCENT,
                linewidth=1.6,
            )
        )
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
    """Precise method schematic for global PCA then within-class moment fitting."""
    airplanes = load_cifar_airplane_images()
    fig = plt.figure(figsize=(14, 6.4))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.05, 1.5, 1.35, 1.25, 1.55],
        height_ratios=[1, 0.36],
        left=0.035,
        right=0.985,
        top=0.84,
        bottom=0.09,
        wspace=0.20,
        hspace=0.22,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    footer = fig.add_subplot(grid[1, :])
    fig.suptitle(
        "From a class-conditioned image distribution to fitted activation datasets",
        fontsize=18,
        fontweight=720,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#cfd4d8")

    airplane_grid = Image.new("RGB", (68, 68), "white")
    for index, airplane in enumerate(airplanes):
        airplane_grid.paste(airplane, ((index % 2) * 36, (index // 2) * 36))
    axes[0].imshow(
        airplane_grid.resize((272, 272), Image.Resampling.NEAREST)
    )
    axes[0].set_title("1 · Condition on one class", loc="left", fontsize=10.5)
    axes[0].text(
        0.5,
        -0.10,
        r"four real CIFAR-10 images,  $y=\mathrm{airplane}$",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )

    axes[1].set_title(
        "2 · Push forward\nthrough the frozen prefix",
        loc="left",
        fontsize=10.5,
        linespacing=1.15,
    )
    glyph = axes[1].inset_axes([0.05, 0.26, 0.90, 0.48])
    draw_network_glyph(glyph, architecture="cnn", cut=3, show_key=True)
    axes[1].text(
        0.50,
        0.12,
        r"$x\sim p(x\mid y)\quad\mapsto\quad z_\ell=f_{\leq\ell}(x)$",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    rng = np.random.default_rng(14)
    class_centres = np.asarray([[-1.2, 0.2], [0.8, 0.8], [0.4, -0.9]])
    class_colors = [GAUSSIAN, "#6f9e3c", "#8b62a8"]
    for centre, color in zip(class_centres, class_colors):
        points = rng.multivariate_normal(
            centre,
            np.asarray([[0.32, 0.14], [0.14, 0.22]]),
            size=34,
        )
        axes[2].scatter(
            points[:, 0],
            points[:, 1],
            s=11,
            color=color,
            alpha=0.55,
            edgecolors="none",
        )
    axes[2].set_title(
        "3 · Fit one global PCA basis\nusing every class",
        loc="left",
        fontsize=10.5,
        linespacing=1.15,
    )
    axes[2].annotate(
        "",
        xy=(1.55, 0.52),
        xytext=(-1.45, -0.32),
        arrowprops={"arrowstyle": "-|>", "color": PROJECTED_REAL, "linewidth": 1.4},
    )
    axes[2].annotate(
        "",
        xy=(-0.25, 1.58),
        xytext=(0.55, -1.35),
        arrowprops={"arrowstyle": "-|>", "color": PROJECTED_REAL, "linewidth": 1.4},
    )
    axes[2].text(
        0.5,
        -0.10,
        "one pooled basis from the declared fit subset;\n"
        "not a separate PCA per class",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        linespacing=1.15,
    )

    airplane_points = rng.multivariate_normal(
        (-0.25, 0.15),
        np.asarray([[0.55, 0.24], [0.24, 0.30]]),
        size=64,
    )
    axes[3].scatter(
        airplane_points[:, 0],
        airplane_points[:, 1],
        s=12,
        color=GAUSSIAN,
        alpha=0.48,
        edgecolors="none",
    )
    axes[3].scatter(
        [airplane_points[:, 0].mean()],
        [airplane_points[:, 1].mean()],
        s=90,
        marker="x",
        linewidths=2.5,
        color=REAL,
        zorder=4,
    )
    axes[3].set_title(
        "4 · Fit class moments\ninside that global basis",
        loc="left",
        fontsize=10.5,
        linespacing=1.15,
    )
    axes[3].text(
        0.5,
        -0.10,
        r"airplane: $\mu_y$ and regularized $\Sigma_y$"
        "\n(all training activations; fixed PCA coordinates)",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        linespacing=1.15,
    )

    axes[4].set_title(
        "5 · Replay four\ncontrolled datasets",
        loc="left",
        fontsize=10.5,
        linespacing=1.15,
    )
    replay_rows = [
        ("real", r"$z_\ell$"),
        ("projected_real", r"$P_k z_\ell$"),
        ("gaussian", r"$\mathcal{N}(\mu_y,\widehat{\Sigma}_y)$"),
        (
            "mean_isotropic",
            r"$\mu_y+\eta,\ \eta\sim\mathcal{N}(0,r^2\bar v I_k)$",
        ),
    ]
    for row_index, (name, formula) in enumerate(replay_rows):
        y = 0.77 - row_index * 0.19
        style = SERIES_STYLE[name]
        axes[4].plot(
            [0.08, 0.28],
            [y, y],
            transform=axes[4].transAxes,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=4,
            solid_capstyle="round",
        )
        axes[4].text(
            0.34,
            y,
            formula,
            transform=axes[4].transAxes,
            ha="left",
            va="center",
            fontsize=10.2,
        )
    axes[4].text(
        0.08,
        0.08,
        "Every suffix is evaluated on held-out real activations.",
        transform=axes[4].transAxes,
        ha="left",
        va="center",
        fontsize=9.3,
        color=MUTED,
    )

    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        fig.add_artist(
            FancyArrowPatch(
                (left_box.x1 + 0.003, (left_box.y0 + left_box.y1) / 2),
                (right_box.x0 - 0.003, (right_box.y0 + right_box.y1) / 2),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=13,
                color=ACCENT,
                linewidth=1.4,
            )
        )

    footer.set_axis_off()
    footer.add_patch(
        FancyBboxPatch(
            (0.02, 0.16),
            0.96,
            0.68,
            boxstyle="round,pad=.015,rounding_size=.02",
            transform=footer.transAxes,
            facecolor="#fff7eb",
            edgecolor="#e8bf8e",
        )
    )
    footer.text(
        0.04,
        0.60,
        "What “isotropic noise” means",
        transform=footer.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight=700,
        color=MEAN_ISOTROPIC,
    )
    footer.text(
        0.04,
        0.34,
        r"$\bar v$ is pooled within-class variance per retained PCA coordinate.  "
        r"At $r=1$, $\mathrm{tr}(r^2\bar v I_k)$ equals the pooled within-class "
        "covariance trace; $r=0$ is exact centroid replay.\n"
        r"This is not tiny numerical $\epsilon$-noise.  We vary $r$ as an ablation.",
        transform=footer.transAxes,
        ha="left",
        va="center",
        fontsize=9.9,
        linespacing=1.4,
    )
    footer.text(
        0.985,
        0.04,
        "Method schematic; point positions are illustrative, not measured activation geometry.",
        transform=footer.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=MUTED,
    )
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
    title = "CNN · Prefix from checkpoint at epoch: 1 · Cut after block 3"
    if epoch1_cnn_uses_legacy_artifact():
        title += "\nLegacy pilot · one separately scheduled epoch-1 model"
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
    bridge_title = "CNN · Prefix from checkpoint at epoch: 1 · Cut after block 3"
    if epoch1_cnn_uses_legacy_artifact():
        bridge_title += "\nLegacy pilot · one separately scheduled epoch-1 model"
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
        "Legacy pilot · separately scheduled runs; x-axis is not one training trajectory",
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
    display_architecture = "CNN" if architecture == "cnn" else "ResNet-18"
    ax.set_title(
        f"{display_architecture} · Prefixes from checkpoint at epoch: {epoch}",
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


# %% Simple two-row tracking versus resolving schematic
def tracking_resolving_figure() -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.09, hspace=0.34)
    fig.suptitle(
        "Two effects mixed together during ordinary training",
        fontsize=18,
        fontweight=720,
    )
    rng = np.random.default_rng(7)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#cfd4d8")

    axes[0].text(
        0.025,
        0.82,
        "TRACKING",
        ha="left",
        va="center",
        fontsize=13,
        fontweight=800,
        color=ACCENT,
    )
    axes[0].text(
        0.025,
        0.64,
        "The prefix changes, so the distribution crossing the cut moves.",
        ha="left",
        va="center",
        fontsize=11,
    )
    centres = [(0.36, 0.44), (0.61, 0.49), (0.84, 0.39)]
    for index, centre in enumerate(centres):
        points = rng.normal(size=(24, 2)) * np.asarray([0.028, 0.055]) + centre
        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            s=13,
            color=GAUSSIAN,
            alpha=0.45 + 0.18 * index,
            edgecolors="none",
        )
        axes[0].text(
            centre[0],
            0.20,
            f"$P_{{t+{index}}}(z_\\ell\\mid y)$",
            ha="center",
            va="center",
            fontsize=9.5,
        )
        if index:
            axes[0].annotate(
                "",
                xy=(centre[0] - 0.04, centre[1]),
                xytext=(centres[index - 1][0] + 0.04, centres[index - 1][1]),
                arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 1.5},
            )
    axes[1].text(
        0.025,
        0.82,
        "RESOLVING",
        ha="left",
        va="center",
        fontsize=13,
        fontweight=800,
        color=GAUSSIAN,
    )
    axes[1].text(
        0.025,
        0.64,
        "Freeze the prefix; measure how the suffix adapts to one fixed distribution.",
        ha="left",
        va="center",
        fontsize=11,
    )
    cloud = rng.normal(size=(34, 2)) * np.asarray([0.032, 0.065]) + (0.44, 0.43)
    axes[1].scatter(
        cloud[:, 0],
        cloud[:, 1],
        s=14,
        color=GAUSSIAN,
        alpha=0.58,
        edgecolors="none",
    )
    axes[1].text(0.44, 0.20, r"fixed $P_t(z_\ell\mid y)$", ha="center", fontsize=9.5)
    axes[1].annotate(
        "",
        xy=(0.61, 0.44),
        xytext=(0.50, 0.44),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.5},
    )
    curve_x = np.linspace(0.65, 0.94, 30)
    curve_y = 0.30 + 0.32 * (1 - np.exp(-14 * (curve_x - 0.65)))
    axes[1].plot(curve_x, curve_y, color=REAL, linewidth=2.4)
    axes[1].plot([0.65, 0.94], [0.30, 0.30], color=LIGHT_MUTED, linewidth=0.8)
    axes[1].text(
        0.80,
        0.20,
        "suffix response time / endpoint",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    axes[1].text(
        0.985,
        0.06,
        "Current experiment: resolving only.  Longer-term aim: compare both timescales.",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


# %% Main suite
def render_static_suite() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs += save_figure(
        conceptual_internal_cut_figure(),
        [
            FIGURE_DIR / "conceptual_internal_cut.svg",
            FIGURE_DIR / "fbd_materials_nn_analogy.svg",
        ],
    )
    outputs += save_figure(
        class_conditioned_pushforward_pca_figure(),
        [
            FIGURE_DIR / "class_conditioned_pushforward_pca.svg",
            FIGURE_DIR / "method_make_datasets.svg",
        ],
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
    args = parser.parse_args()

    outputs = render_static_suite()
    gif_reports: list[dict[str, object]] = []

    relaxation_outputs, relaxation_report = render_cnn_epoch1_relaxation()
    outputs += relaxation_outputs
    gif_reports.append(relaxation_report)

    resnet_changes = load_resnet_changes()
    resnet_outputs, resnet_report = render_depth_checkpoint_gif(
        architecture="resnet",
        changes=resnet_changes,
        epochs=RESNET_CHECKPOINTS,
        output_stem="resnet_over_training_time",
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
                output_stem="cnn_over_training_time_LEGACY_FALLBACK",
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
            output_stem="cnn_over_training_time",
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
