"""Jupytext-style notebook for the Tracking2 talk/post animations.

Run the cells interactively in an editor that understands ``# %%`` cells, or:

    python "LW post/notebooks/animate_training_dynamics.py"

The notebook reads measured JSON artifacts and writes publication assets into
``LW post/figures``. GIF timing is deliberate: three seconds to orient, one second
per time step, then a six-second explanatory hold.
"""

# %%
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# %%
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "LW post" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INK = "#243f65"
MUTED = "#657386"
GRID = "#dbe4ef"
PANEL = "#e8eef6"
TRUE = "#252525"
GAUSSIAN = "#0878b9"
MEAN = "#df6900"
ORANGE = "#df6900"
RED = "#e11d2e"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": PANEL,
        "axes.edgecolor": "none",
        "grid.color": "white",
        "grid.linewidth": 1.25,
    }
)


# %%
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def figure_to_image(fig: plt.Figure, *, dpi: int = 110) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=None, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_gif(
    frames: list[Image.Image],
    durations_ms: list[int],
    output: Path,
) -> None:
    if len(frames) != len(durations_ms):
        raise ValueError("Each GIF frame needs one duration.")
    adaptive = getattr(Image, "Palette", Image).ADAPTIVE
    paletted = [frame.convert("P", palette=adaptive, colors=192) for frame in frames]
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def signed_stacked_bars(
    ax: plt.Axes,
    x: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    width: float = 0.58,
    std_values: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Stack alternative outcomes by sign, ordered from the zero baseline out."""
    bars: dict[str, object] = {}
    series = [
        ("mean", "class mean + trace-matched noise", MEAN),
        ("gaussian", "class Gaussian", GAUSSIAN),
        ("true", "real activations", TRUE),
    ]
    arrays = {
        key: np.asarray(values[key], dtype=float)
        for key, _, _ in series
    }
    bottoms = {
        key: np.zeros_like(x, dtype=float)
        for key, _, _ in series
    }
    # Each segment is a separate outcome, not an additive component. Within
    # each sign, put the outcome closest to zero at the baseline and the most
    # extreme outcome furthest away. This keeps, for example, a less harmful
    # Gaussian outcome visually above a more harmful mean-only outcome.
    for index in range(len(x)):
        positive_keys = sorted(
            (key for key in arrays if arrays[key][index] >= 0),
            key=lambda key: arrays[key][index],
        )
        negative_keys = sorted(
            (key for key in arrays if arrays[key][index] < 0),
            key=lambda key: arrays[key][index],
            reverse=True,
        )
        positive_bottom = 0.0
        for key in positive_keys:
            bottoms[key][index] = positive_bottom
            positive_bottom += arrays[key][index]
        negative_bottom = 0.0
        for key in negative_keys:
            bottoms[key][index] = negative_bottom
            negative_bottom += arrays[key][index]

    for key, label, color in series:
        heights = arrays[key]
        bars[key] = ax.bar(
            x,
            heights,
            width,
            bottom=bottoms[key],
            color=color,
            label=label,
            zorder=3,
        )
        if std_values is not None:
            endpoints = bottoms[key] + heights
            ax.errorbar(
                x,
                endpoints,
                yerr=std_values[key],
                fmt="none",
                capsize=2.4,
                color=color,
                linewidth=1.1,
                zorder=4,
            )
    return bars


def add_depth_cut_inset(ax: plt.Axes, *, architecture: str) -> None:
    """Show that the horizontal axis moves a cut through one architecture."""
    if architecture == "cnn":
        blocks = 4
    elif architecture == "resnet":
        blocks = 8
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    inset = ax.inset_axes((0.655, 0.715, 0.325, 0.235))
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#aab5c1")
        spine.set_linewidth(0.8)

    chain_width = 0.62 if blocks == 4 else 0.80
    gap = 0.045 if blocks == 4 else 0.018
    block_width = (chain_width - gap * (blocks - 1)) / blocks
    position = inset.get_position()
    physical_aspect = (
        inset.figure.get_figwidth() * position.width
        / (inset.figure.get_figheight() * position.height)
    )
    block_height = block_width * physical_aspect
    blocks_left = (1 - chain_width) / 2
    block_y = (1 - block_height) / 2
    inset.plot(
        [blocks_left - 0.055, blocks_left + chain_width + 0.055],
        [0.50, 0.50],
        color=MUTED,
        linewidth=1.0,
        zorder=1,
    )
    centres: list[float] = []
    cut_index = blocks // 2
    for index in range(blocks):
        block_x = blocks_left + index * (block_width + gap)
        centres.append(block_x + block_width / 2)
        prefix_block = index < cut_index
        inset.add_patch(
            FancyBboxPatch(
                (block_x, block_y),
                block_width,
                block_height,
                boxstyle="round,pad=0.005,rounding_size=0.018",
                facecolor="#d6e4ea" if prefix_block else "white",
                edgecolor=MUTED if prefix_block else GAUSSIAN,
                linewidth=1.0,
                zorder=2,
            )
        )
    cut_x = (centres[cut_index - 1] + centres[cut_index]) / 2
    inset.plot(
        [cut_x, cut_x],
        [0.20, 0.80],
        color="#bd4b35",
        linewidth=1.5,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    inset.text(
        cut_x,
        0.83,
        "cut",
        color="#bd4b35",
        fontsize=6.0,
        ha="center",
        va="bottom",
    )


def red_axis_prompts(ax: plt.Axes, *, horizontal: str) -> None:
    """Draw two red direction cues meeting at the plot's upper-left corner."""
    corner = (-0.075, 1.075)
    ax.annotate(
        "",
        xy=(1.0, corner[1]),
        xytext=corner,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.6},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=corner,
        xytext=(corner[0], 0.0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.6},
        annotation_clip=False,
    )
    ax.text(
        0.50,
        1.09,
        horizontal,
        transform=ax.transAxes,
        color=RED,
        fontsize=16,
        fontweight=900,
        ha="center",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        -0.095,
        0.50,
        "ACCURACY GAIN FROM RELAXATION",
        transform=ax.transAxes,
        color=RED,
        fontsize=14,
        fontweight=900,
        rotation=90,
        ha="center",
        va="center",
        clip_on=False,
    )


def gif_badge(fig: plt.Figure) -> None:
    fig.text(
        0.987,
        0.985,
        "GIF",
        ha="right",
        va="top",
        color="white",
        fontsize=11,
        fontweight=900,
        bbox={
            "boxstyle": "round,pad=.34",
            "facecolor": RED,
            "edgecolor": RED,
        },
    )


# %%
RESNET_PATHS = {
    0: PROJECT_ROOT
    / "artifacts/resnet_suffix_statistics/seed0-epoch0/resnet_suffix_statistics.json",
    1: PROJECT_ROOT
    / "artifacts/resnet_suffix_statistics/seed0-epoch1/resnet_suffix_statistics.json",
    5: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch5.json",
    20: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch20.json",
    100: PROJECT_ROOT / "artifacts/resnet_suffix_statistics/seed0-epoch100.json",
}


def resnet_relaxation_changes() -> dict[int, dict[str, np.ndarray]]:
    """Mean accuracy change (percentage points) after suffix relaxation."""
    result: dict[int, dict[str, np.ndarray]] = {}
    for epoch, path in RESNET_PATHS.items():
        artifact = load_json(path)
        per_distribution = {"true": [], "gaussian": [], "mean": []}
        per_distribution_std = {"true": [], "gaussian": [], "mean": []}
        for slice_record in artifact["slices"]:
            records = slice_record["records"]
            final_epoch = max(record["relax_epoch"] for record in records)
            for distribution in per_distribution:
                draw_changes = []
                draws = sorted(
                    {
                        record["draw"]
                        for record in records
                        if record["train_distribution"] == distribution
                    }
                )
                for draw in draws:
                    start = next(
                        record["accuracy"]
                        for record in records
                        if record["draw"] == draw
                        and record["train_distribution"] == distribution
                        and record["eval_distribution"] == "true"
                        and record["relax_epoch"] == 0
                    )
                    final = next(
                        record["accuracy"]
                        for record in records
                        if record["draw"] == draw
                        and record["train_distribution"] == distribution
                        and record["eval_distribution"] == "true"
                        and record["relax_epoch"] == final_epoch
                    )
                    draw_changes.append(100 * (final - start))
                per_distribution[distribution].append(float(np.mean(draw_changes)))
                per_distribution_std[distribution].append(
                    float(np.std(draw_changes, ddof=1)) if len(draw_changes) > 1 else 0.0
                )
        result[epoch] = {
            **{
                distribution: np.asarray(values)
                for distribution, values in per_distribution.items()
            },
            **{
                f"{distribution}_std": np.asarray(values)
                for distribution, values in per_distribution_std.items()
            },
        }
    return result


RESNET_CHANGES = resnet_relaxation_changes()


# %%
def training_time_frame(
    epoch: int,
    *,
    callout_mode: str | None = None,
) -> plt.Figure:
    values = RESNET_CHANGES[epoch]
    fig, ax = plt.subplots(figsize=(12, 6.75), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.86, bottom=0.19)

    x = np.arange(8)
    bars = signed_stacked_bars(
        ax,
        x,
        values,
        std_values={
            key: values[f"{key}_std"] for key in ("true", "gaussian", "mean")
        },
    )

    ax.axhline(0, color="#647383", linewidth=1.4, zorder=2)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.65, 7.65)
    ax.set_ylim(-76, 82)
    ax.set_xticks(x, [str(cut) for cut in range(1, 9)])
    ax.set_ylabel("Final held-out accuracy change (percentage points)")
    ax.set_xlabel("Cut after residual block")
    ax.set_title(
        f"ResNet-18 · Epoch {epoch}",
        loc="left",
        pad=9,
        fontsize=18,
        fontweight=750,
    )
    gif_badge(fig)
    add_depth_cut_inset(ax, architecture="resnet")
    fig.text(
        0.975,
        0.045,
        "Segments are alternative suffix copies, stacked by sign for compact comparison.",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    if callout_mode == "legend":
        ax.annotate(
            "three separate relaxation experiments\nstacked by sign for comparison",
            xy=(3, 55),
            xytext=(4.25, 72),
            ha="left",
            va="top",
            fontsize=14,
            fontweight=800,
            color=RED,
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
        )
    elif callout_mode == "result":
        gaussian_bar = bars["gaussian"][0]
        ax.annotate(
            "simplified internal data hurts most\nat shallow cuts",
            xy=(
                gaussian_bar.get_x() + gaussian_bar.get_width() / 2,
                gaussian_bar.get_y() + gaussian_bar.get_height(),
            ),
            xytext=(1.15, -60),
            ha="left",
            va="center",
            fontsize=14,
            fontweight=850,
            color=RED,
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
        )
        ax.annotate(
            "late cuts barely move",
            xy=(7, float(sum(min(0, values[key][7]) for key in ("true", "gaussian", "mean")))),
            xytext=(5.35, -25),
            ha="left",
            fontsize=14,
            fontweight=850,
            color=RED,
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
        )

    return fig


training_frames = [figure_to_image(training_time_frame(0))]
for checkpoint_epoch in (1, 5, 20):
    training_frames.append(figure_to_image(training_time_frame(checkpoint_epoch)))
training_frames.extend(
    [
        figure_to_image(training_time_frame(100)),
        figure_to_image(training_time_frame(100, callout_mode="result")),
    ]
)

save_gif(
    training_frames,
    [3000, 1000, 1000, 1000, 1000, 6000],
    FIGURE_DIR / "resnet_stacked_over_training_time.gif",
)
training_frames[-1].save(
    FIGURE_DIR / "resnet_stacked_over_training_time_final.png"
)


# %%
CNN_CHECKPOINTS = [0, 1, 5, 10, 20, 30]
CNN_PATHS = {
    epoch: PROJECT_ROOT
    / f"artifacts/suffix_statistics/t{epoch}_cut3_full_seed0/suffix_statistics.json"
    for epoch in CNN_CHECKPOINTS
}


def cnn_relaxation_curves() -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    curves: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for epoch, path in CNN_PATHS.items():
        records = load_json(path)["records"]
        curves[epoch] = {}
        for distribution in ("true", "gaussian", "mean"):
            selected = sorted(
                (
                    record
                    for record in records
                    if record["train_distribution"] == distribution
                    and record["eval_distribution"] == "true"
                ),
                key=lambda record: record["relax_epoch"],
            )
            curves[epoch][distribution] = (
                np.asarray([record["relax_epoch"] for record in selected], dtype=float),
                100 * np.asarray([record["accuracy"] for record in selected], dtype=float),
            )
    return curves


CNN_CURVES = cnn_relaxation_curves()


# %%
def curve_to_stacked_bridge() -> plt.Figure:
    """Connect the epoch-1 cut-3 relaxation curves to one stacked endpoint."""
    trajectories = CNN_CURVES[1]
    fig = plt.figure(figsize=(12, 5.8))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.55, 0.78],
        left=0.075,
        right=0.965,
        top=0.82,
        bottom=0.18,
        wspace=0.34,
    )
    curve_ax = fig.add_subplot(grid[0, 0])
    stack_ax = fig.add_subplot(grid[0, 1])
    styles = {
        "true": (TRUE, "-", "o", "real activations"),
        "gaussian": (GAUSSIAN, "--", "^", "class Gaussian"),
        "mean": (MEAN, ":", "D", "class mean + trace-matched noise"),
    }
    for distribution, (color, linestyle, marker, label) in styles.items():
        epochs, accuracy = trajectories[distribution]
        curve_ax.plot(
            epochs,
            accuracy,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.3,
            markersize=5,
            label=label,
            zorder=3,
        )
        curve_ax.annotate(
            "",
            xy=(epochs[-1], accuracy[-1]),
            xytext=(epochs[-1] - 0.55, accuracy[-1]),
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.5},
        )
    curve_ax.set_xlim(-0.25, 10.5)
    all_accuracy = np.concatenate(
        [trajectories[key][1] for key in ("true", "gaussian", "mean")]
    )
    curve_ax.set_ylim(float(all_accuracy.min() - 3), float(all_accuracy.max() + 8))
    curve_ax.set_xlabel("Suffix-relaxation epoch")
    curve_ax.set_ylabel("Held-out real-activation accuracy (%)")
    curve_ax.set_title(
        "Four-block CNN · Epoch 1 · cut after block 3",
        loc="left",
        fontsize=13,
        fontweight=700,
    )
    curve_ax.grid(zorder=0)
    curve_ax.legend(loc="lower right", frameon=False, fontsize=8.4)

    endpoint_changes = {
        distribution: np.asarray([accuracy[-1] - accuracy[0]])
        for distribution, (_, accuracy) in trajectories.items()
    }
    signed_stacked_bars(
        stack_ax,
        np.asarray([0]),
        endpoint_changes,
        width=0.52,
    )
    stack_ax.axhline(0, color="#647383", linewidth=1.2, zorder=2)
    stack_ax.grid(axis="y", zorder=0)
    stack_total = sum(max(0.0, values[0]) for values in endpoint_changes.values())
    stack_ax.set_ylim(-5, max(30, stack_total * 1.22))
    stack_ax.set_xlim(-0.75, 0.75)
    stack_ax.set_xticks([0], ["cut 3"])
    stack_ax.set_ylabel("Final accuracy change (points)")
    stack_ax.set_title(
        "The same three endpoints\nas one stacked profile",
        fontsize=13,
        fontweight=700,
    )
    fig.text(
        0.632,
        0.50,
        "→",
        color=RED,
        fontsize=34,
        fontweight=800,
        ha="center",
        va="center",
    )
    fig.suptitle(
        "From relaxation curves to the compact depth sweep",
        fontsize=18,
        fontweight=750,
        x=0.075,
        ha="left",
    )
    fig.text(
        0.965,
        0.065,
        "The segments are alternative suffix copies, not additive contributions.",
        ha="right",
        color=MUTED,
        fontsize=9,
    )
    return fig


figure_to_image(curve_to_stacked_bridge(), dpi=125).save(
    FIGURE_DIR / "curve_to_stacked_bridge.png"
)


# %%
CNN_BATCHES_PER_EPOCH = 196
CNN_TRAINING_SPECS = [
    (0, 0, "t0_batch0"),
    (0, 5, "t0_batch5"),
    (0, 20, "t0_batch20"),
    (0, 98, "t0_batch98"),
    (1, None, "t1"),
    (5, None, "t5"),
    (10, None, "t10"),
    (20, None, "t20"),
    (30, None, "t30"),
]


def cnn_epoch_label(epoch: int, batch: int | None) -> str:
    if batch is None:
        return str(epoch)
    if epoch != 0:
        raise ValueError("Within-epoch CNN checkpoints are only defined for epoch 0.")
    if batch == 0:
        return "0"
    return f"{batch}/{CNN_BATCHES_PER_EPOCH}"


def cnn_training_changes() -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for _, _, prefix in CNN_TRAINING_SPECS:
        per_distribution = {"true": [], "gaussian": [], "mean": []}
        for cut in range(1, 5):
            artifact = load_json(
                PROJECT_ROOT
                / (
                    "artifacts/suffix_statistics/"
                    f"{prefix}_cut{cut}_full_seed0/suffix_statistics.json"
                )
            )
            for distribution in per_distribution:
                selected = sorted(
                    (
                        record
                        for record in artifact["records"]
                        if record["train_distribution"] == distribution
                        and record["eval_distribution"] == "true"
                    ),
                    key=lambda record: record["relax_epoch"],
                )
                per_distribution[distribution].append(
                    100 * (selected[-1]["accuracy"] - selected[0]["accuracy"])
                )
        result[prefix] = {
            distribution: np.asarray(values)
            for distribution, values in per_distribution.items()
        }
    return result


CNN_TRAINING_CHANGES = cnn_training_changes()


def cnn_training_time_frame(
    spec: tuple[int, int | None, str],
    *,
    callout_mode: str | None = None,
) -> plt.Figure:
    epoch, batch, prefix = spec
    values = CNN_TRAINING_CHANGES[prefix]
    fig, ax = plt.subplots(figsize=(12, 6.75), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.86, bottom=0.19)

    x = np.arange(4)
    bars = signed_stacked_bars(ax, x, values)

    ax.axhline(0, color="#647383", linewidth=1.4, zorder=2)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.65, 3.65)
    ax.set_ylim(-60, 120)
    ax.set_xticks(x, [str(cut) for cut in range(1, 5)])
    ax.set_ylabel("Final held-out accuracy change (percentage points)")
    ax.set_xlabel("Cut after residual block")
    ax.set_title(
        f"Four-block CNN · Epoch {cnn_epoch_label(epoch, batch)}",
        loc="left",
        pad=9,
        fontsize=18,
        fontweight=750,
    )
    gif_badge(fig)
    add_depth_cut_inset(ax, architecture="cnn")
    fig.text(
        0.975,
        0.045,
        "Segments are alternative suffix copies, stacked by sign for compact comparison.",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    if callout_mode == "legend":
        ax.annotate(
            "three separate suffix copies\nstacked by sign",
            xy=(1, 70),
            xytext=(1.75, 107),
            fontsize=14,
            fontweight=850,
            color=RED,
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
        )
    elif callout_mode == "result":
        ax.annotate(
            "simplified data can damage\nshallow suffixes",
            xy=(0, float(sum(min(0, values[key][0]) for key in ("true", "gaussian", "mean")))),
            xytext=(0.65, -48),
            fontsize=14,
            fontweight=850,
            color=RED,
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
        )
    return fig


cnn_training_frames = [
    figure_to_image(cnn_training_time_frame(CNN_TRAINING_SPECS[0]))
]
for checkpoint_spec in CNN_TRAINING_SPECS[1:-1]:
    cnn_training_frames.append(
        figure_to_image(cnn_training_time_frame(checkpoint_spec))
    )
cnn_training_frames.extend(
    [
        figure_to_image(cnn_training_time_frame(CNN_TRAINING_SPECS[-1])),
        figure_to_image(
            cnn_training_time_frame(
                CNN_TRAINING_SPECS[-1], callout_mode="result"
            )
        ),
    ]
)

save_gif(
    cnn_training_frames,
    [3000] + [1000] * 8 + [6000],
    FIGURE_DIR / "cnn_over_training_time.gif",
)
cnn_training_frames[-1].save(FIGURE_DIR / "cnn_over_training_time_final.png")


# %%
def relaxation_time_frame(through_epoch: int, *, callout: bool = False) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6.75))
    grid = fig.add_gridspec(
        2, 5, height_ratios=[1.7, 1], left=.08, right=.98,
        top=.86, bottom=.09, hspace=.42, wspace=.23
    )
    axes = [fig.add_subplot(grid[0, :])]
    axes += [fig.add_subplot(grid[1, i]) for i in range(5)]
    styles = {
        "true": {"color": TRUE, "linestyle": "-", "label": "real"},
        "gaussian": {
            "color": GAUSSIAN,
            "linestyle": "--",
            "label": "Gaussian",
        },
        "mean": {
            "color": MEAN,
            "linestyle": ":",
            "label": "class means",
        },
    }

    for ax, checkpoint_epoch in zip(axes, CNN_CHECKPOINTS):
        for distribution, style in styles.items():
            x, y = CNN_CURVES[checkpoint_epoch][distribution]
            mask = x <= through_epoch
            ax.plot(
                x[mask], y[mask], marker="o",
                markersize=5 if checkpoint_epoch else 7,
                linewidth=2 if checkpoint_epoch else 3,
                color=style["color"], linestyle=style["linestyle"],
                label=style["label"],
            )
        ax.grid()
        ax.set_xlim(-.3, 10.3)
        ax.set_ylim(5, 60 if checkpoint_epoch == 0 else 86)
        if checkpoint_epoch:
            ax.set_title(f"Epoch {checkpoint_epoch}", fontsize=10, pad=4)
            ax.tick_params(labelsize=8)
        else:
            ax.set_ylabel("held-out accuracy (%)")
            ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=10)

    fig.text(
        0.965,
        0.91,
        "EPOCH 0",
        color=RED,
        fontsize=22,
        fontweight=900,
        ha="right",
    )
    fig.text(
        0.105,
        0.91,
        "RELAX SUFFIX WITH FROZEN PREFIX",
        color=RED,
        fontsize=14,
        fontweight=850,
        ha="left",
    )
    gif_badge(fig)
    if callout:
        x, y = CNN_CURVES[0]["mean"]
        axes[0].annotate(
            "gains accuracy training on mean-only data\n→ has not learned the class means yet",
            xy=(x[-1], y[-1]), xytext=(3.6, 50),
            fontsize=16,
            fontweight=850,
            color=RED,
            ha="left",
            arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 3},
            bbox={
                "boxstyle": "round,pad=.3", "facecolor": "white",
                "edgecolor": RED, "alpha": .94,
            },
        )
    return fig


relaxation_frames = [
    figure_to_image(relaxation_time_frame(epoch)) for epoch in range(11)
]
relaxation_frames.append(
    figure_to_image(relaxation_time_frame(10, callout=True))
)
relaxation_durations = [3000] + [1000] * 10 + [6000]

save_gif(
    relaxation_frames,
    relaxation_durations,
    FIGURE_DIR / "cnn_relaxation_grid_time.gif",
)
relaxation_frames[-1].save(FIGURE_DIR / "cnn_relaxation_grid_time_final.png")


def cnn_relaxation_endpoint_stacked() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.82, bottom=0.17)
    x = np.arange(len(CNN_CHECKPOINTS))
    values = {
        distribution: np.asarray(
            [
                CNN_CURVES[epoch][distribution][1][-1]
                - CNN_CURVES[epoch][distribution][1][0]
                for epoch in CNN_CHECKPOINTS
            ]
        )
        for distribution in ("true", "gaussian", "mean")
    }
    signed_stacked_bars(ax, x, values, width=0.6)
    ax.axhline(0, color="#647383", linewidth=1.4, zorder=2)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(-28, 98)
    ax.set_xticks(x, [str(epoch) for epoch in CNN_CHECKPOINTS])
    ax.set_ylabel("accuracy change (percentage points)")
    ax.set_xlabel("ordinary-training epoch")
    red_axis_prompts(ax, horizontal="ORDINARY TRAINING TIME")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    ax.annotate(
        "mean-only relaxation starts helpful",
        xy=(0, values["mean"][0]),
        xytext=(0.55, 74),
        fontsize=14,
        fontweight=850,
        color=RED,
        arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
    )
    ax.annotate(
        "then becomes harmful",
        xy=(5, values["mean"][5]),
        xytext=(3.55, -21),
        fontsize=14,
        fontweight=850,
        color=RED,
        arrowprops={"arrowstyle": "-|>", "color": RED, "linewidth": 2.8},
    )
    return fig


figure_to_image(cnn_relaxation_endpoint_stacked()).save(
    FIGURE_DIR / "cnn_relaxation_endpoint_stacked.png"
)


# %%
def cut_schematic_svg(
    *,
    architecture: str,
    blocks: int,
    cut: int,
) -> str:
    width = 520
    height = 112
    block_width = 42
    block_gap = 18 if blocks == 4 else 10
    chain_width = blocks * block_width + (blocks - 1) * block_gap
    left = (width - chain_width) / 2
    right = left + chain_width
    block_y = 42
    block_height = 42
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="112" '
        'viewBox="0 0 520 112" role="img" '
        f'aria-label="{architecture} cut after block {cut}">',
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        ".small{font-size:10px;font-weight:700;letter-spacing:.8px}"
        ".label{font-size:12px;font-weight:700}"
        "</style>",
        '<rect width="520" height="112" rx="14" fill="#fffdf8"/>',
        f'<path d="M{left - 20:.2f} 63 H{right + 20:.2f}" '
        'stroke="#8493a1" stroke-width="2"/>',
    ]
    for block_index in range(1, blocks + 1):
        x = left + (block_index - 1) * (block_width + block_gap)
        fill = "#dff2fa" if block_index <= cut else "#eef1f4"
        stroke = "#0b83bd" if block_index <= cut else "#a8b2bb"
        elements.append(
            f'<rect x="{x:.2f}" y="{block_y}" width="{block_width:.2f}" '
            f'height="{block_height}" rx="7" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{x + block_width / 2:.2f}" y="68" '
            f'text-anchor="middle" class="label" fill="#29445e">{block_index}</text>'
        )
    if cut == blocks:
        cut_x = right + block_gap / 2
    else:
        cut_x = left + cut * block_width + (cut - 0.5) * block_gap
    elements.extend(
        [
            f'<path d="M{cut_x:.2f} 23 V95" stroke="#df6900" '
            'stroke-width="3" stroke-dasharray="6 5"/>',
            f'<text x="{cut_x:.2f}" y="16" text-anchor="middle" '
            'class="small" fill="#b84e00">CUT</text>',
            f'<text x="20" y="105" class="small" fill="#657386">{architecture.upper()}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


for block in range(1, 5):
    (FIGURE_DIR / f"cut_cnn_block{block}.svg").write_text(
        cut_schematic_svg(architecture="four-block CNN", blocks=4, cut=block)
    )

for block in (1, 2, 8):
    (FIGURE_DIR / f"cut_resnet_block{block}.svg").write_text(
        cut_schematic_svg(architecture="ResNet-18", blocks=8, cut=block)
    )


# %%
print("Wrote:")
for path in sorted(FIGURE_DIR.glob("*time*")):
    print(f"  {path.relative_to(PROJECT_ROOT)}")
for path in sorted(FIGURE_DIR.glob("cut_*.svg")):
    print(f"  {path.relative_to(PROJECT_ROOT)}")
