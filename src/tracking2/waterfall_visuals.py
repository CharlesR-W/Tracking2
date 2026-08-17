"""Reproducible ResNet-first waterfall viewer and talk-style GIF exports.

The builder reads only hash-pinned saved JSON artifacts. It does not import or
run training code. The signed waterfall stacks are a compact visual profile of
alternative suffix outcomes; they are explicitly not additive components.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs


SERIES_ORDER = ("mean", "gaussian", "true")
SERIES = {
    "true": {
        "label": "real activations",
        "color": "#222222",
        "linestyle": "-",
        "marker": "o",
        "pattern": "",
    },
    "gaussian": {
        "label": "class Gaussian",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "^",
        "pattern": "/",
    },
    "mean": {
        "label": "class mean + trace-matched noise",
        "color": "#E69F00",
        "linestyle": ":",
        "marker": "D",
        "pattern": ".",
    },
}
INK = "#243B5A"
MUTED = "#667085"
GRID = "#DCE3EC"
PANEL = "#F4F6F9"
ACCENT = "#C2412D"

RESNET_DURATIONS_MS = [3000, 1000, 1000, 1000, 1000, 6000]
CNN_DURATIONS_MS = [3000, 1000, 1000, 1000, 1000, 1000, 6000]
RELAXATION_DURATIONS_MS = [3000] + [1000] * 10 + [6000]


class WaterfallDataError(RuntimeError):
    """Raised when a pinned artifact does not satisfy the visual contract."""


@dataclass(frozen=True)
class BuildOutputs:
    html: Path
    figures: tuple[Path, ...]
    gif_reports: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WaterfallDataError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WaterfallDataError(f"Expected a JSON object at {path}.")
    return payload


def load_pinned_inputs(
    project_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate every manifest path/hash and return payloads by stable input id."""
    project_root = project_root.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise WaterfallDataError("Waterfall manifest schema_version must be 1.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise WaterfallDataError("Waterfall manifest needs a non-empty inputs list.")
    result: dict[str, dict[str, Any]] = {}
    for row in inputs:
        if not isinstance(row, dict):
            raise WaterfallDataError("Every waterfall input entry must be an object.")
        input_id = str(row.get("id", ""))
        relative = Path(str(row.get("path", "")))
        expected_hash = str(row.get("sha256", ""))
        if not input_id or input_id in result:
            raise WaterfallDataError(f"Missing or duplicate waterfall input id: {input_id!r}")
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root):
            raise WaterfallDataError(f"Waterfall input escapes project root: {relative}")
        if not path.is_file():
            raise WaterfallDataError(f"Missing waterfall input: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise WaterfallDataError(
                f"Hash mismatch for {relative}: {actual_hash}, expected {expected_hash}"
            )
        result[input_id] = _load_json(path)
    return manifest, result


def _require_measured(payload: Mapping[str, Any], input_id: str) -> None:
    if str(payload.get("status", "")).upper() != "MEASURED":
        raise WaterfallDataError(f"{input_id} is not labeled MEASURED.")


def _endpoint_change(
    records: Sequence[Mapping[str, Any]],
    distribution: str,
) -> tuple[float, float, int]:
    selected = [
        row
        for row in records
        if row.get("train_distribution") == distribution
        and row.get("eval_distribution") == "true"
    ]
    if not selected:
        raise WaterfallDataError(
            f"No held-out true-evaluation records for {distribution}."
        )
    draws = sorted({int(row.get("draw", 0)) for row in selected})
    changes: list[float] = []
    for draw in draws:
        draw_rows = [row for row in selected if int(row.get("draw", 0)) == draw]
        steps = sorted({float(row["relax_epoch"]) for row in draw_rows})
        start_rows = [row for row in draw_rows if float(row["relax_epoch"]) == steps[0]]
        end_rows = [row for row in draw_rows if float(row["relax_epoch"]) == steps[-1]]
        if len(start_rows) != 1 or len(end_rows) != 1:
            raise WaterfallDataError(
                f"Expected one start/end row for {distribution}, draw {draw}."
            )
        changes.append(100.0 * (float(end_rows[0]["accuracy"]) - float(start_rows[0]["accuracy"])))
    mean = float(np.mean(changes))
    std = float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0
    return mean, std, len(changes)


def _depth_axis_range(
    series: Mapping[str, Sequence[Sequence[float]]],
) -> list[float]:
    positive: list[float] = []
    negative: list[float] = []
    individual: list[float] = []
    frame_count = len(next(iter(series.values())))
    for frame_index in range(frame_count):
        cut_count = len(next(iter(series.values()))[frame_index])
        for cut_index in range(cut_count):
            values = [float(series[name][frame_index][cut_index]) for name in SERIES_ORDER]
            positive.append(sum(max(0.0, value) for value in values))
            negative.append(sum(min(0.0, value) for value in values))
            individual.extend(values)
    low = min(negative + individual + [0.0])
    high = max(positive + individual + [0.0])
    span = max(high - low, 1.0)
    return [float(low - 0.10 * span), float(high + 0.12 * span)]


def build_payload(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest, inputs = load_pinned_inputs(project_root, manifest_path)

    resnet_epochs = [int(value) for value in manifest["resnet"]["checkpoints"]]
    resnet_cuts = [int(value) for value in manifest["resnet"]["cuts"]]
    resnet_series = {name: [] for name in SERIES_ORDER}
    resnet_std = {name: [] for name in SERIES_ORDER}
    resnet_draws = {name: [] for name in SERIES_ORDER}
    resnet_modules: list[str] | None = None
    for epoch in resnet_epochs:
        input_id = f"resnet_epoch_{epoch}"
        artifact = inputs[input_id]
        _require_measured(artifact, input_id)
        config = artifact.get("config", {})
        if int(config.get("checkpoint_epoch", -1)) != epoch or int(config.get("seed", -1)) != 0:
            raise WaterfallDataError(f"{input_id} has unexpected checkpoint/seed.")
        slices = sorted(artifact.get("slices", []), key=lambda row: int(row["cut"]))
        if [int(row["cut"]) + 1 for row in slices] != resnet_cuts:
            raise WaterfallDataError(f"{input_id} does not contain the expected eight cuts.")
        modules = [str(row["module"]) for row in slices]
        if resnet_modules is None:
            resnet_modules = modules
        elif modules != resnet_modules:
            raise WaterfallDataError("ResNet module order changes across checkpoints.")
        for name in SERIES_ORDER:
            values: list[float] = []
            stds: list[float] = []
            counts: list[int] = []
            for slice_row in slices:
                mean, std, count = _endpoint_change(slice_row["records"], name)
                values.append(mean)
                stds.append(std)
                counts.append(count)
            resnet_series[name].append(values)
            resnet_std[name].append(stds)
            resnet_draws[name].append(counts)
    if any(count != 3 for frames in resnet_draws.values() for cuts in frames for count in cuts):
        raise WaterfallDataError("Every ResNet cell must contain three surrogate redraws.")

    cnn_epochs = [int(value) for value in manifest["cnn"]["checkpoints"]]
    cnn_cuts = [int(value) for value in manifest["cnn"]["cuts"]]
    cnn_series = {name: [] for name in SERIES_ORDER}
    cnn_std = {name: [] for name in SERIES_ORDER}
    for epoch in cnn_epochs:
        by_cut = []
        for cut in cnn_cuts:
            input_id = f"cnn_epoch_{epoch}_cut_{cut}"
            artifact = inputs[input_id]
            _require_measured(artifact, input_id)
            config = artifact.get("config", {})
            if (
                int(config.get("checkpoint_epoch", -1)) != epoch
                or int(config.get("cut", -1)) != cut
                or int(config.get("seed", -1)) != 0
            ):
                raise WaterfallDataError(f"{input_id} has unexpected checkpoint/cut/seed.")
            by_cut.append(artifact)
        for name in SERIES_ORDER:
            values: list[float] = []
            stds: list[float] = []
            for artifact in by_cut:
                mean, std, count = _endpoint_change(artifact["records"], name)
                if count != 1:
                    raise WaterfallDataError("Legacy CNN cells must each contain one replay bank.")
                values.append(mean)
                stds.append(std)
            cnn_series[name].append(values)
            cnn_std[name].append(stds)

    relaxation_artifact = inputs["cnn_epoch_1_cut_3"]
    relaxation: dict[str, dict[str, list[float]]] = {}
    expected_steps = [float(value) for value in manifest["cnn"]["relaxation_curve"]["epochs"]]
    for name in SERIES_ORDER:
        rows = sorted(
            (
                row
                for row in relaxation_artifact["records"]
                if row.get("train_distribution") == name
                and row.get("eval_distribution") == "true"
            ),
            key=lambda row: float(row["relax_epoch"]),
        )
        steps = [float(row["relax_epoch"]) for row in rows]
        if steps != expected_steps:
            raise WaterfallDataError(f"CNN relaxation steps for {name} are incomplete.")
        relaxation[name] = {
            "steps": steps,
            "accuracy": [100.0 * float(row["accuracy"]) for row in rows],
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "evidence_status": manifest["evidence_status"],
        "canonical_example": manifest["canonical_example"],
        "input_count": len(inputs),
        "resnet": {
            "architecture": "ResNet-18",
            "epochs": resnet_epochs,
            "cuts": resnet_cuts,
            "modules": resnet_modules,
            "series": resnet_series,
            "std": resnet_std,
            "model_units": 1,
            "redraws_per_cell": 3,
            "checkpoint_semantics": "one trained model",
        },
        "cnn": {
            "architecture": "Four-block residual CNN",
            "epochs": cnn_epochs,
            "cuts": cnn_cuts,
            "series": cnn_series,
            "std": cnn_std,
            "model_units_per_frame": 1,
            "checkpoint_semantics": manifest["cnn"]["checkpoint_semantics"],
        },
        "relaxation": {
            "architecture": "Four-block residual CNN",
            "checkpoint_epoch": 1,
            "cut": 3,
            "series": relaxation,
        },
        "inputs": manifest["inputs"],
    }
    payload["resnet"]["y_range"] = _depth_axis_range(resnet_series)
    payload["cnn"]["y_range"] = _depth_axis_range(cnn_series)
    all_accuracy = [value for row in relaxation.values() for value in row["accuracy"]]
    span = max(all_accuracy) - min(all_accuracy)
    payload["relaxation"]["y_range"] = [
        min(all_accuracy) - 0.08 * span,
        max(all_accuracy) + 0.18 * span,
    ]
    return payload


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
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
            "axes.edgecolor": "#AAB5C4",
            "grid.color": "white",
            "grid.linewidth": 1.2,
        }
    )


def _signed_stacked_bars(
    ax: plt.Axes,
    x: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    width: float = 0.62,
) -> None:
    positive = np.zeros_like(x, dtype=float)
    negative = np.zeros_like(x, dtype=float)
    for name in SERIES_ORDER:
        heights = np.asarray(values[name], dtype=float)
        bottoms = np.where(heights >= 0, positive, negative)
        ax.bar(
            x,
            heights,
            width,
            bottom=bottoms,
            color=SERIES[name]["color"],
            edgecolor="white",
            linewidth=0.8,
            label=SERIES[name]["label"],
            zorder=3,
        )
        positive = np.where(heights >= 0, positive + heights, positive)
        negative = np.where(heights < 0, negative + heights, negative)


def _gif_badge(fig: plt.Figure) -> None:
    fig.text(
        0.983,
        0.975,
        "PILOT · GIF",
        ha="right",
        va="top",
        color="white",
        fontsize=10,
        fontweight=700,
        bbox={"boxstyle": "round,pad=.35", "facecolor": ACCENT, "edgecolor": ACCENT},
    )


def _depth_frame(
    model: Mapping[str, Any],
    frame_index: int,
    *,
    callout: bool = False,
) -> plt.Figure:
    architecture = str(model["architecture"])
    epoch = int(model["epochs"][frame_index])
    cuts = np.asarray(model["cuts"], dtype=float)
    values = {
        name: np.asarray(model["series"][name][frame_index], dtype=float)
        for name in SERIES_ORDER
    }
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.84, bottom=0.20)
    _signed_stacked_bars(ax, cuts, values)
    ax.axhline(0, color="#667085", linewidth=1.2, zorder=4)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(float(cuts[0] - 0.65), float(cuts[-1] + 0.65))
    ax.set_ylim(*model["y_range"])
    ax.set_xticks(cuts, [str(int(value)) for value in cuts])
    ax.set_xlabel("Cut after residual block")
    ax.set_ylabel("Final held-out accuracy change (percentage points)")
    unit = "one trained model" if architecture == "ResNet-18" else "separately scheduled model"
    ax.set_title(
        f"{architecture} · checkpoint epoch {epoch}\nHistorical pilot · {unit}",
        loc="left",
        pad=10,
        fontweight=700,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    _gif_badge(fig)
    footer = "Alternative suffix outcomes are stacked by sign for compactness; they are not additive."
    if architecture == "ResNet-18":
        footer += " ResNet whisker data in the viewer are SD across 3 redraws, not model seeds."
    else:
        footer += " Frames are separately scheduled CNN models."
    fig.text(0.975, 0.045, footer, ha="right", va="bottom", color=MUTED, fontsize=8.2)
    if callout:
        early_total = sum(min(0.0, values[name][0]) for name in SERIES_ORDER)
        late_total = sum(min(0.0, values[name][-1]) for name in SERIES_ORDER)
        ax.annotate(
            "Shallow suffixes move much more\n→ strongest sensitivity is early",
            xy=(cuts[0], early_total),
            xytext=(cuts[0] + 0.8, model["y_range"][0] + 0.18 * (model["y_range"][1] - model["y_range"][0])),
            color=ACCENT,
            fontsize=13,
            fontweight=700,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": ACCENT, "alpha": 0.95},
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 2.1},
        )
        ax.annotate(
            "late cut barely moves",
            xy=(cuts[-1], late_total),
            xytext=(cuts[-1] - 2.1, model["y_range"][0] + 0.45 * (model["y_range"][1] - model["y_range"][0])),
            color=ACCENT,
            fontsize=12,
            fontweight=700,
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 2.0},
        )
    return fig


def _relaxation_frame(
    relaxation: Mapping[str, Any],
    through_epoch: int,
    *,
    callout: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.84, bottom=0.20)
    for name in SERIES_ORDER:
        row = relaxation["series"][name]
        steps = np.asarray(row["steps"], dtype=float)
        accuracy = np.asarray(row["accuracy"], dtype=float)
        mask = steps <= through_epoch
        ax.plot(
            steps[mask],
            accuracy[mask],
            color=SERIES[name]["color"],
            linestyle=SERIES[name]["linestyle"],
            marker=SERIES[name]["marker"],
            markersize=6,
            linewidth=2.5,
            label=SERIES[name]["label"],
            zorder=3,
        )
    ax.set_xlim(-0.25, 10.25)
    ax.set_ylim(*relaxation["y_range"])
    ax.set_xlabel("Suffix-relaxation epoch")
    ax.set_ylabel("Held-out real-activation accuracy (%)")
    ax.set_title(
        "Four-block residual CNN · checkpoint epoch 1 · cut after block 3\n"
        "Historical pilot · one checkpoint and one replay bank per condition",
        loc="left",
        pad=10,
        fontweight=700,
    )
    ax.grid(zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )
    _gif_badge(fig)
    if not callout:
        ax.text(
            0.985,
            0.02,
            f"Showing relaxation through epoch {through_epoch}",
            transform=ax.transAxes,
            ha="right",
            color=MUTED,
            fontsize=9,
        )
    else:
        row = relaxation["series"]["mean"]
        ax.annotate(
            "The conditions separate during suffix relaxation\n→ replay choice changes the reachable endpoint",
            xy=(row["steps"][-1], row["accuracy"][-1]),
            xytext=(3.4, relaxation["y_range"][0] + 0.16 * (relaxation["y_range"][1] - relaxation["y_range"][0])),
            color=ACCENT,
            fontsize=12.5,
            fontweight=700,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": ACCENT, "alpha": 0.95},
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 2.0},
        )
    return fig


def _figure_to_image(fig: plt.Figure, *, dpi: int = 120) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def _save_gif(
    frames: Sequence[Image.Image],
    durations_ms: Sequence[int],
    output: Path,
) -> dict[str, Any]:
    if len(frames) != len(durations_ms):
        raise ValueError("Each GIF frame requires one explicit duration.")
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
    return verify_gif(output, durations_ms)


def verify_gif(path: Path, expected_durations_ms: Sequence[int]) -> dict[str, Any]:
    with Image.open(path) as image:
        durations: list[int] = []
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))
        report = {
            "path": str(path),
            "frame_count": image.n_frames,
            "durations_ms": durations,
            "size": list(image.size),
            "inspected_frame_indices": sorted({0, image.n_frames // 2, image.n_frames - 1}),
        }
    if durations != list(expected_durations_ms):
        raise WaterfallDataError(
            f"GIF timing mismatch for {path}: {durations}, expected {list(expected_durations_ms)}"
        )
    return report


def _depth_summary(model: Mapping[str, Any]) -> plt.Figure:
    epochs = list(model["epochs"])
    columns = 3
    rows = int(np.ceil(len(epochs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, 4.0 * rows), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.87, bottom=0.16, hspace=0.38, wspace=0.20)
    cuts = np.asarray(model["cuts"], dtype=float)
    for frame_index, ax in enumerate(axes.flat):
        if frame_index >= len(epochs):
            ax.set_visible(False)
            continue
        values = {
            name: np.asarray(model["series"][name][frame_index], dtype=float)
            for name in SERIES_ORDER
        }
        _signed_stacked_bars(ax, cuts, values, width=0.66)
        ax.axhline(0, color="#667085", linewidth=1.0, zorder=4)
        ax.grid(axis="y", zorder=0)
        ax.set_ylim(*model["y_range"])
        ax.set_xticks(cuts, [str(int(value)) for value in cuts])
        ax.set_title(f"checkpoint epoch {epochs[frame_index]}", fontsize=12, loc="left")
        ax.set_xlabel("cut after block")
        if frame_index % columns == 0:
            ax.set_ylabel("accuracy change (points)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False)
    fig.suptitle(
        f"{model['architecture']} historical pilot · depth by training checkpoint",
        fontsize=18,
        fontweight=700,
        x=0.065,
        ha="left",
    )
    fig.text(
        0.985,
        0.010,
        "Alternative suffix outcomes are stacked by sign; they are not additive components.",
        ha="right",
        color=MUTED,
        fontsize=9,
    )
    return fig


def _html_report(payload: Mapping[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    plotly_js = "\n".join(line.rstrip() for line in get_plotlyjs().splitlines())
    provenance_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['id']))}</td>"
        f"<td><code>{html.escape(str(row['path']))}</code></td>"
        f"<td><code>{html.escape(str(row['sha256']))}</code></td>"
        "</tr>"
        for row in payload["inputs"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracking2 waterfall explorer</title>
<script>{plotly_js}</script>
<style>
*{{box-sizing:border-box}}:root{{--ink:#243b5a;--muted:#667085;--rule:#dce3ec;--soft:#f4f6f9;--accent:#c2412d}}
body{{margin:0;background:#fff;color:var(--ink);font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow-x:hidden}}
main{{max-width:1180px;margin:auto;padding:42px 28px 72px}}h1{{font-size:2.25rem;line-height:1.1;margin:.2rem 0 .7rem}}
h2{{border-top:1px solid var(--rule);padding-top:2rem;margin-top:3rem}}h3{{margin-top:1.7rem}}
.lede{{font-size:1.13rem;max-width:880px}}.status{{display:inline-block;background:#fff1ed;color:#9f2d1b;border:1px solid #efb5a9;border-radius:99px;padding:.3rem .7rem;font-weight:750;font-size:.78rem;letter-spacing:.04em}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:1.4rem 0}}.card{{background:var(--soft);border:1px solid var(--rule);border-radius:10px;padding:16px}}
.card strong{{display:block;font-size:1.08rem;margin-bottom:.25rem}}.guide{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:1rem 0}}
.guide div{{border-left:4px solid var(--accent);background:#fff8f5;padding:12px 15px}}.plot{{height:650px;width:100%;max-width:100%;min-width:0;overflow:hidden;border:1px solid var(--rule);border-radius:10px;background:white}}
.note{{color:var(--muted);font-size:.91rem;max-width:980px}}code{{font-size:.84em}}details{{margin-top:1rem}}
table{{width:100%;border-collapse:collapse;font-size:.78rem}}th,td{{text-align:left;vertical-align:top;padding:7px;border:1px solid var(--rule)}}th{{background:var(--soft)}}td code{{overflow-wrap:anywhere}}
@media(max-width:760px){{main{{padding:24px 14px 48px}}h1{{font-size:1.75rem}}.grid,.guide{{grid-template-columns:1fr}}.plot{{height:560px}}}}
</style>
</head>
<body><main>
<span class="status">MEASURED PILOT · ONE RESNET MODEL</span>
<h1>Where does a suffix remain sensitive to activation structure?</h1>
<p class="lede"><strong>Current observation:</strong> in the ResNet-18 pilot, suffix relaxation moves held-out accuracy much more at shallow cuts than at the final cut. The animation is an explanatory pilot, not an independent-seed confirmation.</p>
<div class="grid">
  <div class="card"><strong>Canonical example</strong>ResNet-18 · eight native post-block cuts</div>
  <div class="card"><strong>Evolution</strong>Five checkpoints from epoch 0 to 100</div>
  <div class="card"><strong>Evidence unit</strong>One model; three surrogate redraws per cell</div>
</div>
<h2>ResNet-18 depth × training waterfall</h2>
<p><strong>Question.</strong> After freezing a prefix, how much does the suffix's held-out accuracy move when relaxed on real, class-Gaussian, or class-mean replay?</p>
<h3>Methodology</h3>
<p>The plotted quantity is endpoint accuracy minus the shared unrelaxed accuracy, in percentage points. Every frame uses the same trained ResNet-18 trajectory and the same fixed y-axis. Redraw variation changes generated activations and minibatch order; it is not model-seed uncertainty.</p>
<div class="guide"><div><strong>Large early stacks, small late stacks</strong> → sensitivity is concentrated in shallow suffixes under this intervention.</div><div><strong>Similar profiles across depth</strong> → weakens the proposed depth contrast.</div></div>
<div id="resnet-plot" class="plot"></div>
<p class="note"><strong>How to read it:</strong> “Waterfall” stacks alternative outcomes by sign for compactness; segments are not additive. Switch to “Common baseline” for direct magnitude comparison. Hover shows each actual endpoint change and, for ResNet, redraw SD.</p>
<h2>CNN comparison</h2>
<span class="status">MEASURED PILOT · SEPARATELY SCHEDULED MODELS</span>
<p>The four-block CNN provides a coarser comparison. Its six checkpoint frames came from separately scheduled models, so apparent training-time motion is not one uninterrupted model trajectory.</p>
<div id="cnn-plot" class="plot"></div>
<h2>One suffix relaxing through time</h2>
<span class="status">MEASURED PILOT · ONE CHECKPOINT</span>
<p>This view holds the epoch-1 CNN prefix and cut fixed, then reveals the ten suffix-relaxation epochs. It shows genuine optimization time for three suffix copies.</p>
<div class="guide"><div><strong>Trajectories separate</strong> → replay choice changes the endpoint reached within the fixed budget.</div><div><strong>Trajectories remain aligned</strong> → weakens sensitivity to the tested replay statistics.</div></div>
<div id="relax-plot" class="plot"></div>
<h2>Interpretation boundary</h2>
<p><strong>Supported as a pilot:</strong> these saved runs display a large shallow-versus-late contrast under finite suffix relaxation. <strong>Not established:</strong> a universal depth law, a representation-only mechanism, or ResNet replication across independently trained models. The separate three-seed CNN battery remains the post's controlled quantitative evidence.</p>
<details><summary>Input provenance · {payload['input_count']} hash-pinned JSON artifacts</summary><table><thead><tr><th>ID</th><th>Path</th><th>SHA-256</th></tr></thead><tbody>{provenance_rows}</tbody></table></details>
</main>
<script>
const DATA={payload_json};
const ORDER=['mean','gaussian','true'];
const STYLE={{mean:{{label:'class mean + trace-matched noise',color:'#E69F00',pattern:'.'}},gaussian:{{label:'class Gaussian',color:'#0072B2',pattern:'/'}},true:{{label:'real activations',color:'#222222',pattern:''}}}};
const CONFIG={{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}};
function tracesFor(model,fi){{return ORDER.map(name=>({{type:'bar',name:STYLE[name].label,x:model.cuts,y:model.series[name][fi],customdata:model.std[name][fi],marker:{{color:STYLE[name].color,pattern:{{shape:STYLE[name].pattern}}}},hovertemplate:'cut after block %{{x}}<br>'+STYLE[name].label+'<br>change %{{y:.2f}} pp<br>redraw SD %{{customdata:.2f}} pp<extra></extra>'}}));}}
function depthPlot(id,model){{
 const mobile=window.innerWidth<760;
 const frames=model.epochs.map((epoch,fi)=>({{name:'epoch-'+epoch,data:tracesFor(model,fi)}}));
 const steps=model.epochs.map(epoch=>({{label:String(epoch),method:'animate',args:[['epoch-'+epoch],{{mode:'immediate',frame:{{duration:0,redraw:true}},transition:{{duration:0}}}}]}}));
 const layout={{height:mobile?610:650,barmode:'relative',paper_bgcolor:'#fff',plot_bgcolor:'#f4f6f9',margin:mobile?{{l:62,r:8,t:135,b:135}}:{{l:80,r:25,t:105,b:90}},font:{{family:'system-ui,sans-serif',size:mobile?10:12,color:'#243b5a'}},title:{{text:model.architecture+' · checkpoint epoch '+model.epochs[0],x:.04}},xaxis:{{title:'Cut after residual block',dtick:1,range:[model.cuts[0]-.7,model.cuts.at(-1)+.7],gridcolor:'#fff'}},yaxis:{{title:mobile?'Accuracy change (points)':'Final held-out accuracy change (percentage points)',range:model.y_range,fixedrange:false,gridcolor:'#fff',zeroline:true,zerolinecolor:'#667085'}},legend:mobile?{{orientation:'h',y:-.30,font:{{size:9}}}}:{{orientation:'h',y:-.18}},updatemenus:[{{type:'buttons',direction:'left',x:0,y:mobile?1.22:1.16,showactive:false,buttons:[{{label:'▶ play',method:'animate',args:[null,{{fromcurrent:true,frame:{{duration:900,redraw:true}},transition:{{duration:220}}}}]}},{{label:'❚❚ pause',method:'animate',args:[[null],{{mode:'immediate',frame:{{duration:0,redraw:false}}}}]}}]}},{{type:mobile?'dropdown':'buttons',direction:mobile?'down':'left',x:1,y:mobile?1.22:1.16,xanchor:'right',buttons:[{{label:'Waterfall',method:'relayout',args:[{{barmode:'relative'}}]}},{{label:'Common baseline',method:'relayout',args:[{{barmode:'group'}}]}}]}}],sliders:[{{active:0,currentvalue:{{prefix:'checkpoint epoch '}},pad:{{t:45}},steps}}]}};
 Plotly.newPlot(id,tracesFor(model,0),layout,CONFIG).then(()=>{{Plotly.addFrames(id,frames);document.getElementById(id).on('plotly_sliderchange',e=>Plotly.relayout(id,{{'title.text':model.architecture+' · checkpoint epoch '+e.step.label}}));}});
}}
function relaxPlot(){{const mobile=window.innerWidth<760;const m=DATA.relaxation;const names=ORDER;const make=(name,n)=>({{type:'scatter',mode:'lines+markers',name:STYLE[name].label,x:m.series[name].steps.slice(0,n),y:m.series[name].accuracy.slice(0,n),line:{{color:STYLE[name].color,width:3,dash:name==='gaussian'?'dash':name==='mean'?'dot':'solid'}},marker:{{symbol:name==='gaussian'?'triangle-up':name==='mean'?'diamond':'circle',size:8}},hovertemplate:STYLE[name].label+'<br>relax epoch %{{x}}<br>accuracy %{{y:.2f}}%<extra></extra>'}});const epochs=m.series.true.steps;const frames=epochs.map((epoch,fi)=>({{name:'relax-'+epoch,data:names.map(name=>make(name,fi+1))}}));const steps=epochs.map(epoch=>({{label:String(epoch),method:'animate',args:[['relax-'+epoch],{{mode:'immediate',frame:{{duration:0,redraw:true}},transition:{{duration:0}}}}]}}));const layout={{height:mobile?610:650,paper_bgcolor:'#fff',plot_bgcolor:'#f4f6f9',margin:mobile?{{l:62,r:8,t:110,b:135}}:{{l:80,r:25,t:80,b:90}},font:{{family:'system-ui,sans-serif',size:mobile?10:12,color:'#243b5a'}},title:{{text:'CNN epoch 1 · cut after block 3',x:.04}},xaxis:{{title:'Suffix-relaxation epoch',range:[-.25,10.25],dtick:1,gridcolor:'#fff'}},yaxis:{{title:mobile?'Held-out accuracy (%)':'Held-out real-activation accuracy (%)',range:m.y_range,gridcolor:'#fff'}},legend:mobile?{{orientation:'h',y:-.30,font:{{size:9}}}}:{{orientation:'h',y:-.18}},updatemenus:[{{type:'buttons',direction:'left',x:0,y:1.12,showactive:false,buttons:[{{label:'▶ play',method:'animate',args:[null,{{fromcurrent:true,frame:{{duration:900,redraw:true}},transition:{{duration:220}}}}]}},{{label:'❚❚ pause',method:'animate',args:[[null],{{mode:'immediate',frame:{{duration:0,redraw:false}}}}]}}]}}],sliders:[{{active:0,currentvalue:{{prefix:'relaxation epoch '}},pad:{{t:45}},steps}}]}};Plotly.newPlot('relax-plot',names.map(name=>make(name,1)),layout,CONFIG).then(()=>Plotly.addFrames('relax-plot',frames));}}
depthPlot('resnet-plot',DATA.resnet);depthPlot('cnn-plot',DATA.cnn);relaxPlot();
</script></body></html>"""


def build_waterfall_outputs(
    project_root: Path,
    manifest_path: Path,
    figure_dir: Path,
    html_path: Path,
) -> BuildOutputs:
    _configure_matplotlib()
    payload = build_payload(project_root, manifest_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    gif_reports: list[dict[str, Any]] = []
    for key, stem, durations in (
        ("resnet", "resnet_over_training_time", RESNET_DURATIONS_MS),
        ("cnn", "cnn_over_training_time", CNN_DURATIONS_MS),
    ):
        model = payload[key]
        frames = [
            _figure_to_image(_depth_frame(model, frame_index))
            for frame_index in range(len(model["epochs"]))
        ]
        frames.append(
            _figure_to_image(_depth_frame(model, len(model["epochs"]) - 1, callout=True))
        )
        gif_path = figure_dir / f"{stem}.gif"
        final_path = figure_dir / f"{stem}_final.png"
        summary_path = figure_dir / f"{stem}_summary.png"
        gif_reports.append(_save_gif(frames, durations, gif_path))
        frames[-1].save(final_path)
        _figure_to_image(_depth_summary(model), dpi=120).save(summary_path)
        outputs.extend((gif_path, final_path, summary_path))

    relaxation = payload["relaxation"]
    relaxation_frames = [
        _figure_to_image(_relaxation_frame(relaxation, epoch)) for epoch in range(11)
    ]
    relaxation_frames.append(
        _figure_to_image(_relaxation_frame(relaxation, 10, callout=True))
    )
    relaxation_gif = figure_dir / "cnn_relaxation_time.gif"
    relaxation_final = figure_dir / "cnn_relaxation_time_final.png"
    gif_reports.append(
        _save_gif(relaxation_frames, RELAXATION_DURATIONS_MS, relaxation_gif)
    )
    relaxation_frames[-1].save(relaxation_final)
    outputs.extend((relaxation_gif, relaxation_final))

    html_path.write_text(_html_report(payload))
    return BuildOutputs(
        html=html_path,
        figures=tuple(outputs),
        gif_reports=tuple(gif_reports),
    )


def output_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: _sha256(path) for path in paths}
