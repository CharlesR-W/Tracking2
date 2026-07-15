from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import textwrap
from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


COLORS = {"plain": "#9ECAE1", "residual": "#08519C", "mean": "#9ECAE1", "covariance": "#4292C6", "true": "#084594"}
DASHES = {"mean": "dot", "covariance": "dash", "true": "solid"}
PROFILE_EPOCHS = (1, 5, 20, 30)


def estimate(rows: list[dict], metric: str) -> tuple[float, float | None, int]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, len(values)
    # Two-sided 95% t critical values for the confirmatory n=2..10 range.
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    half_width = t95.get(len(values), 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width, len(values)


def wrap_annotation(text: str, width: int = 92) -> str:
    return "<br>".join(textwrap.wrap(text, width=width))


def plot_html(figure: go.Figure, div_id: str) -> str:
    return to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )


def experiment_a_figure(rows: list[dict]) -> go.Figure:
    architectures = sorted({row["architecture"] for row in rows})
    facet_titles = ["CIFAR-10 model" if architecture == "residual" else "Optional plain control" for architecture in architectures]
    figure = make_subplots(rows=1, cols=len(architectures), shared_yaxes=True, subplot_titles=facet_titles)
    for column, architecture in enumerate(architectures, 1):
        for train_distribution in ("mean", "covariance", "true"):
            selected = sorted(
                (row for row in rows if row["architecture"] == architecture and row["train_distribution"] == train_distribution and row["test_distribution"] == "true"),
                key=lambda row: row["epoch"],
            )
            if not selected:
                continue
            by_epoch = {epoch: [row for row in selected if row["epoch"] == epoch] for epoch in sorted({row["epoch"] for row in selected})}
            estimates = [estimate(group, "loss") for group in by_epoch.values()]
            figure.add_trace(go.Scatter(
                x=list(by_epoch), y=[value[0] for value in estimates],
                error_y={"type": "data", "array": [value[1] or 0 for value in estimates], "visible": any(value[1] is not None for value in estimates), "thickness": 1.2},
                mode="lines+markers", name=f"$r=\\mathrm{{{train_distribution}}}$", legendgroup=train_distribution,
                showlegend=column == 1, line={"color": COLORS[train_distribution], "dash": DASHES[train_distribution], "width": 2},
                hovertemplate=f"{architecture} · trained on {train_distribution}<br>epoch %{{x}}<br>true loss %{{y:.4f}}<extra></extra>",
            ), row=1, col=column)
    figure.update_layout(
        title={
            "text": "True-CIFAR loss after training on each surrogate · $L_{\\mathrm{true}\\mid r}(t)$",
            "x": 0.0,
        },
        height=510,
        margin={"l": 75, "r": 25, "t": 105, "b": 65},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    figure.update_xaxes(title_text="$t$ · training epoch", dtick=1)
    figure.update_yaxes(title_text="$L_{\\mathrm{true}\\mid r}(t)$ · test cross-entropy", row=1, col=1)
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=1.12, xanchor="left", showarrow=False,
        text=wrap_annotation("Lines vary the training distribution; evaluation stays fixed on true CIFAR. Lower is better."),
        font={"size": 12, "color": "#6d6963"},
    )
    return figure


def experiment_a_reverse_figure(rows: list[dict]) -> go.Figure:
    figure = go.Figure()
    for test_distribution in ("mean", "covariance", "true"):
        selected = [
            row for row in rows
            if row["architecture"] == "residual" and row["train_distribution"] == "true"
            and row["test_distribution"] == test_distribution
        ]
        by_epoch = {epoch: [row for row in selected if row["epoch"] == epoch] for epoch in sorted({row["epoch"] for row in selected})}
        estimates = [estimate(group, "loss") for group in by_epoch.values()]
        figure.add_trace(go.Scatter(
            x=list(by_epoch), y=[value[0] for value in estimates], mode="lines+markers",
            error_y={"type": "data", "array": [value[1] or 0 for value in estimates], "visible": any(value[1] is not None for value in estimates), "thickness": 1.2},
            name=f"$r=\\mathrm{{{test_distribution}}}$",
            line={"color": COLORS[test_distribution], "dash": DASHES[test_distribution], "width": 2},
            hovertemplate=f"trained on true CIFAR · tested on {test_distribution}<br>epoch %{{x}}<br>loss %{{y:.4f}}<extra></extra>",
        ))
    figure.update_layout(
        title={"text": "When does true-CIFAR training solve each surrogate? · $L_{r\\mid\\mathrm{true}}(t)$", "x": 0.0},
        height=500, margin={"l": 75, "r": 25, "t": 105, "b": 65}, template="plotly_white",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    figure.update_xaxes(title_text="$t$ · training epoch", dtick=1)
    figure.update_yaxes(title_text="$L_{r\\mid\\mathrm{true}}(t)$ · test cross-entropy")
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=1.14, xanchor="left", showarrow=False,
        text=wrap_annotation("Training stays fixed on true CIFAR; lines vary the evaluation distribution. Lower is better."),
        font={"size": 12, "color": "#6d6963"},
    )
    return figure


def cifar_context_figure(rows: list[dict]) -> go.Figure:
    selected = [
        row for row in rows
        if row["architecture"] == "residual" and row["train_distribution"] == "true"
        and row["test_distribution"] == "true"
    ]
    by_epoch = {
        epoch: [row for row in selected if row["epoch"] == epoch]
        for epoch in sorted({row["epoch"] for row in selected})
    }
    estimates = [estimate(group, "loss") for group in by_epoch.values()]
    epochs = list(by_epoch)
    means = [value[0] for value in estimates]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=epochs, y=means, mode="lines+markers", name="all B checkpoints",
        line={"color": "#4292C6", "width": 2.5}, marker={"size": 7},
        error_y={"type": "data", "array": [value[1] or 0 for value in estimates], "visible": True, "thickness": 1.2},
        hovertemplate="epoch %{x}<br>true-CIFAR test CE = %{y:.4f}<extra></extra>",
    ))
    profile_epochs = [epoch for epoch in epochs if epoch in PROFILE_EPOCHS]
    figure.add_trace(go.Scatter(
        x=profile_epochs, y=[means[epochs.index(epoch)] for epoch in profile_epochs],
        mode="markers", name="shown in depth profiles",
        marker={"size": 11, "color": "#084594", "symbol": "circle-open", "line": {"width": 2}},
        hovertemplate="depth-profile epoch %{x}<br>true-CIFAR test CE = %{y:.4f}<extra></extra>",
    ))
    figure.update_layout(
        title={"text": "True-CIFAR loss at the B checkpoints · $L_{\\mathrm{true}\\mid\\mathrm{true}}(t_i)$", "x": 0.0},
        height=450, margin={"l": 75, "r": 25, "t": 100, "b": 70}, template="plotly_white",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2},
    )
    figure.update_xaxes(title_text="$t_i$ · training epoch", tickmode="array", tickvals=epochs)
    figure.update_yaxes(title_text="$L_{\\mathrm{true}\\mid\\mathrm{true}}(t_i)$ · test cross-entropy")
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=1.14, xanchor="left", showarrow=False,
        text=wrap_annotation("The same true-trained network used in B, evaluated on the true CIFAR-10 test set. Lower is better."),
        font={"size": 12, "color": "#6d6963"},
    )
    return figure


def tracking_figure(rows: list[dict], metric: str, title: str, y_title: str) -> go.Figure:
    architectures = sorted({row["architecture"] for row in rows})
    facet_titles = ["CIFAR-10 model" if architecture == "residual" else "Optional plain control" for architecture in architectures]
    figure = make_subplots(rows=1, cols=len(architectures), shared_yaxes=True, subplot_titles=facet_titles)
    available_epochs = {row["epoch"] for row in rows}
    epochs = [epoch for epoch in PROFILE_EPOCHS if epoch in available_epochs]
    subtitles = {
        "head_regret": "Held-out loss recovered by refitting only the suffix while holding the current representation fixed. Positive means the refit helped.",
        "tracking_demand": "Penalty for applying the previous checkpoint’s refitted suffix to the current representation. Zero means no tracking burden.",
        "relative_representation_drift": "Relative Euclidean change in cut-layer activations since the previous checkpoint. Zero means a stationary interface.",
        "compensable_fraction": "Fraction of sampled prefix Fisher sensitivity locally absorbable by suffix adjustment. Higher means more compensation.",
        "tracking_alignment": "Cosine between the actual suffix update and movement of the refitted optimum. +1 means directly following it.",
        "resolving_alignment": "Cosine between the actual suffix update and the direction toward the previous refitted optimum. +1 means closing that gap.",
    }
    hover_labels = {
        "head_regret": "$G_\\ell(t_i)$ · signed refit gain",
        "tracking_demand": "$T_\\ell(t_i)$ · tracking demand",
        "relative_representation_drift": "$D_\\ell(t_i)$ · interface drift",
        "compensable_fraction": "$C_\\ell(t_i)$ · compensable fraction",
        "tracking_alignment": "$\\cos(\\Delta b_i,q_{\\mathrm{track}})$",
        "resolving_alignment": "$\\cos(\\Delta b_i,q_{\\mathrm{resolve}})$",
    }
    palette = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    for column, architecture in enumerate(architectures, 1):
        for index, epoch in enumerate(epochs):
            selected = [row for row in rows if row["architecture"] == architecture and row["epoch"] == epoch and row.get(metric) is not None]
            if not selected:
                continue
            by_cut = {cut: [row for row in selected if row["cut"] == cut] for cut in sorted({row["cut"] for row in selected})}
            estimates = [estimate(group, metric) for group in by_cut.values()]
            figure.add_trace(go.Scatter(
                x=list(by_cut), y=[value[0] for value in estimates], mode="lines+markers",
                error_y={"type": "data", "array": [value[1] or 0 for value in estimates], "visible": any(value[1] is not None for value in estimates), "thickness": 1.2},
                name=f"$t_i={epoch}$", legendgroup=str(epoch), showlegend=column == 1,
                line={"color": palette[index % len(palette)], "width": 2},
                hovertemplate=(
                    f"{architecture} CNN · checkpoint {epoch}<br>cut after block %{{x}}"
                    f"<br>{hover_labels[metric]} = %{{y:.4g}}<extra></extra>"
                ),
            ), row=1, col=column)
    figure.update_layout(
        title={"text": title, "x": 0.0},
        height=530,
        margin={"l": 75, "r": 25, "t": 110, "b": 105},
        template="plotly_white",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.24},
    )
    figure.update_xaxes(title_text="cut layer $\\ell$ · after block $\\ell$", dtick=1)
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=1.12, xanchor="left", showarrow=False,
        text=wrap_annotation(subtitles[metric].replace("$", "")),
        font={"size": 12, "color": "#6d6963"},
    )
    if metric in {"head_regret", "tracking_demand", "relative_representation_drift", "tracking_alignment", "resolving_alignment"}:
        figure.add_hline(y=0, line={"color": "#6d6963", "dash": "dot", "width": 1})
    return figure


def time_comparison_figure(
    rows: list[dict],
    cuts: tuple[int, ...],
    metrics: tuple[tuple[str, str, str, str], ...],
    title: str,
    y_title: str,
    div_kind: str,
) -> go.Figure:
    """Compare the same quantities over training at fixed early/late cuts."""
    cut_labels = {cuts[0]: f"Early cut $\\ell={cuts[0]}$", cuts[-1]: f"Late cut $\\ell={cuts[-1]}$"}
    figure = make_subplots(
        rows=1, cols=len(cuts), shared_yaxes=True,
        subplot_titles=[cut_labels.get(cut, f"Cut $\\ell={cut}$") for cut in cuts],
    )
    for column, cut in enumerate(cuts, 1):
        for metric, label, color, dash in metrics:
            selected = [row for row in rows if row["cut"] == cut and row.get(metric) is not None]
            by_epoch = {
                epoch: [row for row in selected if row["epoch"] == epoch]
                for epoch in sorted({row["epoch"] for row in selected})
            }
            estimates = [estimate(group, metric) for group in by_epoch.values()]
            figure.add_trace(go.Scatter(
                x=list(by_epoch), y=[value[0] for value in estimates], mode="lines+markers",
                error_y={"type": "data", "array": [value[1] or 0 for value in estimates], "visible": any(value[1] is not None for value in estimates), "thickness": 1.2},
                name=label, legendgroup=metric, showlegend=column == 1,
                line={"color": color, "dash": dash, "width": 2.3},
                marker={"size": 7},
                hovertemplate=f"cut {cut} · epoch %{{x}}<br>{label} = %{{y:.4g}}<extra></extra>",
            ), row=1, col=column)
    figure.update_layout(
        title={"text": title, "x": 0.0}, height=500,
        margin={"l": 75, "r": 25, "t": 105, "b": 75}, template="plotly_white",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
    )
    figure.update_xaxes(title_text="$t_i$ · training epoch", tickmode="array", tickvals=sorted({row["epoch"] for row in rows}))
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    figure.add_hline(y=0, line={"color": "#6d6963", "dash": "dot", "width": 1})
    figure.add_annotation(
        xref="paper", yref="paper", x=0, y=1.14, xanchor="left", showarrow=False,
        text=wrap_annotation(
            f"The same {div_kind} is followed through training at the first and last residual-stream cuts. "
            "Points are seed means; bars are two-sided 95% t intervals."
        ),
        font={"size": 12, "color": "#6d6963"},
    )
    return figure


def fisher_figure(rows: list[dict]) -> go.Figure:
    figure = tracking_figure(
        rows,
        "compensable_fraction",
        "Suffix-compensable prefix sensitivity · $C_\\ell(t_i)$",
        "$C_\\ell(t_i)$ · compensable fraction",
    )
    profile_rows = [row for row in rows if row["epoch"] in PROFILE_EPOCHS and row.get("compensable_fraction") is not None]
    intervals = []
    for epoch in PROFILE_EPOCHS:
        for cut in sorted({row["cut"] for row in profile_rows}):
            group = [row for row in profile_rows if row["epoch"] == epoch and row["cut"] == cut]
            if group:
                mean, half_width, _ = estimate(group, "compensable_fraction")
                intervals.extend([mean - (half_width or 0), mean + (half_width or 0)])
    lower = max(0.0, min(intervals) - 0.02)
    upper = min(1.0, max(intervals) + 0.02)
    figure.update_yaxes(range=[lower, upper], tickformat=".0%")
    figure.add_annotation(
        xref="paper", yref="paper", x=1, y=0, xanchor="right", yanchor="bottom",
        text=f"Zoomed y-axis: {lower:.0%}–{upper:.0%}", showarrow=False,
        font={"size": 11, "color": "#a43d2f"}, bgcolor="rgba(255,255,255,.8)",
    )
    return figure


def criticality_heatmap(payload: dict) -> go.Figure:
    names = payload["module_names"]
    source_order = ["random"] + [str(epoch) for epoch in payload["config"]["checkpoint_epochs"]]
    final_epoch = max(payload["config"]["checkpoint_epochs"])
    labels = {
        "random": "fresh random draw",
        **{str(epoch): (f"epoch {epoch} (intact final)" if epoch == final_epoch else f"checkpoint {epoch}")
           for epoch in payload["config"]["checkpoint_epochs"]},
    }
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in payload["interventions"]:
        grouped.setdefault((row["source"], row["module"]), []).append(row["delta_error"])
    z = [[statistics.fmean(grouped[(source, name)]) for name in names] for source in source_order]
    custom = [[[len(grouped[(source, name)]), statistics.stdev(grouped[(source, name)])
                if len(grouped[(source, name)]) > 1 else 0.0]
               for name in names] for source in source_order]
    figure = go.Figure(go.Heatmap(
        z=z, customdata=custom, x=names, y=[labels[source] for source in source_order],
        colorscale="Cividis", zmin=0, zmax=max(max(row) for row in z), zauto=False,
        colorbar={"title": "Δ test error"},
        hovertemplate="copy %{y} into this layer<br>%{x}<br>all other layers: intact final model<br>mean Δ test error = %{z:.3f}<br>seed SD = %{customdata[1]:.3f}<br>n = %{customdata[0]}<extra></extra>",
    ))
    figure.update_layout(
        title={"text": "VGG-19 module robustness to checkpoint transplantation", "x": 0},
        height=620, template="plotly_white", margin={"l": 105, "r": 35, "t": 95, "b": 150},
    )
    figure.update_xaxes(title="VGG-19 parametric module · forward order", tickangle=-55)
    # Match Figure 3 of Zhang, Bengio, and Singer: re-randomization at the top,
    # then checkpoint sources in chronological order down to the intact model.
    figure.update_yaxes(
        title=f"value copied into one layer · all others at epoch {final_epoch}",
        autorange="reversed",
    )
    return figure


def criticality_training_figure(payload: dict) -> go.Figure:
    rows = payload["training"]
    grouped = {epoch: [row for row in rows if row["epoch"] == epoch]
               for epoch in sorted({row["epoch"] for row in rows})}
    figure = go.Figure()
    if any("seed" in row for row in rows):
        for seed in sorted({row["seed"] for row in rows}):
            seed_rows = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["epoch"])
            figure.add_trace(go.Scatter(
                x=[row["epoch"] for row in seed_rows], y=[row["accuracy"] for row in seed_rows],
                mode="lines", line={"color": "rgba(0,114,178,.22)", "width": 1},
                name=f"seed {seed}", legendgroup="seeds", showlegend=False,
                hovertemplate=f"seed {seed}<br>epoch %{{x}}<br>accuracy = %{{y:.2%}}<extra></extra>",
            ))
    means, errors = [], []
    for epoch in grouped:
        values = [row["accuracy"] for row in grouped[epoch]]
        means.append(statistics.fmean(values))
        if len(values) > 1:
            # 95% Student-t multiplier for n=5; conservative normal fallback otherwise.
            multiplier = 2.776 if len(values) == 5 else 1.96
            errors.append(multiplier * statistics.stdev(values) / len(values) ** 0.5)
        else:
            errors.append(0.0)
    figure.add_trace(go.Scatter(
        x=list(grouped), y=means, error_y={"type": "data", "array": errors, "visible": len(rows) > len(grouped)},
        mode="lines+markers", name="seed mean", line={"color": "#0072B2", "width": 2.5}, marker={"size": 8},
        hovertemplate="epoch %{x}<br>mean test accuracy = %{y:.2%}<extra></extra>",
    ))
    figure.update_layout(
        title={"text": "VGG-19+BatchNorm learning context", "x": 0}, height=410,
        template="plotly_white", margin={"l": 75, "r": 25, "t": 85, "b": 65},
    )
    figure.update_xaxes(title="training epoch")
    figure.update_yaxes(title="CIFAR-10 test accuracy", tickformat=".0%", range=[0, 1])
    return figure


def render_spec(source: str) -> str:
    # Preserve the complete source in a readable audit surface without adding a
    # second markdown dependency to the artifact builder.
    return f'<pre class="spec-source">{html.escape(source)}</pre>'


def suffix_checkpoint_key(payload: dict) -> tuple[int, int]:
    config = payload["config"]
    batches = config.get("checkpoint_batches")
    return config["checkpoint_epoch"], -1 if batches is None else batches


def suffix_checkpoint_label(payload: dict) -> str:
    config = payload["config"]
    batches = config.get("checkpoint_batches")
    if batches is None:
        return f"epoch {config['checkpoint_epoch']}"
    total_batches = (config["train_size"] + config["batch_size"] - 1) // config["batch_size"]
    if batches == 0:
        return "random init · batch 0"
    return f"batch {batches} · {100 * batches / total_batches:.1f}%"


def suffix_statistics_figures(payloads: list[dict], id_suffix: str = "") -> tuple[str, str, str, str]:
    payloads = sorted(payloads, key=suffix_checkpoint_key)
    order = ["true", "gaussian", "mean"]
    labels = {"true": "true", "gaussian": "Gaussian", "mean": "mean-only"}
    titles = [suffix_checkpoint_label(item) for item in payloads]
    styles = {"mean": ("#D55E00", "dot"), "gaussian": ("#0072B2", "dash"), "true": ("#222222", "solid")}

    trajectories = make_subplots(rows=1, cols=len(payloads), subplot_titles=titles, shared_yaxes=True)
    for column, payload in enumerate(payloads, start=1):
        for train in order:
            rows = sorted(
                (row for row in payload["records"] if row["train_distribution"] == train and row["eval_distribution"] == "true"),
                key=lambda row: row["relax_epoch"],
            )
            color, dash = styles[train]
            trajectories.add_trace(go.Scatter(
                x=[row["relax_epoch"] for row in rows], y=[row["loss"] for row in rows],
                mode="lines+markers", name=labels[train], legendgroup=train,
                showlegend=column == len(payloads), line={"color": color, "dash": dash, "width": 2.3},
                hovertemplate=f"{suffix_checkpoint_label(payload)}<br>relax on {labels[train]}<br>u=%{{x}}<br>true CE=%{{y:.4f}}<extra></extra>",
            ), row=1, col=column)
    trajectories.update_layout(
        title={"text": "Held-out CIFAR-10 cross-entropy through the frozen true prefix", "x": 0},
        height=440, margin={"l": 65, "r": 30, "t": 85, "b": 70}, legend={"title": {"text": "relax on"}},
    )
    trajectories.update_xaxes(title_text="relaxation epoch (u)")
    trajectories.update_yaxes(title_text="CIFAR-10 test CE", row=1, col=1)

    accuracies = make_subplots(rows=1, cols=len(payloads), subplot_titles=titles, shared_yaxes=True)
    for column, payload in enumerate(payloads, start=1):
        for train in order:
            rows = sorted(
                (row for row in payload["records"] if row["train_distribution"] == train and row["eval_distribution"] == "true"),
                key=lambda row: row["relax_epoch"],
            )
            color, dash = styles[train]
            accuracies.add_trace(go.Scatter(
                x=[row["relax_epoch"] for row in rows], y=[100 * row["accuracy"] for row in rows],
                mode="lines+markers", name=labels[train], legendgroup=train,
                showlegend=column == len(payloads), line={"color": color, "dash": dash, "width": 2.3},
                hovertemplate=f"{suffix_checkpoint_label(payload)}<br>relax on {labels[train]}<br>u=%{{x}}<br>CIFAR accuracy=%{{y:.2f}}%<extra></extra>",
            ), row=1, col=column)
    accuracies.update_layout(
        title={"text": "Held-out CIFAR-10 accuracy through the frozen true prefix", "x": 0},
        height=440, margin={"l": 65, "r": 30, "t": 85, "b": 70}, legend={"title": {"text": "relax on"}},
    )
    accuracies.update_xaxes(title_text="relaxation epoch (u)")
    accuracies.update_yaxes(title_text="CIFAR-10 test accuracy (%)", row=1, col=1)

    matrices, all_values = [], []
    for payload in payloads:
        final_epoch = max(row["relax_epoch"] for row in payload["records"])
        final = {(row["train_distribution"], row["eval_distribution"]): row["loss"]
                 for row in payload["records"] if row["relax_epoch"] == final_epoch}
        matrix = [[final[(train, evaluation)] for train in order] for evaluation in order]
        matrices.append(matrix)
        all_values.extend(value for row in matrix for value in row)
    heatmap = make_subplots(rows=1, cols=len(payloads), subplot_titles=titles, shared_yaxes=True)
    for column, (payload, matrix) in enumerate(zip(payloads, matrices), start=1):
        heatmap.add_trace(go.Heatmap(
            z=matrix, x=[labels[name] for name in order], y=[labels[name] for name in order],
            colorscale="Cividis", zmin=min(all_values), zmax=max(all_values),
            text=[[f"{value:.3f}" for value in row] for row in matrix], texttemplate="%{text}",
            showscale=column == len(payloads), colorbar={"title": "test CE", "x": 1.02},
            hovertemplate=f"{suffix_checkpoint_label(payload)}<br>relax on %{{x}}<br>evaluate on %{{y}}<br>CE=%{{z:.4f}}<extra></extra>",
        ), row=1, col=column)
    heatmap.update_layout(
        title={"text": "Final cross-evaluation matrices · rows evaluate, columns relax · true row = CIFAR-10", "x": 0},
        height=430, margin={"l": 90, "r": 65, "t": 85, "b": 75},
    )
    heatmap.update_xaxes(title_text="relaxation distribution")
    heatmap.update_yaxes(autorange="reversed")
    heatmap.update_yaxes(title_text="evaluation distribution", row=1, col=1)

    diagnostic_rows = "".join(
        f"<tr><th>{html.escape(suffix_checkpoint_label(payload))}</th><td>{html.escape(name)}</td>"
        f"<td>{values['class_mean_relative_error']:.3f}</td><td>{values['class_covariance_relative_error']:.3f}</td>"
        f"<td>{payload['explained_variance_fraction']:.1%}</td></tr>"
        for payload in payloads for name, values in payload["moment_diagnostics"].items()
    )
    diagnostic_table = (
        f"<table><thead><tr><th>checkpoint</th><th>surrogate</th><th>class-mean relative error</th>"
        f"<th>class-covariance relative error</th><th>PCA coverage</th></tr></thead><tbody>{diagnostic_rows}</tbody></table>"
        f"<p>Representation shape: <code>{html.escape(str(payloads[0]['representation_shape']))}</code>.</p>"
    )
    return (
        plot_html(trajectories, f"plot-suffix-statistics-time{id_suffix}"),
        plot_html(accuracies, f"plot-suffix-statistics-accuracy{id_suffix}"),
        plot_html(heatmap, f"plot-suffix-statistics-matrix{id_suffix}"),
        diagnostic_table,
    )


def vgg_suffix_statistics_figure(payload: dict) -> go.Figure:
    """Show the C2/C3 bridge without mixing it into Part B's pilot."""
    labels = {"mean": "mean-only", "gaussian": "Gaussian", "true": "true"}
    colors = {"mean": "#D55E00", "gaussian": "#0072B2", "true": "#222222"}
    figure = make_subplots(rows=1, cols=2, subplot_titles=(
        "C2 · native VGG interface", "C3 · checkpoint-0 module transplant"), shared_yaxes=True)
    for column, condition in enumerate(("native", "reset0"), start=1):
        slices = sorted((item for item in payload["slices"] if item["condition"] == condition), key=lambda item: item["cut"])
        for train_distribution in ("mean", "gaussian", "true"):
            x, y, custom = [], [], []
            for item in slices:
                final_epoch = max(row["relax_epoch"] for row in item["records"])
                rows = [row for row in item["records"] if row["relax_epoch"] == final_epoch and row["eval_distribution"] == "true"]
                by_draw = {}
                for row in rows:
                    by_draw.setdefault(row["draw"], {})[row["train_distribution"]] = row["loss"]
                excess = [values[train_distribution] - values["true"] for values in by_draw.values()]
                x.append(item["module"])
                y.append(statistics.fmean(excess))
                custom.append([item["cut"], len(excess)])
            figure.add_trace(go.Scatter(
                x=x, y=y, customdata=custom, mode="lines+markers", name=labels[train_distribution],
                legendgroup=train_distribution, showlegend=column == 2,
                line={"color": colors[train_distribution], "width": 2.4}, marker={"size": 8},
                hovertemplate="cut %{customdata[0]} · %{x}<br>excess true CE=%{y:.4f}<br>draws=%{customdata[1]}<extra></extra>",
            ), row=1, col=column)
    figure.update_layout(title={"text": "VGG suffix dependence on activation statistics", "x": 0},
        template="plotly_white", height=460, margin={"l": 75, "r": 30, "t": 90, "b": 100},
        legend={"title": {"text": "relax on"}})
    figure.update_xaxes(title_text="frozen cut · forward order", tickangle=-25)
    figure.update_yaxes(title_text="excess CE on true activations vs true relaxation", zeroline=True, row=1, col=1)
    return figure


def build_report(
    results_path: Path, output_path: Path, spec_path: Path,
    criticality_path: Path | None = None,
    suffix_statistics_paths: list[Path] | None = None,
    vgg_suffix_statistics_path: Path | None = None,
) -> None:
    payload = json.loads(results_path.read_text())
    seeds = payload["config"].get("seeds", [payload["config"].get("seed", 0)])
    mockup = bool(payload["config"].get("fake_data"))
    pilot = not mockup and (payload["config"].get("train_size", 50000) < 50000 or payload["config"].get("epochs", 20) < 20)
    status = "MOCKUP / PIPELINE SMOKE TEST" if mockup else ("MEASURED CIFAR PILOT — ONE SEED" if pilot else f"MEASURED CIFAR RESULTS — {len(seeds)} SEEDS")
    watermark = '<div class="watermark">MOCKUP — synthetic pipeline check</div>' if mockup else ""
    # The residual stream is the primary scientific object. Older pilot
    # artifacts may also contain a plain-CNN control; keep it in provenance but
    # do not turn the report into an architecture-comparison dashboard.
    a_rows = [row for row in payload["experiment_a"] if row["architecture"] == "residual"]
    b_rows = [row for row in payload["experiment_b"] if row["architecture"] == "residual"]
    fixed_cuts = (min(row["cut"] for row in b_rows), max(row["cut"] for row in b_rows))
    figures = {
        "a": plot_html(experiment_a_figure(a_rows), "plot-a"),
        "a_reverse": plot_html(experiment_a_reverse_figure(a_rows), "plot-a-reverse"),
        "cifar_context": plot_html(cifar_context_figure(a_rows), "plot-cifar-context"),
        "regret": plot_html(tracking_figure(b_rows, "head_regret", "Held-out loss recovered by suffix refitting · $\\Delta L_{\\mathrm{resolve},\\ell}(t_i)$", "$\\Delta L_{\\mathrm{resolve},\\ell}(t_i)$ · test CE"), "plot-regret"),
        "tracking": plot_html(tracking_figure(b_rows, "tracking_demand", "Cost of reusing the previous optimal suffix · $\\Delta L_{\\mathrm{track},\\ell}(t_i)$", "$\\Delta L_{\\mathrm{track},\\ell}(t_i)$ · test CE"), "plot-tracking"),
        "drift": plot_html(tracking_figure(b_rows, "relative_representation_drift", "Cut-layer representation movement · $\\Delta_{\\mathrm{rep},\\ell}(t_i)$", "$\\Delta_{\\mathrm{rep},\\ell}(t_i)$ · relative $L^2$ drift"), "plot-drift"),
        "demand_time": plot_html(time_comparison_figure(
            b_rows, fixed_cuts,
            (
                ("head_regret", "$\\Delta L_{\\mathrm{resolve}}$", "#9ECAE1", "solid"),
                ("tracking_demand", "$\\Delta L_{\\mathrm{track}}$", "#08519C", "dash"),
            ),
            "Resolving and tracking demands at fixed cuts · $\\Delta L_{k,\\ell}(t_i)$",
            "$\\Delta L(t_i)$ · test cross-entropy", "loss decomposition",
        ), "plot-demand-time"),
        "track_align": plot_html(tracking_figure(b_rows, "tracking_alignment", "Alignment with the moving optimum · $A_{\\mathrm{track},\\ell}(t_i)$", "$A_{\\mathrm{track},\\ell}(t_i)$ · cosine alignment"), "plot-track-align"),
        "resolve_align": plot_html(tracking_figure(b_rows, "resolving_alignment", "Alignment toward the previous optimum · $A_{\\mathrm{resolve},\\ell}(t_i)$", "$A_{\\mathrm{resolve},\\ell}(t_i)$ · cosine alignment"), "plot-resolve-align"),
        "alignment_time": plot_html(time_comparison_figure(
            b_rows, fixed_cuts,
            (
                ("tracking_alignment", "$A_{\\mathrm{track}}$", "#08519C", "solid"),
                ("resolving_alignment", "$A_{\\mathrm{resolve}}$", "#9ECAE1", "dash"),
            ),
            "Fixed-cut suffix-update alignments · $A_{k,\\ell}(t_i)$",
            "$A_{k,\\ell}(t_i)$ · cosine alignment", "update-direction alignment",
        ), "plot-alignment-time"),
        # Retained only because the legacy Part B template is assembled before
        # being replaced by the redesigned section below. It is not shown.
        "fisher": plot_html(fisher_figure(b_rows), "plot-fisher"),
    }
    suffix_statistics = [json.loads(path.read_text()) for path in (suffix_statistics_paths or []) if path.exists()]
    criticality = json.loads(criticality_path.read_text()) if criticality_path and criticality_path.exists() else None
    vgg_suffix_statistics = (json.loads(vgg_suffix_statistics_path.read_text())
        if vgg_suffix_statistics_path and vgg_suffix_statistics_path.exists() else None)
    criticality_section = ""
    criticality_provenance = ""
    if criticality:
        figures["criticality"] = plot_html(criticality_heatmap(criticality), "plot-criticality")
        figures["criticality_training"] = plot_html(criticality_training_figure(criticality), "plot-criticality-training")
        baselines = criticality["baselines"] if "baselines" in criticality else [criticality["baseline"]]
        baseline_accuracy = statistics.fmean(row["accuracy"] for row in baselines)
        seed_count = len(baselines)
        reset0: dict[str, list[float]] = {}
        for row in criticality["interventions"]:
            if row["source"] == "0":
                reset0.setdefault(row["module"], []).append(row["delta_error"])
        early = [statistics.fmean(reset0[name]) for name in criticality["module_names"][:9]]
        robust = [statistics.fmean(reset0[name]) for name in criticality["module_names"][9:16]]
        final_epoch = max(criticality["config"]["checkpoint_epochs"])
        uncertainty = (f"Heatmap cells are means across {seed_count} independent seeds; "
                       "hover reports the across-seed standard deviation. Training-curve bars are two-sided 95% Student-t intervals."
                       if seed_count > 1 else "This gate artifact contains one independent seed.")
        criticality_section = rf"""<div class="section"><div class="status">MEASURED · C1 · {seed_count} SEED{'S' if seed_count != 1 else ''}</div><h2>B0 · A critical band appears in VGG-19</h2><p>Following Zhang, Bengio, and Singer, every heatmap cell starts from the <b>final epoch-{final_epoch} model</b> and replaces exactly one module with that module's value from checkpoint $\tau$; every other module stays final. A convolution and its BatchNorm affine parameters and running state are treated as one atomic module. No fine-tuning occurs after transplantation.</p><div class="equation">$$\theta^T_m\leftarrow\theta^\tau_m,\qquad C_m^\tau=L(\theta^T_{{-m}},\theta^\tau_m)-L(\theta^T)$$</div><div class="guide"><b>Interpretation guide.</b> This is a compatibility probe against the final network, not a trajectory showing a checkpoint-$\tau$ model being reset at time $\tau$. Bright cells mean the final predictor is incompatible with that historical version of the module. Compatibility need not be monotone in $\tau$: checkpoint 0 can work while checkpoint 1 does not. The paper explicitly notes this pattern in normalization/weight-decay variants. The final-checkpoint row must be zero.</div>{figures['criticality']}<p><b>Measured result:</b> the intact epoch-{final_epoch} models average {baseline_accuracy:.2%} test accuracy. Resetting modules through <code>stage4.conv1</code> to epoch 0 increases test error by {min(early):.1%}–{max(early):.1%} on average; resetting <code>stage4.conv2</code> through most of stage 5 changes error by {min(robust):.1%}–{max(robust):.1%}. This is the sharp critical/robust transition used to locate C2/C3.</p><p><b>Uncertainty:</b> {uncertainty}</p><p><b>Caveat:</b> the headline VGG experiment in the paper is normalization-free, whereas this positive control uses BatchNorm and treats Conv+BN as atomic. The unusually destructive epoch-1–5 band may therefore reflect early Conv/BN co-adaptation as well as convolutional feature compatibility. Criticality establishes dependence on a learned module, not dependence on higher-order activation statistics.</p>{figures['criticality_training']}</div>"""
        device = criticality.get("device", "multiple / see source artifacts")
        seeds_label = criticality["config"].get("seeds", [criticality["config"].get("seed")])
        criticality_provenance = f"""<h2>VGG critical-module gate</h2><p>Artifact: {html.escape(str(criticality_path))}<br>Device: {html.escape(str(device))}<br>Seeds: {html.escape(str(seeds_label))}<br>Runtime: {criticality['runtime_seconds']:.1f} aggregate seconds<br>Architecture: full-width VGG-19 + BatchNorm; Conv+BN transplanted atomically<br>Intervention records: {len(criticality['interventions'])}</p>"""
    bridge_section = ""
    if vgg_suffix_statistics:
        bridge_plot = plot_html(vgg_suffix_statistics_figure(vgg_suffix_statistics), "plot-vgg-suffix-statistics")
        coverages = [item["explained_variance_fraction"] for item in vgg_suffix_statistics["slices"]]
        draw_count = vgg_suffix_statistics["config"]["surrogate_draws"]
        validation_passed = min(coverages) >= 0.8 and draw_count >= 3
        bridge_status = ("MEASURED · VALIDATION PASSED" if validation_passed else
                         "PRELIMINARY MEASUREMENT · VALIDATION GATE FAILED")
        validation_note = (f"PCA coverage spans {min(coverages):.1%}–{max(coverages):.1%}; "
                           f"the artifact contains {draw_count} surrogate draw{'s' if draw_count != 1 else ''}. "
                           "Do not interpret Gaussian/true gaps until every displayed cut clears 80% coverage and at least three draws are present.")
        bridge_section = f'''<div class="section"><div class="status">{bridge_status}</div><h2>C2–C3 · Activation statistics across the critical boundary</h2><p>C2 repeats Part B's true/mean/Gaussian suffix-relaxation contrast at four VGG cuts bracketing the measured critical-module boundary. C3 repeats the identical analysis after transplanting that cut's Conv+BN module from checkpoint 0. The plotted quantity is excess held-out true-activation cross-entropy relative to a suffix relaxed on true activations; zero therefore means the surrogate was sufficient under this finite relaxation protocol.</p><div class="guide"><b>Validation:</b> {validation_note}</div><div class="guide"><b>Interpretation guide.</b> Read the left facet as statistical sufficiency at the native learned interface. Read the right facet as the bridge to module criticality. A reset-specific Gaussian gap would connect critical damage to information beyond class-conditional first and second moments; module damage alone does not establish that claim.</div>{bridge_plot}<p><b>Controls queued, not yet implemented:</b> C4 selective fresh re-randomization; C5 matched perturbation diagnostics for activation mean, covariance, norm, prediction KL, and representation distance.</p></div>'''
        criticality_provenance += f'''<h2>VGG suffix-statistics bridge</h2><p>Artifact: {html.escape(str(vgg_suffix_statistics_path))}<br>Status: {bridge_status}<br>Device: {html.escape(str(vgg_suffix_statistics['device']))}<br>Runtime: {vgg_suffix_statistics['runtime_seconds']:.1f} seconds</p>'''
    elif criticality:
        bridge_section = '''<div class="section"><div class="status">C2–C3 · RUNNING / RESULT NOT YET LOADED</div><h2>C2–C3 · Activation statistics across the critical boundary</h2><p>The runner is prepared to compare true, class-conditional Gaussian, and mean-only activation relaxation at VGG cuts 7, 8, 9, and 13, first at the native interface (C2), then after checkpoint-0 Conv+BN transplantation (C3). This dashboard contains no C2/C3 measurement yet.</p><div class="guide"><b>Decision rule, fixed before results.</b> A native Gaussian gap means first and second class-conditional moments are insufficient under the finite relaxation protocol. A gap that appears specifically after reset0 would connect that insufficiency to the critical-module intervention. Immediate module damage alone is not evidence for either claim.</div><p><b>Controls TODO:</b> C4 selective fresh re-randomization; C5 matched activation mean/covariance/norm, representation-distance, and prediction-KL diagnostics.</p></div>'''
    config_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in payload["config"].items())
    mathjax = Path("assets/mathjax-tex-svg.js").read_text()
    if mockup:
        current_answer = "The plots validate data flow only; they do not test the hypothesis."
    elif pilot:
        current_answer = (
            "This one-seed, three-epoch residual-network pilot validates the measurements but is too small for a mechanistic conclusion. "
            "At the last checkpoint, tracking loss exceeds resolving loss at every cut, while both are small and do not vary monotonically with depth. "
            "The actual suffix update aligns more strongly with movement of the refitted optimum than with the pre-existing resolving direction at every cut. "
            "True-data training has the best accuracy, while covariance-surrogate training has slightly lower cross-entropy, so the three-epoch distributional result is not yet clean."
        )
    else:
        final_epoch = max(row["epoch"] for row in a_rows)
        final_true = {
            distribution: estimate([
                row for row in a_rows
                if row["epoch"] == final_epoch and row["test_distribution"] == "true"
                and row["train_distribution"] == distribution
            ], "loss")[0]
            for distribution in ("mean", "covariance", "true")
        }
        final_tracking = [row for row in b_rows if row["epoch"] == max(row["epoch"] for row in b_rows)]
        cut_means = {
            cut: {
                metric: estimate([row for row in final_tracking if row["cut"] == cut], metric)[0]
                for metric in ("head_regret", "tracking_demand", "tracking_alignment", "resolving_alignment")
            }
            for cut in sorted({row["cut"] for row in final_tracking})
        }
        positive_tracking = sum(value["tracking_demand"] > value["head_regret"] for value in cut_means.values())
        tracking_positive_with_ci = 0
        alignment_difference_with_ci = 0
        for cut in cut_means:
            cut_rows = [row for row in final_tracking if row["cut"] == cut]
            demand_mean, demand_ci, _ = estimate(cut_rows, "tracking_demand")
            tracking_positive_with_ci += demand_ci is not None and demand_mean - demand_ci > 0
            difference_rows = [{"difference": row["tracking_alignment"] - row["resolving_alignment"]} for row in cut_rows]
            difference_mean, difference_ci, _ = estimate(difference_rows, "difference")
            alignment_difference_with_ci += difference_ci is not None and difference_mean - difference_ci > 0
        current_answer = (
            f"At epoch {final_epoch}, true-CIFAR loss is {final_true['true']:.2f} after true-data training, "
            f"versus {final_true['covariance']:.2f} after covariance-surrogate training and {final_true['mean']:.2f} after mean-only training. "
            f"The layerwise evidence is mixed: point-estimate tracking demand exceeds resolving gain at "
            f"{positive_tracking}/{len(cut_means)} cuts, but positive tracking demand excludes zero at only "
            f"{tracking_positive_with_ci}/{len(cut_means)} cuts. The actual suffix update aligns significantly more with optimum motion than with the old resolving direction at "
            f"{alignment_difference_with_ci}/{len(cut_means)} cuts. Error bars are two-sided 95% t intervals across {len(seeds)} seeds."
        )
    if suffix_statistics:
        coarse_suffix = [item for item in suffix_statistics if item["config"].get("checkpoint_batches") is None]
        zoom_suffix = [item for item in suffix_statistics if item["config"].get("checkpoint_batches") is not None]
        display_suffix = coarse_suffix or suffix_statistics
        suffix_time, suffix_accuracy, suffix_matrix, suffix_diagnostics = suffix_statistics_figures(display_suffix)
        zoom_result = ""
        if zoom_suffix:
            zoom_time, zoom_accuracy, zoom_matrix, zoom_diagnostics = suffix_statistics_figures(zoom_suffix, "-epoch0-zoom")
            zoom_excess = []
            for item in sorted(zoom_suffix, key=suffix_checkpoint_key):
                final_relax_epoch = max(row["relax_epoch"] for row in item["records"])
                final_true_rows = {
                    row["train_distribution"]: row["loss"] for row in item["records"]
                    if row["relax_epoch"] == final_relax_epoch and row["eval_distribution"] == "true"
                }
                zoom_excess.append(final_true_rows["gaussian"] - final_true_rows["true"])
            zoom_result = (
                "<h2>Zoom · motion within epoch 0</h2>"
                "<p>These four facets share axes and protocol. Batch 0 is the exact random initialization; "
                "the remaining checkpoints follow the same deterministic first-epoch minibatch order. "
                "All four zoom slices ran on CPU; compare their motion internally, while treating the separately rendered "
                "epoch 0/1/5 GPU triptych as a distinct device realization.</p>"
                f"{zoom_time}{zoom_accuracy}{zoom_matrix}<h2>Within-epoch surrogate validation</h2>{zoom_diagnostics}"
                f"<p><b>Within-epoch takeaway.</b> Final Gaussian excess true-activation CE is "
                f"{zoom_excess[0]:.3f} at initialization, {zoom_excess[1]:.3f} after five batches, "
                f"{zoom_excess[2]:.3f} at 10%, and {zoom_excess[3]:.3f} at 50%. The gap is present throughout "
                "but non-monotone, revealing motion that the epoch-only slices cannot resolve. These are one-seed "
                "measurements, not uncertainty estimates.</p>"
            )
        suffix_is_mock = any(item["status"].startswith("MOCKUP") for item in suffix_statistics)
        primary_suffix = max(display_suffix, key=suffix_checkpoint_key)
        result_heading = "MOCKUP · pipeline smoke result" if suffix_is_mock else "Measured pilot result"
        result_watermark = '<div class="guide"><b>MOCKUP — fake-data pipeline check, not scientific evidence.</b></div>' if suffix_is_mock else ""
        final_relax = max(row["relax_epoch"] for row in primary_suffix["records"])
        true_final = {
            row["train_distribution"]: row for row in primary_suffix["records"]
            if row["relax_epoch"] == final_relax and row["eval_distribution"] == "true"
        }
        epoch_zero = next((item for item in suffix_statistics if item["config"]["checkpoint_epoch"] == 0), None)
        epoch_zero_note = ""
        if epoch_zero and not suffix_is_mock:
            mean_true = sorted(
                (row for row in epoch_zero["records"] if row["train_distribution"] == "mean" and row["eval_distribution"] == "true"),
                key=lambda row: row["relax_epoch"],
            )
            if len(mean_true) > 1:
                epoch_zero_note = (
                    f"<div class=\"guide\"><b>Why epoch-0 mean-only CE rises.</b> From $u=0$ to $u=1$, "
                    f"CIFAR accuracy rises from {mean_true[0]['accuracy']:.1%} to {mean_true[1]['accuracy']:.1%}, "
                    f"even as CE rises from {mean_true[0]['loss']:.3f} to {mean_true[1]['loss']:.3f}. "
                    "The suffix learns useful mean-based decisions but becomes overconfident on its remaining CIFAR mistakes. "
                    "Also, one plotted relaxation epoch is a full pass over 50,000 surrogate activations, not one optimizer step.</div>"
                )
        measured_takeaway = "" if suffix_is_mock else (
            f"<p><b>Epoch {primary_suffix['config']['checkpoint_epoch']} takeaway.</b> After {final_relax} relaxation epochs, true-activation CE is "
            f"{true_final['true']['loss']:.3f} for true relaxation, {true_final['gaussian']['loss']:.3f} "
            f"for Gaussian relaxation, and {true_final['mean']['loss']:.3f} for mean-only relaxation. "
            f"The Gaussian suffix therefore has excess true-activation loss "
            f"{true_final['gaussian']['loss'] - true_final['true']['loss']:.3f}; mean-only excess is "
            f"{true_final['mean']['loss'] - true_final['true']['loss']:.3f}. This does not support Gaussian sufficiency under the pilot protocol. "
            "However, the conclusion is conditional on the PCA truncation and the held-out covariance mismatch reported below.</p>"
        )
        b_result = (
            f"<h2>{result_heading}</h2>{result_watermark}{suffix_time}{suffix_accuracy}{epoch_zero_note}{suffix_matrix}{measured_takeaway}{zoom_result}"
            f"<h2>Surrogate validation</h2>{suffix_diagnostics}"
            f"<p><b>Epistemic status:</b> {html.escape(', '.join(sorted({item['status'] for item in suffix_statistics})))}. "
            "These checkpoint slices use one seed; sufficiency requires repeat surrogate draws and independent seeds.</p>"
        )
        current_answer = (
            "The redesigned epoch-5, cut-3 suffix-statistics pilot is loaded below. "
            if suffix_is_mock else
            f"At epoch {primary_suffix['config']['checkpoint_epoch']}, cut 3, Gaussian relaxation does not preserve true-activation performance: final excess CE is "
            f"{true_final['gaussian']['loss'] - true_final['true']['loss']:.3f} versus "
            f"{true_final['mean']['loss'] - true_final['true']['loss']:.3f} for mean-only. "
            "This weakens the Gaussian-sufficiency hypothesis, but the activation Gaussian matches only the declared PCA proxy and has substantial held-out covariance error."
        )
    else:
        b_result = (
            '<div class="guide"><b>PLANNED — no result artifact loaded.</b> '
            'If the Gaussian-relaxed suffix nearly matches the true-relaxed suffix on true held-out activations, '
            'while the mean-only suffix remains worse, that supports approximate second-order sufficiency at this slice. '
            'If Gaussian also remains worse, higher-order or out-of-subspace structure is needed.</div>'
        )
        current_answer = (
            "Part B has been respecified, but the existing artifacts do not contain checkpoint weights or cut-layer activations, "
            "so there is not yet a measured answer for the epoch-5, cut-3 suffix-statistics pilot."
        )
    redesigned_b = rf"""<section id="b" class="panel"><div class="section">
<div class="status">REDESIGNED PILOT · EPOCHS 0, 1, 5 · CUT 3</div>
<h2>B · Which statistics of $\phi_{{t,3}}(x)$ can the suffix use?</h2>
<p>At each checkpoint $t\in\{{0,1,5\}}$, freeze the prefix at cut 3. Fit class-conditional mean-only and Gaussian maximum-entropy proxies to its activations. Warm-start three identical suffixes from that checkpoint, relax each on one activation distribution, then cross-evaluate on all three.</p>
<h2>What exactly is the activation surrogate?</h2>
<p>Flatten the cut-3 activation $h=\phi_{{t,3}}(x)\in\mathbb R^D$ and project it into a $k=256$ dimensional PCA subspace:</p>
<div class="equation">$$z=U(h-\bar h),\qquad U\in\mathbb R^{{k\times D}}.$$</div>
<p>The distributions are fitted <b>separately for each CIFAR class</b> $c$. The Gaussian condition retains the full joint covariance in PCA coordinates,</p>
<div class="equation">$$z\mid y=c\sim\mathcal N(\mu_c,\Sigma_c),\qquad \widetilde h=U^\top z+\bar h.$$</div>
<p>Thus it is not componentwise-independent in the native activation tensor: it is a correlated Gaussian in the PCA subspace, inducing a low-rank correlated Gaussian after reconstruction. The mean-only condition keeps only the class-dependent location and uses shared isotropic nuisance:</p>
<div class="equation">$$z\mid y=c=\mu_c+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_{{\mathrm{{pool}}}}^2I_k).$$</div>
<p><b>Where:</b> $\bar h$ is the global activation mean; $U$ contains the fitted PCA directions; $\mu_c$ and $\Sigma_c$ are the class-conditional PCA mean and shrunk covariance; and $\sigma_{{\mathrm{{pool}}}}^2$ is the pooled within-class variance.</p>
<div class="guide"><b>Scope caveat.</b> Directions outside the PCA subspace are reconstructed at the global mean rather than sampled. Also, real cut-3 activations follow a ReLU and are nonnegative, whereas the Gaussian reconstruction is not support-constrained and can be negative. Consequently these plots test sufficiency of the fitted PCA-Gaussian proxy, not unrestricted sufficiency of first and second activation moments.</div>
<div class="equation">$$M_{{s,r}}^{{(t,3)}}(u)=\mathbb E_{{(z,y)\sim Q_s}}\ell(\psi_r^{{(u)}}(z),y),\qquad r,s\in\{{\mathrm{{true}},\mathrm{{Gaussian}},\mathrm{{mean}}\}}.$$</div>
<p><b>Decisive contrast.</b> Read the true-activation row and compare each surrogate-trained suffix with the true-trained suffix. The full matrix checks whether an apparent win is merely caused by an easier surrogate evaluation distribution.</p>
{b_result}</div>
<div class="section"><h2>Secondary diagnostic · suffix update directions</h2>
<p>This retains the useful part of the earlier moving-optimum analysis: whether the suffix update follows movement of the refitted optimum or closes the old resolving gap. It is supporting evidence, not the answer to the activation-statistics question.</p>
<div class="equation">$$A_{{k,\ell}}(t_i)=\frac{{\langle\Delta b_i,q_{{k,i}}\rangle}}{{\|\Delta b_i\|\,\|q_{{k,i}}\|}},\qquad k\in\{{\mathrm{{track}},\mathrm{{resolve}}\}}.$$</div>
{figures['track_align']}{figures['resolve_align']}{figures['alignment_time']}</div>
<div class="section"><h2>Archived supporting views</h2><p>The earlier resolving/tracking loss decomposition remains available for audit, but no longer drives the Part B claim.</p>{figures['regret']}{figures['tracking']}{figures['demand_time']}</div></section>"""
    document = rf"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracking statistical structure — {status}</title><script>{get_plotlyjs()}</script>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}}}};</script><script>{mathjax}</script>
<style>
:root{{--ink:#272522;--muted:#6d6963;--paper:#fbfaf7;--line:#ddd8ce;--accent:#0072b2}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 28px 80px}}h1{{font:700 2.25rem/1.1 Georgia,serif;margin:.2rem 0}}h2{{font:700 1.55rem Georgia,serif;margin-top:0}}.status{{letter-spacing:.09em;font-size:.76rem;font-weight:800;color:#a43d2f}}.hero{{display:grid;grid-template-columns:1.4fr 1fr;gap:28px;align-items:start}}.card,.section{{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:22px 0}}.equation{{border-left:4px solid var(--accent);padding:12px 18px;background:#f3f8fb;margin:18px 0}}nav{{position:sticky;top:0;z-index:4;background:rgba(251,250,247,.95);border-bottom:1px solid var(--line);padding:10px calc((100% - 1120px)/2)}}nav button{{border:0;background:none;padding:9px 15px;font-weight:700;color:var(--muted);cursor:pointer}}nav button.active{{color:var(--accent);border-bottom:2px solid var(--accent)}}.panel{{display:none}}.panel.active{{display:block}}.guide{{background:#fff8e8;border-left:4px solid #e69f00;padding:10px 14px;margin:14px 0}}.watermark{{position:fixed;z-index:20;right:2rem;bottom:1.5rem;transform:rotate(-8deg);font-size:1.35rem;font-weight:800;color:rgba(170,45,35,.28);pointer-events:none}}.spec-source{{white-space:pre-wrap;font:14px/1.55 ui-monospace,monospace}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:7px;vertical-align:top}}th{{width:28%}}@media(max-width:760px){{.hero{{grid-template-columns:1fr}}main{{padding:24px 14px}}nav{{padding:8px}}}}
</style></head><body>{watermark}<nav id="tabs"><button data-tab="overview" class="active">Overview</button><button data-tab="a">A · Surrogates</button><button data-tab="b">B · Tracking</button><button data-tab="spec">Spec</button><button data-tab="provenance">Provenance</button></nav><main>
<section id="overview" class="panel active"><div class="hero"><div><div class="status">{status}</div><h1>Tracking statistical structure through a residual stream</h1><p>At each cut of a residual network, how much downstream learning improves the current representation, and how much merely tracks a representation that is moving upstream?</p><div class="equation">$$\Delta L_{{\mathrm{{resolve}},\ell}}(t)=L(\phi_t,\psi_t)-L(\phi_t,\psi^*[\phi_t])$$</div><p><b>Current answer:</b> {current_answer}</p></div><div class="card"><b>Experiment map</b><p><b>A.</b> Train the residual CNN on class-mean, class-covariance, and true distributions; cross-evaluate all.</p><p><b>B.</b> Warm-refit each suffix at fixed residual-stream cuts, measure resolving and tracking terms, then estimate suffix-compensated prefix Fisher sensitivity.</p></div></div>
<div class="section"><h2>Mathematical formulation</h2><p>Let $P=P_R$ denote the true data distribution and let $P_r$ be an order-$r$ surrogate retaining a prescribed family of class-conditional statistics through order $r$. Training on $P_r$ gives parameters $\theta_r(t)$. The two original cross-distribution questions are</p><div class="equation">$$\begin{{aligned}}L_{{\mathrm{{true}}\mid r}}(t)&=\mathbb E_{{(x,y)\sim P_R}}\,\ell(f_{{\theta_r(t)}}(x),y),\\[2pt]L_{{r\mid\mathrm{{true}}}}(t)&=\mathbb E_{{(x,y)\sim P_r}}\,\ell(f_{{\theta_R(t)}}(x),y).
\end{{aligned}}$$</div><p>The first asks how far training on statistics through order $r$ transfers to the true distribution. The second asks when a model trained on true data becomes able to solve each lower-complexity surrogate. In this pilot, $r$ indexes the class-mean proxy, the class-conditional Gaussian covariance surrogate, and true CIFAR rather than literal finite cumulant truncations.</p><p>At layer cut $\ell$, split the network into an upstream representation and downstream computation,</p><div class="equation">$$f_{{a,b}}(x)=\psi_b^\ell\!\left(\phi_a^\ell(x)\right),\qquad b^*(a)\in\arg\min_b L(a,b).$$</div><p>As the upstream parameters $a(t)$ learn, the optimal suffix $b^*(a(t))$ moves. With suffix error $e(t)=b(t)-b^*(a(t))$, a local quadratic approximation gives</p><div class="equation">$$\begin{{aligned}}\dot e&\approx-\eta_bH_{{bb}}e-J_*(a)\dot a,\\[2pt]J_*(a)=\frac{{db^*}}{{da}}&=-H_{{bb}}^\dagger H_{{ba}}.
\end{{aligned}}$$</div><p>The term $-\eta_bH_{{bb}}e$ is <b>resolution</b>: downstream learning toward the optimum for the representation currently present. The term $-J_*(a)\dot a$ is <b>tracking forcing</b>: downstream work induced because upstream learning moved that optimum. Experiment B approximates these roles with finite checkpoint refits and cross-checkpoint head transfers, then asks how their balance changes across residual-stream depth.</p></div>
<div class="section"><h2>Decisive contrasts</h2><p><b>Distributional progression:</b> later improvement on true data after covariance-only training suggests structure beyond class means/covariances becomes useful. <b>Tracking with depth:</b> tracking demand larger than suffix refit gain suggests downstream computation is spending more of its local adaptation budget following a moving interface than resolving the current one. <b>Compensability:</b> a larger Fisher Schur fraction means more prefix sensitivity can be locally absorbed downstream.</p></div></section>
<section id="a" class="panel"><div class="section"><h2>A · Which retained statistics support true-distribution performance?</h2><div class="equation">$$L_{{s\mid r}}(t)=\mathbb E_{{(x,y)\sim P_s}}\!\left[-\log p_{{\theta_r(t)}}(y\mid x)\right]$$</div><p><b>Plotted object.</b> The residual-network parameters $\theta_r(t)$ are trained on distribution $P_r$ and evaluated on $P_s$. The first slice fixes $s=\mathrm{{true}}$ and varies the training distribution; the second fixes $r=\mathrm{{true}}$ and varies the evaluation distribution. Lower is better.</p><div class="guide"><b>Interpretation guide.</b> Early improvement after covariance-surrogate training supports early use of second-order structure. A later separation favoring true-data training supports subsequent use of unmatched structure. Earlier reduction of $L_{{r\mid\mathrm{{true}}}}$ for simpler surrogates would be the complementary “progressive resolution” signature.</div>{figures['a']}{figures['a_reverse']}<p><b>Caveat:</b> “mean” is a class-mean plus isotropic-nuisance proxy; “covariance” is Gaussian matching in the fitted PCA space. Neither deletes literal pixel cumulants.</p></div></section>
<section id="b" class="panel">{criticality_section}<div class="section"><h2>B · Resolving versus tracking at each cut</h2><p>At cut $\ell$, write $f_t=\psi_t^\ell\circ\phi_t^\ell$. The finite refit protocol defines $\widehat\psi_t^{{*,\ell}}$ by freezing $\phi_t^\ell$, warm-starting from $\psi_t^\ell$, and optimizing only the suffix.</p><div class="equation">$$\begin{{aligned}}\Delta L_{{\mathrm{{resolve}},\ell}}(t_i)&=L(\phi_{{t_i}}^\ell,\psi_{{t_i}}^\ell)-L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_i}}^{{*,\ell}}),\\[2pt]\Delta L_{{\mathrm{{track}},\ell}}(t_i)&=L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_{{i-1}}}}^{{*,\ell}})-L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_i}}^{{*,\ell}}),\\[2pt]\Delta_{{\mathrm{{rep}},\ell}}(t_i)&=\frac{{\|\phi_{{t_i}}^\ell(X)-\phi_{{t_{{i-1}}}}^\ell(X)\|_2}}{{\|\phi_{{t_{{i-1}}}}^\ell(X)\|_2}}.
\end{{aligned}}$$</div><p><b>Plotted objects.</b> $\Delta L_{{\mathrm{{resolve}},\ell}}$ is signed held-out loss recovered by suffix refitting; $\Delta L_{{\mathrm{{track}},\ell}}$ is the penalty from using the previous checkpoint’s refitted suffix on the current representation; $\Delta_{{\mathrm{{rep}},\ell}}$ is relative representation movement. The ideal resolving gap is nonnegative, but its finite held-out estimate can be negative.</p><div class="guide"><b>Interpretation guide.</b> A larger positive resolving loss means more performance remains available at fixed representation. A larger tracking loss means more downstream adaptation is required solely because the representation moved.</div><h2>Training-loss context</h2><p>This is the ordinary true-CIFAR test loss of the same model at every checkpoint where B was measured.</p>{figures['cifar_context']}<p><b>Depth profiles.</b> These show only $t_i\in\{{1,5,20,30\}}$ to make the cross-layer comparison legible; color runs from light blue (early) to dark blue (late).</p>{figures['regret']}{figures['tracking']}{figures['drift']}<h2>Fixed-cut trajectories</h2><p>This complementary view fixes the first and last residual-stream cuts and compares resolving with tracking over all measured checkpoints.</p>{figures['demand_time']}</div><div class="section"><h2>Update-direction decomposition</h2><div class="equation">$$\begin{{aligned}}\Delta b_i&=b_i-b_{{i-1}},\\q_{{\mathrm{{track}},i}}&=\widehat b_i^*-\widehat b_{{i-1}}^*,\\q_{{\mathrm{{resolve}},i}}&=\widehat b_{{i-1}}^*-b_{{i-1}},\\A_{{k,\ell}}(t_i)&=\frac{{\langle\Delta b_i,q_{{k,i}}\rangle}}{{\|\Delta b_i\|\,\|q_{{k,i}}\|}},\qquad k\in\{{\mathrm{{track}},\mathrm{{resolve}}\}}.
\end{{aligned}}$$</div><p><b>Notation in plain language.</b> $b_i$ is the vector of all suffix weights actually present at checkpoint $t_i$; $\widehat b_i^*$ is the suffix obtained by freezing the representation at that checkpoint and refitting the suffix. Thus $\Delta b_i$ is the update the network really made. $A_{{\mathrm{{track}},\ell}}$ is simply the cosine of the angle between that real update and the direction in which the refitted optimum moved. It is $+1$ for perfect tracking, $0$ for no directional relation, and $-1$ for movement in the opposite direction.</p><p><b>What the dot products mean.</b> Tracking alignment asks whether the actual suffix update follows movement of the refitted optimum. Resolving alignment asks whether it points toward the optimum that existed at the start of the interval. These directions are generally non-orthogonal, so the artifact also records their joint two-vector regression coefficients and explained fraction. The current plots use the Euclidean parameter metric; a Fisher-metric dot product is a useful robustness check.</p>{figures['track_align']}{figures['resolve_align']}<h2>Fixed-cut alignment trajectories</h2>{figures['alignment_time']}</div><div class="section"><h2>Empirical-Fisher compensation</h2><div class="equation">$$\begin{{aligned}}F&=\begin{{pmatrix}}F_{{aa}}&F_{{ab}}\\F_{{ba}}&F_{{bb}}\end{{pmatrix}},\\[2pt]F_{{\mathrm{{eff}},a}}&=F_{{aa}}-F_{{ab}}(F_{{bb}}+\gamma I)^{{-1}}F_{{ba}},\\[2pt]C_\ell&=1-\frac{{\operatorname{{tr}}F_{{\mathrm{{eff}},a}}}}{{\operatorname{{tr}}F_{{aa}}}}.
\end{{aligned}}$$</div><p><b>Block notation.</b> Here $a$ means every parameter before cut $\ell$, while $b$ means every parameter after the cut—the same suffix-weight vector denoted $b_i$ at checkpoint $t_i$. $F_{{aa}}$ measures prefix sensitivity with the suffix held fixed; the Schur term subtracts sensitivity that a local suffix adjustment can absorb. $C_\ell$ is the resulting absorbable fraction.</p><p><b>Why the ridge is present.</b> With only $m=32$ per-example gradients, $F_{{bb}}$ is low-rank and cannot generally be inverted. The ridge $\gamma I$, with $\gamma=0.1\,\operatorname{{tr}}(K_b)/m$ and $K_b=G_bG_b^\top$, makes the finite-sample inverse stable. It is not mathematically essential: one could use $F_{{bb}}^\dagger$ instead, but that zero-ridge pseudoinverse is much more sensitive to poorly estimated small singular directions. Consequently $C_\ell$ should be read as a ridge-dependent diagnostic, and a $\gamma$ sensitivity sweep is the appropriate robustness check.</p><div class="guide"><b>Interpretation guide.</b> A larger compensable fraction means more prefix output sensitivity lies in directions that downstream parameters can locally absorb. The y-axis is zoomed to the observed range and does not begin at zero.</div>{figures['fisher']}</div></section>
<section id="spec" class="panel"><div class="section"><h2>Embedded design specification</h2>{render_spec(spec_path.read_text())}</div></section>
<section id="provenance" class="panel"><div class="section"><h2>Run provenance</h2><p>Artifact: {html.escape(str(results_path))}<br>Device: {html.escape(str(payload['device']))}<br>Independent seeds: {len(seeds)} ({html.escape(str(seeds))})<br>Aggregate GPU runtime: {payload['runtime_seconds']:.1f} seconds</p>{criticality_provenance}<table>{config_rows}</table><p>Points are seed means; error bars are two-sided 95% Student-t intervals across independent seeds. Plotly, MathJax, the measured data, and the specification are embedded for offline use.</p></div></section>
</main><script>document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab));window.dispatchEvent(new Event('resize'));if(window.MathJax?.typesetPromise) MathJax.typesetPromise();}});</script></body></html>"""
    if criticality_section or bridge_section:
        document = document.replace(
            '<button data-tab="b">B · Tracking</button>',
            '<button data-tab="b">B · Tracking</button><button data-tab="c">C · Critical modules</button>',
        )
        document = document.replace(
            f'<section id="b" class="panel">{criticality_section}',
            '<section id="b" class="panel">',
        )
        document = document.replace(
            '<section id="spec" class="panel">',
            f'<section id="c" class="panel">{criticality_section.replace("B0 ·", "C1 ·")}{bridge_section}</section>'
            '<section id="spec" class="panel">',
        )
    document = document.replace(
        "This is the ordinary true-CIFAR test loss of the same model at every checkpoint where B was measured.",
        "This is the ordinary true-CIFAR test loss of the same model at every checkpoint where B was measured. "
        "It requires no new run; a dense per-epoch curve between these checkpoints would require additional logged evaluations.",
    )
    b_start = document.index('<section id="b" class="panel">')
    b_end_marker = '<section id="c" class="panel">' if '<section id="c" class="panel">' in document else '<section id="spec" class="panel">'
    b_end = document.index(b_end_marker)
    document = document[:b_start] + redesigned_b + document[b_end:]
    document = document.replace('<button data-tab="b">B · Tracking</button>', '<button data-tab="b">B · Suffix statistics</button>')
    document = document.replace(
        "At each cut of a residual network, how much downstream learning improves the current representation, and how much merely tracks a representation that is moving upstream?",
        "At a frozen internal interface, which class-conditional statistics of the representation are sufficient for the downstream suffix?",
    )
    document = document.replace(
        r'<div class="equation">$$\Delta L_{\mathrm{resolve},\ell}(t)=L(\phi_t,\psi_t)-L(\phi_t,\psi^*[\phi_t])$$</div>',
        r'<div class="equation">$$\Delta_{\mathrm{true}\mid r}^{(t,\ell)}=M_{\mathrm{true},r}^{(t,\ell)}-M_{\mathrm{true},\mathrm{true}}^{(t,\ell)}$$</div>',
    )
    document = document.replace(
        "<b>B.</b> Warm-refit each suffix at fixed residual-stream cuts, measure resolving and tracking terms, then estimate suffix-compensated prefix Fisher sensitivity.",
        "<b>B.</b> Freeze the epoch-0, epoch-1, and epoch-5 prefixes at cut 3; relax matched suffixes on true, Gaussian, or mean-only activations; compare aligned triptychs. Update-direction dot products remain secondary; Fisher is on hold.",
    )
    formulation_start = document.index('<div class="section"><h2>Mathematical formulation</h2>')
    formulation_end = document.index('<div class="section"><h2>Decisive contrasts</h2>')
    formulation = r'''<div class="section"><h2>Mathematical formulation</h2>
<p>Part A asks which input statistics support performance. Part B pushes the same intervention through a frozen prefix and asks what the suffix can learn from the resulting internal distribution.</p>
<div class="equation">$$z=\phi_{t,\ell}(x),\qquad M_{s,r}^{(t,\ell)}(u)=\mathbb E_{(z,y)\sim Q_s^{(t,\ell)}}\ell(\psi_{r}^{(u)}(z),y).$$</div>
<p>$Q_r$ is the true, class-mean, or class-conditional Gaussian activation distribution. Every $\psi_r$ has the same checkpoint warm start; only its relaxation distribution changes. The true-activation row of the cross-matrix is the primary comparison.</p></div>
'''
    document = document[:formulation_start] + formulation + document[formulation_end:]
    old_contrasts = document.index('<div class="section"><h2>Decisive contrasts</h2>')
    old_contrasts_end = document.index('</div></section>', old_contrasts) + len('</div>')
    contrasts = '''<div class="section"><h2>Decisive contrasts</h2><p><b>Gaussian sufficiency:</b> Gaussian relaxation matches true relaxation on true held-out activations while mean-only does not. <b>Higher-order need:</b> Gaussian remains detectably worse than true. <b>Validation gate:</b> neither verdict is interpretable unless the intended held-out activation moments match and PCA coverage is reported. Update-direction dot products remain secondary; Fisher is on hold.</p></div>'''
    document = document[:old_contrasts] + contrasts + document[old_contrasts_end:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("report.html"))
    parser.add_argument("--spec", type=Path, default=Path("SPEC.md"))
    parser.add_argument("--criticality", type=Path)
    parser.add_argument("--suffix-statistics", type=Path, nargs="+")
    parser.add_argument("--vgg-suffix-statistics", type=Path)
    args = parser.parse_args()
    build_report(args.results, args.output, args.spec, args.criticality, args.suffix_statistics,
                 args.vgg_suffix_statistics)
    print(args.output)


if __name__ == "__main__":
    main()
