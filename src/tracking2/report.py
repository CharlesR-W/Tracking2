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


COLORS = {"plain": "#0072B2", "residual": "#D55E00", "mean": "#999999", "covariance": "#009E73", "true": "#CC79A7"}
DASHES = {"mean": "dot", "covariance": "dash", "true": "solid"}


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
    facet_titles = ["Residual stream" if architecture == "residual" else "Optional plain control" for architecture in architectures]
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
            "text": "True-CIFAR loss after training on each surrogate",
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
        title={"text": "When does true-CIFAR training solve each surrogate?", "x": 0.0},
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


def tracking_figure(rows: list[dict], metric: str, title: str, y_title: str) -> go.Figure:
    architectures = sorted({row["architecture"] for row in rows})
    facet_titles = ["Residual stream" if architecture == "residual" else "Optional plain control" for architecture in architectures]
    figure = make_subplots(rows=1, cols=len(architectures), shared_yaxes=True, subplot_titles=facet_titles)
    epochs = sorted({row["epoch"] for row in rows})
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
    palette = ["#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]
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


def fisher_figure(rows: list[dict]) -> go.Figure:
    figure = tracking_figure(
        rows,
        "compensable_fraction",
        "How much prefix sensitivity can the suffix compensate?",
        "$C_\\ell(t_i)$ · compensable fraction",
    )
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    return figure


def render_spec(source: str) -> str:
    # Preserve the complete source in a readable audit surface without adding a
    # second markdown dependency to the artifact builder.
    return f'<pre class="spec-source">{html.escape(source)}</pre>'


def build_report(results_path: Path, output_path: Path, spec_path: Path) -> None:
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
    figures = {
        "a": plot_html(experiment_a_figure(a_rows), "plot-a"),
        "a_reverse": plot_html(experiment_a_reverse_figure(a_rows), "plot-a-reverse"),
        "regret": plot_html(tracking_figure(b_rows, "head_regret", "How much held-out loss does suffix refitting recover?", "$\\Delta L_{\\mathrm{resolve},\\ell}(t_i)$ · test CE"), "plot-regret"),
        "tracking": plot_html(tracking_figure(b_rows, "tracking_demand", "How costly is it to reuse the previous optimal suffix?", "$\\Delta L_{\\mathrm{track},\\ell}(t_i)$ · test CE"), "plot-tracking"),
        "drift": plot_html(tracking_figure(b_rows, "relative_representation_drift", "How far does the residual-stream interface move?", "$\\Delta_{\\mathrm{rep},\\ell}(t_i)$ · relative $L^2$ drift"), "plot-drift"),
        "track_align": plot_html(tracking_figure(b_rows, "tracking_alignment", "Does the suffix update follow the moving optimum?", "$A_{\\mathrm{track},\\ell}(t_i)$ · cosine alignment"), "plot-track-align"),
        "resolve_align": plot_html(tracking_figure(b_rows, "resolving_alignment", "Does the suffix update close its existing optimum gap?", "$A_{\\mathrm{resolve},\\ell}(t_i)$ · cosine alignment"), "plot-resolve-align"),
        "fisher": plot_html(fisher_figure(b_rows), "plot-fisher"),
    }
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
<section id="b" class="panel"><div class="section"><h2>B · Resolving versus tracking at each cut</h2><p>At cut $\ell$, write $f_t=\psi_t^\ell\circ\phi_t^\ell$. The finite refit protocol defines $\widehat\psi_t^{{*,\ell}}$ by freezing $\phi_t^\ell$, warm-starting from $\psi_t^\ell$, and optimizing only the suffix.</p><div class="equation">$$\begin{{aligned}}\Delta L_{{\mathrm{{resolve}},\ell}}(t_i)&=L(\phi_{{t_i}}^\ell,\psi_{{t_i}}^\ell)-L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_i}}^{{*,\ell}}),\\[2pt]\Delta L_{{\mathrm{{track}},\ell}}(t_i)&=L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_{{i-1}}}}^{{*,\ell}})-L(\phi_{{t_i}}^\ell,\widehat\psi_{{t_i}}^{{*,\ell}}),\\[2pt]\Delta_{{\mathrm{{rep}},\ell}}(t_i)&=\frac{{\|\phi_{{t_i}}^\ell(X)-\phi_{{t_{{i-1}}}}^\ell(X)\|_2}}{{\|\phi_{{t_{{i-1}}}}^\ell(X)\|_2}}.
\end{{aligned}}$$</div><p><b>Plotted objects.</b> $\Delta L_{{\mathrm{{resolve}},\ell}}$ is signed held-out loss recovered by suffix refitting; $\Delta L_{{\mathrm{{track}},\ell}}$ is the penalty from using the previous checkpoint’s refitted suffix on the current representation; $\Delta_{{\mathrm{{rep}},\ell}}$ is relative representation movement. The ideal resolving gap is nonnegative, but its finite held-out estimate can be negative.</p><div class="guide"><b>Interpretation guide.</b> A larger positive resolving loss means more performance remains available at fixed representation. A larger tracking loss means more downstream adaptation is required solely because the representation moved.</div>{figures['regret']}{figures['tracking']}{figures['drift']}</div><div class="section"><h2>Update-direction decomposition</h2><div class="equation">$$\begin{{aligned}}\Delta b_i&=b_i-b_{{i-1}},\\q_{{\mathrm{{track}},i}}&=\widehat b_i^*-\widehat b_{{i-1}}^*,\\q_{{\mathrm{{resolve}},i}}&=\widehat b_{{i-1}}^*-b_{{i-1}},\\A_{{k,\ell}}(t_i)&=\frac{{\langle\Delta b_i,q_{{k,i}}\rangle}}{{\|\Delta b_i\|\,\|q_{{k,i}}\|}},\qquad k\in\{{\mathrm{{track}},\mathrm{{resolve}}\}}.
\end{{aligned}}$$</div><p><b>What the dot products mean.</b> Tracking alignment asks whether the actual suffix update follows movement of the refitted optimum. Resolving alignment asks whether it points toward the optimum that existed at the start of the interval. These directions are generally non-orthogonal, so the artifact also records their joint two-vector regression coefficients and explained fraction. The current plots use the Euclidean parameter metric; a Fisher-metric dot product is a useful robustness check.</p>{figures['track_align']}{figures['resolve_align']}</div><div class="section"><h2>Empirical-Fisher compensation</h2><div class="equation">$$\begin{{aligned}}F&=\begin{{pmatrix}}F_{{aa}}&F_{{ab}}\\F_{{ba}}&F_{{bb}}\end{{pmatrix}},\\[2pt]F_{{\mathrm{{eff}},a}}&=F_{{aa}}-F_{{ab}}(F_{{bb}}+\gamma I)^{{-1}}F_{{ba}},\\[2pt]C_\ell&=1-\frac{{\operatorname{{tr}}F_{{\mathrm{{eff}},a}}}}{{\operatorname{{tr}}F_{{aa}}}}.
\end{{aligned}}$$</div><p><b>Plotted object.</b> $a$ denotes parameters before cut $\ell$, $b$ the suffix parameters, and $C_\ell$ the fraction of sampled prefix Fisher sensitivity locally absorbable by the suffix. Here $F$ is the per-example empirical Fisher and $\gamma=0.1\,\operatorname{{tr}}(K_b)/m$, where $K_b=G_bG_b^\top$ is the suffix sample-gradient Gram matrix for $m$ examples.</p><div class="guide"><b>Interpretation guide.</b> A larger compensable fraction means more prefix output sensitivity lies in directions that downstream parameters can locally absorb. It is a ridge-dependent, low-rank Schur diagnostic, not a complete definition of plasticity.</div>{figures['fisher']}</div></section>
<section id="spec" class="panel"><div class="section"><h2>Embedded design specification</h2>{render_spec(spec_path.read_text())}</div></section>
<section id="provenance" class="panel"><div class="section"><h2>Run provenance</h2><p>Artifact: {html.escape(str(results_path))}<br>Device: {html.escape(str(payload['device']))}<br>Independent seeds: {len(seeds)} ({html.escape(str(seeds))})<br>Aggregate GPU runtime: {payload['runtime_seconds']:.1f} seconds</p><table>{config_rows}</table><p>Points are seed means; error bars are two-sided 95% Student-t intervals across independent seeds. Plotly, MathJax, the measured data, and the specification are embedded for offline use.</p></div></section>
</main><script>document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab));window.dispatchEvent(new Event('resize'));if(window.MathJax?.typesetPromise) MathJax.typesetPromise();}});</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("report.html"))
    parser.add_argument("--spec", type=Path, default=Path("SPEC.md"))
    args = parser.parse_args()
    build_report(args.results, args.output, args.spec)
    print(args.output)


if __name__ == "__main__":
    main()
