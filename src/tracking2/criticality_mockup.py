from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from .models import InstrumentedVGG19


def watermark() -> dict:
    return {
        "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.52,
        "text": "MOCKUP — SCHEMATIC, NOT RESULTS", "showarrow": False,
        "textangle": -18, "font": {"size": 30, "color": "rgba(130,45,45,.18)"},
    }


def heatmap(module_names: list[str]) -> go.Figure:
    rng = np.random.default_rng(12)
    sources = ["fresh random", "epoch 0", "epoch 1", "epoch 5", "epoch 20", "final"]
    depth = np.linspace(0, 1, len(module_names))
    critical_profile = 0.72 * np.exp(-2.7 * depth) + 0.08
    source_strength = np.array([1.0, 0.92, 0.78, 0.53, 0.22, 0.0])[:, None]
    z = np.clip(source_strength * critical_profile + rng.normal(0, 0.018, (len(sources), len(module_names))), 0, None)
    z[-1] = 0
    figure = go.Figure(go.Heatmap(
        z=z, x=module_names, y=sources, colorscale="Cividis", zmin=0, zmax=0.85,
        colorbar={"title": "Δ test error"},
        hovertemplate="%{y}<br>%{x}<br>schematic Δ error=%{z:.3f}<extra>MOCKUP</extra>",
    ))
    figure.update_layout(
        title="FIGURE B1 — MOCKUP · checkpoint transplantation reveals a critical band",
        height=560, template="plotly_white", margin={"l": 105, "r": 40, "t": 90, "b": 145},
        annotations=[watermark()],
    )
    figure.update_xaxes(title="VGG-19 parametric module · forward order", tickangle=-55)
    figure.update_yaxes(title="transplant source")
    return figure


def recovery(module_names: list[str]) -> go.Figure:
    steps = np.array([0, 10, 50, 100, 200])
    selections = [(module_names[0], 0.74, 35), (module_names[8], 0.42, 75), (module_names[12], 0.25, 150)]
    colors = ["#D55E00", "#0072B2", "#6d6963"]
    dashes = ["solid", "dash", "dot"]
    figure = go.Figure()
    for (name, initial, tau), color, dash in zip(selections, colors, dashes):
        remaining = initial * (0.18 + 0.82 * np.exp(-steps / tau))
        figure.add_trace(go.Scatter(
            x=steps, y=remaining, mode="lines+markers", name=name,
            line={"color": color, "dash": dash, "width": 2.5},
            hovertemplate=name + "<br>step %{x}<br>schematic remaining Δ error=%{y:.3f}<extra>MOCKUP</extra>",
        ))
    figure.update_layout(
        title="FIGURE B2 — MOCKUP · downstream-only recovery separates trackable criticality",
        height=500, template="plotly_white", margin={"l": 80, "r": 30, "t": 90, "b": 70},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22},
        annotations=[watermark()],
    )
    figure.update_xaxes(title="suffix refit updates")
    figure.update_yaxes(title="remaining transplant damage · Δ test error", rangemode="tozero")
    return figure


def build(output: Path) -> None:
    model = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625)
    names = [name for name, _ in model.intervention_modules()]
    figures = {
        "heat": to_html(heatmap(names), full_html=False, include_plotlyjs=False, div_id="mockup-heat", config={"displaylogo": False}),
        "recovery": to_html(recovery(names), full_html=False, include_plotlyjs=False, div_id="mockup-recovery", config={"displaylogo": False}),
    }
    payload = json.dumps({"mockup": True, "module_names": names})
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Panel B critical modules — MOCKUP</title><script>{get_plotlyjs()}</script>
<style>
:root{{--ink:#24221f;--muted:#6d6963;--paper:#fbfaf7;--accent:#9b3b2f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.7rem max(1rem,calc((100vw - 1180px)/2));z-index:3}}
button{{border:0;background:none;padding:.55rem .8rem;font-weight:650;cursor:pointer}} button.active{{color:var(--accent);border-bottom:2px solid var(--accent)}}
main{{max-width:1180px;margin:auto;padding:2.4rem 1.2rem 5rem}} .panel{{display:none}} .panel.active{{display:block}}
.status{{display:inline-block;color:#8d3026;border:1px solid #b45f55;padding:.25rem .55rem;border-radius:4px;font-weight:750;letter-spacing:.04em}}
h1{{font:650 2.35rem/1.1 Georgia,serif;margin:.8rem 0}} h2{{font:650 1.65rem/1.2 Georgia,serif;margin-top:2.2rem}}
.lede{{max-width:820px;font-size:1.12rem}} .guide{{border-left:4px solid #0072B2;background:#eef6fa;padding:.8rem 1rem;margin:1rem 0 1.3rem}}
.warning{{border:1px solid #d5aaa4;background:#fff5f3;padding:1rem;margin:1.2rem 0}} .equation{{padding:1rem;border:1px solid #ddd;background:white;font-family:serif;font-size:1.12rem}}
.plot{{background:white;border:1px solid #e5e0d8;margin:1rem 0 2rem}} code{{background:#eee9e1;padding:.12rem .3rem}}
@media(max-width:700px){{h1{{font-size:1.8rem}} main{{padding:1.4rem .7rem}}}}
</style></head><body><nav><button data-tab='overview' class='active'>Overview</button><button data-tab='panelb'>Panel B · MOCKUP figures</button></nav><main>
<section id='overview' class='panel active'><span class='status'>MOCKUP — NO EXPERIMENTAL DATA</span><h1>Which VGG-19 modules are critical, and which are trackable?</h1>
<p class='lede'>This isolated page is the proposed Panel B reading surface. All values are deterministic schematic placeholders. It does not modify or claim evidence from the live report.</p>
<div class='equation'>C<sub>m</sub><sup>τ</sup> = L(θ<sup>(m←τ)</sup>) − L(θ<sup>T</sup>)</div>
<div class='warning'><strong>Interpretation:</strong> large immediate damage identifies functional criticality. Rapid downstream-only recovery means the changed interface is trackable; persistent damage indicates an irreducibly critical learned map.</div>
<p>Planned artifact: <code>artifacts/criticality/results.json</code>. Fixed module order: 16 convolutions, two hidden classifiers, and the final linear classifier.</p></section>
<section id='panelb' class='panel'><span class='status'>ALL FIGURES BELOW ARE MOCKUPS</span><h1>Panel B figures — MOCKUP</h1>
<h2>FIGURE B1 — MOCKUP · criticality heatmap</h2><div class='guide'><strong>Interpretation guide:</strong> a bright vertical band under early resets means those modules are critical. Uniformly dark rows weaken the critical-module hypothesis. A zero final-checkpoint row is an implementation invariant.</div><div class='plot'>{figures['heat']}</div>
<h2>FIGURE B2 — MOCKUP · suffix recovery</h2><div class='guide'><strong>Interpretation guide:</strong> steep decay means downstream layers can track the transplanted interface. A high plateau means the module remains critical even after downstream adaptation. Compare curves by position and linestyle, not colour alone.</div><div class='plot'>{figures['recovery']}</div></section>
<script>const PAYLOAD={payload}; document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab));setTimeout(()=>window.dispatchEvent(new Event('resize')),20)}});</script>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("MOCKUP/criticality_panel_b_MOCKUP.html"))
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
