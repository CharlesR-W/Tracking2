"""Compact presentation-first mockup for the planned Part E experiment.

Part E is deliberately a direct transformer port of Part B: freeze a prefix,
fit three activation distributions, warm-start matched suffixes, and collect the
full relax-by-evaluate matrix. Transformer-specific machinery is kept in a
collapsed operationalization block rather than presented as another experiment.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


__all__ = ["part_e_dashboard_html", "part_e_mockup_html"]


def part_e_mockup_html() -> str:
    """Return the complete, self-contained Part E dashboard section."""

    return r"""
<section id="e" class="panel part-e-mockup" data-epistemic-status="MOCKUP-PLANNED-UNRUN">
<style>
.part-e-mockup {
  --pe-ink: #282521;
  --pe-muted: #6d675f;
  --pe-line: #ddd6ca;
  --pe-paper: #fffefb;
  --pe-blue: #0072b2;
  --pe-green: #009e73;
  --pe-orange: #d55e00;
  --pe-status: #983e35;
  color: var(--pe-ink);
}
.part-e-mockup * { box-sizing: border-box; }
.part-e-mockup .pe-hero,
.part-e-mockup .pe-panel {
  position: relative;
  overflow: hidden;
  margin: 22px 0;
  padding: 22px;
  border: 1px solid var(--pe-line);
  border-radius: 12px;
  background: var(--pe-paper);
}
.part-e-mockup .pe-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(280px, .8fr);
  gap: 26px;
  border-top: 5px solid var(--pe-status);
}
.part-e-mockup .pe-eyebrow,
.part-e-mockup .pe-status {
  color: var(--pe-status);
  font-size: .75rem;
  font-weight: 850;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.part-e-mockup .pe-status {
  display: inline-block;
  padding: .28rem .62rem;
  border: 1px solid #d9aaa5;
  border-radius: 999px;
  background: #fff5f3;
}
.part-e-mockup h1,
.part-e-mockup h2,
.part-e-mockup h3 { color: var(--pe-ink); font-family: Georgia, serif; }
.part-e-mockup h1 { margin: .45rem 0 .8rem; font-size: 1.95rem; line-height: 1.14; }
.part-e-mockup h2 { margin: .15rem 0 .7rem; font-size: 1.5rem; line-height: 1.22; }
.part-e-mockup h3 { margin: 0 0 .45rem; font-size: 1.05rem; line-height: 1.28; }
.part-e-mockup p { max-width: 94ch; }
.part-e-mockup .pe-lede { font-size: 1.06rem; line-height: 1.62; }
.part-e-mockup .pe-muted { color: var(--pe-muted); }
.part-e-mockup .pe-watermark {
  position: absolute;
  top: 1.25rem;
  right: .55rem;
  transform: rotate(12deg);
  color: rgba(152, 62, 53, .15);
  font-size: .82rem;
  font-weight: 900;
  letter-spacing: .07em;
  pointer-events: none;
  text-transform: uppercase;
}
.part-e-mockup .pe-grid-2,
.part-e-mockup .pe-grid-3 {
  display: grid;
  gap: 14px;
  margin: 16px 0;
}
.part-e-mockup .pe-grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.part-e-mockup .pe-grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.part-e-mockup .pe-card {
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--pe-line);
  border-top: 4px solid var(--pe-blue);
  border-radius: 9px;
  background: #fcfcfa;
}
.part-e-mockup .pe-card.pe-green { border-top-color: var(--pe-green); }
.part-e-mockup .pe-card.pe-orange { border-top-color: var(--pe-orange); }
.part-e-mockup .pe-guide,
.part-e-mockup .pe-warning,
.part-e-mockup .pe-equation,
.part-e-mockup .pe-decision {
  margin: 14px 0;
  padding: .85rem 1rem;
  border-left: 4px solid var(--pe-blue);
  background: #eef7fb;
}
.part-e-mockup .pe-warning { border-left-color: #e69f00; background: #fff8e8; }
.part-e-mockup .pe-decision { border-left-color: var(--pe-green); background: #eef8f4; }
.part-e-mockup .pe-equation {
  overflow-x: auto;
  border: 1px solid #cadde7;
  border-left: 4px solid var(--pe-blue);
  background: #f4f9fc;
  text-align: center;
}
.part-e-mockup .pe-map {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
  margin: 18px 0;
}
.part-e-mockup .pe-map > div:not(.pe-arrow) {
  padding: 14px;
  border: 1px solid var(--pe-line);
  border-radius: 9px;
  background: #fcfcfa;
}
.part-e-mockup .pe-map strong { display: block; margin-bottom: .35rem; }
.part-e-mockup .pe-arrow { align-self: center; color: var(--pe-muted); font-size: 1.35rem; }
.part-e-mockup table { width: 100%; margin: 14px 0; border-collapse: collapse; }
.part-e-mockup th,
.part-e-mockup td {
  padding: 9px;
  border-bottom: 1px solid var(--pe-line);
  text-align: left;
  vertical-align: top;
}
.part-e-mockup th { color: #4a453f; font-size: .88rem; }
.part-e-mockup .pe-table-scroll { overflow-x: auto; }
.part-e-mockup .pe-matrix { min-width: 650px; table-layout: fixed; }
.part-e-mockup .pe-matrix th,
.part-e-mockup .pe-matrix td { text-align: center; }
.part-e-mockup .pe-matrix thead th:first-child,
.part-e-mockup .pe-matrix tbody th { text-align: left; }
.part-e-mockup .pe-matrix .pe-diagonal { background: #eef7f2; }
.part-e-mockup .pe-matrix .pe-decisive { background: #fff1df; font-weight: 750; }
.part-e-mockup details {
  margin-top: 16px;
  border: 1px solid var(--pe-line);
  border-radius: 9px;
  background: #fbfaf7;
}
.part-e-mockup summary { padding: 13px 15px; cursor: pointer; font-weight: 800; }
.part-e-mockup details > div { padding: 0 15px 13px; }
.part-e-mockup code { padding: .1rem .3rem; border-radius: 4px; background: #f0ede7; }
.part-e-mockup .pe-no-data {
  margin: .5rem 0 0;
  color: var(--pe-status);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .04em;
  text-transform: uppercase;
}
@media (max-width: 860px) {
  .part-e-mockup .pe-hero,
  .part-e-mockup .pe-grid-2,
  .part-e-mockup .pe-grid-3 { grid-template-columns: 1fr; }
  .part-e-mockup .pe-map { grid-template-columns: 1fr; }
  .part-e-mockup .pe-arrow { transform: rotate(90deg); justify-self: center; }
  .part-e-mockup h1 { font-size: 1.7rem; }
}
</style>

  <div class="pe-hero" id="part-e-hero">
    <span class="pe-watermark" data-mockup-watermark="true">MOCKUP · PLANNED · UNRUN</span>
    <div>
      <div class="pe-eyebrow">Part E · direct transformer replication · mockup / planned / unrun</div>
      <h1>E · Does Part B's suffix-statistics result survive next-token prediction? — MOCKUP</h1>
      <p class="pe-lede"><strong>Same experiment, new interface.</strong> Freeze a GPT-2 prefix, cache its residual-stream sequences, fit mean-only and Gaussian proxies, relax matched suffix copies, and cross-evaluate them against held-out true activations.</p>
      <div class="pe-warning"><strong>MOCKUP · PLANNED · UNRUN.</strong> This tab defines the experiment and contains no measured, simulated, or illustrative outcome values.</div>
    </div>
    <aside>
      <span class="pe-status">No transformer run yet</span>
      <p><strong>Backbone:</strong> GPT-2 small, context length 256, trained on deterministic TinyStories manifests.</p>
      <p><strong>Primary sweep:</strong> scratch-training checkpoints; residual-stream cuts after blocks 0, 5, and 11.</p>
      <p class="pe-muted"><strong>Primary endpoint:</strong> held-out next-token cross-entropy in nats per loss-bearing token. Uncertainty resamples stories, not tokens.</p>
    </aside>
  </div>

  <div class="pe-panel" id="part-e-primary-protocol">
    <span class="pe-watermark" data-mockup-watermark="true">MOCKUP · PLANNED · UNRUN</span>
    <div class="pe-eyebrow">Primary question · the Part B intervention, unchanged in spirit</div>
    <h2>Freeze once, relax three matched suffixes, evaluate all nine pairs</h2>

    <div class="pe-map" aria-label="Part E suffix relaxation protocol">
      <div><strong>1 · Freeze and cache</strong>At checkpoint $t$ and cut $\ell$, cache complete causal sequences $H=\phi_{t,\ell}(x)$ with masks, positions, and next-token targets.</div>
      <div class="pe-arrow" aria-hidden="true">→</div>
      <div><strong>2 · Fit three $Q_r$</strong>Use the same held-out-safe fit bank to define mean-only, sequence-Gaussian, and empirical true activation distributions.</div>
      <div class="pe-arrow" aria-hidden="true">→</div>
      <div><strong>3 · Relax and cross-evaluate</strong>Warm-start identical copies of $\psi_{t,\ell}$, change only the relaxation distribution, then evaluate every copy on every $Q_s$.</div>
    </div>

    <div class="pe-equation">$$M^{(t,\ell)}_{s,r}(u)=\mathbb E_{(H,Y)\sim Q_s}\!\left[\ell_{\mathrm{NTP}}\!\left(\psi^{(u)}_{t,\ell;r}(H),Y\right)\right],\qquad r,s\in\{\mathrm{mean},\mathrm{seqG},\mathrm{true}\}.$$</div>

    <div class="pe-grid-3">
      <article class="pe-card">
        <h3>$Q_{\mathrm{mean}}$ · conditional mean</h3>
        <p>Next-token-group and position-conditioned mean plus calibrated isotropic nuisance in the fitted subspace. This is Part B's mean-only rung.</p>
      </article>
      <article class="pe-card pe-green">
        <h3>$Q_{\mathrm{seqG}}$ · sequence Gaussian</h3>
        <p>The same mean plus fitted channel covariance and cross-position covariance. This is Part B's Gaussian rung adapted to an ordered sequence.</p>
      </article>
      <article class="pe-card pe-orange">
        <h3>$Q_{\mathrm{true}}$ · cached activations</h3>
        <p>Complete measured residual-stream sequences with their original masks and targets. This is the full-space reference.</p>
      </article>
    </div>

    <h3>Complete three-by-three relax-by-evaluate matrix</h3>
    <div class="pe-table-scroll">
      <table class="pe-matrix" aria-label="Planned three by three transformer suffix relaxation matrix">
        <thead><tr><th>evaluate ↓ / relax →</th><th>$Q_{\mathrm{mean}}$</th><th>$Q_{\mathrm{seqG}}$</th><th>$Q_{\mathrm{true}}$</th></tr></thead>
        <tbody>
          <tr><th>$Q_{\mathrm{mean}}$</th><td class="pe-diagonal" data-matrix-cell="planned">planned CE</td><td data-matrix-cell="planned">planned CE</td><td data-matrix-cell="planned">planned CE</td></tr>
          <tr><th>$Q_{\mathrm{seqG}}$</th><td data-matrix-cell="planned">planned CE</td><td class="pe-diagonal" data-matrix-cell="planned">planned CE</td><td data-matrix-cell="planned">planned CE</td></tr>
          <tr><th>$Q_{\mathrm{true}}$</th><td class="pe-decisive" data-matrix-cell="planned">planned true|mean</td><td class="pe-decisive" data-matrix-cell="planned">planned true|seqG</td><td class="pe-diagonal" data-matrix-cell="planned">planned true|true</td></tr>
        </tbody>
      </table>
    </div>

    <div class="pe-equation">$$\Delta_{\mathrm{true}\mid r}^{(t,\ell)}(u)=M_{\mathrm{true},r}^{(t,\ell)}(u)-M_{\mathrm{true},\mathrm{true}}^{(t,\ell)}(u).$$</div>
    <div class="pe-guide"><strong>Interpretation guide — planned, not observed.</strong> Near-zero sequence-Gaussian excess loss together with larger mean-only excess loss would support the bounded claim that modeled second-order sequence statistics suffice for this suffix, checkpoint, cut, and relaxation budget. A positive sequence-Gaussian gap would show that the fitted Gaussian proxy is missing useful structure; it would not by itself identify that structure as higher-order.</div>
    <p class="pe-no-data">Mockup / planned / unrun — every matrix entry is a label, not a value.</p>
  </div>

  <div class="pe-panel" id="part-e-ntp-adaptations">
    <span class="pe-watermark" data-mockup-watermark="true">MOCKUP · PLANNED · UNRUN</span>
    <div class="pe-eyebrow">NTP adaptation · only the machinery needed to make B valid for sequences</div>
    <h2>What changes for a transformer—and what does not</h2>
    <table aria-label="Mapping from Part B to Part E">
      <thead><tr><th>Part B object</th><th>Part E replacement</th><th>Why it is necessary</th></tr></thead>
      <tbody>
        <tr><td>Activation vector $z$</td><td>Complete residual-stream sequence $H\in\mathbb R^{T\times d}$ with attention mask and positions</td><td>The suffix's computation is causal and order-sensitive.</td></tr>
        <tr><td>Class label $y$</td><td>Next-token target $y_p=x_{p+1}$ at each loss-bearing position</td><td>The surrogate models the joint training distribution; it is not an inference-time generator.</td></tr>
        <tr><td>Class-conditional Gaussian</td><td>Bounded matrix-normal proxy with channel and cross-position covariance</td><td>IID token rows would delete the sequence structure under test.</td></tr>
        <tr><td>Example-level held-out accuracy</td><td>Cross-entropy in nats/token, with intervals from story-level resampling</td><td>Tokens within one story are dependent.</td></tr>
        <tr><td>Independent suffix copy</td><td>Suffix blocks + final LayerNorm + cloned, untied LM head</td><td>Updating a tied head must not alter the frozen input embedding.</td></tr>
      </tbody>
    </table>

    <div class="pe-equation">$$Z\mid Y\sim\mathcal{MN}\!\left(M(Y),K_{\mathrm{pos}},\Sigma_{\mathrm{chan}}\right),\qquad M_p(Y)=\bar z+a_{g(y_p)}+b_{\operatorname{posbin}(p)}.$$</div>
    <p>The target-conditioned mean is the direct analogue of Part B's class-conditional fit. Rare tokens use preregistered groups. A history-only mean is retained as an ablation, and changing future targets must leave all earlier generated rows unchanged.</p>

    <div class="pe-decision"><strong>Small first gate:</strong> one scratch-training seed, selected early/middle/final checkpoints, cuts 0/5/11, one surrogate draw, and 64 suffix updates. Confirm with three model seeds and three surrogate draws only if replay parity, surrogate fit, and a preregistered true-row contrast all pass.</div>

    <details>
      <summary>Operational checks and audit artifacts</summary>
      <div>
        <ul>
          <li><strong>Replay parity:</strong> the untouched cached suffix reproduces the intact model's logits and cross-entropy at update 0.</li>
          <li><strong>Immutable banks:</strong> surrogate fitting, suffix relaxation, and held-out evaluation use disjoint, hashed story sets with fixed masks and tokenization.</li>
          <li><strong>PCA guard:</strong> report held-out variance coverage and the projected-true versus full-true gap. A material gap blocks a full-space sufficiency claim.</li>
          <li><strong>Surrogate validation:</strong> check conditional means, channel covariance, lag covariance, support, and the future-target permutation guard.</li>
          <li><strong>Matched relaxation:</strong> identical warm start, labels, sample count, minibatch order, optimizer, learning rate, update count, and evaluation banks.</li>
        </ul>
        <p>Persist <code>manifest.json</code>, <code>replay_manifest.json</code>, <code>sequence_surrogates.npz</code>, <code>surrogate_diagnostics.parquet</code>, and <code>suffix_statistics.json</code>. These are implementation requirements, not additional scientific panels.</p>
      </div>
    </details>

    <details>
      <summary>Useful diagnostics that stay secondary</summary>
      <div>
        <p>An IID-token Gaussian can test whether modeled cross-position covariance matters. A projected-true replay can diagnose PCA truncation. Neither needs to enlarge the headline $3\times3$ matrix. Finite resolving/tracking quantities remain secondary Part B diagnostics, and Part D retains ownership of moment velocity, Fisher/GGN, reset, and bandwidth experiments.</p>
      </div>
    </details>

    <details>
      <summary>Deferred extensions</summary>
      <div>
        <p>Instruction continuation, response-only SFT, LoRA, and preference/RL branches are separate adaptation studies. They should reuse this validated suffix-statistics protocol later, not be prerequisites for answering Part E's first question.</p>
      </div>
    </details>
    <p class="pe-no-data">Mockup / planned / unrun — the first deliverable is the bounded scratch-training replication.</p>
  </div>
</section>
""".strip()


def _load_measured(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "MEASURED" or payload.get("schema_version", 0) < 2:
        raise ValueError(f"Part E dashboard requires a measured schema-v2 artifact: {path}")
    return payload


def _part_e_payload(matched: dict, equal_lr: dict) -> dict:
    distributions = ["mean", "token_gaussian", "seq_gaussian", "projected_true", "true"]
    labels = {
        "mean": "mean only",
        "token_gaussian": "token-IID Gaussian",
        "seq_gaussian": "sequence Gaussian",
        "projected_true": "projected true",
        "true": "full true",
    }

    def result_rows(payload: dict) -> dict[str, dict]:
        rows = payload["records"]
        updates = sorted({int(row["update"]) for row in rows})
        cuts = sorted({int(row["cut"]) for row in rows})
        curves: dict[str, dict] = {}
        matrices: dict[str, list[list[float]]] = {}
        endpoints: dict[str, list[float]] = {name: [] for name in distributions[:-1]}
        for cut in cuts:
            cut_rows = [row for row in rows if int(row["cut"]) == cut]
            curve = {"updates": updates}
            for train in distributions:
                values = []
                for update in updates:
                    cell = next(
                        row for row in cut_rows
                        if row["train_distribution"] == train
                        and row["eval_distribution"] == "true"
                        and int(row["update"]) == update
                    )
                    baseline = next(
                        row for row in cut_rows
                        if row["train_distribution"] == "true"
                        and row["eval_distribution"] == "true"
                        and int(row["update"]) == update
                    )
                    values.append(float(cell["loss_nats_per_token"] - baseline["loss_nats_per_token"]))
                curve[train] = values
                if train != "true":
                    endpoints[train].append(values[-1])
            curves[str(cut)] = curve

            final_update = updates[-1]
            matrix = []
            for evaluate in distributions:
                baseline = next(
                    row for row in cut_rows
                    if row["train_distribution"] == "true"
                    and row["eval_distribution"] == evaluate
                    and int(row["update"]) == final_update
                )
                matrix.append([
                    float(next(
                        row for row in cut_rows
                        if row["train_distribution"] == train
                        and row["eval_distribution"] == evaluate
                        and int(row["update"]) == final_update
                    )["loss_nats_per_token"] - baseline["loss_nats_per_token"])
                    for train in distributions
                ])
            matrices[str(cut)] = matrix
        return {
            "cuts": cuts, "updates": updates, "curves": curves,
            "matrices": matrices, "endpoints": endpoints,
        }

    diagnostics = sorted(matched["diagnostics"], key=lambda row: int(row["cut"]))
    return {
        "mockup": False,
        "labels": labels,
        "distributions": distributions,
        "matched": result_rows(matched),
        "equal": result_rows(equal_lr),
        "diagnostics": [{
            "cut": int(row["cut"]),
            "components": int(row["pca_components"]),
            "coverage": float(row["pca_coverage"]),
            "projected_gap": float(row["projected_true_excess_loss"]),
            "parity": float(row["parity"]["max_abs_logit_error"]),
            "gradient_norms": {
                key: float(value) for key, value in row["initial_gradient_norms"].items()
            },
            "lr_scales": {
                key: float(value) for key, value in row["learning_rate_scales"].items()
            },
        } for row in diagnostics],
        "config": matched["config"],
        "runtime_seconds": float(matched["runtime_seconds"] + equal_lr["runtime_seconds"]),
    }


def part_e_dashboard_html(
    matched_path: Path | None = None,
    equal_lr_path: Path | None = None,
) -> str:
    """Return measured Part E results, falling back to the planned mockup."""

    if matched_path is None or equal_lr_path is None:
        return part_e_mockup_html()
    if not matched_path.exists() or not equal_lr_path.exists():
        return part_e_mockup_html()

    matched = _load_measured(matched_path)
    equal_lr = _load_measured(equal_lr_path)
    payload = _part_e_payload(matched, equal_lr)
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    matched_label = html.escape(str(matched_path))
    equal_label = html.escape(str(equal_lr_path))

    return rf"""
<section id="e" class="panel part-e-measured" data-epistemic-status="MEASURED-DIAGNOSTIC-ONE-SEED">
<style>
.part-e-measured {{
  --pe-ink:#282521; --pe-muted:#6d675f; --pe-line:#ddd6ca; --pe-paper:#fffefb;
  --pe-blue:#0072b2; --pe-sky:#56b4e9; --pe-green:#009e73; --pe-orange:#d55e00;
  --pe-purple:#8c67a5; --pe-warn:#a55a00; color:var(--pe-ink);
}}
.part-e-measured * {{ box-sizing:border-box; }}
.part-e-measured .pe-hero,.part-e-measured .pe-section {{
  margin:22px 0; padding:22px; border:1px solid var(--pe-line);
  border-radius:12px; background:var(--pe-paper);
}}
.part-e-measured .pe-hero {{
  display:grid; grid-template-columns:minmax(0,1.45fr) minmax(270px,.75fr);
  gap:24px; border-top:5px solid var(--pe-orange);
}}
.part-e-measured h1,.part-e-measured h2,.part-e-measured h3 {{
  color:var(--pe-ink); font-family:Georgia,serif;
}}
.part-e-measured h1 {{ margin:.4rem 0 .75rem; font-size:1.95rem; line-height:1.14; }}
.part-e-measured h2 {{ margin:.1rem 0 .65rem; font-size:1.48rem; }}
.part-e-measured h3 {{ margin:.1rem 0 .35rem; font-size:1.05rem; }}
.part-e-measured .pe-eyebrow {{
  color:var(--pe-orange); font-size:.75rem; font-weight:850; letter-spacing:.09em;
  text-transform:uppercase;
}}
.part-e-measured .pe-lede {{ font-size:1.08rem; line-height:1.62; }}
.part-e-measured .pe-status {{
  display:inline-block; padding:.3rem .65rem; border:1px solid #e0b177;
  border-radius:999px; color:#814607; background:#fff8ea; font-size:.76rem;
  font-weight:850; letter-spacing:.06em; text-transform:uppercase;
}}
.part-e-measured .pe-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:13px; margin:16px 0; }}
.part-e-measured .pe-grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:13px; margin:16px 0; }}
.part-e-measured .pe-card {{ padding:14px; border:1px solid var(--pe-line); border-top:4px solid var(--pe-blue); border-radius:9px; background:#fcfcfa; }}
.part-e-measured .pe-card:nth-child(2) {{ border-top-color:var(--pe-green); }}
.part-e-measured .pe-card:nth-child(3) {{ border-top-color:var(--pe-orange); }}
.part-e-measured .pe-number {{ display:block; margin:.25rem 0; font:700 1.35rem Georgia,serif; }}
.part-e-measured .pe-muted {{ color:var(--pe-muted); }}
.part-e-measured .pe-guide,.part-e-measured .pe-takeaway,.part-e-measured .pe-warning,.part-e-measured .pe-equation {{
  margin:14px 0; padding:.85rem 1rem; border-left:4px solid var(--pe-blue); background:#eef7fb;
}}
.part-e-measured .pe-takeaway {{ border-left-color:var(--pe-green); background:#eef8f4; }}
.part-e-measured .pe-warning {{ border-left-color:#e69f00; background:#fff8e8; }}
.part-e-measured .pe-equation {{ overflow-x:auto; border:1px solid #cadde7; border-left:4px solid var(--pe-blue); background:#f4f9fc; text-align:center; }}
.part-e-measured .pe-controls {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin:12px 0; }}
.part-e-measured .pe-controls label {{ font-weight:750; }}
.part-e-measured select {{ margin-left:.35rem; padding:.35rem .5rem; border:1px solid var(--pe-line); border-radius:6px; background:white; }}
.part-e-measured .pe-plot {{ width:100%; min-height:430px; }}
.part-e-measured table {{ width:100%; border-collapse:collapse; margin:12px 0; }}
.part-e-measured th,.part-e-measured td {{ padding:9px; border-bottom:1px solid var(--pe-line); text-align:left; }}
.part-e-measured th {{ color:#4a453f; }}
.part-e-measured details {{ margin-top:14px; border:1px solid var(--pe-line); border-radius:9px; background:#fbfaf7; }}
.part-e-measured summary {{ padding:13px 15px; cursor:pointer; font-weight:800; }}
.part-e-measured details>div {{ padding:0 15px 14px; overflow-x:auto; }}
.part-e-measured .pe-flow {{
  display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr);
  gap:10px; align-items:stretch; margin:18px 0;
}}
.part-e-measured .pe-flow>div:not(.pe-arrow) {{
  padding:14px; border:1px solid var(--pe-line); border-radius:9px; background:#fcfcfa;
}}
.part-e-measured .pe-flow strong {{ display:block; margin-bottom:.3rem; }}
.part-e-measured .pe-arrow {{ align-self:center; color:var(--pe-muted); font-size:1.35rem; }}
.part-e-measured .pe-where {{ margin-top:-.4rem; color:var(--pe-muted); font-size:.92rem; }}
.part-e-measured .pe-surrogate th:first-child {{ width:18%; }}
.part-e-measured .pe-surrogate th:nth-child(2) {{ width:34%; }}
@media(max-width:860px) {{
  .part-e-measured .pe-hero,.part-e-measured .pe-grid,.part-e-measured .pe-grid-2,.part-e-measured .pe-flow {{ grid-template-columns:1fr; }}
  .part-e-measured .pe-arrow {{ transform:rotate(90deg); justify-self:center; }}
  .part-e-measured h1 {{ font-size:1.68rem; }}
  .part-e-measured .pe-plot {{ min-height:390px; }}
}}
</style>

<div class="pe-hero">
  <div>
    <div class="pe-eyebrow">Part E · internal sequence experiment · GPT-2 small</div>
    <h1>E · What structure in a residual-stream sequence does the rest of a transformer need?</h1>
    <p class="pe-lede">A transformer passes a <strong>sequence of hidden vectors</strong> from one block
    to the next. We freeze that sequence at an internal cut, replace its distribution with controlled
    surrogates, and ask what a fresh copy of the downstream blocks can still learn.</p>
    <div class="pe-takeaway"><strong>Current answer:</strong> preserving channel and cross-token covariance
    is not enough to reproduce learning on real hidden sequences at the early and middle cuts. It gives a
    modest improvement over a mean-only surrogate only at the final cut.</div>
  </div>
  <aside>
    <span class="pe-status">Measured diagnostic · one seed</span>
    <p><strong>Model:</strong> 12-block GPT-2 small, trained from scratch for 50M TinyStories tokens.</p>
    <p><strong>Where measured:</strong> residual streams after blocks 0, 5, and 11.</p>
    <p><strong>Readout:</strong> held-out next-token cross-entropy after 32 downstream-only updates.</p>
    <p class="pe-muted">One model seed and one surrogate draw: enough to diagnose the mechanism,
    not enough for a population-level claim.</p>
  </aside>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">1 · From language modeling to a frozen internal interface</div>
  <h2>A token sequence becomes a sequence of residual-stream vectors</h2>
  <p>Let $X=(x_1,\ldots,x_T)$ be a tokenized story. At cut $\ell$, the embedding and
  transformer blocks through $\ell$ form a <strong>prefix</strong> $\phi_\ell$; the remaining blocks,
  final LayerNorm, and language-model head form a <strong>suffix</strong> $\psi_\ell$:</p>
  <div class="pe-equation">$$
    H_\ell=\phi_\ell(X)\in\mathbb R^{{T\times d}},\qquad
    O=\psi_\ell(H_\ell)\in\mathbb R^{{T\times |\mathcal V|}}.
  $$</div>
  <p class="pe-where">$H_{{\ell,p}}$ is the hidden vector at position $p$; $O_p$ contains logits
  for the next token. Causality means $H_{{\ell,p}}$ may depend on $x_1,\ldots,x_p$, but not on
  later tokens.</p>
  <p>The supervised target at position $p$ is $Y_p=x_{{p+1}}$. The suffix is scored by average
  next-token cross-entropy:</p>
  <div class="pe-equation">$$
    \mathcal L_{{\mathrm{{NTP}}}}(\psi;H,Y)
    =-\frac1{{T-1}}\sum_{{p=1}}^{{T-1}}
      \log\operatorname{{softmax}}\!\left(\psi(H)_p\right)_{{Y_p}}.
  $$</div>
  <div class="pe-flow" aria-label="Frozen-interface sequence protocol">
    <div><strong>Real stories</strong>Tokenize 256-token TinyStories sequences and retain each
    next-token target.</div><div class="pe-arrow">→</div>
    <div><strong>Frozen prefix</strong>Run through block $\ell$ once and cache the complete
    causal hidden sequence $H_\ell$.</div><div class="pe-arrow">→</div>
    <div><strong>Trainable suffix copy</strong>Clone blocks $\ell+1\!:\!11$, final LayerNorm, and
    an untied output head; update only this copy.</div>
  </div>
  <div class="pe-guide"><strong>Why freeze the interface?</strong> Every suffix copy begins with
  identical weights and sees the same targets. Only the distribution of hidden sequences changes,
  so differences in downstream learning can be attributed to information preserved by that distribution.</div>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">2 · A hierarchy of synthetic hidden-sequence distributions</div>
  <h2>Each surrogate preserves one more kind of structure</h2>
  <p>The raw residual stream has width $d=768$. At each cut we fit a local orthonormal PCA basis
  $U\in\mathbb R^{{d\times k}}$ and work with retained coordinates
  $Z=(H-\bar H)U\in\mathbb R^{{T\times k}}$. The dimension $k$ is chosen adaptively to retain at
  least 95% of held-out variance.</p>
  <p>The shared conditional mean at position $p$ is</p>
  <div class="pe-equation">$$
    M_p(Y)=\bar z+a_{{g(Y_p)}}+b_{{\operatorname{{posbin}}(p)}}.
  $$</div>
  <p class="pe-where">$a_{{g(Y_p)}}$ captures the average hidden state associated with the token
  being predicted (frequent tokens get their own group; rarer tokens are pooled), while
  $b_{{\operatorname{{posbin}}(p)}}$ captures coarse position effects.</p>
  <div class="pe-warning"><strong>This is a supervised distributional test, not a text generator.</strong>
  During suffix training the next-token label is available, just as a class label is available in a
  class-conditional image experiment. A generated row may depend on its own label $Y_p$, but the
  leakage test verifies that it cannot depend on future labels $Y_{{p+1:T}}$.</div>
  <table class="pe-surrogate" aria-label="Part E hidden-sequence surrogate hierarchy">
    <thead><tr><th>Distribution</th><th>Mathematical form in PCA coordinates</th><th>Structure preserved</th></tr></thead>
    <tbody>
      <tr><th>Mean only</th><td>$Z=M(Y)+\sigma E$, $E_{{p,:}}\overset{{iid}}\sim\mathcal N(0,I)$</td><td>Token-conditioned mean, coarse position, and overall nuisance scale.</td></tr>
      <tr><th>Token-IID Gaussian</th><td>$Z_p=M_p(Y)+\varepsilon_p$, $\varepsilon_p\overset{{iid}}\sim\mathcal N(0,\Sigma_{{\mathrm{{chan}}}})$</td><td>Adds covariance between hidden channels, but treats positions independently.</td></tr>
      <tr><th>Sequence Gaussian</th><td>$Z\mid Y\sim\mathcal{{MN}}\!\left(M(Y),K_{{\mathrm{{pos}}}},\Sigma_{{\mathrm{{chan}}}}\right)$</td><td>Adds both channel covariance and covariance between token positions.</td></tr>
      <tr><th>Projected true</th><td>$\widetilde H=\bar H+((H-\bar H)U)U^\top$</td><td>All empirical structure inside the retained PCA subspace; diagnoses truncation.</td></tr>
      <tr><th>Full true</th><td>$H$ from the frozen model</td><td>The complete measured residual-stream sequence.</td></tr>
    </tbody>
  </table>
  <div class="pe-equation">$$
    \operatorname{{Cov}}\!\left(\operatorname{{vec}} Z\mid Y\right)
    =\Sigma_{{\mathrm{{chan}}}}\otimes K_{{\mathrm{{pos}}}}.
  $$</div>
  <p>This Kronecker product is the key sequence-modeling assumption:
  $\Sigma_{{\mathrm{{chan}}}}$ describes which hidden features vary together at a position;
  $K_{{\mathrm{{pos}}}}$ describes which positions vary together across the story. It is a
  <strong>separable approximation</strong>, not an arbitrary covariance over all $Tk$ coordinates.</p>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">3 · The intervention and the comparison</div>
  <h2>Train on one hidden distribution; evaluate on every hidden distribution</h2>
  <p>For each relaxation distribution $Q_r$, clone the same checkpoint suffix and update it for
  $u$ minibatches. Then evaluate that suffix on a held-out bank from every distribution $Q_s$:</p>
  <div class="pe-equation">$$
    M_{{s,r}}^{{(\ell)}}(u)
    =\mathbb E_{{(H,Y)\sim Q_s}}
      \left[\mathcal L_{{\mathrm{{NTP}}}}\!\left(\psi_{{\ell;r}}^{{(u)}};H,Y\right)\right].
  $$</div>
  <p class="pe-where">$r$ indexes what the suffix learned from; $s$ indexes what it is tested on.
  The full cross-evaluation matrix matters because a suffix can perform well on an easy synthetic
  distribution without transferring to real residual streams.</p>
  <p>The decisive row evaluates every suffix on held-out <strong>full true</strong> activations.
  Lower loss is better. We subtract the loss of the suffix trained on true activations:</p>
  <div class="pe-equation">$$\Delta_{{\mathrm{{true}}\mid r}}^{{(\ell)}}(u)
  =M_{{\mathrm{{true}},r}}^{{(\ell)}}(u)-M_{{\mathrm{{true}},\mathrm{{true}}}}^{{(\ell)}}(u).$$</div>
  <p class="pe-where">Thus $\Delta=0$ means the surrogate trained the suffix as effectively as real
  activations at this update budget; $\Delta>0$ is the remaining surrogate shortfall.</p>
  <div class="pe-guide"><strong>Interpretation guide.</strong> Zero is sufficient at this update budget.
  If sequence Gaussian lies below mean-only, modeled covariance adds useful structure; if they remain
  aligned or sequence Gaussian is higher, the covariance model has not explained the missing signal.
  Projected true near zero shows that PCA truncation is not driving the conclusion.</div>
  <div class="pe-grid-2">
    <article class="pe-card"><h3>Primary: gradient-scale normalization</h3><p>Different hidden
    distributions produce different raw loss-gradient magnitudes. We rescale each learning rate by
    $\eta_r=\eta_{{\mathrm{{true}}}}\|g_{{\mathrm{{true}}}}\|/\|g_r\|$ (clipped to $[0.25,4]$)
    before running AdamW. This is a practical scale control; because AdamW normalizes coordinates
    adaptively, it does not guarantee identical parameter-update norms.</p></article>
    <article class="pe-card"><h3>Control: equal nominal learning rate</h3><p>Every suffix uses the
    same $\eta=5\times10^{{-5}}$. This preserves the literal optimizer setting but allows the
    surrogate to induce a larger or smaller functional intervention.</p></article>
  </div>
  <p><strong>Measured protocol:</strong> 50M-token checkpoint; cuts 0, 5, and 11; 512 fit stories;
  64 held-out evaluation stories; batch size 8; 32 suffix updates; one model seed and one
  independently keyed surrogate draw.</p>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">4 · Primary result</div>
  <h2>Does adding sequence covariance close the true-activation learning gap?</h2>
  <div id="pe-endpoint-plot" class="pe-plot" aria-label="Part E endpoint excess loss by cut"></div>
  <div class="pe-takeaway"><strong>Observed:</strong> under gradient-scale normalization, sequence covariance
  does not beat mean-only at cuts 0 and 5. At cut 11 it lowers excess loss
  from 0.027 to 0.014 nats/token. Filled solid traces are gradient normalized; open dotted traces retain
  equal nominal LR.</div>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">5 · Relaxation trajectory and optimizer control</div>
  <h2>The conclusion develops over downstream-only training</h2>
  <p>The solid curves use gradient-scale-normalized learning rates; dotted curves use equal nominal learning rate.
  Showing both answers two distinct questions: whether the surrogate contains useful information under a
  comparable intervention, and what happens under literally identical optimizer hyperparameters.</p>
  <div id="pe-curve-plot" class="pe-plot" aria-label="Part E relaxation trajectories"></div>
  <div class="pe-warning"><strong>Why two regimes?</strong> At cuts 0 and 5, the initial
  sequence-Gaussian gradient is 3.52× and 2.45× the true-activation gradient. Equal nominal LR therefore
  moves those suffixes much farther on their first step. Cut 11 is better matched at 1.19×.</div>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">6 · Validity checks</div>
  <h2>The analyzer clears PCA, replay, leakage, and parity checks</h2>
  <div class="pe-grid">
    <article class="pe-card"><h3>Adaptive PCA</h3><span class="pe-number">95.02–97.52%</span><p>Held-out variance coverage using 128, 349, and 409 components across cuts 0, 5, and 11.</p></article>
    <article class="pe-card"><h3>Projected replay</h3><span class="pe-number">0.0077–0.0144</span><p>Excess nats/token from PCA reconstruction, small relative to the surrogate shortfalls.</p></article>
    <article class="pe-card"><h3>FP32 parity</h3><span class="pe-number">≤ 1.24×10⁻⁵</span><p>Maximum absolute cached-suffix logit error; future-target leakage is exactly zero.</p></article>
  </div>
  <table aria-label="Part E mechanical diagnostics">
    <thead><tr><th>Cut</th><th>PCA components</th><th>Coverage</th><th>Projected excess</th><th>FP32 parity max</th><th>seqG / true gradient</th></tr></thead>
    <tbody id="pe-diagnostic-rows"></tbody>
  </table>
</div>

<div class="pe-section">
  <div class="pe-eyebrow">Audit matrix · details on demand</div>
  <h2>Full five-by-five relax-by-evaluate matrix</h2>
  <p>Each cell is endpoint loss relative to relaxation on full true activations for the same evaluation
  distribution. Rows select the evaluation bank; columns select the suffix-relaxation bank.</p>
  <div class="pe-controls">
    <label>Cut <select id="pe-matrix-cut"><option value="0">block 0</option><option value="5">block 5</option><option value="11">block 11</option></select></label>
    <label>LR regime <select id="pe-matrix-regime"><option value="matched">gradient normalized</option><option value="equal">equal nominal LR</option></select></label>
  </div>
  <div id="pe-matrix-plot" class="pe-plot" aria-label="Part E five by five cross-evaluation matrix"></div>
  <div class="pe-guide"><strong>How to read the matrix.</strong> The full-true column is zero by construction.
  Warm colors are worse than true relaxation; cool colors are better on that evaluation proxy. The
  true-activation row is the primary scientific evidence; other rows expose distribution-specific overfitting.</div>
</div>

<div class="pe-section">
  <h2>Decision</h2>
  <div class="pe-warning"><strong>Do not launch the multi-seed confirmation yet.</strong> These checks
  reject PCA failure and numerical replay as explanations, but do not establish broad sequence-Gaussian
  sufficiency. The next experiment should test richer nonseparable or history-conditioned structure rather
  than repeat this separable proxy at larger scale.</div>
  <details><summary>Protocol and provenance</summary><div>
    <p><strong>Gradient-matched artifact:</strong> <code>{matched_label}</code><br>
    <strong>Equal-LR artifact:</strong> <code>{equal_label}</code></p>
    <p>Both reuse the verified 50M-token checkpoint from <code>gate_seed0_20260722</code>.
    Five distributions: conditional mean, token-IID Gaussian, sequence Gaussian, projected true, and full true.
    Three residual-stream cuts; 32 suffix updates; 64 held-out stories; schema v2; total diagnostic GPU runtime
    {payload["runtime_seconds"]:.1f} seconds. Saved artifacts rebuild this section without importing training code.</p>
  </div></details>
</div>

<script>
window.PART_E_DATA={data_json};
(function(){{
  const D=window.PART_E_DATA;
  const colors={{mean:'#d55e00',token_gaussian:'#8c67a5',seq_gaussian:'#0072b2',projected_true:'#009e73',true:'#555'}};
  const dashes={{mean:'dash',token_gaussian:'dot',seq_gaussian:'solid',projected_true:'dashdot',true:'solid'}};
  const names=D.labels;
  const base={{
    paper_bgcolor:'white',plot_bgcolor:'white',font:{{family:'Inter,system-ui,sans-serif',color:'#282521'}},
    margin:{{l:72,r:25,t:52,b:62}},hovermode:'closest',
    xaxis:{{gridcolor:'#eee9e1',zeroline:false}},yaxis:{{gridcolor:'#eee9e1',zerolinecolor:'#777',zerolinewidth:1}}
  }};
  const cfg={{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}};
  function merge(a,b){{return Object.assign({{}},a,b);}}
  function renderEndpoint(){{
    const regimes=[['equal','equal nominal LR','circle-open'],['matched','gradient normalized','circle']];
    const traces=[];
    for(const [key,label,symbol] of regimes){{
      for(const dist of ['mean','token_gaussian','seq_gaussian','projected_true']){{
        traces.push({{type:'scatter',mode:'lines+markers',x:D[key].cuts,y:D[key].endpoints[dist],
          name:names[dist]+' · '+label,legendgroup:dist,showlegend:key==='matched',
          line:{{color:colors[dist],dash:key==='matched'?dashes[dist]:'dot',width:key==='matched'?2.5:1.4}},
          marker:{{symbol,size:key==='matched'?9:7,color:colors[dist]}},
          hovertemplate:'cut %{{x}}<br>'+names[dist]+'<br>'+label+'<br>excess %{{y:.4f}} nats/token<extra></extra>'}});
      }}
    }}
    Plotly.react('pe-endpoint-plot',traces,merge(base,{{
      title:{{text:'True-activation excess loss after 32 updates',font:{{size:16}}}},
      xaxis:{{title:'residual-stream cut after block',tickvals:D.matched.cuts,gridcolor:'#eee9e1'}},
      yaxis:{{title:'excess next-token loss (nats/token)',rangemode:'tozero',gridcolor:'#eee9e1',zeroline:true}},
      legend:{{orientation:'h',y:-.22,x:0}}
    }}),cfg);
  }}
  function renderCurves(){{
    const traces=[];
    const cuts=D.matched.cuts;
    cuts.forEach((cut,i)=>{{
      for(const regime of ['equal','matched']){{
        for(const dist of ['mean','seq_gaussian']){{
          traces.push({{type:'scatter',mode:'lines+markers',x:D[regime].curves[String(cut)].updates,
            y:D[regime].curves[String(cut)][dist],xaxis:'x'+(i?i+1:''),yaxis:'y'+(i?i+1:''),
            name:names[dist]+' · '+(regime==='matched'?'gradient normalized':'equal LR'),
            legendgroup:dist+'-'+regime,showlegend:i===0,
            line:{{color:colors[dist],dash:regime==='matched'?dashes[dist]:'dot',width:regime==='matched'?2.5:1.3}},
            marker:{{size:5,symbol:regime==='matched'?'circle':'circle-open'}},
            hovertemplate:'cut '+cut+'<br>update %{{x}}<br>excess %{{y:.4f}}<extra></extra>'}});
        }}
      }}
    }});
    const domains=[[0,.29],[.355,.645],[.71,1]];
    const layout=merge(base,{{title:{{text:'Surrogate shortfall grows during relaxation',font:{{size:16}}}},
      height:470,legend:{{orientation:'h',y:-.24,x:0}},annotations:cuts.map((c,i)=>({{
        text:'cut '+c,xref:'paper',yref:'paper',x:(domains[i][0]+domains[i][1])/2,y:1.05,showarrow:false,font:{{size:13}}
      }}))}});
    domains.forEach((dom,i)=>{{
      const s=i?String(i+1):'';
      layout['xaxis'+s]={{domain:dom,title:'suffix update',gridcolor:'#eee9e1',zeroline:false}};
      layout['yaxis'+s]={{title:i===0?'true-data excess loss':'',range:[-.01,.28],gridcolor:'#eee9e1',zeroline:true}};
    }});
    Plotly.react('pe-curve-plot',traces,layout,cfg);
  }}
  function renderMatrix(){{
    const cut=document.getElementById('pe-matrix-cut').value;
    const regime=document.getElementById('pe-matrix-regime').value;
    const z=D[regime].matrices[cut],labels=D.distributions.map(x=>names[x]);
    let bound=.01; z.forEach(row=>row.forEach(v=>bound=Math.max(bound,Math.abs(v))));
    Plotly.react('pe-matrix-plot',[{{type:'heatmap',z,x:labels,y:labels,zmin:-bound,zmax:bound,zmid:0,
      colorscale:'RdBu',
      colorbar:{{title:{{text:'excess<br>nats/token'}}}},
      text:z.map(row=>row.map(v=>v.toFixed(3))),texttemplate:'%{{text}}',
      hovertemplate:'evaluate on %{{y}}<br>relax on %{{x}}<br>relative loss %{{z:.4f}}<extra></extra>'}}],
      merge(base,{{title:{{text:'Endpoint cross-evaluation · cut '+cut+' · '+(regime==='matched'?'gradient normalized':'equal nominal LR'),font:{{size:16}}}},
      margin:{{l:140,r:25,t:52,b:90}},height:520,xaxis:{{title:'relax suffix on →',tickangle:-20}},
      yaxis:{{title:'evaluate suffix on →',autorange:'reversed'}}}}),cfg);
  }}
  function renderDiagnostics(){{
    document.getElementById('pe-diagnostic-rows').innerHTML=D.diagnostics.map(d=>{{
      const ratio=d.gradient_norms.seq_gaussian/d.gradient_norms.true;
      return `<tr><td>${{d.cut}}</td><td>${{d.components}}</td><td>${{(100*d.coverage).toFixed(2)}}%</td><td>${{d.projected_gap.toFixed(4)}}</td><td>${{d.parity.toExponential(2)}}</td><td>${{ratio.toFixed(2)}}×</td></tr>`;
    }}).join('');
  }}
  let rendered=false;
  window.renderPartE=function(){{
    if(!rendered){{renderDiagnostics();rendered=true;}}
    renderEndpoint();renderCurves();renderMatrix();
    if(window.MathJax?.typesetPromise) MathJax.typesetPromise([document.getElementById('e')]);
  }};
  document.getElementById('pe-matrix-cut').addEventListener('change',renderMatrix);
  document.getElementById('pe-matrix-regime').addEventListener('change',renderMatrix);
  const button=document.querySelector('[data-tab="e"]');
  if(button) button.addEventListener('click',()=>setTimeout(window.renderPartE,40));
  if(document.getElementById('e').classList.contains('active')) setTimeout(window.renderPartE,40);
}})();
</script>
</section>
""".strip()
