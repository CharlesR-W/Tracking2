"""Dependency-free Part D theory panel for tangent response at a network cut.

The fragment contains derivations and proposed readouts, but no numerical
outcomes; every independently screenshot-able panel is marked THEORY / UNRUN.
"""

from __future__ import annotations

__all__ = ["part_d_moment_mockup_html"]


def part_d_moment_mockup_html() -> str:
    """Return the planned Part D tangent-response tab as an HTML fragment."""

    return r"""<section id="d" class="panel part-d-tangent" data-epistemic-status="THEORY" data-run-status="PLANNED-UNRUN" aria-labelledby="pd-title">
<style>
.part-d-tangent {
  --pd-ink:#282521; --pd-muted:#69635c; --pd-line:#ddd5c8; --pd-paper:#fffefb;
  --pd-soft:#f6f3ed; --pd-blue:#0072b2; --pd-orange:#d55e00;
  --pd-green:#008b68; --pd-purple:#6f5aa6; --pd-red:#9d3329;
  color:var(--pd-ink);
}
.part-d-tangent * { box-sizing:border-box; }
.part-d-tangent h1,.part-d-tangent h2,.part-d-tangent h3 { color:var(--pd-ink); font-family:Georgia,serif; }
.part-d-tangent h1 { font-size:clamp(1.85rem,3.4vw,2.65rem); line-height:1.08; margin:.55rem 0 .8rem; }
.part-d-tangent h2 { font-size:1.55rem; line-height:1.2; margin:.45rem 0 .7rem; }
.part-d-tangent h3 { font-size:1.08rem; line-height:1.25; margin:0 0 .45rem; }
.part-d-tangent p,.part-d-tangent li { line-height:1.55; }
.part-d-tangent p { max-width:98ch; }
.part-d-tangent .pd-muted { color:var(--pd-muted); }
.part-d-tangent .pd-kicker { color:var(--pd-red); font:800 .76rem/1.35 system-ui,sans-serif; letter-spacing:.11em; text-transform:uppercase; }
.part-d-tangent .pd-status { display:inline-block; color:var(--pd-red); background:#fff5f2; border:1px solid #d9a39d; border-radius:999px; padding:.3rem .65rem; font:800 .73rem/1.3 system-ui,sans-serif; letter-spacing:.07em; text-transform:uppercase; }
.part-d-tangent .pd-hero,.part-d-tangent .pd-panel { position:relative; overflow:hidden; border:1px solid var(--pd-line); border-radius:13px; background:var(--pd-paper); margin:22px 0; padding:clamp(18px,3vw,28px); }
.part-d-tangent .pd-hero { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(250px,.72fr); gap:25px; border-top:6px solid var(--pd-purple); }
.part-d-tangent .pd-panel { border-top:4px solid #b7afa2; }
.part-d-tangent .pd-watermark { position:absolute; top:1.15rem; right:-.75rem; transform:rotate(12deg); color:rgba(145,48,40,.16); font:900 .88rem/1 system-ui,sans-serif; letter-spacing:.08em; text-transform:uppercase; pointer-events:none; white-space:nowrap; }
.part-d-tangent .pd-guide,.part-d-tangent .pd-caveat,.part-d-tangent .pd-equation { margin:14px 0; padding:.9rem 1rem; line-height:1.5; }
.part-d-tangent .pd-guide { border-left:4px solid var(--pd-blue); background:#eef7fb; }
.part-d-tangent .pd-caveat { border-left:4px solid #e69f00; background:#fff8e8; }
.part-d-tangent .pd-equation { overflow-x:auto; border:1px solid #cbdde7; border-left:4px solid var(--pd-blue); background:#f4f9fc; text-align:center; }
.part-d-tangent .pd-equation.pd-central { border-left-color:var(--pd-purple); background:#f6f3fa; }
.part-d-tangent .pd-grid-2,.part-d-tangent .pd-grid-3 { display:grid; gap:14px; margin:16px 0; }
.part-d-tangent .pd-grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
.part-d-tangent .pd-grid-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
.part-d-tangent .pd-card { min-width:0; border:1px solid var(--pd-line); border-top:4px solid var(--pd-blue); border-radius:9px; background:#fcfbf8; padding:14px; }
.part-d-tangent .pd-card.orange { border-top-color:var(--pd-orange); }
.part-d-tangent .pd-card.green { border-top-color:var(--pd-green); }
.part-d-tangent .pd-card.purple { border-top-color:var(--pd-purple); }
.part-d-tangent .pd-num { display:inline-grid; place-items:center; width:1.65rem; height:1.65rem; margin-right:.35rem; border-radius:50%; color:white; background:#4d4943; font:800 .8rem/1 system-ui,sans-serif; }
.part-d-tangent .pd-scroll { overflow-x:auto; }
.part-d-tangent table { width:100%; border-collapse:collapse; margin:14px 0; }
.part-d-tangent th,.part-d-tangent td { border-bottom:1px solid var(--pd-line); padding:9px; text-align:left; vertical-align:top; line-height:1.45; }
.part-d-tangent th { color:#4d4943; font-size:.88rem; }
.part-d-tangent thead th { background:var(--pd-soft); }
.part-d-tangent .pd-flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:9px; margin:16px 0; align-items:stretch; }
.part-d-tangent .pd-flow > div:not(.pd-arrow) { border:1px solid var(--pd-line); border-radius:9px; padding:13px; background:#fcfbf8; }
.part-d-tangent .pd-arrow { align-self:center; color:#5f5a54; font-size:1.4rem; }
.part-d-tangent .pd-spectrum { width:100%; min-height:238px; border:1px solid var(--pd-line); border-radius:10px; background:#fffdfa; }
.part-d-tangent .pd-checks { columns:2; column-gap:30px; padding-left:1.25rem; }
.part-d-tangent .pd-checks li { break-inside:avoid; margin:0 0 .58rem; }
@media (max-width:850px) {
  .part-d-tangent .pd-hero,.part-d-tangent .pd-grid-2,.part-d-tangent .pd-grid-3 { grid-template-columns:1fr; }
  .part-d-tangent .pd-flow { grid-template-columns:1fr; }
  .part-d-tangent .pd-arrow { transform:rotate(90deg); justify-self:center; }
  .part-d-tangent .pd-checks { columns:1; }
}
</style>

<header class="pd-hero" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <div>
    <div class="pd-kicker">Part D · local dynamics at one checkpoint</div>
    <h1 id="pd-title">Which movements of an activation distribution can the suffix track?</h1>
    <p>At training time $t$ and cut $\ell$, freeze a local chart around the prefix–suffix pair and its implied optimal suffix $\psi^*[\phi]$. Linearize <em>the update rule</em>, then treat a small movement of the activation distribution as a driven input. This gives poles, response modes, and a frequency-dependent residual in prediction space.</p>
    <div class="pd-equation pd-central">
      $$\boxed{\mathcal R_\ell(e^{i\omega})=\mathcal T_u+\mathcal T_b(e^{i\omega}I-A_{\rm suf})^{-1}B_u},
      \qquad
      \boxed{F_{\rm eff}(\omega)=\mathcal R_\ell(e^{i\omega})^*\mathcal R_\ell(e^{i\omega})}.$$
    </div>
    <p><strong>Proposed Part-D object.</strong> The eigenvectors of the domain-normalized $F_{\rm eff}(\omega)$ are activation-<em>distribution</em> perturbations; its eigenvalues are their residual predictive KL gains after the suffix responds at frequency $\omega$.</p>
  </div>
  <aside>
    <span class="pd-status">Theory / experiment design · unrun</span>
    <p><strong>Poles and stability:</strong> optimizer Jacobian $A$.</p>
    <p><strong>Behavioral size:</strong> model-Fisher/GGN output metric.</p>
    <p><strong>Static compensation:</strong> the $\omega=0$ Schur-complement limit.</p>
    <p><strong>What is deferred:</strong> products of time-varying Jacobians, finite-time Lyapunov exponents, and optimizer-noise forcing.</p>
    <div class="pd-caveat"><strong>No result is shown here.</strong> All diagrams are definitions or interpretation guides; no eigenvalues or response curves have yet been measured.</div>
  </aside>
</header>

<article class="pd-panel" id="pd-local-chart" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <span class="pd-status">1 · define the local chart</span>
  <h2>The perturbation is a movement of a distribution, not one activation vector</h2>
  <div class="pd-equation">
    $$h^\ell=\phi_a^\ell(x),\qquad Q_{a,\ell}=\operatorname{Law}(h^\ell,y),\qquad
    b^*(a)\in\arg\min_b\;\mathbb E_{Q_{a,\ell}}\ell(\psi_b(h),y).$$
  </div>
  <p>Use an immutable probe bank and define a finite tangent basis for nearby distributions. A transport chart is especially concrete:</p>
  <div class="pd-equation">
    $$h_i(u)=h_i+\sum_{j=1}^{m}u_jv_j(h_i,y_i),\qquad
    \delta h_i^\ell=V_i\,\delta u,\qquad
    M_u=\mathbb E_i[V_i^\top V_i].$$
  </div>
  <p>The columns $v_j$ may encode class-conditional translations, covariance deformations, higher-order Hermite transports, or data-driven smooth directions. $M_u$ declares equal RMS transport cost. A reweighting/score chart is also possible, but it is a different intervention and must not be silently mixed with transport.</p>
  <div class="pd-grid-3">
    <section class="pd-card"><h3><span class="pd-num">1</span>Anchor</h3><p>Refit or tightly relax $b_0=b^*(a_0)$ at checkpoint $t$. Otherwise a nonzero base gradient is confused with response to $u$.</p></section>
    <section class="pd-card orange"><h3><span class="pd-num">2</span>Coordinates</h3><p>Fix probe examples, transport basis, normalization, model mode, and cut. The answer is conditional on this chart.</p></section>
    <section class="pd-card green"><h3><span class="pd-num">3</span>Readout</h3><p>Stack Fisher-whitened logit changes on the probe bank so squared norm is locally $2D_{\rm KL}$.</p></section>
  </div>
  <div class="pd-guide"><strong>Interpretation guide.</strong> A mode may overlap “mean,” “covariance,” or “skew” directions, but the response eigenvector itself is an empirical distributional transport. Label it by a moment only after reporting that overlap and leakage into other moments.</div>
</article>

<article class="pd-panel" id="pd-poles" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <span class="pd-status">2 · compare full and clamped dynamics</span>
  <h2>Compare eigenvalues of update maps—not raw Fisher eigenvalues</h2>
  <p>Write prefix and suffix parameters as $a,b$, let $H=\nabla^2L(a_0,b_0)$, and first consider vanilla gradient descent with block learning rates. After subtracting the unperturbed step,</p>
  <div class="pd-equation pd-central">
    $$\begin{bmatrix}\delta a_{k+1}\\\delta b_{k+1}\end{bmatrix}
    =A_{\rm full}\begin{bmatrix}\delta a_k\\\delta b_k\end{bmatrix},\qquad
    A_{\rm full}=I-
    \begin{bmatrix}\eta_aI&0\\0&\eta_bI\end{bmatrix}
    \begin{bmatrix}H_{aa}&H_{ab}\\H_{ba}&H_{bb}\end{bmatrix},
    \qquad A_{\rm suf}=I-\eta_bH_{bb}.$$
  </div>
  <div class="pd-grid-2">
    <div class="pd-scroll"><table aria-label="Full versus suffix-only eigenspectra">
      <thead><tr><th>Operator</th><th>Question</th><th>Primary readout</th></tr></thead>
      <tbody>
        <tr><th>$A_{\rm full}$</th><td>How do coupled prefix and suffix perturbations decay or grow?</td><td>Unit-disk eigenvalues, spectral radius, and mode localization across the cut</td></tr>
        <tr><th>$A_{\rm suf}$</th><td>How quickly can the suffix relax when the activation distribution is clamped?</td><td>Suffix poles and relaxation times $\tau_j=-1/\log|\lambda_j|$</td></tr>
        <tr><th>Difference</th><td>Does coupling introduce slow or unstable joint modes absent from suffix relaxation?</td><td>Matched spectral densities plus full-mode prefix/suffix participation</td></tr>
      </tbody>
    </table></div>
    <svg class="pd-spectrum" viewBox="0 0 500 270" role="img" aria-labelledby="pd-spectrum-title pd-spectrum-desc">
      <title id="pd-spectrum-title">Schematic full and suffix eigenvalue comparison</title>
      <desc id="pd-spectrum-desc">A unit disk and symbolic eigenvalue sets; locations are illustrative and contain no measured data.</desc>
      <rect width="500" height="270" fill="#fffdfa"/>
      <circle cx="165" cy="137" r="102" fill="#f7f4ee" stroke="#77716a" stroke-width="2"/>
      <line x1="45" y1="137" x2="285" y2="137" stroke="#bbb3a7"/><line x1="165" y1="22" x2="165" y2="252" stroke="#bbb3a7"/>
      <text x="165" y="18" text-anchor="middle" font-size="13" fill="#57514a">stable update eigenvalues: $|\lambda|&lt;1$</text>
      <circle cx="125" cy="112" r="6" fill="#0072b2"/><circle cx="205" cy="162" r="6" fill="#0072b2"/>
      <path d="M235 92 l9 9 m0-9 l-9 9" stroke="#d55e00" stroke-width="4"/>
      <path d="M278 137 l9 9 m0-9 l-9 9" stroke="#d55e00" stroke-width="4"/>
      <text x="318" y="89" font-size="14" fill="#0072b2">● suffix-only poles</text>
      <text x="318" y="117" font-size="14" fill="#d55e00">× full-system poles</text>
      <text x="318" y="164" font-size="13" fill="#69635c">SCHEMATIC ONLY</text>
      <text x="318" y="184" font-size="13" fill="#69635c">positions are not values</text>
    </svg>
  </div>
  <p>The block coupling is related to a Schur complement, but the static Schur complement is <strong>not</strong> the full-system spectrum. In continuous time, eliminating the suffix at Laplace frequency $s$ gives</p>
  <div class="pd-equation">
    $$\mathcal S_a(s)=sI+\eta_aH_{aa}-\eta_aH_{ab}(sI+\eta_bH_{bb})^{-1}\eta_bH_{ba}.$$
  </div>
  <p>Only at $s=0$ does this reduce, up to $\eta_a$, to $H_{aa}-H_{ab}H_{bb}^{\dagger}H_{ba}$. The full poles therefore come from a frequency-dependent block-elimination problem.</p>
  <div class="pd-caveat"><strong>Stability language.</strong> A stable discrete tangent model has $\rho(A)&lt;1$. A sinusoid does not make such an LTI model unstable; it may be strongly amplified near a pole, especially for a non-normal $A$. If momentum, Adam, or normalization state is active, include that state in the Jacobian. Report resolvent norm or pseudospectral checks beside eigenvalues when non-normality is appreciable.</div>
</article>

<article class="pd-panel" id="pd-response" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <span class="pd-status">3 · derive the suffix response</span>
  <h2>The suffix is a driven linear system</h2>
  <p>Let $u_k$ drive the activation-distribution chart. Linearizing the suffix optimizer and the Fisher-whitened prediction vector $y$ gives</p>
  <div class="pd-equation pd-central">
    $$\delta b_{k+1}=A_{\rm suf}\delta b_k+B_u u_k,\quad B_u=-\eta_bH_{bu},
    \qquad y_k=\mathcal T_b\delta b_k+\mathcal T_u u_k.$$
    $$\mathcal R_\ell(z)=\mathcal T_u+\mathcal T_b(zI-A_{\rm suf})^{-1}B_u.$$
  </div>
  <p>The first term is the <strong>frozen-suffix</strong> response. The resolvent term is what suffix learning cancels, ignores, or amplifies. Evaluating $z=e^{i\omega}$ gives a Bode-like response without assuming one timescale.</p>
  <div class="pd-grid-2">
    <section class="pd-card purple">
      <h3>Moving-optimum view</h3>
      <div class="pd-equation">$$K_H=\frac{db^*}{du}=-H_{bb}^{\dagger}H_{bu}.$$</div>
      <p>If $H_{bb}q_j=h_jq_j$ and the optimizer is scalar-step GD, suffix mode $j$ responds to its moving target with</p>
      <div class="pd-equation">$$\Gamma_j(e^{i\omega})=\frac{\eta_bh_j}{e^{i\omega}-(1-\eta_bh_j)}.$$</div>
      <p>At DC, $\Gamma_j(1)=1$ for an identifiable positive-curvature mode. Above its relaxation rate, tracking gain falls and phase lag grows.</p>
    </section>
    <section class="pd-card green">
      <h3>Distribution-mode view</h3>
      <div class="pd-equation">$$\widehat F_{\rm eff}(\omega)=M_u^{-1/2}\mathcal R(e^{i\omega})^*\mathcal R(e^{i\omega})M_u^{-1/2}.$$</div>
      <p>Its right eigenvectors answer “<em>which distributional perturbations?</em>”; its eigenvalues answer “<em>how much predictive response per unit RMS transport?</em>”</p>
      <p>Also report $c(v,\omega)=\|\mathcal Rv\|/\|\mathcal T_uv\|$: $c&lt;1$ means attenuation, $c\approx1$ failure to compensate, and $c&gt;1$ adaptive amplification.</p>
    </section>
  </div>
  <div class="pd-guide"><strong>Interpretation guide.</strong> “Trackable” should mean small residual predictive response and small moving-optimum error at the same frequency—not merely large parameter motion. A small suffix update can be correct if the network has learned invariance.</div>
</article>

<article class="pd-panel" id="pd-fisher-schur" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <span class="pd-status">4 · place Fisher and Schur correctly</span>
  <h2>Fisher measures visible residuals; it does not set the poles</h2>
  <p>Stack categorical-logit Jacobians whitened by $W_i^{1/2}$, where $W_i=\operatorname{diag}(p_i)-p_ip_i^\top$. Then $G_{bb}=\mathcal T_b^*\mathcal T_b$, $G_{bu}=\mathcal T_b^*\mathcal T_u$, and $G_{uu}=\mathcal T_u^*\mathcal T_u$.</p>
  <p>If suffix parameters may change arbitrarily to minimize instantaneous predictive KL, $K_G=-G_{bb}^{\dagger}G_{bu}$ and</p>
  <div class="pd-equation pd-central">
    $$\min_{\delta b}\|\mathcal T_u\delta u+\mathcal T_b\delta b\|^2
    =\delta u^*\underbrace{\left(G_{uu}-G_{ub}G_{bb}^{\dagger}G_{bu}\right)}_{\text{static Fisher/GGN Schur complement}}\delta u.$$
  </div>
  <div class="pd-grid-3">
    <section class="pd-card"><h3>Frozen sensitivity</h3><p>$G_{uu}$: what the perturbation does before the suffix moves.</p></section>
    <section class="pd-card orange"><h3>Geometric DC limit</h3><p>The Schur complement: what an unconstrained best local suffix change could leave behind.</p></section>
    <section class="pd-card green"><h3>Realized response</h3><p>$F_{\rm eff}(\omega)=\mathcal R^*\mathcal R$: what the declared optimizer leaves behind on its timescale.</p></section>
  </div>
  <p>The optimizer’s DC adjustment is instead $K_H=-H_{bb}^{\dagger}H_{bu}$. The two coincide only when the loss Hessian is adequately represented by the same model Fisher/GGN and the same damping/metrics are used. This equality must be checked, not assumed.</p>
  <div class="pd-caveat"><strong>Crucial separation.</strong> Use the exact update Jacobian (or a declared Hessian/preconditioner approximation) for dynamics; use Fisher/GGN to measure predictive KL. Small Fisher curvature does not imply fast relaxation, and a stable Hessian does not imply that a perturbation is behaviorally unimportant.</div>
</article>

<article class="pd-panel" id="pd-dashboard" data-panel-status="THEORY-PLANNED-UNRUN">
  <span class="pd-watermark" data-theory-watermark="true">THEORY · PLANNED · UNRUN</span>
  <span class="pd-status">5 · proposed Part-D measurement</span>
  <h2>What the measured Part D should show</h2>
  <div class="pd-flow" aria-label="Part D evidence flow">
    <div><h3>Poles</h3><p>Full-system versus suffix-only spectra at the same anchor; mode participation and relaxation times.</p></div>
    <div class="pd-arrow">→</div>
    <div><h3>Response</h3><p>Frequency × perturbation-mode heatmap of residual KL gain, attenuation, and phase.</p></div>
    <div class="pd-arrow">→</div>
    <div><h3>Meaning</h3><p>Project the most/least trackable modes onto named mean, covariance, and higher-order transports.</p></div>
  </div>
  <div class="pd-scroll"><table aria-label="Minimum plots for measured Part D">
    <thead><tr><th>View</th><th>Comparison</th><th>What would be evidence?</th></tr></thead>
    <tbody>
      <tr><th>Complex eigenvalue plane</th><td>$\sigma(A_{\rm full})$ against $\sigma(A_{\rm suf})$</td><td>Coupled poles nearer the unit circle identify slow joint adaptation absent when the prefix is clamped</td></tr>
      <tr><th>Relaxation spectrum</th><td>Timescale and prefix/suffix participation across cut and time</td><td>Shows whether slow modes live in the suffix or in co-adaptation across the cut</td></tr>
      <tr><th>Response singular-value map</th><td>Frozen $\mathcal T_u$ versus realized $\mathcal R(e^{i\omega})$</td><td>Low-frequency attenuation with roll-off is tracking; uniformly small frozen response is invariance</td></tr>
      <tr><th>Mode portraits</th><td>Top and bottom response directions versus the transport dictionary</td><td>Identifies distribution movements that are rejected, passed, or amplified</td></tr>
      <tr><th>DC triangle</th><td>Finite refit, optimizer DC prediction, and Fisher Schur bound</td><td>Agreement validates the local model; disagreement localizes Hessian/GGN, damping, or nonlinearity error</td></tr>
    </tbody>
  </table></div>
  <h3>Minimum validity checks</h3>
  <ul class="pd-checks">
    <li>Same checkpoint, suffix optimum, probe bank, and coordinate metric throughout.</li>
    <li>Finite-difference JVP and one-step update-Jacobian checks.</li>
    <li>Transport moment-target and off-target leakage audit.</li>
    <li>Amplitude sweep establishing a local linear regime.</li>
    <li>Dense eigensolver agreement on a toy network before matrix-free scaling.</li>
    <li>Finite nonlinear sinusoidal rollouts compared prospectively with the tangent prediction.</li>
    <li>Zero-drive and frozen-suffix controls.</li>
    <li>Uncertainty across independent training seeds; probe resampling is not a seed.</li>
  </ul>
  <div class="pd-guide"><strong>Recommended first experiment.</strong> Use the normalization-free residual MLP with a small declared transport dictionary and plain SGD. Start with exact dense Jacobians on a deliberately tiny model, then scale only after the spectrum, DC Schur limit, and finite sinusoidal response agree.</div>
  <div class="pd-caveat"><strong>Deferred extension.</strong> Products $A_{k+K-1}\cdots A_k$ and finite-time Lyapunov exponents answer whether a time-varying training trajectory expands perturbations over several steps. That is valuable, but it is a different question from this frozen-chart transfer function and should follow only after the one-step operator is validated.</div>
</article>
</section>
"""
