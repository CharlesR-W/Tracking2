# Tangent models as a unifying theory of local neural response

> **Status: THEORY / EXPERIMENT DESIGN — UNRUN.** This document does not report a
> Fisher, tangent-model, control, or LLM result. It is a companion to the current
> Parts D and E designs. Any dashboard surface derived from it must remain labeled
> **MOCKUP · PLANNED · UNRUN** until a validated artifact replaces it.

## Executive conclusion

The tangent model is the right unifying object only if “unifying” means a common
local response map, not an identity among every matrix used in learning theory.

At a declared checkpoint, probe distribution, and intervention class, the Jacobian
maps a small intervention to a change in model outputs. The output-distribution
metric turns this map into a Fisher/GGN geometry. The same map, viewed from output
space, gives a weighted tangent kernel. When the optimizer update is also
linearized, the checkpoint becomes a local dynamical system with separate notions
of:

1. **visibility:** does a perturbation change the predictive distribution?
2. **reachability:** can the declared intervention or optimizer produce it?
3. **stability:** does it contract, amplify, or exhibit transient growth?
4. **trackability:** can a suffix compensate a moving interface on its timescale?
5. **validity:** for what amplitudes and horizons does the tangent system predict
   the nonlinear network?

This is the theory target:

> A trained network is locally described by a sequence of response charts—a
> **tangent atlas**. Fisher metrizes behavioral visibility inside each chart;
> optimizer dynamics determine reachability and stability; chart drift and Taylor
> remainder mark where representation change exceeds the local account.

The decisive empirical standard is prospective. Geometry measured before a fork
must predict adaptation, forgetting, or intervention persistence on held-out forks.
Retrospective correlation is not enough.

## Relationship to the current Tracking2 program

- Parts B/C ask which fitted activation distributions support suffix relaxation at
  a frozen interface. They do not measure a tangent model.
- Part D already separates moment motion, directional model-Fisher/GGN sensitivity,
  geometric compensation, and measured bandwidth. The present document derives
  those quantities from one state-space object and adds a tangent-validity atlas.
- Part E specifies GPT-2 sequence surrogates and matched P/C/I/S branches. The first
  LLM tangent gate below is deliberately smaller than the full Part-E surrogate
  matrix.
- The existing `empirical_fisher_schur_trace` routine is an observed-label gradient
  second moment with a small sample-rank cap. It must remain archived and must not
  be renamed as the model Fisher/GGN defined here.

The worktree is currently substantially dirty and some checkpoint/JSON provenance
is not represented by `HEAD`. Implement this design only in a fresh namespace with
hashed checkpoint bytes; do not retrofit new semantics onto old artifacts.

## 1. Declare the local response problem

Let $Q$ be an immutable probe distribution and let $s$ denote the output object of
interest. For a classifier, $s$ may be logits on held-out examples. For an
autoregressive model, stack logits only at declared loss-bearing token positions,
retaining example, mask, position, and context identity.

Let $u$ be coordinates for one intervention class. Examples include:

- a full-parameter or module-restricted perturbation;
- LoRA coordinates or another trainable subspace;
- a continuous soft prompt;
- a residual-stream intervention at cut $\ell$;
- a suffix parameter adjustment;
- a controlled data or interface-moment transport $\lambda$.

An intervention map $R_u$ embeds these coordinates in the relevant parameter,
activation, or input space. At the anchor $u=0$,

$$
s(R_u\delta u)
=
s_0+J_u\delta u+r_2(\delta u),
\qquad
J_u
=
\left.\frac{\partial s}{\partial u}\right|_{0}.
$$

Every claim is conditional on:

- the anchor checkpoint and optimizer state;
- the probe distribution and output mask;
- the intervention coordinates and their metric;
- the output metric;
- the amplitude and rollout horizon; and
- the model mode, mutable normalization state, and stochasticity policy.

There is no context-free “the Fisher of the model” or “the tangent space of the
behavior” suitable for the empirical claims below.

### 1.1 Required chart registry

No experiment may say only “at cut $\ell$.” Each local operator must reference a
versioned `chart_id`. The initial registry is:

| `chart_id` family | Coordinate vector | Intervention timing |
|---|---|---|
| `parameter/full` | all declared trainable parameters in canonical name/offset order | persistent optimizer state/update |
| `parameter/suffix@cut` | suffix parameters only, same canonical ordering | persistent optimizer state/update |
| `activation@cut` | declared flattened activation tensor or low-rank basis coordinates | one-shot or per-step injection, stored explicitly |
| `moment@cut` | coefficients of a validated interface transport $T_\lambda$ | level or increment drive, stored explicitly |
| `lora@targets` | pinned target modules and a nondegenerate local factorization | persistent optimizer update |

Every registry row must also pin the embedding/intervention map, coordinate shape
and flatten ordering, cost metric $M_u$, probe bank and output metric, legal
direction families, model train/eval mode, mutable normalization state, and whether
the perturbation is an initial condition or a driven input. The actual update, a
new-task gradient, a generalized Fisher mode, and an activation mode do not live in
one coordinate space unless a declared map relates them.

For LoRA, the usual initialization with one zero factor has a degenerate first-order
map in one factor. Freeze a nonzero factor or linearize in the induced low-rank
weight-update coordinates; record that anchor convention before comparing LoRA
tangent images.

## 2. The whitened tangent operator

For categorical predictions with probabilities $p$, the logit-space model-Fisher
metric is

$$
W(p)
=
\operatorname{diag}(p)-pp^\top.
$$

For an autoregressive bank, use the block-diagonal $W_i$ at each valid token and
average with the declared per-token or per-example normalization. Stack the
per-observation operators as

$$
\mathcal T_u
=
\begin{bmatrix}
\sqrt{w_1}W_1^{1/2}J_{u,1}\\
\vdots\\
\sqrt{w_n}W_n^{1/2}J_{u,n}
\end{bmatrix}.
$$

If $M_u\succ0$ is the cost metric on intervention coordinates, use the
domain-whitened map

$$
\widehat{\mathcal T}_u
=
\mathcal T_u M_u^{-1/2}.
$$

The two Gram operators are

$$
M_u^{-1/2}G_{uu}M_u^{-1/2}
=
\widehat{\mathcal T}_u^\top\widehat{\mathcal T}_u,
$$

$$
K_{W,u}
=
\widehat{\mathcal T}_u\widehat{\mathcal T}_u^\top.
$$

The two Gram operators share the same nonzero eigenvalues $\sigma_i^2$. The right
and left singular vectors of $\widehat{\mathcal T}_u$ pair intervention directions
with predictive-output directions.

For a small intervention, this normalization obeys the local expansion

$$
D_{\mathrm{KL}}\!\left(p_u\,\|\,p_{u+\delta u}\right)
=
\frac12\delta u^\top G_{uu}\delta u+o(\|\delta u\|^2),
$$

under the declared probe weighting and regularity conditions. Finite-KL checks
below compare the measured KL with one half of the directional quadratic, not with
the unscaled quadratic.

This exact duality supplies a useful vocabulary:

- **parameter/intervention modes:** right singular vectors;
- **behavioral response modes:** left singular vectors;
- **response strength per declared intervention cost:** squared singular values;
- **null directions:** conditional on the probe, output metric, numerical tolerance,
  and intervention parameterization.

It does not imply that a Fisher-null direction is globally safe. It may be a model
symmetry, a hole in probe coverage, an output-saturation artifact, or a direction
with a large finite/OOD effect.

### 2.1 A dictionary of related objects

| Name | Definition / role | Permitted interpretation |
|---|---|---|
| Parameter NTK | $J_\theta M^{-1}J_\theta^\top$ | Output response under a declared parameter/update metric |
| Weighted tangent kernel | $W^{1/2}J M^{-1}J^\top W^{1/2}$ | Predictive-output response in Fisher-whitened coordinates |
| Model Fisher / GGN | $\mathbb E[J^\top WJ]$ | Local predictive KL geometry; GGN equality requires the matching likelihood/output construction |
| Exact Hessian | $J^\top H_s\ell J+\sum_a(\partial\ell/\partial s_a)\nabla^2s_a$ | Local optimizer/loss dynamics, potentially indefinite |
| Observed-label empirical Fisher | $\mathbb E[\nabla\ell_y\nabla\ell_y^\top]$ | Noncentral gradient second moment, not the model Fisher in general |
| Gradient-noise covariance | $\mathbb E[(g-\bar g)(g-\bar g)^\top]$ | Centered stochastic forcing of optimizer state |
| Data Fisher | $\mathbb E[\nabla_\lambda\log q_\lambda\nabla_\lambda\log q_\lambda^\top]$ | Distinguishability/normalization of input-distribution paths, not suffix use |

The paper should derive these from the same local scaffold while preserving their
different domains, expectations, and semantics.

### 2.2 Layerwise forward--backward factorization

For a scalar-output fully connected affine layer with input activation
$h_{\ell-1}(x)$ and backpropagated output sensitivity $\delta_\ell(x)$, the
weight-block tangent contribution factors as

$$
K_\ell(x,x')
=
\langle h_{\ell-1}(x),h_{\ell-1}(x')\rangle
\langle\delta_\ell(x),\delta_\ell(x')\rangle.
$$

Vector outputs, convolutions, tied weights, and attention require the corresponding
index contractions. This factorization is an exact local bridge from activation
geometry to credit/output geometry. It does not say that activation similarity
alone determines learning or that weights and activations are globally equivalent.

## 3. Local equivalence between intervention classes

Let $a$ and $b$ denote two intervention classes, such as LoRA and a residual-stream
shift. For a declared $b$ intervention $v$, define the best local match from class
$a$ by

$$
\epsilon_{a\leftarrow b}(v)
=
\min_u
\left\|
\mathcal T_a u-\mathcal T_bv
\right\|_2^2
+
\gamma u^\top M_a u.
$$

Report:

- normalized residual response energy;
- regularization and solver residual;
- canonical correlations or principal angles between the two tangent images;
- the matched intervention's finite predictive KL;
- performance on the fit bank, held-out in-domain bank, and OOD bank; and
- persistence after continued training.

Only a small residual plus finite nonlinear transfer supports bounded equivalence.
Matching on one prompt bank does not establish that a system prompt, activation
shift, adapter, and fine-tune are the same mechanism.

## 4. Task-conditioned plasticity at one checkpoint

Let $g_1$ be the new-task gradient in the declared intervention coordinates. Let
$M_0\succeq0$ represent local old-task damage, usually an old-task model Fisher,
GGN, PSD Hessian approximation, or an explicitly Euclidean cost. Let
$A_0=M_0+\lambda R\succ0$. Define

$$
\mathcal P_\lambda(g_1)
=
g_1^\top A_0^{-1}g_1.
$$

Under the local approximation, with $g_1=\nabla L_1$ and improvement measured as
$-\Delta L_1$,

$$
\Delta L_0
\approx
\frac12\delta u^\top A_0\delta u,
\qquad
\Delta L_1
\approx
g_1^\top\delta u,
$$

the constrained result is

$$
-\min_{\frac12\delta u^\top A_0\delta u\le\varepsilon}
g_1^\top\delta u
=
\sqrt{2\varepsilon\mathcal P_\lambda},
\qquad
\delta u^*
=
-\sqrt{\frac{2\varepsilon}{\mathcal P_\lambda}}A_0^{-1}g_1.
$$

The $\Delta L_0$ expression approximates actual old-task loss change only at an
old-task stationary point, or when the omitted old-task linear term is negligible.
Otherwise $\tfrac12\delta u^\top A_0\delta u$ is a declared quadratic damage proxy,
not the full old-loss change. The result is standard constrained quadratic
mathematics. The research contribution is to test whether it predicts finite
adaptation across tasks, checkpoints, and intervention classes better than simple
baselines.

For distribution-level response directions, solve

$$
G_{\mathrm{new}}v
=
\mu(G_{\mathrm{old}}+\lambda R)v.
$$

Use support-restricted or matrix-free solvers, report damping sensitivity, and do
not interpret $\mu$ as beneficial new-task improvement without a directional
objective.

## 5. From a response chart to a dynamical system

Let $x_k$ include every state variable needed to determine the next update:
parameters, momentum, adaptive moments, scheduler state if relevant, and mutable
normalization statistics. Let $u_k$ be an allowed adaptation input and $d_k$ an
exogenous disturbance such as a moving interface moment or task mixture. The exact
update is

$$
x_{k+1}
=
\Psi_k(x_k,u_k,d_k,\zeta_k).
$$

Linearizing along a reference trajectory gives

$$
\delta x_{k+1}
=
A_k\delta x_k+B_k\delta u_k+E_k\delta d_k+\xi_k+r_k,
$$

$$
\delta y_k
=
C_k\delta x_k+D^u_k\delta u_k+D^d_k\delta d_k+q_k.
$$

$\delta y_k$ is the predictive output whitened by the declared $W_k^{1/2}$.
$D^u_k$ and $D^d_k$ allow controls such as activation/soft-prompt interventions and
interface disturbances to affect the current output directly. $\xi_k$ is centered
stochastic forcing, while $r_k$ and $q_k$ are state/output linearization errors.
Keeping them separate prevents optimizer noise from being used as a synonym for
tangent-model failure.

### 5.1 Reachability and observability

For a time-invariant gate and zero direct term, define

$$
W_c(H)
=
\sum_{j=0}^{H-1}A^jBB^\top(A^j)^\top,
$$

$$
W_o(H)
=
\sum_{j=0}^{H-1}(A^j)^\top C^\top CA^j.
$$

This writes the control coordinates as cost-whitened. For a raw control metric
$M_u$, replace $BB^\top$ by $BM_u^{-1}B^\top$.

- $W_c$ depends on the allowed control/update coordinates and optimizer dynamics.
- $W_o$ depends on the retained-behavior probe and predictive metric.
- $C^\top C$ is the instantaneous state Fisher/GGN only after the output map is
  defined this way.
- Under the displayed timing, a one-step control has output kernel
  $C_{k+1}B_kM_u^{-1}B_k^\top C_{k+1}^\top$. An instantaneous parameter,
  activation, or prompt intervention has the kernel of its matched same-time
  coordinate-injection map. These recover the same weighted tangent kernel only
  when timing, injection, and metric conventions match.

Balanced or Hankel-like modes from $W_c$ and $W_o$ are optional theory targets:
they identify local state directions that are both reachable and consequential.
They should not be computed at LLM scale until the small-system estimators beat
directional plasticity and tangent-spectrum baselines.

### 5.2 Stability is not Fisher curvature

For plain gradient descent on a fixed local loss in suffix parameters $b$,

$$
b_{k+1}=b_k-\eta\nabla_bL(b_k,\lambda_k),
$$

so

$$
A=I-\eta H_{bb},
\qquad
B=-\eta H_{b\lambda}.
$$

If $H_{bb}$ is symmetric positive definite, the frozen linear system is stable for
$0<\eta<2/\lambda_{\max}(H_{bb})$. Flat directions give semistability; negative
curvature can produce instability. Momentum, adaptive preconditioning, alternating
updates, and time variation enlarge the state and can introduce non-normal
transient growth even when pointwise eigenvalues look benign.

For the linear time-varying system, define

$$
\Phi(k,j)
=
A_{k-1}A_{k-2}\cdots A_j.
$$

Uniform exponential stability over the full declared time domain requires constants
$M<\infty$ and $0<\rho<1$ such that

$$
\|\Phi(k,j)\|
\le
M\rho^{k-j}
$$

for every admissible $k\ge j$ in the declared norm. Pointwise spectral radii of
$A_k$ do not imply this when the matrices vary or do not commute. On a finite
experimental window, report the transition-product gains
$\max_{j<k}\|\Phi(k,j)\|$ and finite-horizon contraction/amplification directly;
do not promote an inequality fit on that finite window to uniform exponential
stability.

Under this condition, a standard input-to-state bound is

$$
\|\delta x_k\|
\le
M\rho^k\|\delta x_0\|
+
M\sum_{j=0}^{k-1}
\rho^{k-1-j}
\left(
\|B_j\delta u_j\|
+
\|E_j\delta d_j\|
+
\|\xi_j\|
+
\|r_j\|
\right).
$$

The bound makes the empirical burden explicit: even a stable suffix can exhibit
large error under strong/fast forcing, weak contraction, transient amplification,
or large tangent remainder.

### 5.3 Fisher-weighted separation is not automatically a Lyapunov function

For $V_k=\delta x_k^\top P_k\delta x_k$,

$$
V_{k+1}-V_k
=
\delta x_k^\top
\left(A_k^\top P_{k+1}A_k-P_k\right)
\delta x_k
+
\text{input and cross terms}.
$$

Calling $V$ Lyapunov requires $P_k\succ0$ and a verified negative drift or
dissipativity inequality. Choosing $P_k=F_k$ is not sufficient: neural-network
Fishers are often singular, change with the probe/predictions, and measure output
visibility rather than state contraction. Still report Fisher-weighted predictive
separation as an output-energy observable, under that name.

## 6. Moving interfaces as a disturbance-rejection problem

At cut $\ell$, write the suffix logits as $f_b(z)$ and introduce a validated
interface transport $T_\lambda(z)$. Around a fixed anchor,

$$
\delta b_{k+1}
=
A\delta b_k+B\delta\lambda_k,
$$

$$
\delta f_k
=
J_b\delta b_k+J_\lambda\delta\lambda_k.
$$

For the level input $\delta\lambda_k$, the local transfer function is

$$
T(z)
=
J_\lambda
+
J_b(zI-A)^{-1}B.
$$

For plain gradient descent,

$$
T(1)
=
J_\lambda
-
J_bH_{bb}^{-1}H_{b\lambda},
$$

when the inverse and equilibrium exist.

This exposes an important distinction in the current Part-D language.

### 6.1 Geometric compensation versus optimizer DC response

The joint model-Fisher/GGN blocks are

$$
G
=
\begin{pmatrix}
G_{bb} & G_{b\lambda}\\
G_{\lambda b} & G_{\lambda\lambda}
\end{pmatrix}.
$$

For $R\succeq0$, define the regularized geometric problem

$$
\min_{\delta b}
\left\|\mathcal T_b\delta b+\mathcal T_\lambda\delta\lambda\right\|_2^2
+
\gamma\delta b^\top R\delta b.
$$

Its optimizer is

$$
\delta b_F
=
-(G_{bb}+\gamma R)^{-1}G_{b\lambda}\delta\lambda,
$$

with minimized penalized quadratic

$$
G_{\mathrm{pen}}
=
G_{\lambda\lambda}
-
G_{\lambda b}(G_{bb}+\gamma R)^{-1}G_{b\lambda}.
$$

For $\gamma=0$, when the inverse or declared pseudoinverse is well defined, this is
also the residual predictive-output metric. For $\gamma>0$, the pure residual
output metric is instead

$$
G_{\mathrm{out}}
=
G_{\lambda\lambda}
-2G_{\lambda b}QG_{b\lambda}
+G_{\lambda b}QG_{bb}QG_{b\lambda},
\qquad
Q=(G_{bb}+\gamma R)^{-1},
$$

equivalently
$G_{\mathrm{out}}=G_{\mathrm{pen}}-
\gamma G_{\lambda b}QRQG_{b\lambda}$ under the stated symmetric quadratic
conventions.

Store $G_{\mathrm{pen}}$ and $G_{\mathrm{out}}$ separately; the former includes the
regularization cost.

The optimizer's quasi-static equilibrium response is instead

$$
\delta b_H
=
-H_{bb}^{-1}H_{b\lambda}\delta\lambda.
$$

These coincide only when the relevant Hessian and Fisher/GGN compensation maps
agree, with compatible damping/regularization. Therefore:

- $\delta b_F$ and $G_{\mathrm{out}}$ answer **how much predictive disturbance is
  left by the declared regularized geometric compensation?**
- $G_{\mathrm{pen}}$ answers **what is the optimal output-error/intervention-cost
  tradeoff under that regularizer?**
- $T(1)$ answers **what cancellation does the declared optimizer converge to in
  the frozen local system?**
- $T(e^{i\omega})$ answers **what cancellation is realized at frequency
  $\omega$?**

The Fisher Schur complement must not be called the optimizer's zero-frequency
response without this agreement check.

### 6.2 Moving-optimum coordinates

If

$$
b^*(\lambda)
=
\arg\min_bL(b,\lambda),
\qquad
K
=
\frac{db^*}{d\lambda}
=
-H_{bb}^{-1}H_{b\lambda},
$$

and $e_k=b_k-b^*(\lambda_k)$, then

$$
e_{k+1}
\approx
Ae_k-K\Delta\lambda_k+\xi_k+r_k.
$$

The input is now moment *increment* rather than moment level. This form is useful
for tracking-error bounds, but it should not be mixed algebraically with the level
input transfer function without declaring the convention.

The displayed derivative requires an isolated differentiable optimum branch and
invertible $H_{bb}$. If $H_{bb}$ is singular, a pseudoinverse supplies a declared
minimum-norm projected sensitivity only under explicit range, gauge, and branch
assumptions; it is not an unconditional implicit-function derivative.

### 6.3 Frequency-domain burden

For an internally stable LTI approximation, fixed whitening/output coordinates,
and a zero-mean stationary disturbance with declared spectral density
$S_d(\omega)$, the local output energy is

$$
\mathbb E\|\delta y\|^2
\approx
\frac{1}{2\pi}
\int_{-\pi}^{\pi}
\operatorname{tr}
\left[
T(e^{i\omega})
S_d(\omega)
T(e^{i\omega})^*
\right]
d\omega.
$$

This is the precise control-theoretic version of “Fisher-weighted motion”: motion
matters according to its amplitude and spectrum, the direction's predictive
visibility, and the suffix transfer response. $H_\infty$-style worst-case gain and
$H_2$-style noise energy are possible summaries only after the local system and
input normalization pass finite-response checks.

## 7. A tangent atlas across training

At checkpoints $t_0<t_1<\cdots<t_m$, keep the same probe examples, output mask,
intervention coordinates, and metric conventions. Store the local chart

$$
\mathfrak a_t
=
\left(
J_t,
G_t,
K_t,
A_t,
B_t,
C_t,
\mathcal V_t
\right),
$$

where $\mathcal V_t$ contains empirical validity diagnostics.

### 7.1 Three meanings of “more linear”

Do not collapse these into one curve.

1. **Amplitude linearity:** the Taylor approximation remains accurate over a larger
   Fisher/output-normalized intervention radius.
2. **Temporal linearity:** a frozen checkpoint chart predicts more optimizer steps.
3. **Frame stationarity:** the weighted tangent image and important response modes
   rotate more slowly across checkpoints.

A network can have a large instantaneous linearity radius while its Jacobian drifts
quickly under training, or a stable tangent frame with a small finite-amplitude
radius.

### 7.2 Required validity statistics

For a perturbation distribution $\nu_r$ at radius $r$, record a robust relative
output error such as

$$
E_{\mathrm{Taylor}}(r)
=
\operatorname{median}_{\delta u\sim\nu_r}
\frac{
\|W^{1/2}(s(u+\delta u)-s(u)-J_u\delta u)\|_2
}{
\|W^{1/2}(s(u+\delta u)-s(u))\|_2+\tau
}.
$$

Define $r_\varepsilon$ as the endpoint of the largest contiguous-from-zero
preregistered radius interval passing both the median and upper-quantile error
thresholds. Also measure:

- finite predictive KL versus one half of the quadratic Fisher prediction;
- superposition defect for pairs of interventions;
- frozen-tangent rollout error at horizons $1,2,4,8,16,32$;
- relinearized-tangent rollout error at the same horizons;
- raw logit-Jacobian image drift, drift under a fixed reference metric
  $W_{\mathrm{ref}}$, and contemporaneously weighted tangent-image
  projector distance/principal angles;
- drift of $W_t$ and any intervention-cost metric $M_t$ as separate objects; and
- normalized kernel alignment on the identical probe bank.

Low-response directions need an absolute-error floor; otherwise the relative ratio
is ill-conditioned. The output Fisher itself can collapse under confident softmax
predictions, so retain unweighted logit and loss views as diagnostics.

### 7.3 Frozen versus relinearized versus nonlinear

Every forked rollout should compare:

1. actual nonlinear fork;
2. fixed chart at the fork checkpoint;
3. chart sequence measured along the unperturbed reference trajectory; and
4. a simple constant-output or loss-linear baseline.

Reference and fork must use common random numbers: identical minibatch order,
augmentation and dropout masks, stochastic-layer seeds, and any sampled forcing.
Alternatively use a deterministic full-batch gate. Condition the tangent predictor
on the realized noise/input sequence and store batch and RNG identities; otherwise
ordinary stochastic divergence is confounded with tangent-model error.

Interpretation:

- fixed and relinearized both work: stable local response regime;
- only relinearized works: smooth chart drift is essential;
- neither works at small amplitude: nonlinear response, optimizer-state omission,
  stochastic coupling, or numerical failure;
- both work on the fit bank but not OOD: probe-conditioned local validity only.

The difference between predictor errors is not an exact causal decomposition of
feature learning.

## 8. Hypotheses and falsifiers

### H0. Dual-Gram implementation

**Mechanical prediction:** matrix-free Fisher/GGN and weighted output-tangent
operators agree on nonzero spectral probes and bilinear forms within tolerance.

**Failure:** JVP/VJP, masking, normalization, tied weights, or metric implementation
is wrong. No scientific interpretation is permitted.

### H1. Tangent-atlas maturation

**Hypothesis:** one or more of $r_\varepsilon$, frozen-chart prediction horizon, or
frame stationarity increases during training, possibly non-monotonically and by
layer.

**Falsifier:** no seed-robust change, or any apparent change disappears after
output-amplitude, confidence, and optimizer-step normalization.

### H2. Prospective plasticity

**Hypothesis:** task-conditioned tangent/Fisher quantities measured before
adaptation predict new-task gain and old-task damage across checkpoint, task, and
intervention class better than simple baselines.

**Falsifier:** no held-out predictive improvement, strong damping/probe sensitivity,
or only retrospective fit.

### H3. Layer tracking is bandwidth-limited

**Hypothesis:** a validated local transfer function predicts step recovery and the
onset of lag under periodic moment forcing.

**Falsifier:** shrinking-amplitude responses do not approach the tangent prediction,
or measured lag is better explained by changing forcing amplitude, mutable state,
or surrogate failure.

### H4. Critical periods are a loss of reachable low-damage directions

**Hypothesis:** the same capability-relevant late intervention has a persistent
deficit predicted by the pre-intervention reachability/old-observability geometry,
and a targeted rescue reopens it.

**Falsifier:** late training catches up under matched optimization, the geometry
does not predict the deficit, or a generic extra-budget control explains the rescue.

## 9. Experimental ladder

### Gate -1. Resolve the existing projected-subspace confound

This is not a tangent experiment, but it is the cheapest way to make current B/C
results scientifically legible.

Add a deterministic projected-true activation rung

$$
z_{\mathrm{proj}}
=
((z-\mu)V^\top)V+\mu
$$

and a Gaussian-versus-Gaussian finite-sample diagnostic floor. The four scientific
rungs are full true, projected true, Gaussian, and mean; independent Gaussian draws
are diagnostic replicates rather than an unlabeled fifth rung. Rerun only:

- bounded residual CNN: epochs 1 and 30 at cut 3;
- ResNet-18: epochs 0 and 100 at cuts 0, 4, and 7.

Fit $\mu,V$ on the fit split only. Define held-out pooled PCA coverage as

$$
\frac{\sum_i\|(z_i-\mu)V^\top\|_2^2}
{\sum_i\|z_i-\mu\|_2^2},
$$

and report a per-example distribution secondarily. Compare two independent
Gaussian draws both in fitted-coordinate moments and in downstream suffix
train/evaluate outcomes. Reuse identical checkpoint bytes across cuts; the bounded
CNN therefore needs a train-once/load-many-cuts path rather than its current
cut-specific checkpoints. Measure direct PCA-coordinate error, update-0 suffix/logit
parity, and the full $4\times4$ cross-evaluation matrix.

Mechanical tests cover projection idempotence, retained-coordinate equality,
matrix-key completeness, update-0 parity, and cross-cut checkpoint-hash equality.
This gate separates discarded-subspace utility from within-subspace
misspecification. Stop broad B/C expansion if the sentinel result remains
unidentifiable.

### Gate 0. Numerical tangent primitives

Freeze a CPU/float64 classifier fixture, fixed batch, parameter initialization,
probe bank, and seeds. Implement the model and plain-SGD update as pure functions
with a documented state flattening. Apply categorical $W$ implicitly through
matrix-vector products; do not require an explicit $W^{1/2}$ at scale. On this
fixture:

1. JVPs match centered finite differences at shrinking amplitudes.
2. VJPs pass adjoint tests.
3. $2D_{\mathrm{KL}}/\epsilon^2$ converges to the model-Fisher/GGN directional
   quadratic.
4. $\mathcal T^\top\mathcal T$ and $\mathcal T\mathcal T^\top$ spectral probes
   agree.
5. Parameter, activation, and LoRA coordinate maps reproduce direct interventions
   at first order.
6. Linearization of one optimizer step matches a finite fork, including optimizer
   state.
7. Frozen and relinearized rollout code closes on an exactly linear model.

Default mechanical tolerances are: VJP adjoint and dense/implicit Gram relative
error $\le10^{-10}$; pure-functional versus framework SGD and exact-linear closure
$\le10^{-10}$; centered-JVP minimum relative error $\le10^{-7}$ over a declared
shrinking grid; and finite-KL relative error $\le10^{-3}$ in at least two
consecutive radii before roundoff dominates. Record absolute errors and denominators
and fail low-signal cases rather than dividing through them. A platform-specific
tolerance change requires a recorded calibration artifact, not an inline edit.

### Gate 1. Static synthetic validity

Use Part D's normalization-free architecture, cuts, optimizer, and schedules as the
starting point. Its current prose does **not** yet fully specify a data generator.
Before implementation, add a generator/transport appendix that pins exact sampling
equations, normalized Hermite convention, channel allocation, label conditional,
Bayes-advantage calibration objective, train/calibration/probe splits, nuisance law,
mean and order-4 transport equations, common-random-number coupling across
$\lambda$, support policy, and numerical lower-moment leakage thresholds. Gate 1
is blocked—not silently tunable—until that appendix exists.

Initially run only:

- one class-independent mean nuisance transport;
- one task-relevant order-4 transport;
- cuts after early, middle, and late residual blocks; and
- one phenomenon-gate seed after unit/smoke tests.

For each transport:

- validate target moment motion and lower-order preservation on held-out samples;
- compare the signed finite logit response with $J_\lambda$ and the predictive KL
  with $\tfrac12\delta\lambda^\top G_{\lambda\lambda}\delta\lambda$ at three
  shrinking amplitudes;
- compute damped Fisher/GGN compensation;
- compute the linearized optimizer-equilibrium response when it exists;
- compare both with an actual short suffix refit; and
- expose when Fisher compensation and optimizer equilibrium disagree.

The primary short refit uses the anchor optimizer and learning rate with identical
batches/RNG, recording updates 1, 4, and 16; a 64-update equilibrium diagnostic is
secondary. Because $H_{bb}$ can be indefinite or singular, the optimizer-step
Jacobian is primary. An equilibrium comparison is valid only with a stable local
update, a declared damped MINRES/trust-region or projected solver, relative residual
$\le10^{-6}$, and explicit negative-curvature handling. Otherwise store it as
undefined rather than forcing a CG/inverse result.

Do not proceed if the transport violates its preregistered leakage/support checks,
finite KL does not approach the quadratic, or solver/damping choices change the
qualitative ordering.

### Gate 2. One-seed step, reversal, and anchored return

Test the control-theoretic centerpiece on the same validated Hermite system before
building a broad atlas. At one exact stationary checkpoint, define the primary
state as suffix parameters under plain SGD. Treat the scheduler, ordered minibatches,
and sampled transport as known exogenous inputs. Store the unperturbed reference
trajectory used to evaluate $A_k,B_k,C_k,D^u_k,D^d_k$.

Drive the two validated moment channels with:

- a static offset;
- an abrupt step;
- a reversal;
- an anchored $A\rightarrow B\rightarrow A$ sequence; and
- zero-amplitude, frozen-suffix, stationary, and oracle-recentered controls.

Always evaluate current and previous environments. For signed gain/phase, use a
preregistered scalar readout: the output response projected onto the normalized
direct forcing-response direction at the anchor. Report vector predictive-output
energy separately. Fit or differentiate the local state-space system before
revealing the full nonlinear response. Compare immediate disturbance,
finite-horizon transition-product gain, recovery time, steady residual, and
anchored retention.

Gate 4 unlocks only if, at the two smallest nonzero step amplitudes, median tangent-
trajectory relative error is $\le0.20$, the 90th percentile is $\le0.35$, halving
amplitude changes normalized gain by $\le10\%$, and no declared transition-product
gain is nonfinite or exceeds the preregistered safety cap. These are finite-window
operational checks, not proof of uniform stability.

### Gate 3. Slim tangent atlas, then five-seed confirmation

Separate two questions:

1. **maturation:** does local validity change on the current training task?
2. **plasticity:** does a pre-fork score predict transfer from a declared old task
   A to new task B?

Before the second claim, add an A/B protocol fixing the old and new data laws,
held-out gradient/probe banks, adaptation stream, update/token budget, new-task
endpoint, and retained-task forgetting metric.

The implementation benchmark is one seed, two sentinel cuts, checkpoints
$0,1,10,30$, and only:

- the actual next-interval update, used solely for retrospective chart validity;
- a held-out current-task gradient; and
- module- and metric-matched random directions.

Use fixed radii and horizons $1,2,4,8,16$. Do not compute generalized or trailing
modes in this benchmark. Save a complete continuation bundle: model, optimizer,
scheduler, RNG states, batch cursor, and ordered future-batch hashes. The nonlinear
fork and reference share the realized future batches and RNG; the frozen predictor
uses $A_0,C_0$ throughout, while the relinearized predictor uses $A_k,C_k$ along the
unperturbed reference.

If this benchmark is valid and discriminative, freeze its cells and expand to five
independently trained seeds, shared checkpoint bytes at dense early updates and
epochs $0,1,5,10,20,30$, and the separately declared A/B task. Only then add top-
$k$ weighted-tangent or generalized modes above the numerical floor. Predeclare
$k$, solver tolerance, spectral-gap rule, and captured response energy; literal
trailing modes in singular Fishers are not a target.

Primary normalization is predicted output KL/Fisher norm, with parameter norm and
actual finite KL retained. Report $r_\varepsilon$, frozen/relinearized prediction
horizons, raw-$J$/fixed-$W$/current-$W$ drift, separate metric drift,
superposition error, and actual A/B outcomes. This is the direct experiment for the
three distinct meanings of “more linear over time.”

### Gate 4. Frequency response

Unlock only if Gate 2 passes its finite-radius linearity and finite-horizon gain
criteria.
Use the periods and amplitude checks already preregistered in Part D. For a MIMO
response, predeclare scalar readouts or singular-value/cross-spectral summaries
before reporting gain, phase, and coherence. Also report recovery time and
cross-environment loss. Compare the measured response with:

- frozen suffix response;
- the Fisher-optimal static compensation as a geometric lower-bound/baseline, not
  as a frequency-dependent transfer function;
- Hessian/optimizer DC response; and
- the full tangent state-space prediction.

The comparison, not a raw curvature spectrum, determines bandwidth.

### Gate 5. Noise causality

Fork multiple replicas from identical checkpoint and optimizer-state bytes. Compare
full batch, ordinary minibatches, deterministic cycling, and calibrated centered
noise. If feasible, rotate noise between behaviorally visible/reachable and matched
orthogonal directions at fixed covariance trace or robust scale.

Separate mean drift from centered diffusion. Test whether observed output energy is
predicted by the transfer-filtered disturbance covariance. The archived empirical
Fisher is not an allowed noise-covariance estimator.

### Gate 6. Prospective GPT-2 adaptation

There is currently no GPT-2 model, tokenizer, dataset, LoRA stack, or transformer
artifact in this repository. Gate 6a is therefore an engineering prerequisite, not
an assumed dependency. Prefer consuming the hashed full-parameter `P` endpoint from
Part E. Record exact source/revision and licenses, tokenizer files, raw/processed
data hashes, token-bank manifests, checkpoint/optimizer hashes, dependency lock,
hardware, peak memory, directional-JVP cost, and a smoke adaptation. If a released
checkpoint is used instead, assign a separate study ID and state that it does not
instantiate Part E.

The first scientific pilot uses one exact GPT-2-small/TinyStories anchor and one
deterministic TinyStoriesInstruct stream, not the full five-distribution Part-E
battery.

Intervention classes:

- full parameters;
- the last two transformer blocks; and
- only after the Part-E full-parameter gate succeeds, LoRA with a separately pinned
  target/rank/scaling and the nondegenerate tangent-coordinate convention in the
  chart registry.

Soft-prompt and residual-stream classes are expansions, not pilot requirements.
Pin adaptation learning rates, probe/mask sizes, token/update budgets, continuation
length, and OOD manifest in the Gate-6 run protocol. Restrict the pilot to
directional matrix-free quantities; a full GPT-2 tangent spectrum is out of scope.

Before adaptation, estimate task-gradient capture, old/new directional plasticity,
weighted tangent-image overlap, early tangent-predicted learning, and finite-radius
validity. Then run matched adaptation streams and measure:

- instruction held-out loss/behavior;
- retained plain-story loss;
- OOD prompt transfer;
- actual versus fixed/relinearized tangent trajectories;
- output-subspace and Jacobian rotation; and
- persistence after a short plain-continuation phase.

The first gate is a prospective one-cell case study, not predictor validation or a
critical-period result. For an LLM-centered headline, freeze a confirmatory matrix
with at least two checkpoints, three adaptation task families, and three
intervention classes across independent training seeds where compute permits. One
task family must be an explicit alignment-relevant adapt/retain pair, ideally
shared with Brianna's static case study.

Predeclare $\mathcal P_\lambda(g_{\mathrm{new}})$, with damping and probe choices
selected on calibration cells, as the primary score; use new-task loss decrease at
a fixed processed-token/update budget as the primary outcome and retained-bank
damage as the safety endpoint. Hold out task-family $\times$ seed cells for the
final ranking/calibration test. If this matrix is not run, label the LLM result a
pilot and keep the paper title/generalization claims correspondingly broad.

## 10. Measurements and baselines

### Direction families

- new-task gradient and natural-gradient-like direction;
- actual future update, used only for retrospective chart validation and excluded
  from prospective fitting/selection;
- generalized old/new Fisher modes;
- weighted-tangent singular directions;
- random directions matched by module and metric norm;
- activation/LoRA matched-response solutions; and
- moment transports with held-out fidelity checks.

### Outcome metrics

- predictive KL and cross-entropy in nats per example/token;
- new-task gain at fixed updates and fixed processed tokens;
- retained-task damage and OOD damage;
- updates to a preregistered criterion;
- Taylor and rollout error;
- intervention persistence/decay rate;
- step recovery time and periodic gain/phase/coherence; and
- tangent-frame rotation/principal angles.

### Required simple baselines

- current loss/perplexity;
- gradient norm and gradient cosine;
- parameter/update norm;
- trainable parameter count/rank;
- Fisher/GGN trace and top eigenvalue;
- linear probe or representation similarity where meaningful;
- last-layer/frozen-feature adaptation;
- directly trained frozen linearized model; and
- a constant-Jacobian or constant-output dynamical predictor.

## 11. Statistical design

- Model seeds are the independent units for small-model claims.
- Prompt rows, probe examples, cuts, checkpoints, directions, surrogate draws, and
  fork replicas are repeated measurements, not independent trained models.
- Fit thresholds, damping, probe banks, and direction seeds before viewing outcomes.
- Use held-out task/intervention cells for the prospective prediction test.
- Report both rank prediction and calibrated outcome prediction.
- Hierarchical analyses may use within-seed cells, but uncertainty on the headline
  must aggregate at model seed.
- A one-checkpoint LLM gate is a phenomenon demonstration. Do not attach a
  critical-period or universal post-training claim to it.

## 12. Artifact contract

Write new outputs under `artifacts/tangent_model/<run_id>/`. Do not reuse the
archived empirical-Fisher fields.

- `manifest.json`: schema version/status, mandatory source-bundle file/hash
  manifest, model/checkpoint hashes, optimizer-state hash, probe hashes,
  intervention maps, output metric, normalization, dtype, hardware/runtime, and
  exclusions. A git commit or diff hash alone is insufficient in the current dirty
  worktree.
- `chart_registry.json`: the exact coordinate, metric, injection, state, and probe
  declarations from Section 1.1.
- `schemas/*.json`: versioned field enums, primary keys, units, nullability, array
  sidecar references, and one-row-per-angle/eigenvalue conventions.
- `static_response.parquet`: checkpoint, cut, intervention class/direction, radius,
  predicted/actual logit response, finite KL, Fisher quadratic, Taylor error, and
  solver diagnostics.
- `tangent_atlas.parquet`: checkpoint pairs, fixed/relinearized horizons, frame
  drift, principal angles, kernel alignment, and validity thresholds.
- `state_space.parquet`: $A/B/C/D^u/D^d$ operator probes,
  stability/transient-growth
  summaries, state definition, and finite-step validation.
- `forcing_response.parquet`: step/sine condition, input convention, amplitude,
  frequency, gain, phase, coherence, recovery, and current/previous environment
  losses.
- `plasticity_prediction.parquet`: pre-adaptation scores/baselines and held-out
  realized gain, forgetting, persistence, and calibration split.
- `intervention_equivalence.parquet`: source/target class, fit/held-out residual,
  subspace overlap, actual finite transfer, and persistence.
- `vectors/*.npz`: hashed response vectors, operator probes, or spectra referenced
  by scalar table rows; do not place an unspecified vector into one Parquet cell.
- `_SUCCESS.json`: written atomically only after the fail-closed verifier passes,
  containing verifier version and hashes of every required file.

Every row must carry `run_id`, model seed, exact checkpoint and optimizer-state
hashes, probe bank, intervention-map identifier, output-metric identifier, and
epistemic status.

Implement `scripts/verify_tangent_artifacts.py` before a scientific run. It checks
schema/status compatibility, finite values, unique primary keys, requested-cell
completeness, source/checkpoint/optimizer/probe hashes across files, sidecar hashes,
solver validity and tolerance fields, and the atomic completion marker. Mock,
smoke, failed, exploratory, and confirmatory rows must fail closed when mixed.

## 13. Mechanical acceptance

1. JVP finite differences and VJP adjoint tests pass at shrinking amplitudes.
2. One half of the model-Fisher/GGN directional quadratic matches finite predictive
   KL over a declared local range.
3. Parameter- and output-space Gram probes share their nonzero spectrum within
   tolerance.
4. Every compared cut/checkpoint uses the intended identical model bytes.
5. Optimizer-state linearization includes all mutable state and predicts a one-step
   fork.
6. Frozen and relinearized simulators close exactly on a linear positive control.
7. Fisher-optimal compensation and Hessian/optimizer equilibrium are stored under
   different field names and compared rather than conflated.
8. No stability claim is made from Fisher eigenvalues alone.
9. No bandwidth claim is made before step response, amplitude linearity, and
   coherence pass.
10. No critical-period claim is made without a persistent matched late deficit and
    a prospective/rescue test.
11. Mock, smoke, failed, exploratory, and confirmatory artifacts fail closed when
    mixed.
12. Headline uncertainty uses independently trained seeds or is explicitly labeled
    a one-model phenomenon gate.
13. Every rollout fork has matching batch/RNG identities or is a deterministic
    full-batch condition.
14. Tangent-frame claims report raw-$J$, fixed-$W_{\mathrm{ref}}$, current-$W_t$,
    and separate metric drift.
15. The chart registry, source bundle, schemas, sidecars, and completion marker pass
    `verify_tangent_artifacts.py`.

## 14. Recommended execution order

1. Add projected true and the diagnostic floor at sentinel B/C cells.
2. Implement tangent/Fisher/pure-optimizer primitives and the artifact verifier.
3. Finish the Hermite generator/transport appendix and run one-seed static validity.
4. Run one-seed step/reversal/anchored-return forcing on that system.
5. Run the slim one-seed tangent-atlas benchmark.
6. Freeze discriminative cells and run five-seed confirmation.
7. Build and benchmark the chosen GPT-2/Part-E anchor path.
8. Run the one-checkpoint prospective LLM pilot.
9. Choose among the confirmatory LLM matrix, frequency response, and noise causality
   according to which earlier gate is valid and discriminative; the LLM matrix is
   mandatory for an LLM-centered headline.
10. Attempt a critical-period rescue only after a pre-adaptation predictor works.

Do not spend the next tranche finishing 20 Part-A seeds, launching a broad Bode
battery, or running the full Part-E surrogate matrix before these gates. The current
11 Part-A seed files already establish robustness of that bounded pattern; the new
information lies in identifiability and prospective response.

## Primary-source starting points and access note

Local full-text copies already present in `papers/` include Martens on natural
gradient/model Fisher, Kunstner et al. on empirical-Fisher limitations, Achille et
al. on critical periods, and Zhao et al. on dynamic regret. They were available for
this design, but this document is not a line-level citation audit of every theorem.

Additional primary sources to audit before publication:

- Arthur Jacot, Franck Gabriel, and Clément Hongler, [*Neural Tangent Kernel:
  Convergence and Generalization in Neural Networks*](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html).
- Jaehoon Lee et al., [*Wide Neural Networks of Any Depth Evolve as Linear Models
  Under Gradient Descent*](https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html).
- Wesley Maddox et al., [*Fast Adaptation with Linearized Neural
  Networks*](https://proceedings.mlr.press/v130/maddox21a.html).
- Daniel LeJeune and Sina Alemohammad, [*An Adaptive Tangent Feature Perspective of
  Neural Networks*](https://proceedings.mlr.press/v234/lejeune24a.html).
- Chaoyue Liu, Libin Zhu, and Misha Belkin, [*On the Linearity of Large Non-linear
  Models*](https://proceedings.neurips.cc/paper/2020/hash/b7ae8fecf15b8b6c3c69eceae636d203-Abstract.html).
- Michael Kleinman, Alessandro Achille, and Stefano Soatto, [*Critical Learning
  Periods Emerge Even in Deep Linear Networks*](https://openreview.net/forum?id=Aq35gl2c1k).

The February 2026 preprint [*Linearization Explains Fine-Tuning in Large Language
Models*](https://arxiv.org/abs/2602.08239) is a direct novelty-overlap risk for the
LLM/LoRA branch. Only its abstract was inspected here. Read the full paper before
claiming that LLM fine-tuning linearization or NTK spectra predict adaptation as a
new contribution. The likely distinct contribution must be prospective old/new
task geometry, multiple intervention classes, time-varying stability, moving
interfaces, or a validated failure boundary.
