# Tracking statistical structure through a deep network

Status: Experiment A and the bounded residual-CNN version of Experiment B are
implemented on CIFAR-10. Experiment C now repeats B's frozen-interface
activation-statistics experiment on ResNet-18 and VGG-19 across depth and
training time. The first C gate uses one training seed and three surrogate draws;
critical-module transplantation is supporting context at the bottom of C, not
its organizing hypothesis. Part D (tangent stability and suffix frequency
response) and Part E (the GPT-2 sequence-statistics extension) are specified but
unrun.

## 1. Core question

During training, a deep network must do two things at every cut layer:

1. improve the downstream map for the representation it currently receives; and
2. keep that downstream map adapted while the upstream representation moves.

The proposed program asks whether the second burden increasingly dominates in
later hidden blocks, and whether the transition is related to (i) the progressive
use of higher-order distributional structure, (ii) loss of plasticity, and (iii)
the stochastic dynamics of SGD.

The main contribution should be a measurable **moving-optimum decomposition**, not
merely another sequence of probes. Cumulant-controlled data and periodic forcing
are complementary interventions with which to identify that decomposition.

## 2. Important correction: what is a cumulant surrogate?

Let \(P(x,y)\) be the true distribution. The relevant statistics are
**class-conditional** cumulants of a chosen feature vector \(z=T(x)\), not
unconditional cumulants of raw pixels. Unconditional matching can destroy the
class signal while still satisfying the nominal constraint.

The phrase "preserve only cumulants through order \(r\)" cannot in general mean
setting every higher cumulant to zero. A distribution with an exactly finite
cumulant-generating polynomial above degree two generally need not exist. Nor is
the full order-\(r\) cumulant tensor of a 3,072-dimensional CIFAR image estimable.

Use one of two well-defined constructions:

- **Maximum-entropy surrogate** \(P_r\): among distributions on a fixed bounded
  support matching selected class-conditional moments through degree \(r\), choose
  the maximum-entropy distribution. This removes structure not forced by the
  constraints without claiming its higher cumulants literally vanish.
- **Cumulant-controlled generator** \(G_{\le r}(u,y)\): generate from Gaussian
  latents using a Hermite expansion whose coefficients provide controlled
  low-order statistical channels. This gives better causal control but is a model
  family, not a unique projection of CIFAR.

Call these *order-\(r\) surrogates*, not datasets preserving "only" order-\(r\)
cumulants.

## 3. One shared experimental object

Construct a family \(P_{\boldsymbol\lambda}\) where independent coordinates of
\(\boldsymbol\lambda=(\lambda_1,\ldots,\lambda_R)\) control label information in
statistics of increasing Hermite degree. A useful binary task is

$$
z = \sum_{k=1}^{R}\lambda_k a_k(y) H_k(g_k)v_k + \epsilon,
\qquad g_k\sim\mathcal N(0,1),
$$

with orthogonal directions \(v_k\), centered/normalized Hermite polynomials
\(H_k\), and nuisance noise \(\epsilon\). In practice the exact construction must
be checked empirically because nonlinear mixing can create cross-cumulants.

Two variants are scientifically useful:

- **independent channels:** separate \(g_k,v_k\), testing pure order-wise sample
  complexity;
- **aligned/correlated channels:** shared or correlated latents/directions,
  testing the "sliding down the stairs" acceleration mechanism.

This same \(P_{\boldsymbol\lambda}\) supports static truncation, abrupt switches,
and sinusoidal forcing. That makes the four ideas parts of one project.

## 4. Experiment A: distributional-complexity learning

### A1. Clean causal benchmark (first experiment)

For \(r=1,2,3,4\), define \(P_{\le r}\) by activating only channels through degree
\(r\). Train identical small MLPs and CNNs on each distribution. At logarithmically
spaced checkpoints, evaluate every model on every test distribution:

$$
M_{s,r}(t)=\mathbb E_{(x,y)\sim P_{\le s}}
  \ell(f_{\theta_r(t)}(x),y).
$$

The user's two requested curves are slices of this matrix:

- \(L_{\mathrm{true}|r}(t)=M_{R,r}(t)\): true/full-distribution loss for a model
  trained on order-\(r\) data;
- \(L_{r|\mathrm{true}}(t)=M_{r,R}(t)\): order-\(r\) loss for a model trained on
  full data.

Plot the whole matrix, because the two slices alone conflate transfer asymmetry,
task difficulty, and distribution shift. Also report excess loss relative to a
model trained and tested on the same distribution:

$$
\Delta_{s|r}(t)=M_{s,r}(t)-M_{s,s}(t).
$$

Primary tests:

- Does the time at which \(L_{r|\mathrm{true}}\) improves increase with \(r\)?
- Does alignment between low- and high-order channels reduce that delay?
- Does training on \(P_{\le r}\) help on \(P_R\) beyond a matched sample/entropy
  baseline?

### A2. Real-data extension (not the starting point)

Use CIFAR-10 only after A1 works. Compare:

1. class-conditional Gaussian surrogates in a reduced feature space (mean and
   covariance; the established \(r=2\) case);
2. maximum-entropy surrogates matching selected cubic/quartic statistics; and
3. a generator with a low-dimensional Hermite bottleneck fitted to CIFAR.

Choose \(T(x)\) before looking at outcomes: e.g. PCA-whitened pixels, fixed
wavelet coefficients, or frozen random-convolution features. Wavelets are the
preferred first choice because they preserve spatial scale/locality while keeping
the statistic set manageable. Run sensitivity across at least two choices of
\(T\); conclusions are about the selected statistic family, not "all CIFAR
cumulants."

Validate every surrogate with held-out estimates of the constrained moments,
unconstrained moments, support/range, class balance, and a discriminator trained to
separate surrogate from real data.

## 5. Experiment B: which statistics of a frozen representation can the suffix use?

The primary Part B question now mirrors Experiment A at an internal interface.
At checkpoint (t) and cut (ell), freeze

$$
z=\phi_{t,\ell}(x), \qquad f_t(x)=\psi_{t,\ell}(z).
$$

Fit three class-conditional distributions to training representations (z):

- **true:** the empirical frozen-prefix activations;
- **mean:** class means plus class-independent isotropic nuisance in a declared
  PCA subspace;
- **Gaussian:** class-conditional means and covariances in that same PCA
  subspace, sampled from a Gaussian maximum-entropy model.

Warm-start three identical suffix copies from the checkpoint suffix
(\psi_{t,\ell}). Relax each copy while holding the prefix fixed, using one of
the three activation distributions. Cross-evaluate every relaxed suffix on
every held-out activation distribution:

$$
M^{(t,\ell)}_{s,r}(u)=
\mathbb E_{(z,y)\sim Q^{(t,\ell)}_s}
\ell\!\left(\psi^{(u)}_{t,\ell;r}(z),y\right),
\qquad r,s\in\{\text{mean},\text{Gaussian},\text{true}\}.
$$

Here (r) is the activation distribution used to relax the suffix, (s) is
the evaluation distribution, and (u) is suffix-relaxation time. The full
matrix is required: a Gaussian-trained suffix doing well on Gaussian samples
alone could merely show that the surrogate task is easier.

The decisive statistic is true-activation excess loss,

$$
\Delta_{\mathrm{true}\mid r}^{(t,\ell)}(u)
=M^{(t,\ell)}_{\mathrm{true},r}(u)
-M^{(t,\ell)}_{\mathrm{true},\mathrm{true}}(u).
$$

Small Gaussian excess loss together with materially larger mean-only excess
loss supports the bounded claim that, at that checkpoint and cut, second-order
class-conditional activation statistics are approximately sufficient for the
suffix under this relaxation protocol. It does **not** show that the suffix
uses no higher-order statistic, nor that the PCA-truncated Gaussian preserves
all second-order structure in the full activation space.

### Primary bounded sweep and controls

The primary Part B measurement uses the four-block residual CNN at epochs 0, 1,
5, 10, 20, and 30 and after each nonempty residual prefix. Report the
layer-by-training-time map together with the full 3x3 cross-evaluation matrices.
Before interpreting it:

1. verify held-out class-mean and class-covariance errors for both surrogates;
2. report PCA dimension and explained variance;
3. use identical labels, sample counts, minibatch order, optimizer, learning
   rate, update count, and warm start across the three suffixes;
4. repeat surrogate sampling at least three times before claiming sufficiency;
5. compare warm-started relaxation with a suffix reinitialization control if
   the pilot is positive, to distinguish retained suffix knowledge from what
   can be relearned from the surrogate.

Use small multiples or a shared-scale heatmap so both depth and training time
remain comparable. Part B establishes the protocol in a bounded system; Part C
tests whether its depth pattern survives in larger standard architectures.

This design adapts the maximum-entropy evaluation logic of Belrose et al. to an
internal representation, but adds suffix relaxation and a train/evaluation
cross-matrix. Belrose et al. evaluate checkpoints on low-order maximum-entropy
inputs; they do not perform this frozen-prefix suffix-refit experiment.

### Supporting moving-optimum diagnostics

Keep the existing suffix update-direction dot products as a secondary view:
they ask whether the actual suffix update follows movement of the refitted
optimum or closes a pre-existing gap. The earlier finite refit/tracking losses
may remain in audit/provenance, but they no longer define Part B's headline.

The existing empirical-Fisher/Schur-complement branch remains **archived** and
should not appear in the active evidence chain. It uses outer products of
observed-label loss gradients. Those are useful gradient-second-moment
diagnostics, but are not in general the model Fisher/GGN required below.

### Bridge to Part D

Parts B and C measure which static activation statistics a frozen suffix can use;
they do not identify how a suffix responds when its activation distribution
moves. Their PCA bases are fitted independently at each cell, so the current
artifacts cannot define a dynamical response. Part D instead fixes one local
transport chart, compares the full and prefix-clamped update Jacobians, and
measures the suffix transfer function in a model-Fisher/GGN output metric.

### Prior local differential formulation (supporting only)

Near a stable optimum with block Hessian \(H\), implicit differentiation gives

$$
J_*(a)=\frac{db^*}{da}=-H_{bb}^{\dagger}H_{ba}.
$$

Writing \(e=b-b^*(a)\), gradient flow approximately obeys

$$
\dot e=-\eta_b H_{bb}e-J_*(a)\dot a.
$$

The first term is **resolution/relaxation**; the second is **tracking forcing**.
Estimate their norms and cosine, preferably in prediction-space or Fisher metric,
at each layer and time. Verify the local prediction against the finite checkpoint
grid above; do not trust Hessian algebra alone.

Prior moving-optimum hypothesis (not the current Part B headline):

> The ratio of tracking forcing to head relaxation increases through hidden
> blocks, possibly falling again at the final classifier, and rises when the data
> distribution begins exposing a new higher-order channel.

This is more precise than "middle layers are ineffective." A competing hypothesis
is that middle blocks have small envelope curvature and therefore change little;
another is that they move substantially but mostly induce compensatory downstream
updates. The measurements distinguish these.

## Experiment C: does the Part-B pattern persist in larger architectures?

Part C repeats the Part-B intervention rather than introducing a new criticality
metric. Its data and relaxation budgets are reduced for feasibility, but the
decisive excess-loss statistic is unchanged. For every architecture, checkpoint, and cut, freeze the native prefix,
fit the same class-mean and class-Gaussian activation surrogates, warm-start
matched suffixes, relax on true/mean/Gaussian activations, and cross-evaluate the
full 3x3 matrix. The decisive statistic and sign convention are unchanged:

$$
\Delta_{\mathrm{true}\mid r}^{(t,\ell)}
=M_{\mathrm{true},r}^{(t,\ell)}
-M_{\mathrm{true},\mathrm{true}}^{(t,\ell)},
\qquad r\in\{\mathrm{mean},\mathrm{Gaussian}\}.
$$

Positive values mean the surrogate-trained suffix is worse on held-out true
activations. Near-zero Gaussian excess loss together with positive mean-only
excess loss is the bounded second-order-sufficiency signature; positive Gaussian
excess loss is diagnostic only unless surrogate fidelity is established.
Held-out accuracy gaps may be reported as an intuitive secondary quantity but
must not silently replace this primary contrast. When accuracy is shown, report
both the absolute relaxation effect from the common warm start,

$$
\Delta A_r^{(t,\ell)}=A_{\mathrm{true}\mid r}^{\mathrm{after}}-A^{\mathrm{before}},
$$

and the surrogate shortfall relative to matched true-data relaxation,

$$
S_r^{(t,\ell)}=A_{\mathrm{true}\mid\mathrm{true}}^{\mathrm{after}}
-A_{\mathrm{true}\mid r}^{\mathrm{after}}.
$$

Thus $\Delta A_r>0$ means that relaxation improves the original checkpoint,
whereas $S_r>0$ means that the surrogate endpoint is worse than the true-data
endpoint. True-data relaxation is a measured finite-protocol control, not an
assumed improvement.

### C1. ResNet-18 depth-by-training-time replication

Use the normalization-free CIFAR ResNet-18 V2 at epochs **0, 1, 5, 20, and
100**, measuring the output of all eight residual blocks. These cuts include
four central blocks as well as early and late blocks. Dashed stage boundaries
are orientation aids only. The first gate uses one trained model, three
independent surrogate draws, 10,000 train activations, 2,000 held-out
activations, a PCA fit on 5,000 examples capped at 512 components, and five
suffix-relaxation epochs.

### C2. VGG-19 depth-by-training-time replication

Use CIFAR VGG-19+BatchNorm at the same five checkpoints and the seven native
post-convolution interfaces `3, 5, 7, 9, 11, 13, 15`. This samples the end of
stage 2 and the middle/end of stages 3–5 while avoiding the exceptionally large
stage-1 activation tensor in the first feasibility gate. Use the same data,
PCA, draw, optimizer, and relaxation budgets as C1. Do not transplant or reset a
module in the primary C2 experiment.

Display C1 and C2 with the same two heatmaps, sign convention, and symmetric
color range. Compare depth only after normalizing each cut by its architecture's
forward order; a VGG post-convolution cut and a ResNet post-block cut are not
identical computational objects. Keep the full 3x3 matrices and draw-level
dispersion available in audit detail.

### Feasibility and validation gate

The one-seed sweep is a phenomenon gate, not cross-training-seed evidence. PCA
coverage, held-out moment errors in the fitted PCA target space, and the number
of surrogate draws are part of the result. The earlier full-native-space moment
diagnostic is retained only for legacy artifacts because it incorrectly counts
deliberately discarded PCA directions as matching failures. Expand to multiple
training seeds only if both architectures produce a coherent depth pattern and
the surrogate-fidelity diagnostics are calibrated.

### Supporting critical-module context (bottom of C only)

The completed VGG and ResNet checkpoint-transplant heatmaps may appear after the
C1/C2 suffix-statistics results. They ask a different question—whether the final
co-adapted network tolerates an old module—and must not determine the C layout,
stage boundaries, or verdict. No targeted recovery run is active.

Primary source for that supporting intervention:
`papers/20-069-are-all-layers-created-equal.pdf` (open-access JMLR version,
accessed in full).

## Experiment D: tangent stability and suffix frequency response

**Status: theory / planned / unrun.** At checkpoint $t$ and cut $\ell$, write
$h^\ell=\phi_a^\ell(x)$ and refit the local suffix optimum

$$
b^*(a)\in\arg\min_b\mathbb E_{Q_{a,\ell}}
\ell(\psi_b(h),y),\qquad Q_{a,\ell}=\operatorname{Law}(h^\ell,y).
$$

Part D freezes a local tangent chart around $(a,b^*(a))$. It compares the
linearized full training system with the prefix-clamped suffix system, then treats
a small movement of the activation distribution as a driven input. The target is
not another moment-velocity dashboard; it is a frequency-resolved account of
which distributional perturbations the suffix rejects, passes, or amplifies.

### D1. Full-system versus suffix-only poles

For vanilla gradient descent with block learning rates and loss Hessian
$H=\nabla^2L$, the one-step perturbation maps are

$$
A_{\mathrm{full}}=I-
\begin{bmatrix}\eta_aI&0\\0&\eta_bI\end{bmatrix}
\begin{bmatrix}H_{aa}&H_{ab}\\H_{ba}&H_{bb}\end{bmatrix},
\qquad
A_{\mathrm{suf}}=I-\eta_bH_{bb}.
$$

Compare their eigenvalues in the same complex unit disk, their spectral radii,
and the prefix/suffix participation of full-system modes. For a stable discrete
mode, report $\tau_j=-1/\log|\lambda_j|$. If momentum, Adam, normalization state,
or another persistent optimizer variable is active, it belongs in the state and
therefore in the update Jacobian. Eigenvalues of the Fisher are not stability
eigenvalues. When the update is non-normal, resolvent or pseudospectral checks are
required because eigenvalues alone can miss large transient amplification.

Block elimination gives a useful but limited Schur connection. In continuous
time the frequency-dependent prefix operator is

$$
\mathcal S_a(s)=sI+\eta_aH_{aa}
-\eta_aH_{ab}(sI+\eta_bH_{bb})^{-1}\eta_bH_{ba}.
$$

At $s=0$ this contains $H_{aa}-H_{ab}H_{bb}^{\dagger}H_{ba}$, but that static
Schur complement is not the full-system eigenspectrum.

### D2. Activation-distribution coordinates

Define a finite transport chart on an immutable probe bank,

$$
h_i(u)=h_i+\sum_{j=1}^m u_jv_j(h_i,y_i),qquad
M_u=\mathbb E_i[V_i^\top V_i].
$$

$M_u$ declares intervention cost. The transport dictionary may include
class-conditional mean shifts, covariance deformations, higher-order Hermite
directions, and data-driven smooth directions. Each direction must pass
finite-amplitude target-moment and off-target leakage checks. A response
eigenvector is called a mean/covariance/higher-order mode only after its overlap
with this dictionary is reported.

### D3. The suffix transfer function

Linearizing the suffix update and a Fisher-whitened predictive readout gives

$$
\delta b_{k+1}=A_{\mathrm{suf}}\delta b_k+B_uu_k,
\qquad B_u=-\eta_bH_{bu},
$$

$$
y_k=\mathcal T_b\delta b_k+\mathcal T_uu_k,
\qquad
\mathcal R_\ell(z)=\mathcal T_u+
\mathcal T_b(zI-A_{\mathrm{suf}})^{-1}B_u.
$$

The first term is frozen-suffix sensitivity and the resolvent term is realized
suffix adaptation. On the unit circle, the central response object is

$$
\widehat F_{\mathrm{eff}}(\omega)=M_u^{-1/2}
\mathcal R_\ell(e^{i\omega})^*\mathcal R_\ell(e^{i\omega})M_u^{-1/2}.
$$

Its eigenvectors are distributional perturbation modes and its eigenvalues are
residual predictive-KL gains per unit transport cost at frequency $\omega$. Also
report attenuation relative to the frozen suffix,
$c(v,\omega)=\|\mathcal Rv\|/\|\mathcal T_uv\|$, and phase. For the
moving optimum,

$$
K_H=\frac{db^*}{du}=-H_{bb}^{\dagger}H_{bu}.
$$

If $H_{bb}q_j=h_jq_j$ under scalar-step gradient descent, the response of suffix
mode $j$ to its moving target is

$$
\Gamma_j(e^{i\omega})=
\frac{\eta_bh_j}{e^{i\omega}-(1-\eta_bh_j)}.
$$

This supplies the one-pole benchmark, but measured systems need not be one-pole.
A stable linear system is not “destabilized” by a sinusoid; it can be resonantly
or non-normally amplified. Genuine local instability means
$\rho(A)\geq1$.

### D4. Fisher/GGN and the static Schur limit

Let $\mathcal T_b$ and $\mathcal T_u$ stack logit Jacobians whitened by
$W(p)=\operatorname{diag}(p)-pp^\top$. Their Gram blocks are the corresponding
model-Fisher/GGN blocks. The best instantaneous predictive compensation obeys

$$
\min_{\delta b}\|\mathcal T_u\delta u+\mathcal T_b\delta b\|^2
=\delta u^*\left(G_{uu}-G_{ub}G_{bb}^{\dagger}G_{bu}\right)\delta u.
$$

Thus the Fisher/GGN Schur complement is the ideal static geometric limit. The
optimizer's DC change is instead $K_H=-H_{bb}^{\dagger}H_{bu}$. They coincide
only when the same model Fisher/GGN adequately approximates the loss Hessian and
the metrics/damping agree. Use the update Jacobian for poles; use Fisher/GGN to
measure behavioral size.

### D5. Required dashboard and validity gate

The measured dashboard should contain:

1. a shared-unit-circle eigenvalue plot for $A_{\mathrm{full}}$ and
   $A_{\mathrm{suf}}$, plus mode participation and relaxation times;
2. frequency-by-mode maps of residual KL gain, attenuation, and phase;
3. portraits of the most rejected and amplified distribution transports, with
   moment-family overlaps and leakage;
4. a DC comparison among finite suffix refit, optimizer prediction, and the
   Fisher/GGN Schur bound; and
5. held-out nonlinear sinusoidal rollouts testing prospective tangent predictions.

Start with a deliberately tiny normalization-free residual model and plain SGD so
dense eigensolvers can validate matrix-free JVP/VJP implementations. Require
finite-difference Jacobian checks, a perturbation-amplitude linearity sweep,
zero-drive and frozen-suffix controls, and uncertainty across independent model
seeds. Products of several time-varying Jacobians and finite-time Lyapunov
exponents are a separate extension and are deferred until this frozen-chart
one-step operator is validated. The companion derivation is
[`docs/tangent_model_control_design.md`](docs/tangent_model_control_design.md).

## Experiment E: the Part B suffix-statistics experiment for next-token prediction

**Status: planned / unrun. Every dashboard panel is a labeled MOCKUP until a
validated artifact is loaded.** The primary Part E experiment is now a direct
port of Part B: at GPT-2 scratch-training checkpoints, freeze a residual-stream
prefix, fit mean-only, sequence-Gaussian, and true activation distributions,
warm-start three matched suffix copies, and collect the full 3x3
relax-by-evaluate matrix. The NTP-specific changes are complete sequence replay,
causal masks and positions, sequence covariance, cloned tied-head handling,
story-level uncertainty, and PCA/leakage validity checks. The concise current
contract is in [`docs/part_e_gpt2_design.md`](docs/part_e_gpt2_design.md).

The expanded branch design below is retained as a **deferred follow-up**, not as
part of the first Part E gate. Ordinary continuation, instruction CLM,
response-only SFT, LoRA, and RL may reuse the validated protocol later. Likewise,
token-IID Gaussian, projected-true replay, and finite transfer quantities are
secondary diagnostics; they do not enlarge the headline 3x3 matrix.

### Deferred E1. Model, data, and matched training branches

Use GPT-2 small (12 blocks, 12 heads, width 768) at context length 256 on a
deterministic 50M GPT-2-token TinyStories subset, with a separate 2M-token
validation set. Run one trunk and three matched branches:

| branch | initialization | data and objective | exposure rule |
|---|---|---|---|
| P · scratch | random | ordinary TinyStories full-token LM | 50M loss tokens |
| C · continuation | exact P endpoint; reset optimizer | disjoint ordinary-story full-token LM | branch updates and processed-token batch matched to I/S |
| I · instruction CLM | exact P endpoint; reset optimizer | TinyStoriesInstruct full-token LM | exact ordered sequences/batches used by S |
| S · full SFT | exact P endpoint; reset optimizer | the same TinyStoriesInstruct stream, response-only loss | exact ordered sequences/batches used by I; 10M processed non-padding tokens |

C/I/S share checkpoint bytes, optimizer reset, learning-rate schedule, processed
sequence batch, update count, precision, and evaluation cadence. I and S also see
the exact same examples in the exact same order; only S masks prompt labels. Thus
C $\rightarrow$ I is explicitly a bundled domain/format/content contrast, while
I $\rightarrow$ S isolates the response-loss mask. C $\rightarrow$ S is never
called a pure fine-tuning effect. P uses an absolute loss-token axis; branch plots
use update count and annotate processed plus arm-specific loss-bearing tokens.

Full-parameter SFT comes first so the parameter partition remains comparable;
LoRA is a second-stage trainable-subspace intervention, and RL is deferred because
rollout, reward, and credit-assignment noise add several confounds at once. The 50M
P endpoint is called "pretrained" only if held-out loss and learning-curve slope
show a useful learned regime.

Save P checkpoints at $0,0.25,1,4,16,50$M loss tokens. Save C/I/S at common branch
updates corresponding to processed non-padding-token milestones $0,0.1,0.5,2,10$M. Record
the actual update, sequence, processed-token, loss-bearing-token, and compute count
for every arm; masked prompt processing is not free.

### E2. Interfaces and immutable replay banks

Measure `resid_post` after blocks 0, 2, 5, 8, and 11, plus a LayerNorm-normalized
companion because raw pre-LN scale can move while the next block's normalized
input stays stable. Gate suffix-relaxation work at blocks 0, 5, and 11.

Cache complete sequences, masks, token IDs, and targets from two immutable banks:
2,048 train plus 512 held-out plain-story sequences, and the same counts for
instruction sequences. Split and bootstrap by story, not token. GPT-2 ties input
and output embeddings; cached interfaces make the prefix embedding inactive, so
clone the LM-head weight into each suffix copy and record this operational
untying. All copies must agree exactly at relaxation step $u=0$.

### E3. Sequence-preserving surrogate hierarchy

An IID token Gaussian is only a destructive control: it removes temporal
covariance, position, story length, and causal trajectory structure. For suffix
relaxation, fit a local PCA projection capped at 384 components and require at
least 90% held-out variance coverage. This locally fitted PCA is for the
surrogate only; it is not interpreted as a cross-checkpoint motion basis.

Let $Z\in\mathbb R^{T\times k}$ be a projected activation sequence and
$y_p=x_{p+1}$. Use an additive conditional mean

$$
M_p(Y)=\bar z+a_{g(y_p)}+b_{\operatorname{posbin}(p)},
$$

where the most frequent 1,024 next-token types have singleton groups and the
tail uses declared frequency/token-shape groups. A row may depend on its own
next-token class $y_p$, matching B/C's class-conditional construction, but never
on $y_{p+1:}$. Fit the bounded matrix-normal proxy

$$
Z\mid Y\sim\mathcal{MN}\!\left(M(Y),K_{\mathrm{pos}},
\Sigma_{\mathrm{chan}}\right).
$$

The target-conditioned mean is a B/C-style joint-distribution diagnostic, not an
inference-time generator because $y_p$ is the token being predicted. Pair it with
a history-only mean ablation using visible input-token groups. Fix the
matrix-normal scale by $\operatorname{tr}(K)/n_{\mathrm{valid}}=1$, use only valid
within-story position pairs, declare shrinkage, and validate held-out lag
covariances at 1, 2, 4, 8, and 16 tokens.

Compare five distributions: mean plus isotropic noise; token-Gaussian with channel
covariance but IID positions; sequence-Gaussian with position and channel covariance;
empirical true sequences projected into and reconstructed from the PCA space; and
full true cached sequences. Warm-start matched suffixes and record the complete
$5\times5$ relaxation/evaluation matrix

$$
M_{s,r}^{(t,\ell,d)}(u)=
\mathbb E_{(H,Y)\sim Q_s}
\ell_{\mathrm{token}}\!\left(\psi_r^{(u)}(H),Y\right).
$$

The seqG-to-projected-true gap identifies unmatched structure within the retained
subspace; the projected-to-full-true gap separately identifies useful discarded
directions. Cross-entropy in nats/token is primary. Perplexity ratios, top-$k$
accuracy, and common/tail-token strata are secondary. A future-label permutation
test must prove that changing $y_{>p}$ cannot alter generated rows $H_{\le p}$.

### E4. Finite suffix resolving and transfer demand

Keep the finite B/C-style suffix quantities, but use them as operational transfer
readouts rather than as a claim about moment velocity:

$$
\Delta L_{\mathrm{resolve}}
=L(H_t,\psi_t)-L(H_t,\widehat\psi_t^{(U)}),
\qquad
\Delta L_{\mathrm{transfer}}
=L(H_t,\widehat\psi_{t-\Delta}^{(U)})-L(H_t,\widehat\psi_t^{(U)}).
$$

The first asks how much useful computation the intact checkpoint suffix has not
yet realized at its own interface. The second asks how much a suffix fitted at the
previous checkpoint falls behind at the new interface. Record complete relaxation
curves, update-0 loss, a validated plateau (or final endpoint plus `unresolved`),
area above that reference, updates to a preregistered threshold, refit path length,
actual suffix-update norm, and Euclidean update/refit alignment. Unequal checkpoint
gaps always carry update, processed-token, and arm-specific loss-token intervals;
the finite endpoint is not called an optimum without a convergence check.

The sequence-surrogate matrix asks *which fitted sequence structure the suffix can
use*; transfer demand asks *how stale an adjacent-checkpoint fitted suffix is*.
Their association is descriptive. Update eigenspectra, activation-distribution
transport modes, Fisher-weighted transfer functions, and Schur-limit comparisons
remain Part D measurements.

### E5. Gate, artifact contract, and dashboard

Begin with a 1M-P-loss-token runtime/storage smoke. The first scientific gate uses
one model seed, the matched branches, both replay domains, sentinel cuts 0, 5, and
11, the full $5\times5$ matrix, one surrogate draw, and 64 suffix-relaxation updates at selected early,
branch-point, and endpoint checkpoints. The old 6--10 A100-80GB-hour sketch assumed
three arms and four distributions and is obsolete; benchmark component timings must
produce a new four-arm, $5\times5$ estimate. Confirmatory work requires three model
seeds, three surrogate draws, and 128 updates only after all mechanical gates pass.

Persist `manifest.json`, `training_metrics.jsonl`, `replay_manifest.json`,
`sequence_surrogates.npz`, `surrogate_diagnostics.parquet`,
`suffix_statistics.json`, and `tracking_statistics.parquet`. The manifest fixes
dataset/tokenizer revisions, exact branch lineage, replay hashes, checkpoint
hashes, tied-head policy, PCA/surrogate definitions, dtype, and exclusions. It
must be impossible to mix mock, smoke, failed, and measured evidence.

The Part E dashboard shows the branch/control logic, sequence-surrogate hierarchy,
the full cross-evaluation matrix, shared-bank branch contrasts, finite resolving
and transfer demand, and collapsed audit gates. It contains no dedicated
tangent-spectrum, frequency-response, or Fisher-Schur result panel. The complete protocol is
[`docs/part_e_gpt2_design.md`](docs/part_e_gpt2_design.md).

## 6. Original minimal viable sequence and current scope

The sequence below remains the strongest causal program. The current experiment
deliberately begins with the requested CIFAR extension: class-mean and
class-covariance PCA surrogates for A and B, plus archived moving-optimum and
observed-label gradient diagnostics. These are exploratory proxies for cumulant
order, not a substitute for the synthetic identifiability checks in steps 1--2
or the planned Part-D model-Fisher transport measurement.

1. **Static synthetic matrix:** independent versus aligned order-1--4 channels;
   produce \(M_{s,r}(t)\). This validates the statistical hierarchy.
2. **Tangent audit:** one tiny residual MLP, three cuts, checkpoint suffix refits,
   dense full/suffix update Jacobians, and finite-difference validation.
3. **Distributional response:** drive a small declared transport dictionary;
   compare tangent transfer functions with finite sinusoidal rollouts.
4. **Scale-up:** replace dense operators with matrix-free eigensolvers only after
   their spectra and DC response agree on the tiny model.
5. **CIFAR extension:** only after the metrics discriminate the synthetic cases.

Recommended synthetic follow-up scale: binary task, dimension 64--128, 4--6 block
residual MLP, 5 seeds, four cumulant orders, and three cut layers. Use a full-batch
or deterministic baseline before SGD.

## 7. Required plots

1. train-distribution by test-distribution loss matrix over time;
2. onset time of useful order-$r$ information versus $r$;
3. frozen-interface surrogate shortfall by layer/time;
4. full-system versus suffix-only update eigenvalues in a shared unit disk;
5. relaxation times and prefix/suffix participation of the slow modes;
6. frequency-by-transport-mode residual KL gain, attenuation, and phase;
7. portraits of the most rejected and amplified activation-distribution modes;
8. finite-refit, optimizer-DC, and Fisher/GGN Schur-limit comparison;
9. prospective nonlinear sinusoidal rollouts against tangent predictions;
10. Part E's branch comparison and sequence-surrogate shortfall matrix.

Every mock plot must be visibly labeled MOCKUP and include an interpretation guide.

## 8. Failure criteria and confounds

- Surrogates do not match their targeted held-out statistics.
- The declared activation transport does not preserve its off-target constraints.
- Results disappear when task Bayes difficulty or mutual information is matched.
- Head refits find different basins and make parameter distances meaningless.
- BatchNorm state changes while a "frozen" representation is supposedly frozen.
- Data augmentation silently changes the intended cumulants.
- The suffix refit is not sufficiently close to $\psi^*[\phi]$ for a stable local
  chart.
- Periodic response is nonlinear in amplitude or nonstationary across cycles.
- The measured update Jacobian fails finite-difference or one-step checks.
- Fisher is used as the stability operator, or the Hessian is used as the
  predictive-KL metric.

## 9. Decision points before implementation

The first implementation should settle only three choices:

1. binary classification or teacher-student regression (classification is closer
   to CIFAR; regression is analytically cleaner);
2. residual MLP or small CNN (start with residual MLP, then test architecture
   robustness);
3. whether the order channels differ in Bayes information. Prefer calibrating
   channel strengths so each isolated channel has comparable Bayes advantage,
   then add an uncalibrated natural-strength condition.

## 10. Primary sources currently in the project

- Alessandro Achille, Matteo Rovere, and Stefano Soatto, *Critical Learning
  Periods in Deep Neural Networks* (ICLR 2019), `papers/achille-critical-periods.pdf`.
- Lorenzo Bardone and Sebastian Goldt, *Sliding down the stairs: how correlated
  latent variables accelerate learning with neural networks* (NeurIPS 2024),
  `papers/sliding-down-stairs.pdf`.
- Nora Belrose et al., *Neural Networks Learn Statistics of Increasing
  Complexity* (ICML 2024), `papers/increasing-complexity.pdf`.
- Onat Ure, Samet Demir, and Zafer Dogan, *Learning Beyond the Gaussian Data:
  Learning Dynamics of Neural Networks on an Expressive and
  Cumulant-Controllable Data Model* (arXiv 2026),
  `papers/cumulant-controllable.pdf`.
- James Martens, *New Insights and Perspectives on the Natural Gradient Method*
  (JMLR 2020), `papers/martens-natural-gradient-2020.pdf`.
- Frederik Kunstner, Lukas Balles, and Philipp Hennig, *Limitations of the
  Empirical Fisher Approximation for Natural Gradient Descent* (NeurIPS 2019),
  `papers/kunstner-empirical-fisher-2019.pdf`.
- Alec Radford et al., *Language Models are Unsupervised Multitask Learners*
  (2019), `papers/radford-gpt2-2019.pdf`.
- Ronen Eldan and Yuanzhi Li, *TinyStories: How Small Can Language Models Be and
  Still Speak Coherent English?* (2023), `papers/eldan-tinystories-2023.pdf`.
- Edward Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021),
  `papers/hu-lora-2021.pdf`.
