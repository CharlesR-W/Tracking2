# Tracking statistical structure through a deep network

Status: Experiment A is implemented on CIFAR-10. Experiment B is being
redesigned around a frozen-prefix activation-statistics battery; the existing
five-seed moving-optimum artifact is supporting evidence, not the main result. A measured seed-0
VGG-19+BatchNorm critical-module gate reached 90.73% test accuracy at epoch 40
and found a sharp checkpoint-0 reset boundary between `stage4.conv1` and
`stage4.conv2`; a five-seed, 100-epoch criticality/recovery battery is running.
The sinusoidal branch remains deferred.

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

### Pilot slice and controls

The first measured slices are **epochs 0, 1, and 5 at cut 3**, residual CNN,
seed 0. Report each relaxation trajectory and final 3x3 loss matrix as aligned
triptychs with shared axes and color scale. Before interpreting them:

1. verify held-out class-mean and class-covariance errors for both surrogates;
2. report PCA dimension and explained variance;
3. use identical labels, sample counts, minibatch order, optimizer, learning
   rate, update count, and warm start across the three suffixes;
4. repeat surrogate sampling at least three times before claiming sufficiency;
5. compare warm-started relaxation with a suffix reinitialization control if
   the pilot is positive, to distinguish retained suffix knowledge from what
   can be relearned from the surrogate.

Epoch 10 is the next time slice only after these pilots are legible. Time slices
must remain separate facets with shared axes, not overlaid into one crowded plot.

This design adapts the maximum-entropy evaluation logic of Belrose et al. to an
internal representation, but adds suffix relaxation and a train/evaluation
cross-matrix. Belrose et al. evaluate checkpoints on low-order maximum-entropy
inputs; they do not perform this frozen-prefix suffix-refit experiment.

### Supporting moving-optimum diagnostics

Keep the existing suffix update-direction dot products as a secondary view:
they ask whether the actual suffix update follows movement of the refitted
optimum or closes a pre-existing gap. The earlier finite refit/tracking losses
may remain in audit/provenance, but they no longer define Part B's headline.

The empirical-Fisher/Schur-complement branch is **on hold** and should not appear
in the active dashboard evidence chain. Its artifacts can remain archived.

## Experiment C: VGG critical modules and activation statistics

Part C uses VGG-19+BatchNorm as a positive-control system in which a sharp
critical-module boundary is already observed, then replicates Part B's
representation-surrogate analysis over the intact VGG training trajectory. A
convolution and its BatchNorm affine parameters and running state are one atomic
module in the C1 criticality evaluation.

### C1. Criticality and recovery atlas — implemented / running

For avoidance of ambiguity, this follows the paper's post-training probe. Each
cell begins from the final model and performs

$$
\theta_m^T \leftarrow \theta_m^\tau,
$$

with every other module held at its final value. It is not the different
experiment that starts from checkpoint-$\tau$ and resets a layer at that time.
Consequently the heatmap measures compatibility with the final co-adapted
network and need not be monotone in $\tau$. The paper itself notes cases where
checkpoint 1 is more destructive than checkpoint 0 in normalization/weight-
decay variants. Our VGG+BatchNorm result should be described as a positive
control variant, not as an exact replication of the paper's normalization-free
headline VGG experiment.

**Rerun TODO — match the paper more closely.** If C1 is rerun, the primary
replication should use the paper's normalization-free VGG architecture and
paper-matched training protocol: SGD with momentum 0.9, 100 epochs, batch size
128, and a piecewise-constant learning rate multiplied by 0.2 at epochs 30, 60,
and 90. Preserve the paper's post-training intervention exactly: replace one
parametric layer of the final model with its checkpoint-$\tau$ value or a fresh
draw, leave every other layer final, and perform no fine-tuning for the primary
heatmap. Match the paper's layer boundaries rather than folding BatchNorm into
the convolution. Report the current VGG+BatchNorm Conv+BN-atomic battery only as
a separate robustness variant, not as the headline replication.

Before interpreting moving-optimum quantities, establish a positive-control
layerwise effect comparable to Zhang, Bengio, and Singer, *Are All Layers Created
Equal?* (JMLR 2022). Train a CIFAR-sized VGG-19+BatchNorm and retain
checkpoints at initialization and throughout training. For each of its 19
parametric modules, replace only that module in the final model with either a
fresh random draw or its value from an earlier checkpoint, with no immediate
fine-tuning. The primary Panel B view is a module-by-source heatmap of held-out
test-error increase relative to the intact final model.

Then freeze the transplanted module and everything below it and refit only the
downstream suffix. Report the recovery curve, fraction of damage recovered, and
steps to fixed recovery fractions. This distinguishes modules that are
immediately critical but downstream-trackable from modules whose learned map
cannot be compensated by the suffix. The confirmatory battery uses five
independent 100-epoch seeds and saves initialization, intermediate, and final
checkpoints so C2 can reuse the same trained models. Fisher measurements are
not part of this VGG branch.

### C2. VGG replication of Part B — active

Replicate Part B on the intact VGG training trajectory before introducing any
new module intervention. At checkpoint epochs 0, 1, 5, 20, and 100 and at cuts
7–10 (`stage3.conv4` through `stage4.conv3`), freeze the native prefix, encode
train/test activations, and fit the same class-conditional mean-only and
PCA-Gaussian surrogates. Warm-start matched suffix copies from that checkpoint,
relax them on true, Gaussian, or mean-only activations, and cross-evaluate the
full 3-by-3 matrix.

The primary result is a cut-by-training-time map of held-out true-activation
accuracy gaps relative to true-activation relaxation,

$$
G_r(t,\ell)=100\left[A_{\mathrm{true}\mid r}(t,\ell)
-A_{\mathrm{true}\mid\mathrm{true}}(t,\ell)\right],
\qquad r\in\{\mathrm{mean},\mathrm{Gaussian}\}.
$$

Display this map beside, but do not combine it mathematically with, the C1
criticality boundary. The descriptive question is whether the sharp transition
between `stage4.conv1` and `stage4.conv2` coincides with a change in which
class-conditional activation statistics support suffix relearning. Criticality
and statistical sufficiency remain distinct measurements; an aligned boundary
is evidence for a relationship, not causality.

Report PCA coverage and held-out class-mean/covariance errors at every cell.
Gaussian-versus-true gaps are not interpretable as higher-order dependence when
the fitted Gaussian misses its intended moments materially. The first gate uses
one independently trained VGG seed and three surrogate draws. Expand across
training seeds only if this layer-by-time map is coherent.

### Deferred intervention — not active

Checkpoint transplantation, fresh re-randomization, and downstream recovery
experiments are archived rather than displayed. Revisit a targeted intervention
only after C2 establishes where the statistical-tracking signature changes.

Primary source: `papers/20-069-are-all-layers-created-equal.pdf` (open-access
JMLR version, accessed in full).

At cut \(\ell\), write

$$
f_{a,b}(x)=\psi_b^\ell(\phi_a^\ell(x)),\qquad
b^*(a)=\arg\min_b L(a,b).
$$

The optimum is not necessarily unique, so parameter distance
\(\|b-b^*\|\) is gauge-dependent and should not be the main metric. Define
\(b^*(a)\) operationally as the endpoint of a fixed downstream refit protocol,
warm-started from the current \(b\), with \(a\) frozen. Compare in function and
loss space:

- **head regret:** \(R_\ell=L(a,b)-L(a,b^*(a))\);
- **prediction gap:** mean KL divergence between \(\psi_b(\phi_a(x))\) and
  \(\psi_{b^*}(\phi_a(x))\);
- **refit effort:** optimizer steps or path length needed to reach a fixed fraction
  of the available loss reduction.

At adjacent checkpoints \(a_t,a_{t+\Delta}\), fit the 2x2 counterfactual grid

$$
b_t^*=b^*(a_t),\qquad b_{t+\Delta}^*=b^*(a_{t+\Delta}),
$$

and evaluate each head on both representations. This separates:

- **representation drift / tracking demand:** performance lost when \(b_t^*\) is
  placed on \(a_{t+\Delta}\);
- **residual head suboptimality / resolving demand:** gap from the actual \(b_t\)
  to \(b_t^*\) on fixed \(a_t\).

Normalize both by the total loss improvement over the interval. Report signed
terms as well as magnitudes: representation changes can help an old head, so an
absolute-only "effort fraction" can be misleading.

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

## 6. On hold: Fisher, plasticity, and effective curvature

Partition the empirical Fisher or generalized Gauss--Newton matrix at the same cut:

$$
F=\begin{pmatrix}F_{aa}&F_{ab}\\F_{ba}&F_{bb}\end{pmatrix}.
$$

After allowing the downstream head to re-equilibrate, the local curvature seen by
the upstream block is the damped Schur complement

$$
F_{\mathrm{eff},a}=F_{aa}-F_{ab}(F_{bb}+\gamma I)^{-1}F_{ba}.
$$

Interpretation: \(F_{aa}\) counts output-sensitive upstream directions if the head
is held fixed; the subtracted term is sensitivity the head can absorb. This is a
candidate measure of **uncompensated plasticity**, not plasticity by definition.

Do not materialize these matrices. Estimate traces, leading eigenvalues, and
quadratic forms with Jacobian-vector/vector-Jacobian products and conjugate
gradient. Include:

- \(\mathrm{tr}(F_{aa})\), \(\mathrm{tr}(F_{\mathrm{eff},a})\), and their ratio;
- effective rank and top eigenvalues;
- gradient energy in high- versus low-curvature eigenspaces;
- agreement between Fisher/GGN predictions and actual small perturbations;
- empirical head-relearning time after a controlled representation perturbation.

Run Hessian versions only for small models. For cross-entropy, Fisher/GGN is PSD
and makes the Schur computation more stable, but it omits non-Gauss--Newton
curvature and is not interchangeable with the Hessian.

## 7. Deferred: sinusoidal system identification

This branch is out of scope for the current implementation. Experiments A and B,
including the optional Fisher/GGN measurements, should be stabilized before any
periodic-forcing work begins.

Modulate a **distribution parameter with clear semantic/statistical meaning**, not
global image amplitude:

$$
\lambda_k(t)=\lambda_{k,0}+A\sin(\omega t).
$$

Best first inputs are the strength of one Hermite/cumulant channel, the correlation
between low- and high-order latent channels, or a nuisance feature's label
correlation. In the CIFAR extension, candidates are color-label correlation,
Fourier/wavelet texture power in a chosen band, or mixture weight between original
and transformed examples. Keep label semantics fixed.

After a stationary burn-in, sweep logarithmically spaced \(\omega\), with at least
5--10 periods per frequency and small enough \(A\) to verify linearity using
\(A/2,A,2A\). For each layer measure the complex first-harmonic response of:

- predictions and loss;
- representation statistics aligned with the driven channel;
- \(b_t\), \(b^*(a_t)\), and their prediction-space discrepancy;
- tracking and relaxation terms from Section 5.

Estimate gain and phase by regressing each observable on
\(\sin\omega t,\cos\omega t\), averaging complex responses across seeds rather than
averaging phases. Define a critical frequency operationally, e.g. the first
frequency with gain below \(-3\) dB relative to the low-frequency plateau, and also
report phase-lag and coherence. A single critical frequency is expected only for a
one-pole response; otherwise fit a small state-space model or report the empirical
Bode curve without forcing that story.

Controls: shuffled phase, \(A=0\), quasi-static ramps, matched abrupt switches,
different batch sizes, learning rates, momentum, and update ratios for upstream
versus downstream blocks.

## 8. Noise is an intervention, not a post-hoc explanation

To test whether stochasticity causes the observed tracking/plasticity dynamics,
match mean drift while varying gradient-noise covariance:

- full batch versus minibatch;
- batch size with learning-rate adjustments reported both with fixed \(\eta\) and
  approximately fixed noise scale;
- with-replacement sampling versus deterministic cycling;
- explicit Gaussian gradient noise calibrated to the measured minibatch covariance;
- optional projection of added noise onto leading Fisher eigenspaces versus their
  orthogonal complement.

Log per-example gradient covariance sketches, parameter-update covariance, Hessian
or Fisher spectral summaries, and frequency response. The falsifiable claim is not
"noise matters," but that altering noise along particular curvature eigenspaces
predictably shifts tracking regret, loss of plasticity, or response bandwidth.

## 9. Original minimal viable sequence and current scope

The sequence below remains the strongest causal program. The current experiment
deliberately begins with the requested CIFAR extension: class-mean and
class-covariance PCA surrogates for A, plus the moving-optimum and empirical-Fisher
measurements for B. These are exploratory proxies for cumulant order, not a
substitute for the synthetic identifiability checks in steps 1--2.

1. **Static synthetic matrix:** independent versus aligned order-1--4 channels;
   produce \(M_{s,r}(t)\). This validates the statistical hierarchy.
2. **Moving-optimum audit:** one small residual MLP, three cuts, checkpoint refits,
   then compare finite differences to the implicit-Hessian prediction.
3. **Periodic forcing:** drive one order-2 and one order-4 channel; estimate Bode
   curves and critical frequencies across cuts.
4. **Noise ablation:** batch size plus calibrated explicit noise, holding the
   deterministic setup fixed.
5. **CIFAR extension:** only after the metrics discriminate the synthetic cases.

Recommended synthetic follow-up scale: binary task, dimension 64--128, 4--6 block
residual MLP, 5 seeds, four cumulant orders, and three cut layers. Use a full-batch
or deterministic baseline before SGD.

## 10. Required plots

1. train-distribution by test-distribution loss matrix over time;
2. onset time of useful order-\(r\) information versus \(r\);
3. head regret and prediction gap by layer/time;
4. finite tracking demand versus resolving demand by layer/time;
5. suffix update-direction dot products as a secondary diagnostic;
6. activation-surrogate held-out moment validation and PCA coverage;
7. Bode gain, phase, and coherence by layer and driven cumulant order;
8. the above under controlled noise interventions.

Every mock plot must be visibly labeled MOCKUP and include an interpretation guide.

## 11. Failure criteria and confounds

- Surrogates do not match their targeted held-out statistics.
- Higher-order estimators are dominated by sampling error; use unbiased k-statistic
  estimates or held-out uncertainty intervals.
- Results disappear when task Bayes difficulty or mutual information is matched.
- Head refits find different basins and make parameter distances meaningless.
- BatchNorm state changes while a "frozen" representation is supposedly frozen.
- Data augmentation silently changes the intended cumulants.
- Tracking and resolving terms depend qualitatively on checkpoint spacing.
- Periodic response is nonlinear in amplitude or nonstationary across cycles.
- Fisher is described as Hessian, or empirical-Fisher behavior is overinterpreted
  without perturbation checks.

## 12. Decision points before implementation

The first implementation should settle only three choices:

1. binary classification or teacher-student regression (classification is closer
   to CIFAR; regression is analytically cleaner);
2. residual MLP or small CNN (start with residual MLP, then test architecture
   robustness);
3. whether the order channels differ in Bayes information. Prefer calibrating
   channel strengths so each isolated channel has comparable Bayes advantage,
   then add an uncalibrated natural-strength condition.

## 13. Primary sources currently in the project

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
