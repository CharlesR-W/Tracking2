# Measurement supplement proposal for Parts B, C, and E

**Status:** Mixed. The July 26 CNN battery completed several controls proposed
here, while the dynamics, Fisher, standardized-reader, and curvature program
remains open. This document is a follow-up roadmap, not the post's evidence
index.

**Date:** 2026-07-23

## Status amendment after the July 26 control battery

| Proposed item | Status | Current boundary |
|---|---|---|
| Projected-true replay and PCA rank check | **Partly complete** | Complete for the canonical CNN cells and a seed-0 rank sweep; rank 2,048 fails the shallow predictive-KL gate, while the exploratory seed-0 rank-4,096 cell passes. Fisher-weighted PCA coverage remains open. |
| First-update optimizer control | **Complete for the publication battery** | Fixed LR inflated shallow surrogate updates; the canonical comparison matches the first update only, not the whole trajectory. |
| Mean-noise radius and covariance shrinkage | **Complete for the publication battery** | Radius sensitivity changes the Gaussian-versus-mean conclusion; 5% covariance shrinkage is a separate, small sensitivity. |
| Reinitialized suffix control | **Partly complete** | Five-epoch reinitialization preserves the early/late contrast but does not capacity-match receivers. Standardized readers remain open. |
| Relaxation horizon | **Partly complete** | Five- and 20-epoch endpoints are measured. Plateau-aware kinetics, adaptive continuation, and defensible relaxation times remain open. |
| Representation motion / stale-suffix splice | **Open** | The canonical post battery covers epoch 30 and cuts 1/4 only; it cannot support a depth-by-time motion claim. |
| Activation Fisher, finite noise curves, and parameter Fisher | **Open** | No publication claim depends on these proposed measurements. |
| Hessian/GGN curvature sentinel | **Open** | Still gated on a mechanically validated small-model implementation. |
| Part E sequence diagnostic | **Partly complete; negative** | The repaired 50M-token one-seed endpoint passes mechanical gates, but sequence Gaussian does not consistently beat mean replay. Redesign and time-dependent/multi-seed work remain open. |

The older one-model ResNet trajectory is historical pilot context, not a
completed architecture replication or a canonical supplement result.

## Recommendation

The supplement should contain four linked measurements:

1. **relaxation speed**;
2. **representation motion**;
3. **local predictive sensitivity**; and
4. **usable capacity**.

No single one should be presented as *the* tracking metric. Together they can
distinguish several explanations that currently look similar:

- the suffix is slow because its input representation moves quickly;
- the suffix is locally insensitive to most changes at its interface;
- the representation no longer contains useful structure accessible to a
  downstream reader;
- the native suffix is too simple or insufficiently plastic to use the
  available structure; or
- the apparent effect is caused by the PCA approximation.

For the first pass, Fisher and Hessian quantities should be measured only at
the intact base checkpoints, before any relaxation. A suffix-restricted Hessian
evaluated at the common warm start is still a base-model measurement.
Curvature during the mean/Gaussian relaxations would multiply the cost and add
surrogate- and learning-rate-specific confounds before the base-model
measurements have established a useful signal.

| Question | Proposed object | What it actually tells us |
|---|---|---|
| How quickly can the suffix adapt? | Plateau-validated relaxation time | Optimization speed under the declared relaxation protocol |
| How quickly is its input moving? | CKA geometry speed plus an old-suffix/new-prefix splice | Geometric motion and motion that matters to the downstream computation |
| Are later suffixes simply insensitive? | Activation Fisher plus finite noise curves | How much predictions notice small perturbations at each cut |
| Is learning stopping because of limited capacity or plasticity? | Fixed-decoder probes plus native-suffix reinitialization | Accessible information versus native-suffix optimization/capacity |
| Is PCA causing the result? | Projected-true control plus Fisher-weighted PCA coverage | Whether discarded directions matter to the suffix, not merely how much variance they contain |

## 1. Relaxation time: separate speed from ceiling

The literal “time to 90% of true-loss improvement” will usually fail here. In
the saved B and ResNet paths, training on mean or Gaussian activations often
makes true-activation loss worse rather than better.

Two different quantities should therefore be recorded.

### 1.1 Fitting time

On held-out data from the distribution being trained on, measure how many
suffix updates are required to realize 90% of the eventual loss reduction:

$$
q_r(u)
=
\frac{L_{r\mid r}(0)-L_{r\mid r}(u)}
     {L_{r\mid r}(0)-L_{r\mid r}^{\mathrm{plateau}}}.
$$

Define $\tau_{90}$ as the first update at which $q_r(u)\geq 0.9$ and remains
within the declared plateau band.

### 1.2 True-data consequence

Over the same updates, separately report:

- whether true-activation loss improves or deteriorates;
- the final plateau-relative shortfall;
- area under the true-loss trajectory; and
- an optional true-data settling time.

A suffix can therefore be described as **fast but wrong**: it rapidly learns
the mean-only task while moving away from the true task. That is scientifically
important and would be hidden by a statistic that collapsed speed and endpoint.

If a true-data timescale is wanted, define **settling time** as the first update
after which true loss remains within 10% of its eventual start-to-plateau
change. This works whether the eventual change is beneficial or harmful and
should not be called “time to improvement.”

### 1.3 Protocol

- Save losses at updates $0,1,2,4,8,\ldots$, then more densely around the
  transition.
- Continue relaxation in chunks until the final portion of the curve is flat
  relative to held-out uncertainty.
- If it does not stabilize within the declared cap, label it `UNRESOLVED`;
  do not call the final sampled point “converged.”
- For non-monotone curves, report trajectory area as a robust companion; do
  not force an exponential fit.
- Use cross-entropy as the primary quantity. Accuracy is too coarse and
  step-like for estimating a timescale.

For Part E, timescale comparisons should use equal learning rates and the same
minibatch-index stream. The gradient-matched runs intentionally use different
effective learning rates. They are useful outcome controls but do not provide a
clean comparison of speed.

## 2. Representation velocity: CKA is useful but not sufficient

There is direct precedent for comparing fixed layers across training. SVCCA was
used to show that representations tend to converge “bottom-up,” and later work
introduced CKA as a robust similarity measure:

- [Raghu et al., *SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability*](https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-understanding-and-improvement)
- [Kornblith et al., *Similarity of Neural Network Representations Revisited*](https://proceedings.mlr.press/v97/kornblith19a)

### 2.1 Geometry speed

For adjacent base checkpoints, use

$$
d_{\mathrm{CKA}}(t,t+\Delta)
=
\arccos\!\left(\operatorname{CKA}(H_t,H_{t+\Delta})\right),
$$

$$
v_{\mathrm{CKA}}
=
\frac{d_{\mathrm{CKA}}(t,t+\Delta)}
     {\Delta\text{ optimizer updates or processed tokens}}.
$$

The angle is preferable to $(1-\mathrm{CKA})/\Delta$. Near identical
representations, $1-\cos\theta$ is quadratic in the small movement, whereas
$\theta$ behaves like an ordinary first-order distance. Angular CKA also
supports meaningful path-length and wandering calculations:

- [Williams et al., *Generalized Shape Metrics on Neural Representations*](https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html)
- [Lange et al., *Deep Networks as Paths on the Manifold of Neural Representations*](https://proceedings.mlr.press/v221/lange23a.html)

Linear CKA is normalized linear-kernel HSIC. Linear CKA and linear HSIC should
therefore not be presented as two independent confirmations. Unnormalized HSIC
also confounds motion with activation scale.

### 2.2 Sampling and activation layout

Use exactly the same immutable held-out examples at every checkpoint, with
augmentation disabled.

- For CNNs, use corresponding image/spatial locations as observations and
  channels as features. This permits channel-coordinate rotations while
  preserving spatial correspondence. A globally pooled view can be included as
  a sensitivity analysis.
- For GPT-2, use fixed story/token positions as observations and the 768
  residual channels as features. Bootstrap uncertainty by story, never by
  treating correlated tokens as independent samples.
- Compute both raw and class-/position-mean-residualized CKA. The difference
  indicates whether motion is dominated by mean structure, but it is not an
  exact cumulant decomposition.
- Compute CKA in the native representation rather than the independently fitted
  surrogate PCA spaces.

The dashboard should show:

1. a cut-by-training-interval heatmap of angular CKA speed;
2. checkpoint-by-checkpoint distance matrices at sentinel cuts; and
3. cumulative path length divided by initial-to-final distance.

A path-length ratio substantially above one indicates wandering, reversal, or
backtracking. With sparse checkpoints it is only a lower bound on the actual
path length.

### 2.3 CKA’s central blind spot

CKA deliberately ignores orthogonal channel rotations and isotropic scaling.
This is useful when comparing independently trained networks, but potentially
misleading for tracking. If

$$
H_{t+\Delta}=H_tQ
$$

for an orthogonal rotation $Q$, CKA reports no movement even though a
coordinate-fixed suffix may need to rotate its first weight matrix by
$Q^{-1}$.

CKA can also miss functionally important low-variance directions:

- [Ding et al., *Grounding Representation Similarity Through Statistical Testing*](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97-Abstract.html)

CKA should therefore be paired with a direct functional measurement.

### 2.4 The stale-suffix splice

For adjacent checkpoints, form the cross-time matrix

| | Old suffix | New suffix |
|---|---:|---:|
| Old prefix | intact old model | reverse splice |
| New prefix | **stale-suffix splice** | intact new model |

The focal cell feeds the new prefix’s activations into the old frozen suffix.
Report its true loss and predictive KL relative to the two intact models.

This is a deliberately strict form of model stitching. No learned adapter is
allowed in the primary measurement because learning an adapter would remove the
coordinate mismatch that the suffix itself had to track. Model stitching is an
established task-facing complement to representational similarity:

- [Bansal et al., *Revisiting Model Stitching to Compare Neural Representations*](https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)

An optional diagnostic can fit an orthogonal Procrustes map on a separate fit
bank:

- if an orthogonal map restores performance, the incompatibility is largely
  basis motion;
- if it does not, the representation’s geometry or task-relevant content has
  changed.

For VGG, freeze the old suffix’s BatchNorm buffers in the primary splice and
include a “recalibrate BN statistics only” control.

### 2.5 Interpretation

| CKA motion | Stale-suffix damage | Interpretation |
|---|---|---|
| High | Low | Much of the geometry change is irrelevant to this suffix |
| Low | High | CKA is hiding a basis/scale change or small sensitive direction |
| High | High | The interface is moving in a functionally consequential way |
| Low | Low | The interface is plausibly easy for this suffix to track |

During active learning, call this **representation motion** rather than
representational drift. “Drift” is normally used for motion accompanied by
stable task performance:

- [Pashakhanloo and Koulakov, *Stochastic Gradient Descent-Induced Drift of Representation in a Two-Layer Neural Network*](https://proceedings.mlr.press/v202/pashakhanloo23a.html)

### 2.6 Eventual timescale comparison

After both clocks are validated in comparable units, an exploratory
tracking-load proxy is

$$
\chi_{t,\ell}
=
v_{\mathrm{interface}}(t,\ell)\,
\tau_{90}(t,\ell).
$$

It reads as “approximately how far the interface moves while the suffix takes
one relaxation time to adapt.”

This product should not be reported from the current protocols because base
training and suffix relaxation use different learning rates, optimizer states,
and sometimes minibatch streams. Initially, show the two quantities beside one
another. A dimensionally defensible $\chi$ requires a matched continuation
control using the base optimizer and the same processed-example/token clock.

## 3. Noise robustness and Fisher

The literature recalled in the conversation consists of two distinct papers:

- [Arora et al., *Stronger Generalization Bounds for Deep Nets via a Compression Approach*](https://proceedings.mlr.press/v80/arora18b.html)
  defines layer and interlayer cushions and directly studies propagation of
  Gaussian activation noise through VGG-19 on CIFAR-10.
- [Zhang, Bengio, and Singer, *Are All Layers Created Equal?*](https://jmlr.org/papers/v23/20-069.html)
  identifies critical and robust modules by replacing final-model layers with
  their initialized or newly randomized versions.

The existing critical-module transplantation corresponds to the second paper.
It is useful context, but it is not a noise cushion, a capacity measurement, or
an HOC-dependence measurement.

### 3.1 Main local object: cut-level activation Fisher

At a base checkpoint and frozen representation $h$, compute the model Fisher
with respect to the cut activation. Operationally:

1. sample an output from the model’s predictive distribution;
2. backpropagate its log probability only as far as $h$;
3. record the squared activation-gradient norm; and
4. normalize by activation norm and dimension.

For an isotropic perturbation with
$\lVert\delta h\rVert=\alpha\lVert h\rVert$, the small-noise prediction is

$$
\mathbb E\,D_{\mathrm{KL}}
\left(
p(\cdot\mid h)
\;\middle\|\;
p(\cdot\mid h+\delta h)
\right)
\approx
\frac{\alpha^2}{2}
\frac{\lVert h\rVert^2}{d}
\operatorname{tr}G_h.
$$

In plain language, this measures:

> How much does this frozen suffix notice a 1% perturbation at its input?

This is well aligned with B/C because it lives at the exact interface under
study, has predictive-KL units at every cut, requires no retraining, and
directly tests the hypothesis that later suffixes are noise robust.

It remains a local measurement. A suffix can be locally smooth while depending
on finite, global higher-order structure.

### 3.2 Validate with finite noise curves

At every base checkpoint and cut, inject Gaussian activation noise with relative
norms

$$
\alpha\in\{0.01,0.03,0.10,0.30\}.
$$

Use several fixed-seed draws and record:

- predictive KL;
- change in true-label cross-entropy;
- accuracy or prediction disagreement as an intuitive secondary readout; and
- the small-noise Fisher prediction.

Use at least these noise geometries:

1. full activation-space isotropic noise;
2. noise restricted to the retained PCA space; and
3. noise restricted to the discarded PCA space.

For convolutional cuts, include a channel-coherent or covariance-shaped
control. Completely IID spatial noise may be removed simply by downstream
averaging and could make a suffix appear robust for an uninteresting reason.

The finite curves are a validity check on the Fisher approximation. If the
small-$\alpha$ KL does not follow the quadratic prediction, the local Fisher
should not be extrapolated to those perturbation sizes.

### 3.3 Proper parameter Fisher at the base checkpoints

Also estimate the model Fisher of the intact network:

$$
F_\theta
=
\mathbb E_x\mathbb E_{y\sim p_\theta(\cdot\mid x)}
\left[
\nabla_\theta\log p_\theta(y\mid x)
\nabla_\theta\log p_\theta(y\mid x)^\top
\right].
$$

Report, by parameter module:

- total trace;
- trace per parameter; and
- fraction of whole-model trace.

The suffix trace at a cut is the sum over its modules, so all cuts receive a
base-only Fisher quantity without a separate eigensolver.

This reproduces the main object used by Achille, Rovere, and Soatto to study
layerwise “effective connectivity” during critical periods:

- [Achille, Rovere, and Soatto, *Critical Learning Periods in Deep Networks*](https://openreview.net/forum?id=BkeStsCcKQ)

Important controls:

- track predictive entropy, because categorical Fisher can shrink as the model
  becomes confident;
- report raw trace, per-parameter trace, and relative trace rather than treating
  any one as parameterization-independent;
- keep the probe-bank size fixed; and
- do not compare raw ResNet and VGG traces as though architecture and
  parameterization were irrelevant.

The existing observed-label gradient outer product is an empirical Fisher. It
should remain separate because it is not generally the model Fisher, GGN, or
Hessian:

- [Kunstner, Hennig, and Balles, *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*](https://proceedings.neurips.cc/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)

### 3.4 Fisher-weighted PCA coverage

Using the fitted orthonormal PCA basis $V$, calculate

$$
\rho_{\mathrm{PCA}}
=
\frac{
\mathbb E\left\lVert
V^\top\nabla_h\log p(y\mid h)
\right\rVert^2
}{
\mathbb E\left\lVert
\nabla_h\log p(y\mid h)
\right\rVert^2
}.
$$

This asks:

> What fraction of the suffix’s local predictive sensitivity lies inside the
> retained PCA subspace?

This is a proposed proxy for this project, derived directly from the
predictive-KL/Fisher geometry. It is more relevant than explained variance
alone: a discarded direction may have low activation variance but high
predictive importance.

Report both:

- total sensitivity fraction retained; and
- sensitivity per retained dimension versus per discarded dimension.

High Fisher-weighted coverage is still only a local result and cannot prove that
PCA preserves finite non-Gaussian structure.

## 4. Hessian feasibility and scope

Approximate model sizes are:

| Model | Parameters |
|---|---:|
| B residual CNN | 0.61M |
| ResNet-18 | 11.2M |
| VGG-19+BN | 20.6M |
| GPT-2 small | 124M |

Dense Hessians are impossible except for the tiny terminal classifiers.
Matrix-free Hessian-vector products and Lanczos are empirically realistic for
the CIFAR models. Ghorbani et al. used stochastic Lanczos quadrature on
networks at and beyond this scale:

- [Ghorbani, Krishnan, and Xiao, *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density*](https://proceedings.mlr.press/v97/ghorbani19b.html)

### 4.1 Proposed CIFAR pilot

At early, middle, and late checkpoints and cuts:

- evaluate the suffix-restricted Hessian $H_{bb}$ at relaxation update zero;
- estimate its largest positive and most negative eigenvalues;
- estimate the corresponding model-Fisher/GGN directions;
- record curvature along the actual initial relaxation gradient,
  $g^\top H_{bb}g/g^\top g$; and
- validate the matrix-free implementation against the deepest B and ResNet
  classifier suffixes, where exact dense calculations are feasible.

This may help explain initial relaxation behavior, but Hessian eigenvalues do
not automatically predict $\tau_{90}$:

- the largest eigenvalue controls stiffness, not the slowest response;
- slow response depends on the small-curvature modes that the actual gradient
  occupies;
- momentum changes the recurrence;
- Adam introduces optimizer state and preconditioning; and
- the relaxation path can leave the local quadratic region.

A later, selected-cell extension could start Lanczos from the actual
true/mean/Gaussian initial gradient. This would estimate the curvature spectrum
seen by each relaxation direction without measuring any post-update Hessian.

### 4.2 Part E

For GPT-2, port the activation-space measurements first:

- per-token relative activation noise;
- predictive KL averaged over valid positions;
- activation Fisher; and
- sensitivity inside versus outside the PCA subspace.

A 124M-parameter double-backward spectral calculation is technically possible,
but it should follow mechanical validation on CIFAR and a one-cell runtime and
memory benchmark. It is not the first E measurement to run.

### 4.3 VGG caveat

For VGG, use frozen evaluation-mode BatchNorm buffers for the primary
curvature measurement. This is curvature of the fixed predictor and does not
include the BatchNorm-state evolution that occurs during a training-mode
relaxation. It can be reported descriptively but should not be sold as a
complete predictor of VGG relaxation speed.

## 5. Usable capacity: controlled readers rather than one scalar

The native suffix becomes dramatically smaller and less nonlinear with depth:

| Architecture | Early measured suffix | Deepest measured suffix |
|---|---:|---:|
| B residual CNN | about 601k parameters | about 1.3k parameters |
| ResNet-18 | about 11.1M | about 5.1k |
| VGG-19 | about 20.3M | about 0.53M |
| GPT-2 | about 116.6M | about 38.6M |

The deepest B and ResNet suffixes are essentially pooling plus a linear
classifier. GPT cut 11 retains many parameters because of the vocabulary head,
but functionally it is LayerNorm plus a linear map.

Late-cut mean sufficiency may therefore mean either:

1. the prefix has converted the task into a simple, low-order form; or
2. the remaining suffix is too simple to exploit additional structure.

### 5.1 Decoder ladder

At every cut, train:

1. the same standardized linear reader;
2. the same standardized small nonlinear reader;
3. the native suffix from its checkpoint warm start; and
4. the native suffix from a fresh initialization.

Keep the standardized readers’ depth, width, fit set, optimizer, update budget,
and validation rule fixed within an architecture family.

This follows the idea of **predictive usable information**: information is
measured relative to a declared class of readers rather than as an
unqualified high-dimensional mutual information:

- [Xu et al., *A Theory of Usable Information Under Computational Constraints*](https://arxiv.org/abs/2002.10689)
- [Kleinman et al., *Usable Information and Evolution of Optimal Representations During Training*](https://openreview.net/forum?id=dmVxElcgKZd)

Interpretation:

- A linear reader succeeding late supports “the prefix has already resolved
  the task into an accessible form.”
- A freshly initialized native suffix outperforming the checkpoint warm start
  suggests an optimization or plasticity problem.
- A standardized nonlinear reader succeeding where the native suffix fails
  suggests a native-suffix capacity limitation.
- No reader succeeding means only that the information is unusable by the
  declared reader families; it does not prove that the information is absent.

The first pass should train standardized readers on true activations only. If
this reveals an informative depth pattern, repeat the surrogate cross-evaluation
at a small set of sentinel cuts rather than immediately multiplying the full
B/C/E matrix.

### 5.2 Secondary audits

From the same activation banks, report:

- effective or entropy rank of the centered representation;
- within-class effective rank;
- fraction of variance attributable to class means;
- dormant ReLU fraction where applicable; and
- native suffix parameter count and number of nonlinear blocks.

Effective rank and dormant-unit fraction have been used as correlates of
plasticity loss:

- [Dohare et al., *Loss of Plasticity in Deep Continual Learning*](https://www.nature.com/articles/s41586-024-07711-7)

They should remain supporting diagnostics. Neither rank nor parameter count is,
by itself, a functional measure of capacity.

## 6. PCA ablation

PCA is operationally necessary because early internal activations can have tens
of thousands of coordinates. A full class-conditional covariance would be both
enormous and severely under-sampled.

The present PCA surrogate reconstructs discarded directions at the global mean.
Consequently, “Gaussian” currently means a Gaussian in the retained PCA
subspace, not a full-space Gaussian activation distribution.

### 6.1 Projected-true control

Project real held-out activations into the retained PCA subspace and reconstruct
them without Gaussianizing. Compare:

1. full true activations;
2. projected true activations;
3. Gaussian samples in the same retained space; and
4. mean-only samples.

This decomposes:

- **projection damage:** full true versus projected true; and
- **within-subspace Gaussian mismatch:** projected true versus Gaussian.

Part E already contains adaptive PCA and projected-true machinery. B/C should
gain the same decomposition.

### 6.2 Dimension sweep

At sentinel cells, select PCA dimension by held-out variance targets such as
80%, 90%, 95%, and 99%, subject to the sample-rank ceiling. At every dimension,
show both:

- ordinary variance coverage; and
- Fisher-weighted sensitivity coverage $\rho_{\mathrm{PCA}}$.

If variance coverage is high but Fisher coverage is low, PCA is discarding
low-variance directions that the suffix cares about. If both are high and
projected-true replay is faithful, PCA is less likely to explain the surrogate
shortfall.

A later control could use a full-space low-rank-plus-diagonal Gaussian, retaining
the class mean and residual marginal variance outside the PCA subspace. This is
not needed for the first gate.

## 7. Critical-module context

The existing final-target transplantation results can remain at the bottom of
Part C. They ask whether the final co-adapted network depends on a particular
learned parameter block.

They should not be interpreted as:

- capacity;
- activation noise robustness;
- HOC dependence; or
- a critical period.

If checkpoint alignment becomes important, a later extension can target each
base checkpoint $t$ and replace one module with:

- its checkpoint-0 version;
- a fresh random version; or
- its immediately preceding checkpoint version.

Measure predictive KL, cross-entropy damage, and accuracy damage. For VGG,
separate resetting learned weights/affine parameters from resetting BatchNorm
buffers, and include BN-statistics recalibration without gradient updates.

This would be a project-specific extension of Zhang et al., not the exact
published experiment.

## 8. Proposed dashboard views

Each B/C/E section should receive the same compact dynamics-and-capacity grammar
rather than a large generic diagnostics tab.

### View 1: Can the suffix catch up?

- depth-by-training-time heatmap of $\tau_{90}$;
- speed-versus-ceiling scatter;
- true-loss trajectory area; and
- clear `PLATEAU` versus `UNRESOLVED` status.

Interpretation guide:

- fast plus small shortfall: adapts quickly and successfully;
- fast plus large shortfall: quickly learns the wrong surrogate task;
- slow plus small shortfall: useful but difficult adaptation;
- slow plus large shortfall: both optimization and sufficiency are problematic.

### View 2: How quickly does the interface move?

- angular CKA speed heatmap;
- sentinel checkpoint-distance matrices;
- stale-suffix splice damage; and
- optional path-length ratio.

The CKA/splice 2-by-2 interpretation table should appear before the measured
plots.

### View 3: What does the suffix notice?

- finite activation-noise curves;
- activation-Fisher small-noise prediction;
- Fisher-weighted PCA retained/discarded sensitivity; and
- proper modulewise model-Fisher allocation.

### View 4: Is usable reader capacity changing?

- standardized linear/nonlinear reader losses;
- native warm-start versus reinitialized suffix;
- effective-rank audit; and
- native suffix depth/parameter count.

All uncertainty labels must distinguish:

- surrogate-draw variation;
- image/story bootstrap uncertainty; and
- independent model-seed variation.

Images, tokens, cuts, and checkpoints from one training run are not independent
model-level evidence.

## 9. Repository feasibility and provenance

### Part B

The older Part-B depth-by-time runner trains a fresh base model for each nominal
checkpoint/cut and sets the cosine schedule length using the requested endpoint:

- [`src/tracking2/suffix_statistics.py`](../src/tracking2/suffix_statistics.py)

Thus its epochs 1, 5, 10, 20, and 30 are not checkpoints from one training
trajectory, and nominally identical epoch/cut cells can contain different
weights. Those historical artifacts cannot support a longitudinal CKA or
velocity claim.

The July 26 publication battery instead uses uninterrupted, independently trained
30-epoch CNN trajectories with audited training manifests, but measures only the
epoch-30 endpoints at cuts 1 and 4. It resolves the publication comparison, not
the proposed motion measurement. Before measuring B motion, use checkpoints from
one fixed trajectory and have every cut reference identical checkpoint bytes.

The older B relaxation paths can still be used for exploratory kinetic
analysis, with the provenance limitation stated, but their endpoints are not
currently demonstrated plateaus.

### Part C

- ResNet has a genuine seed-0 trajectory at epochs 0, 1, 5, 20, and 100 and can
  support checkpoint-only motion/Fisher/noise analysis now.
- VGG base checkpoints exist, but the corrected seven-cut suffix-statistics
  sweep is not yet complete. Checkpoint-only measurements remain possible.
- Sparse 0/1/5/20/100 checkpoints give coarse secant motion, not instantaneous
  velocity or a trustworthy measure of oscillatory path length.

### Part E

- Exact 0/4M/16M/50M checkpoints and immutable replay banks exist.
- Base-model motion, activation Fisher, and noise measurements can be computed
  without retraining.
- The repaired schema-v2 surrogate analysis currently covers the 50M endpoint;
  comparable 4M and 16M repairs are needed for a clean time-dependent
  surrogate comparison.
- The existing cadence gives coarse interval motion. Future training should
  compute lightweight activation Gram snapshots more frequently if genuine
  velocity is a headline.

## 10. Recommended implementation order

1. **Repair provenance**
   - preserve existing artifacts;
   - establish canonical checkpoint hashes and immutable probe banks;
   - rerun B as one shared trajectory before longitudinal claims.

2. **Add plateau-aware relaxation kinetics**
   - derive exploratory quantities from current paths;
   - add adaptive continuation and `UNRESOLVED` status to new runs;
   - use equal-LR/shared-index timing controls for E.

3. **Run the checkpoint-only dynamics pass**
   - angular CKA;
   - stale-suffix splices;
   - effective-rank audits;
   - finite activation-noise curves;
   - activation Fisher; and
   - Fisher-weighted PCA coverage.

4. **Add controlled readers**
   - true-activation standardized linear/nonlinear readers;
   - native warm-start and reinitialized suffix;
   - expand to surrogate readers only at informative sentinel cuts.

5. **Add proper parameter Fisher**
   - exact ten-class validation on a small CIFAR subset;
   - sampled scalable estimator on the full probe bank;
   - modulewise trace maps across base training.

6. **Run the curvature sentinel**
   - exact tiny-head validation;
   - B and ResNet suffix-restricted Hessian/GGN at selected cells;
   - benchmark VGG separately because of BatchNorm;
   - defer GPT-2 spectra until CIFAR is mechanically validated.

7. **Expand only if predictive**
   - scale curvature across depth/time only if the sentinel quantities explain
     observed relaxation behavior;
   - do not add curvature during relaxation until base-checkpoint curvature has
     proved useful.

## 11. Decisions requested

1. For “capacity,” should the first standardized-reader pass use true
   activations only, with surrogate readers added only if it reveals an
   informative pattern?
   - **Recommendation:** yes.

2. Is it acceptable to rerun B from one canonical shared trajectory before
   making any representation-velocity claim?
   - **Recommendation:** yes; the current B artifacts cannot support that
     longitudinal claim.

3. Should the first Hessian pilot cover B and ResNet only, or include VGG
   immediately despite its BatchNorm complications?
   - **Recommendation:** B and ResNet first; benchmark VGG only after the
     matrix-free implementation agrees with exact tiny-head calculations.

## Literature access

The literature descriptions above were prepared from complete open-access PDFs,
not abstract-only or paywalled summaries.
