# Free-body diagrams for neural networks

*Work in progress. This is an exploratory CIFAR-10 research note, not a settled
claim about how deep networks represent concepts. The strongest comparison
uses three independently trained small CNNs; several important controls still
use one model.[^ai]*

## The short version

- A network prefix turns images into a new, learned data distribution for the
  suffix. I freeze a prefix, simplify that internal distribution, and ask what
  the suffix can learn from each replacement.
- After the first of four CNN blocks, replay on real projected activations,
  a fitted class Gaussian, and class means plus isotropic noise produces
  clearly different results. After the final block, their accuracies are
  almost indistinguishable.
- That depth contrast survives three model seeds, a longer training horizon,
  and reinitializing the suffix. But moving the cut also changes the suffix's
  size and trainability, so this is not a representation-only effect.
- Two controls substantially weakened my original interpretation. A shared
  learning rate made the shallow synthetic updates hundreds of times larger
  than the real-data update. And whether Gaussian replay beats mean-only replay
  depends on how much isotropic noise I add to the class means.
- The remaining result is narrower: under this finite relaxation protocol,
  the native late suffix is much less sensitive to the replay distribution
  than the native early suffix is. I do **not** yet have a clean measurement of
  “how much covariance” or “how many higher moments” each layer uses.

The full matrices, provenance, PCA gates, and ablations are in the
[interactive data appendix](../free-body-diagrams-for-neural-networks.html).

## Why look at this?

Most descriptions of neural networks start either from individual parameters
or from the input-output function. I want a useful middle scale: learned
interfaces, activation distributions, and the optimization pressures acting
between parts of a network.

This is tentative, but it seems relevant to alignment. Methods such as
[activation steering](https://proceedings.mlr.press/v235/singh24d.html) act on
representations at a particular layer, often with one affine direction. Their
success can therefore depend on the local geometry where that direction is
applied. Representations can also drift while behaviour stays stable, in both
[artificial networks](https://proceedings.mlr.press/v202/pashakhanloo23a.html)
and biological systems: Driscoll et al. tracked mouse parietal neurons during
a stable virtual-maze task and found that individual neurons' relationships to
the task [continually reorganized](https://pmc.ncbi.nlm.nih.gov/articles/PMC5718200/).

Those observations make me wary of an ontology in which a concept is simply
“stored in these weights” or “is this fixed direction.” A more useful account
might describe how a representation is produced, what downstream computation
currently relies on, and how both change during training. The present
experiment is a small frozen-interface test of that broader programme.

## Every prefix is a data generator

Write a checkpointed network as

$$
x \xrightarrow{\phi_{t,\ell}} z
\xrightarrow{\psi_{t,\ell}} \hat y .
$$

The prefix $\phi_{t,\ell}$ pushes the labelled image distribution forward to
an activation distribution $P_{t,\ell}(z,y)$. For the suffix
$\psi_{t,\ell}$, those activations are its effective dataset.

A free-body diagram in mechanics isolates one part of a system and records
what crosses its boundary. I make the same organizational move here: cut the
network, freeze the prefix, and study the data crossing that cut. This is an
analogy, not a claim that neural networks obey a force-balance law.

![A macroscopic distributional law motivates placing a cut inside a neural
network and treating the activations crossing it as data.](figures/conceptual_internal_cut.svg)

*Figure 1. Distributional simplicity bias is normally asked about the external
dataset. An internal cut lets us ask the analogous operational question about
the learned distribution presented to a suffix.*

The motivating background is
[distributional simplicity bias](https://proceedings.mlr.press/v202/refinetti23a.html):
networks often exploit simple statistics of an input distribution before more
complicated ones. The first cumulant is the mean; the second is covariance. A
Gaussian is fixed by those two statistics, while real activation distributions
also contain non-Gaussian structure.[^cumulants] My question is whether this
mean/covariance/full-distribution ladder is useful at internal interfaces.

## The frozen-interface experiment

I trained a four-block residual CNN with GroupNorm on CIFAR-10. For each
checkpoint and cut I cache the prefix activations, then construct four replay
datasets:

1. **True:** the empirical activations.
2. **Projected true:** the same activations reconstructed through a fitted PCA
   basis.
3. **Gaussian:** class-conditional samples with the fitted mean and exact
   empirical covariance in PCA coordinates.
4. **Mean + isotropic noise:** samples around each class mean with a declared
   class-independent spherical noise scale.

![A class-conditioned image distribution is pushed through a fixed prefix,
fitted in PCA coordinates, and replaced by projected-real, Gaussian, or
mean-plus-noise replay.](figures/class_conditioned_pushforward_pca.svg)

*Figure 2. The PCA basis is fitted on the first 10,000 ordered training
activations. Class moments are then estimated from all 50,000 training
activations. The held-out evaluation bank contains 10,000 examples.*

I copy the checkpoint suffix for each condition, train it on one replay
dataset, and evaluate it on held-out true activations. The primary readable
quantity is held-out cross-entropy; accuracy is secondary. The dashboard also
retains the full train-distribution × evaluation-distribution matrix.

Projected true matters because a low-rank Gaussian otherwise mixes two
changes: discarding activation directions and Gaussianizing the retained
coordinates. It does not magically “isolate PCA loss”—training on projected
activations and deploying on true activations is itself a train/deploy
intervention. It does provide the matched empirical baseline for comparisons
inside the retained subspace.

### Why not use every PCA component?

The input image has $32\times32\times3=3{,}072$ values, but the activation
after block 1 has $32\times32\times32=32{,}768$ coordinates. PCA acts on that
activation tensor, not on input pixels or network parameters.

There are also only about 5,000 training examples per class. A per-class
empirical covariance therefore has rank at most about 4,999, even before
storage and factorization costs. “All 32,768 components” would not produce a
well-estimated full-rank class covariance.

I instead predeclared a functional projection gate:

- absolute update-0 cross-entropy change at most $0.05$ nat; and
- predictive KL from true to projected predictions at most $0.02$ nat.

For seed 0 after block 1:

| PCA rank | Total variance | Within-class variance | CE change | Predictive KL | Gate |
|---:|---:|---:|---:|---:|:---:|
| 2,048 | 88.0% | 87.6% | +0.0154 | 0.0373 | Fail |
| 3,072 | 91.2% | 90.9% | +0.0083 | 0.0211 | Fail |
| 4,096 | 93.0% | 92.8% | +0.0044 | 0.0146 | Pass |

Rank 4,096 was selected after looking at this held-out gate, so it is
exploratory rather than an independent confirmation. The cheaper three-seed
rank-2,048 comparison below is explicitly a **projected-subspace** result.

### The optimizer-scale control

A shared learning rate sounds fair, but it did not imply comparable
interventions. At seed 0, cut 1, rank 2,048, the first Gaussian update was
about 184 times the true-replay update; mean-$r1$ was about 287 times larger.
Much of the apparent synthetic-data failure was an optimizer shock.

I therefore report two regimes:

- **Fixed LR:** the literal intervention in which every copy uses the same
  optimizer and learning rate.
- **Matched first update:** a sensitivity that rescales each condition's
  learning rate so the measured first update norm matches true replay.

The latter uses different learning rates and only matches the first batch; it
is a scale control, not a uniquely correct optimizer.

## Results

### One shallow cell that passes the PCA gate

At seed 0, cut 1, rank 4,096, after five matched-update epochs:

| Replay training data | Held-out true accuracy | Held-out true CE |
|---|---:|---:|
| True | 78.06% | 0.948 |
| Projected true | 78.03% | 0.938 |
| Exact Gaussian | 74.61% | 1.065 |
| Mean + isotropic noise ($r=1$) | 72.09% | 1.233 |

Inside this adequacy-passing cell, Gaussian replay is 0.128 nat worse than
projected-real replay, and mean-$r1$ is another 0.168 nat worse than Gaussian.
This says the fitted Gaussian and the trace-matched mean control do not support
the same five-epoch update as empirical projected activations. It does not tell
me which higher-order feature causes the gap.

### The depth contrast across three trained CNNs

For cross-seed replication I used rank 2,048 and the matched-update regime.
The table reports paired held-out CE differences after five epochs,
mean ± sample SD across three independently trained CNNs. Positive means the
distribution on the right did worse.

| Cut | Gaussian − projected true | Mean-$r1$ − Gaussian |
|---|---:|---:|
| After block 1 | $+0.183 \pm 0.055$ | $+0.144 \pm 0.019$ |
| After block 4 | $-0.012 \pm 0.002$ | $+0.023 \pm 0.011$ |

After block 1, the replay conditions separate consistently. After block 4,
their accuracies are indistinguishable, and Gaussian replay actually has
slightly lower cross-entropy than projected-real replay. The result is not
“simplified data always falls behind”; it is that the late native suffix is
insensitive to these replay changes under this protocol.

![Measured CNN controls showing the three-seed depth contrast, optimizer-scale
sensitivity, and mean-noise-radius ablation.](figures/cnn_measured_controls.png)

*Figure 3. Current measured controls only. Rank 2,048 shallow results are
projected-subspace comparisons; the rank-4,096 seed-0 cell above is the
adequacy-passing shallow check.*

### Controls that changed the interpretation

**Optimizer regime.** At the same seed-0 cut-1 checkpoint, matching the first
update changed Gaussian CE from 1.805 to 1.045 and mean-$r1$ CE from 3.313 to
1.193. The ordering survived, but most of its fixed-LR magnitude did not. At
cut 4, all fixed/matched differences were small. Effect-size claims are
therefore optimizer-regime-dependent.

**Mean-noise radius.** Radius 1 is trace-matched: its spherical covariance has
the same trace as the average fitted within-class covariance. Radius 0 uses
exact centroids; radius 0.5 has one quarter of the trace, and radius 2 has four
times the trace.

At seed 0, cut 1, rank 2,048:

| Replay | Accuracy | CE |
|---|---:|---:|
| Exact Gaussian | 75.27% | 1.045 |
| Mean, $r=0$ | 76.07% | 0.991 |
| Mean, $r=0.5$ | 70.84% | 1.324 |
| Mean, $r=1$ | 71.93% | 1.193 |
| Mean, $r=2$ | 70.33% | 1.041 |

The centroid condition beats the Gaussian, while mean-$r1$ loses to it.
Therefore the tempting sentence “Gaussian beats mean, so class covariance
helped” is not robust to the nuisance-noise definition. The late cut remains
insensitive across all four radii.

**Covariance estimator.** The primary Gaussian uses exact empirical covariance
in PCA space with no hidden diagonal jitter. Replacing it with 5% spherical
shrinkage changes seed-0 CE by only $-0.0084$ at cut 1 and $+0.0005$ at cut 4.
Shrinkage is not driving the result. A finite generated bank still does not
exactly reproduce held-out moments, so “targets the fitted covariance” is more
accurate than “matches the held-out covariance.”

**Warm start.** Reinitializing the suffix preserves the qualitative seed-0
depth contrast: after block 1, true/projected/Gaussian/mean accuracies are
68.7%, 64.5%, 43.4%, and 37.8%; after block 4 they are all about 77.9%. But
the fresh shallow true suffix itself only reaches 68.7% in five epochs. This
shows that inherited suffix knowledge is not required for the contrast while
also making the capacity/trainability confound more obvious.

**Longer horizon.** Extending matched replay from five to twenty epochs does
not close the shallow gaps. Gaussian-minus-projected CE grows from 0.120 to
0.147 nat; mean-minus-Gaussian grows from 0.148 to 0.462 nat. At the final cut,
accuracies remain within 0.08 percentage points. Longer replay increasingly
mixes mismatch and forgetting, so this is a sensitivity rather than an
asymptotic capability measurement.

![Held-out true cross-entropy through twenty matched-update replay epochs at
the shallow and final CNN cuts.](figures/cnn_measured_horizon.png)

*Figure 4. The shallow replay conditions continue to separate; the final-cut
accuracies remain nearly unchanged even as calibration loss drifts.*

## What I think survives

The robust observation is operational and modest:

> A native suffix after the first CNN block is much more sensitive to the
> replay distribution than the final linear suffix is, under both warm and
> reinitialized finite-budget relaxation.

Several stronger stories do not survive:

- The fixed-LR gap is not a representation-only effect; update scale explains
  much of its size.
- “Covariance helps” is not well-defined without specifying the mean-control
  noise distribution.
- Moving the cut changes the receiver. The final cut leaves a linear
  classifier; the first cut leaves three nonlinear blocks and many more
  parameters.
- A Gaussian gap does not identify a particular higher-order cumulant or prove
  that such structure is absent/present elsewhere.

The older ResNet pilot showed a superficially similar depth pattern, but it
uses one trained model, incomplete source lineage, and independently sampled
crop/flip augmentations at different cuts. I have excluded it from the
canonical dashboard and from the evidence above rather than calling it an
architecture replication.

## From frozen cuts to tracking

In ordinary training, the prefix is not frozen. It continually changes the
activation distribution seen by the suffix. That suggests two jobs:

- **Resolving:** learning useful structure in the current activation
  distribution.
- **Tracking:** staying adapted while that distribution moves.

![A simple tracking-versus-resolving decomposition for future
measurements.](figures/tracking_resolving.svg)

*Figure 5. The current experiment freezes movement and measures a relaxation
response. The longer-term project is to compare representation movement with
the suffix's adaptation time.*

This is where the middle-out framing may become useful. Rather than choose
between a parameter-level story and a static feature-geometry story, we can
measure the evolving distribution at an interface and the downstream system's
response to it. That could eventually help explain representation drift or
when a steering direction remains valid. The present note only establishes
the measurement problem more carefully.

## Next tests

1. Attach the same capacity-matched receiver at every cut. This is the most
   important missing control for the depth comparison.
2. Repeat the adequacy-passing rank-4,096 shallow cell across independent
   model seeds and a fresh validation population. The current rank was chosen
   exploratorily on the test bank.
3. Replace endpoint-only first-update matching with optimizer controls that
   match update scale over time, or compare learning-rate sweeps directly.
4. Test alternative mean-only controls that do not require an arbitrary
   spherical nuisance distribution.
5. Measure representation movement on a fixed image bank during ordinary
   training, then compare its timescale with suffix relaxation.

I would especially welcome suggestions for a clean capacity-matched receiver
or a mean-only baseline whose nuisance geometry is less arbitrary.

[^ai]: Codex assisted with experiment engineering, checks, dashboard code, and
    drafting. The experimental choices, interpretation, and final text remain
    the author's responsibility.

[^cumulants]: For a Gaussian distribution, cumulants above order two vanish.
    Real activation distributions are not Gaussian, so matching a mean and
    covariance does not match their full distribution.
