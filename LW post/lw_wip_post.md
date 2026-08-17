# Free-body diagrams for neural networks

*Work in progress. This is an exploratory CIFAR-10 research note, not a settled
claim about how deep networks represent concepts. The strongest comparison
uses three independently trained small CNNs; several important controls still
use one model. The ResNet result is an earlier one-model pilot, not an
architecture-level replication.[^ai]*

## The short version

- A network prefix turns images into a new, learned data distribution for the
  suffix. I freeze a prefix, simplify that internal distribution, and ask what
  the suffix can learn from each replacement.
- After the first of four CNN blocks, replay on real projected activations, a
  fitted class Gaussian, and class means plus isotropic noise produces clearly
  different cross-entropies. After the final block, the differences are small.
- This early-versus-late contrast repeats across three independently trained
  CNNs. In seed-0 sensitivities, it also survives a longer relaxation horizon
  and reinitializing the suffix. But moving the cut changes the suffix's size
  and trainability, so this is not a representation-only effect.
- Two controls weakened my original interpretation. A shared learning rate
  made the shallow Gaussian and mean-replay first updates about 184 and 287
  times larger than the true-replay update. And whether Gaussian replay beats
  mean-only replay depends on how much isotropic noise I add to the class
  means.
- The result I am willing to keep is narrower: under this finite relaxation
  protocol, the native late suffix is much less sensitive to the replay
  distribution than the native early suffix is. I do **not** yet have a clean
  measurement of “how much covariance” or “how many higher moments” each layer
  uses.

The full matrices, provenance, PCA gates, and ablations are in the
[interactive ablation appendix](free-body-diagrams-for-neural-networks.html).

## Introduction and motivation

A deep network makes its own data.

After any block, the prefix has turned the external dataset into a new, learned
distribution. For the remaining suffix, that distribution is the training
data. As the prefix learns, its output distribution moves.

I want to study this internal data directly. The organizing analogy is a
free-body cut. Continuum mechanics gets a useful macroscopic description of
matter by cutting it into pieces and asking what crosses their boundaries. I
make the same move here: move a cut through a network, describe the activation
distribution crossing it, and test what the suffix can learn from controlled
replacements.

![Particles become a continuum description by moving a local cut through a
material; a whole network becomes a layerwise description by moving an
internal cut through its blocks.](figures/conceptual_internal_cut.svg)

*Figure 1. Moving a boundary turns one global problem into a family of local
ones. The analogy is about scale and method, not about forces.*

This is a middle-out question. Most descriptions of neural networks start
either from individual parameters or from the whole input-output function. I
want useful objects at an intermediate scale: learned interfaces, activation
distributions, and the optimization pressures acting between parts of a
network.

Calling an activation distribution “mesoscale” does not make it substantive.
It would have to earn that status through useful predictive or interventional
invariances. The present experiment is a small frozen-interface test of that
broader programme.

### Why simplify distributions?

The immediate experiment comes from
[distributional simplicity bias](https://proceedings.mlr.press/v202/refinetti23a.html).
Networks often exploit simple statistics of their input distribution before
more complicated ones.

The first cumulant of a distribution is its mean. The second is its covariance.
A Gaussian is fixed by those two statistics; a real image or activation
distribution also has non-Gaussian structure.[^cumulants] This gives a crude
ladder:

$$
\text{class mean}
\quad\longrightarrow\quad
\text{class mean and covariance}
\quad\longrightarrow\quad
\text{the empirical distribution}.
$$

[Refinetti, Ingrosso, and Goldt](https://proceedings.mlr.press/v202/refinetti23a.html)
found that Gaussian replacements could reproduce the early part of learning
on CIFAR-10, while more realistic generated data matched learning for longer.
[Belrose et al.](https://proceedings.mlr.press/v235/belrose24a.html) asked the
complementary question: after training on real data, which statistics suffice
to reproduce a checkpoint's behaviour?

Both studies intervene at the network input. I apply the same comparison at
learned interfaces inside the network.

## The frozen-interface experiment

I trained a four-block residual CNN with GroupNorm on CIFAR-10. At an
ordinary-training checkpoint $t$, split the network after block $\ell$:

$$
f_t=\psi_t^\ell\circ\phi_t^\ell.
$$

If $P_c=\operatorname{Law}(X\mid Y=c)$ is the image distribution for class
$c$, the prefix pushes it forward to

$$
Q_{t,\ell,c}=(\phi_t^\ell)_\#P_c.
$$

This $Q_{t,\ell,c}$ is the activation distribution crossing the cut. The
notation is heavier than the idea: the prefix makes a dataset, and the suffix
consumes it.

At each cut I cache the prefix activations, then construct four replay
datasets:

1. **True:** the empirical activations.
2. **Projected true:** the same activations reconstructed through a fitted PCA
   basis.
3. **Gaussian:** class-conditional samples with the fitted mean and exact
   empirical covariance in PCA coordinates.
4. **Mean + isotropic noise:** samples around each class mean with a declared
   class-independent spherical noise scale $\rho$.

![A class-conditioned image distribution is pushed through a fixed prefix,
fitted in PCA coordinates, and replaced by projected-real, Gaussian, or
mean-plus-noise replay.](figures/class_conditioned_pushforward_pca.svg)

*Figure 2. The PCA basis is fitted on the first 10,000 ordered training
activations. Class moments are then estimated from all 50,000 training
activations. The held-out evaluation bank contains 10,000 examples.*

The PCA fit is explicit. Flatten each activation to $z_i\in\mathbb R^D$, fit
one global basis $U_k$ with centre $m$, and define

$$
T_k(z)=U_k^\top(z-m),
\qquad
R_k(a)=m+U_ka.
$$

Within each class, fit a mean $\mu_c$ and empirical covariance $\Sigma_c$ in
PCA coordinates. The main Gaussian uses that exact covariance, with no default
shrinkage. The mean control is

$$
\mathcal N(\mu_c,\rho^2\bar v I_k),
\qquad
\bar v=\frac{1}{Ck}\sum_{c=1}^{C}\operatorname{tr}(\Sigma_c).
$$

Thus $\rho=1$ gives the spherical distribution the across-class average
within-class covariance trace; $\rho=0$ gives exact class centroids. Projected
true matters because a low-rank Gaussian comparison would otherwise mix two
changes: discarding activation directions and Gaussianizing the retained
coordinates. It is the matched empirical baseline for comparisons inside the
retained subspace.

For each replay condition, I make an identical copy of the checkpoint suffix,
train it for a finite budget, and evaluate it on held-out *true* activations.
The primary readable quantity is held-out cross-entropy; accuracy is secondary.

This is not a damage test on one network. It asks what further learning each
version of the internal data supports from the same checkpoint.

### Why not use every PCA component?

The input image has $32\times32\times3=3{,}072$ values, but the activation
after block 1 has $32\times32\times32=32{,}768$ coordinates. PCA acts on that
activation tensor, not on input pixels or network parameters.

There are also only about 5,000 training examples per class. A per-class
empirical covariance therefore has rank at most about 4,999, even before
storage and factorization costs. “All 32,768 components” would not produce a
well-estimated full-rank class covariance.

I predeclared a functional projection gate:

- absolute projected-minus-true update-0 cross-entropy at most $0.05$ nat; and
- predictive KL from true to projected predictions at most $0.02$ nat.

For seed 0 after block 1:

| PCA rank | Total variance | Within-class variance | Projected − true CE | Predictive KL | Gate |
|---:|---:|---:|---:|---:|:---:|
| 2,048 | 88.0% | 87.6% | +0.0154 | 0.0373 | Fail |
| 3,072 | 91.2% | 90.9% | +0.0083 | 0.0211 | Fail |
| 4,096 | 93.0% | 92.8% | +0.0044 | 0.0146 | Pass |

Rank 4,096 was selected after looking at this held-out gate, so it is
exploratory rather than an independent confirmation. The cheaper three-seed
rank-2,048 comparison below is explicitly a **projected-subspace** result, not
a claim about full-space sufficiency.

### The optimizer-scale control

A shared learning rate sounds fair, but it did not imply comparable
interventions. At seed 0, cut 1, rank 2,048, the first projected, Gaussian, and
mean-$r1$ update norms were about 2.5, 184, and 287 times the true-replay
update. Much of the apparent synthetic-data failure was an optimizer shock.

I therefore keep two regimes distinct:

- **Fixed LR:** every copy uses the same optimizer and learning rate.
- **Matched first update:** each replay condition's learning rate is rescaled
  so its measured first-update norm matches true replay.

At this shallow cell, the endpoint held-out cross-entropies were:

| Regime | Projected true | Exact Gaussian | Mean-$r1$ |
|---|---:|---:|---:|
| Fixed LR | 0.967 | 1.805 | 3.313 |
| Matched first update | 0.925 | 1.045 | 1.193 |

The ordering survives, but most of the fixed-LR magnitude does not. Matching
uses different learning rates and only matches the first batch. It is a scale
control, not whole-trajectory optimizer matching or a uniquely correct
optimizer.

## Results

### The depth contrast across three trained CNNs

The primary comparison uses three independently trained epoch-30 CNNs, cuts
after blocks 1 and 4, rank 2,048, warm suffixes, matched first updates, one
surrogate draw, and five relaxation epochs. The table reports paired held-out
true cross-entropy differences, mean $\pm$ sample standard deviation across
the three models. Positive means the distribution on the right did worse.

| Cut | Gaussian − projected true | Mean-$r1$ − Gaussian |
|---|---:|---:|
| After block 1 | $+0.183 \pm 0.055$ | $+0.144 \pm 0.019$ |
| After block 4 | $-0.012 \pm 0.002$ | $+0.023 \pm 0.011$ |

After block 1, the replay conditions separate consistently. After block 4,
the cross-entropy differences are small, and Gaussian replay actually finishes
slightly below projected-real replay. Accuracy tells the same coarse story:
the three conditions differ at the shallow cut and finish in nearly the same
place at the final cut. The result is not “simplified data always falls
behind”; it is an early-versus-late sensitivity contrast under this finite,
projected-subspace protocol.

![Measured CNN controls showing the three-seed depth contrast, optimizer-scale
sensitivity, and mean-noise-radius ablation.](figures/cnn_measured_controls.png)

*Figure 3. Current measured controls only. Rank-2,048 shallow results are
projected-subspace comparisons; the rank-4,096 seed-0 cell below is the
adequacy-passing shallow check.*

### One shallow cell that passes the PCA gate

At seed 0, cut 1, rank 4,096, after five matched-update epochs:

| Replay training data | Held-out true CE | Held-out true accuracy |
|---|---:|---:|
| Projected true | 0.938 | 78.03% |
| Exact Gaussian | 1.065 | 74.61% |
| Mean + isotropic noise ($r=1$) | 1.233 | 72.09% |

The shallow ordering persists in this adequacy-passing cell. But this is one
model seed, and the rank was chosen after inspecting the same held-out
population. I treat it as exploratory support, not confirmation.

### Controls that changed the interpretation

**Mean-noise radius.** Radius 1 trace-matches the average fitted within-class
covariance. Radius 0 uses exact centroids; radius 0.5 has one quarter of that
trace, and radius 2 has four times the trace.

At seed 0, cut 1, rank 2,048, with matched first updates:

| Replay | Held-out true CE |
|---|---:|
| Exact Gaussian | 1.044668 |
| Mean, $r=0$ | 0.991185 |
| Mean, $r=0.5$ | 1.323755 |
| Mean, $r=1$ | 1.192704 |
| Mean, $r=2$ | 1.041234 |

The centroid condition beats the Gaussian, while mean-$r1$ loses to it.
Therefore the tempting sentence “Gaussian beats mean, so class covariance
helped” is not robust to the nuisance-noise definition. The late cut remains
relatively insensitive across the declared radii.

**Covariance estimator.** The primary Gaussian uses exact empirical covariance
in PCA space with no hidden diagonal jitter. Replacing it with 5% spherical
shrinkage changes seed-0 cross-entropy by only $-0.0084$ at cut 1 and
$+0.0005$ at cut 4. Shrinkage is not driving the contrast. A finite generated
bank still does not exactly reproduce held-out moments, so “targets the fitted
covariance” is more accurate than “matches the held-out covariance.”

**Warm start.** Reinitializing the suffix preserves the qualitative seed-0
contrast. After block 1, true/projected/Gaussian/mean-$r1$ accuracies are
68.70%, 64.46%, 43.39%, and 37.79%; after block 4 they are all about 77.9%.
Inherited suffix knowledge is not required for the contrast. This does not
equalize receiver capacity: the fresh shallow suffix itself only reaches 68.7%
in five epochs.

**Longer horizon.** Extending matched replay from five to twenty epochs does
not close the shallow gaps. Gaussian-minus-projected cross-entropy grows from
0.120 to 0.147 nat; mean-minus-Gaussian grows from 0.148 to 0.462 nat. At the
final cut, epoch-20 accuracies span only 77.88% to 77.96%. Longer replay
increasingly mixes mismatch and forgetting, so this is a sensitivity rather
than an asymptotic capability measurement.

![Held-out true cross-entropy through twenty matched-update replay epochs at
the shallow and final CNN cuts.](figures/cnn_measured_horizon.png)

*Figure 4. The shallow replay conditions continue to separate. The final-cut
accuracies remain nearly unchanged even as calibration loss drifts.*

## What I think survives

The safe claim is operational and modest:

> Under matched first-update norm, three independently trained epoch-30 CNNs
> show a robust contrast between cut 1 and cut 4 in the rank-2,048 retained
> subspace after five relaxation epochs. Under the additional one-model
> sensitivities, reinitialization and a twenty-epoch horizon preserve that
> qualitative contrast.

The adequacy-passing rank-4,096 seed-0 cell keeps the same shallow ordering.
The late native suffix is comparatively insensitive under this finite
protocol.

Several stronger interpretations are unsafe:

- Rank 2,048 fails the predictive-KL PCA gate at the shallow cut, so the
  three-seed result is not a full-space sufficiency result.
- “Higher moments appear later” is not licensed. A Gaussian gap does not
  identify a particular higher-order cumulant, and a small gap does not prove
  that higher-order information is absent.
- “Covariance helps” is not stable without specifying the mean-control noise
  distribution.
- Moving the cut changes the receiver. The final cut leaves nearly a linear
  classifier; the first cut leaves three nonlinear blocks and many more
  trainable parameters. Reinitialization does not remove that confound.
- Matching the first update does not match later optimizer dynamics.
- The one-model, test-selected rank-4,096 cell is not confirmatory evidence.
- The current battery covers epoch 30 and cuts 1 and 4. It does not establish a
  canonical depth-by-time trajectory.

### Earlier pilots and methods demonstrations

Earlier CNN depth-by-time plots used a different protocol: rank 256, fixed
learning rate, covariance shrinkage, no projected-real control, and separately
scheduled checkpoints. The ResNet animation likewise comes from one trained
model with incomplete lineage and differing augmentation draws across cuts.
Those figures remain useful as methods demonstrations, but I have removed them
from the quantitative evidence chain above. They are not inputs to the
canonical dashboard, which is CNN-only. The ResNet pilot is not
architecture-level replication. The talk remains an archival “as delivered”
account of those preliminary pilot results.

A separate one-model sequence diagnostic passed its repaired mechanical gates
but was scientifically negative: Gaussian replay did not reliably beat the
mean control. I do not fold it into the CIFAR evidence. Any redesigned
sequence-model study is future work.

## From resolving to tracking

The present experiment freezes the prefix. It measures how the suffix resolves
structure in one fixed activation distribution.

Ordinary training adds a second problem. The prefix changes, so the activation
distribution moves; the suffix must track that movement while continuing to
learn.

![At time t a prefix produces an activation cloud for a suffix; at a later
time the prefix, cloud, and adapting suffix have all
moved.](figures/tracking_resolving.svg)

*Figure 5. Ordinary training mixes movement of the internal data with the
suffix's response. The current experiment removes the movement: it freezes the
prefix and measures suffix relaxation on one fixed activation distribution.
Tracking the moving cloud is future work.*

This is the version of the middle-out programme I find most promising. Rather
than choose between a parameter-level story and a static feature-geometry
story, measure the evolving distribution at an interface and the downstream
system's response to it. That could eventually help explain representation
drift or when a steering direction remains valid. The present note only
establishes the measurement problem more carefully.

## Next tests

1. Attach the same capacity-matched receiver at every cut. This is the most
   important missing control for the depth comparison.
2. Repeat the adequacy-passing rank-4,096 shallow cell across independent model
   seeds and a fresh validation population. The current rank was chosen
   exploratorily on the test bank.
3. Replace first-update matching with optimizer controls that match update
   scale over time, or compare learning-rate sweeps directly.
4. Test alternative mean-only controls that do not require an arbitrary
   spherical nuisance distribution.
5. Measure representation movement on a fixed image bank during ordinary
   training, then compare its timescale with suffix relaxation.

The immediate result is small but concrete: the same statistical replacement
has a very different effect depending on where it enters the network.

[^ai]: Experiment engineering, checks, figures, and drafting were assisted by
    Claude/Codex. The experimental choices, interpretation, and final text are
    the author's responsibility.

[^cumulants]: For a Gaussian distribution, cumulants above order two vanish.
    Real activation distributions are not Gaussian, so matching a mean and
    covariance does not match their full distribution.
