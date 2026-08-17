# Free-body diagrams for neural networks

*This is an exploratory CIFAR-10 analysis. The animated depth-by-time plots are
historical pilots that motivated the experiment; the strongest quantitative
comparison is a later controlled follow-up across three independently trained
small CNNs. I see a consistent difference across depth, but I do not yet know
what causes it.[^ai]*

## Summary

- Previous work on **distributional simplicity bias** finds that neural
  networks tend to use simple statistics of their input data before using more
  complicated ones.
- I apply the same question inside a network. Pick a layer, hold everything
  before it fixed, and treat its outputs as the training data for everything
  after it.
- I replace those internal activations with PCA-projected empirical, fitted Gaussian,
  or class-mean-plus-noise replay. I then continue training copies of the later
  part and test every copy on held-out real activations.
- The pilot animations make the qualitative pattern visible: simplified replay
  changes early suffixes much more than late suffixes. Those CNN checkpoints
  came from separately scheduled models, and the ResNet used one trained model,
  so the animations are demonstrations rather than independent replications.
- In the controlled follow-up, all three independently trained CNNs show the
  same shallow-versus-final contrast after matched first updates. Moving the
  cut also changes the receiver's size and trainability, so this is not a
  representation-only depth effect.
- The result I am willing to keep is narrow: under this finite suffix-training
  protocol, the native late suffix is less sensitive to replay distribution
  than the native early suffix. Stronger interpretations are unsafe.

## Looking at one part of a network at a time

Training curves tell us how the whole network improves, but not how learning is
divided among its parts.

Draw a line after one layer. Everything before the line turns images into
activation vectors. For everything after it, those vectors are data. As the
first part learns, that data changes.

A free-body diagram makes a similar move in mechanics: isolate one piece of a
system and describe what crosses its boundary. Repeating this across small
pieces is one route from Newtonian mechanics to continuum mechanics. Here I
move a cut through the network and ask what the layers after each cut can learn
from the data crossing it.

There is no force balance or conservation law in this experiment. During
ordinary training both sides of the cut change at once. I hold the earlier side
fixed so I can study further learning on the later side.

![A comparison between using free-body diagrams to build a local description
of a material and moving an internal cut through a neural
network.](https://charlesr-w.github.io/Tracking2/figures/fbd_materials_nn_analogy.svg)

*Figure 1. The shared move is to isolate one boundary, measure what crosses it,
and repeat at different positions. This is a guide for organizing the
experiment, not a claim that neural networks obey continuum mechanics.*

## Distributional simplicity bias

The question I ask at each cut comes from work on
[distributional simplicity bias](https://proceedings.mlr.press/v202/refinetti23a.html).

A data distribution can be described by statistics of increasing order. Its
first cumulant is its mean. Its second cumulant is its covariance, which records
how pairs of variables vary together. Higher-order cumulants describe
dependencies that are not fixed by the mean and covariance.[^cumulants]

A Gaussian distribution is fully determined by its mean and covariance. If we
replace a dataset with a fitted Gaussian, we target those low-order statistics
while removing its empirical non-Gaussian structure. Comparing learning on
PCA-projected empirical and Gaussian replay asks when the missing structure matters
inside the retained PCA subspace. It does not identify a particular
higher-order cumulant, and a finite generated bank does not exactly reproduce
held-out moments.

[Refinetti, Ingrosso, and Goldt](https://proceedings.mlr.press/v202/refinetti23a.html)
compared real CIFAR with several generated training sets. A
class-conditional Gaussian matching each class's mean and covariance reproduced
the early part of the real-data learning curve. More realistic generated data
matched it for longer. [Belrose et
al.](https://proceedings.mlr.press/v235/belrose24a.html) turned this around:
they trained on real data, then tested checkpoints on maximum-entropy datasets
that preserved selected low-order statistics. Early checkpoints worked
relatively well on these replacements. As training continued, performance on
real data improved while performance on the low-order replacements fell.

There is also a practical reason to care about the order. In simple
high-dimensional models, a useful direction encoded only in higher-order
statistics can take much more data to find than one encoded in the mean or
covariance. [Székely et
al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f8af4eebc4e50994e0490898d891c96-Abstract-Conference.html)
study this gap directly.

[Bardone and Goldt's *Sliding Down the
Stairs*](https://proceedings.mlr.press/v235/bardone24a.html) studies why a
higher-order direction can be slow to learn and what can speed it up. In the
basic version of their model, class information is placed in a mean direction,
a covariance direction, and a direction available only through higher-order
statistics. Networks tend to learn these in stages, with plateaus between them.
When the covariance signal and higher-order signal come from correlated latent
variables, the covariance gives an early clue to the higher-order direction and
the long plateau largely disappears.

This line of work studies statistics at the input to the network. My question
is what happens if we carry the same analysis through the network, one cut at a
time. I am not yet testing Bardone and Goldt's explanation. I am asking whether
the same mean $\rightarrow$ covariance $\rightarrow$ higher-order breakdown is
useful inside a trained network. The mean-only comparison turns out to depend
on its noise definition, so I use that ladder as motivation rather than a
conclusion.

## The experiment

The historical pilot used two CIFAR-10 image classifiers: a small four-block
residual CNN and a normalization-free pre-activation ResNet-18. The controlled
follow-up reran the small CNN across three independently trained model
seeds.[^protocol]

![Schematic architectures of the four-block residual CNN and normalization-free
pre-activation ResNet-18, with every measured cut
marked.](https://charlesr-w.github.io/Tracking2/figures/cnn_resnet_architectures.svg)

*Figure 2. The small CNN has four residual blocks. The ResNet has eight
residual blocks arranged in four stages. A cut freezes the prefix and turns its
saved activations into training data for the remaining suffix.*

At an ordinary-training checkpoint, I repeat the following procedure after a
chosen block:

1. Hold the prefix fixed and save its labelled training activations.
2. Fit one global PCA basis at that cut.
3. Construct up to four replay datasets:

   - **Raw empirical:** the saved activations themselves.
   - **PCA-projected empirical:** those activations reconstructed through the
     fitted PCA basis.
   - **Gaussian:** class-conditional samples using each class's fitted mean and
     empirical covariance in PCA coordinates.
   - **Mean + isotropic noise:** samples centred on each class mean with the
     same spherical noise scale for every class.

![Labelled images pass through a frozen prefix; its activations are projected,
fitted class by class, and replayed as empirical, Gaussian, or mean-plus-noise
datasets.](https://charlesr-w.github.io/Tracking2/figures/method_make_datasets.svg)

*Figure 3. The cut turns one internal interface into a labelled dataset.
PCA-projected empirical replay separates the effect of PCA truncation from
Gaussianizing the retained coordinates. This delivered-talk schematic says
“regularized covariance” because it depicts the fixed-LR pilot; the controlled
headline instead uses exact empirical covariance. The point clouds are
explanatory, not measured geometry.*

For the mean control I use

$$
\mathcal N(\mu_c,r^2\bar v I),
$$

where $\bar v$ is the pooled average within-class variance in retained PCA
coordinates. Only $r=0$ is centroid-only; $r=1$ trace-matches the average fitted
within-class covariance, and other radii change the nuisance-noise scale.

I then compare what further learning each replay dataset supports:

1. Clone the suffix from the checkpoint.
2. Train each copy for a short relaxation horizon on one replay dataset.
3. Evaluate every copy on the same held-out true activations.

![Four activation datasets train identical copies of the remaining layers,
which are all evaluated on the same held-out real
activations.](https://charlesr-w.github.io/Tracking2/figures/method_train_compare.svg)

*Figure 4. In the fixed-LR pilot depicted here, only the suffix-training
distribution changes. Every copy begins at the same checkpoint and is
evaluated on the same held-out empirical interface. In the controlled
headline, condition-specific learning rates also change to match each first
update to raw empirical replay.*

This is not an immediate corruption test. It asks what further learning each
version of the internal data supports from the network's current state.

Two controls matter for interpreting the comparison.

First, a Gaussian fitted after PCA can differ from true activations because of
discarded directions as well as non-Gaussian structure. PCA-projected empirical is the
matched empirical baseline inside the retained subspace.[^pca]

Second, a shared learning rate produced wildly different first-update norms at
the shallow cut: the Gaussian and mean-$r=1$ updates were about 184 and 287
times the true-replay update. The controlled follow-up therefore rescales each
condition's learning rate to match its first update to raw empirical replay. This is a
scale control, not whole-trajectory optimizer matching. The pilot animations
predate both controls and should be read as phenomenon demonstrations.

## What happens across depth

### The waterfall pilots

The plots that first made the phenomenon legible were not endpoint tables but
waterfalls. For each cut, they show how much held-out accuracy changed during a
short burst of suffix relaxation on real, Gaussian, or mean-plus-noise replay.

![Animated CNN waterfall across ordinary-training checkpoints and four internal
cuts.](https://charlesr-w.github.io/Tracking2/figures/cnn_over_training_time.gif)

*Figure 5. Four-block CNN pilot. Each frame is an ordinary-training checkpoint;
the horizontal axis moves the cut from block 1 to block 4. The coloured
segments are outcomes for alternative suffix copies, stacked by sign only to
make comparison compact. They are not additive components. The checkpoint
models were separately scheduled runs, so this animation is not one model's
training trajectory. “Real activations” in the pilot legend means raw empirical
replay; the pilot has no PCA-projected empirical condition.[^runs]*

The striking part is the geometry of the picture. At early cuts, relaxation can
move held-out accuracy by tens of percentage points, and the replay conditions
separate. By the final cut, all three copies barely move. The change across the
nominal checkpoints is non-monotonic; because these are different CNN runs, I
do not interpret that sequence as learning dynamics.

The underlying suffix curves show what one bar compresses:

![Animated CNN suffix-relaxation curves at the cut after block 3 for one
epoch-1 checkpoint.](https://charlesr-w.github.io/Tracking2/figures/cnn_relaxation_time.gif)

*Figure 6. One epoch-1 pilot checkpoint unpacked. Each curve traces held-out
raw-activation accuracy while a suffix copy trains on one replay distribution;
the animation reveals successive suffix-relaxation epochs from 0 to 10.*

The ResNet pilot repeats the visual sweep with eight cuts from one trained
model:

![Animated ResNet comparison across eight internal cuts and five checkpoints.](https://charlesr-w.github.io/Tracking2/figures/resnet_over_training_time.gif)

*Figure 7. One-model ResNet-18 pilot. The bands share a zero baseline; they are
not stacked contributions. Repeated analyses redraw generated activations and
minibatch order, not independently trained ResNets. Fresh crop and flip draws
also differ across cuts, so this is not architecture-level replication.*

The animations are the most informative view of the original experiment, but
their provenance sets a hard limit: they show the phenomenon that motivated the
controlled battery, not the uncertainty estimate for the final claim.

The follow-up below tests the same qualitative shallow-versus-final contrast
with a different endpoint: paired final cross-entropy differences between
conditions, not the pilot waterfalls' accuracy change from each warm start. The
two numerical scales are not directly comparable.

### Controlled follow-up across three CNNs

The controlled comparison uses three independently trained epoch-30 CNNs, cuts
after blocks 1 and 4, rank 2,048, warm suffixes, matched first updates, one
generated replay bank per condition and model, and five relaxation epochs. The
table reports paired held-out true cross-entropy differences, mean $\pm$ sample
standard deviation across the three model seeds. Positive means the first
named replay distribution finished with higher cross-entropy.

| Cut | Gaussian $-$ PCA-projected empirical | Mean ($r=1$) $-$ Gaussian |
|---|---:|---:|
| After block 1 | $+0.183 \pm 0.055$ | $+0.144 \pm 0.019$ |
| After block 4 | $-0.012 \pm 0.002$ | $+0.023 \pm 0.011$ |

After block 1, the replay conditions separate consistently. After block 4,
the differences are small, and Gaussian replay finishes slightly below
PCA-projected empirical replay. The independent units are trained model seeds. There is
only one replay bank per condition and model, so within-model redraw uncertainty
is unmeasured.

The same PCA rank is asymmetric in native terms. After block 1 it retains 2,048
of 32,768 coordinates and fails the declared predictive-KL projection gate;
after block 4 the native dimension is itself 2,048 and projection is effectively
exact. PCA-projected empirical replay controls truncation *within* each cut but
does not equalize intervention severity *across* cuts. A seed-0 rank-4,096 shallow check
passes the projection gate and keeps the same ordering, but that rank was
selected after looking at the gate and was not repeated across seeds.

The predeclared gate requires both an initial PCA-projected-minus-raw empirical
cross-entropy excess of at most $0.05$ nat and predictive KL from raw to
projected predictions of at most $0.02$ nat. At shallow rank 2,048 the
cross-entropy excess is $0.0154$ nat but predictive KL is $0.0373$ nat, so it
passes the first threshold and fails the second. This does not erase the
within-subspace replay comparison; it prevents treating it as evidence that the
projection preserves the full shallow interface.

The mean-versus-Gaussian result is also sensitive to the nuisance-noise radius.
At seed 0 and cut 1, $r=1$ loses to the Gaussian, $r=2$ is approximately tied,
and the $r=0$ centroid cell finishes below it. That centroid cell is only
approximately update-matched: its learning-rate multiplier hits the $0.001$
floor, leaving its first update $9.463\%$ larger than raw empirical replay. I therefore
do not infer that class covariance has a radius-independent advantage.

In separate seed-0 sensitivities, reinitializing the suffix and extending the
relaxation horizon to twenty epochs preserve the shallow-versus-final contrast.
Those are one-model checks, and longer replay increasingly mixes mismatch with
forgetting.

The full matrices, exact values, PCA gates, and provenance are in the
[interactive ablation appendix](https://charlesr-w.github.io/Tracking2/).

The result I am willing to keep is:

> With matched first-update norms, all three epoch-30 CNN seeds showed the same
> qualitative contrast under the rank-2,048 protocol: replay conditions
> separated after block 1 and were close after block 4.

This compares each cut's native receiver. It does not isolate representation
depth from receiver capacity, and it does not establish a canonical
depth-by-time trajectory.

## What could cause the difference across depth?

For now the observation is descriptive: after late cuts, the three copies end
much closer together than they do after early cuts. The experiment does not yet
tell us why.

There are several possible reasons.

One possibility is that the early blocks have made the classes easier to
separate. Near the output, differences between class means may already contain
most of what the remaining layers use.

But moving the cut also changes how much network remains. An early cut leaves
several nonlinear blocks; a late cut may leave little more than the classifier.
I should therefore repeat the comparison with the same small classifier
attached at every cut.

The primary suffixes begin from the current checkpoint rather than from
scratch. A seed-0 reinitialization check preserves the qualitative contrast,
but it also shows that the fresh shallow receiver learns much less within the
same budget. The experiment does not measure every piece of information present
in the activations, or everything a newly initialized receiver could eventually
learn from them.

Every comparison uses a finite training budget. The pilots use ten extra epochs
for the small CNN and five for the ResNet; the controlled headline uses five
matched-update epochs. Some early-cut curves are still moving, and some copies
trained on simplified data get worse on real activations. Even at twenty
epochs, the endpoint mixes adaptation speed, distribution mismatch, and
forgetting.

Finally, PCA removes part of the activation space before synthetic replay. The
controlled battery adds PCA-projected empirical replay, which separates PCA truncation from
Gaussianization within a cut, but the shallow rank-2,048 projection still fails
the predictive-KL gate. I should not call the real-versus-Gaussian difference a
measurement of higher-order cumulants alone.

## From static cuts to training dynamics

During ordinary training, the earlier part of the network is not fixed. It
continually changes the activation distribution seen by the later part. This
suggests two different jobs:

- **Resolving:** learning more of the useful structure in the current
  activation distribution.
- **Tracking:** staying adapted while that distribution moves.

This suggests two measurements at each cut: how quickly the activations move,
and how long the remaining layers take to adjust. Comparing those times across
depth is the longer-term goal. The current experiment takes a first step by
stopping the activations from moving and measuring further learning on that
fixed distribution.

![A proposed comparison between movement of the activation distribution and
the time the remaining layers need to adjust.](https://charlesr-w.github.io/Tracking2/figures/tracking_resolving_only.png)

*Figure 8. In normal training, the data crossing a cut moves while the later
layers learn. The proposed analysis compares the speed of that movement with
the time the later layers need to adjust.*

![The longer-term programme cycles between measuring a moving internal
distribution, perturbing it, and measuring the suffix response.](https://charlesr-w.github.io/Tracking2/figures/closing_cycle.svg)

*Figure 9. The proposed middle-out loop: measure the interface, intervene on
its distribution, and compare the downstream response over training.*

## Next steps

1. Attach the same capacity-matched receiver at every cut. This is the most
   important missing control for separating the interface from the native
   suffix that consumes it.
2. Repeat the adequacy-passing rank-4,096 shallow comparison across independent
   model seeds and a fresh validation population. The present rank was selected
   after inspecting the projection gate.
3. Replace first-update matching with optimizer controls that match update scale
   over time, or compare explicit learning-rate sweeps.
4. Test mean-only controls that do not require choosing an arbitrary spherical
   nuisance distribution.
5. Run an uninterrupted depth-by-time battery. The pilot waterfall is visually
   suggestive, but its CNN frames are not one model trajectory.
6. Measure representation movement on a fixed image bank and compare its
   timescale with suffix relaxation.

My next goal is to measure reliably which statistics support learning after
each cut, then ask whether movement of the internal data helps explain the
differences across depth. I would especially welcome suggestions for a
capacity-matched comparison across depth, or for a measure of representation
movement that is meaningful to the later layers.

[^ai]: Drafting and data checks were assisted by Codex. The experimental
    choices, interpretation, and final text are the author's responsibility.

[^cumulants]: For a Gaussian distribution, all cumulants above order two are
    zero. Real image and activation distributions are not Gaussian, so matching
    their mean and covariance does not generally match the full distribution.

[^protocol]: The pilot ResNet is a normalization-free pre-activation CIFAR
    ResNet-18, not the torchvision BatchNorm model. It used all 50,000 CIFAR-10
    training images, 10,000 saved training activations per suffix analysis, a
    5,000-example PCA fit, and 2,000 held-out examples. The CNN batteries used
    50,000 training and 10,000 held-out activations, with PCA fitted on the
    first 10,000 ordered training activations. ResNet cuts were re-encoded with
    fresh augmentation draws. Exact controlled-battery inputs and provenance
    are recorded in the public appendix.

[^pca]: Flattened activation tensors can have tens of thousands of coordinates.
    A dense per-class covariance is expensive and sample-rank limited. The
    controlled Gaussian uses exact empirical covariance in retained PCA
    coordinates; the historical pilots used 5% spherical shrinkage. A finite
    generated bank adds sampling error in either case.

[^runs]: The CNN pilot frames use separately trained/scheduled models and one
    generated activation dataset of each type. The ResNet pilot uses one trained
    network and three redraw analyses per cut; those redraw generated
    activations and minibatch order. The controlled headline instead uses three
    independently trained CNN seeds, with one replay bank per condition and
    seed.
