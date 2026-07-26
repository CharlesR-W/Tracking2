# Free-body diagrams for neural networks

*Research note; work in progress.* This is an exploratory CIFAR-10 experiment,
not a settled result. The current evidence comes from several small residual
CNN runs and one ResNet-18 training trajectory. A cleaner uninterrupted CNN
run, plus PCA and noise ablations, is being incorporated. I am publishing the
note now partly to make the work easier to criticize while it is still
changeable.[^ai]

## The result in one paragraph

[Distributional simplicity
bias](https://proceedings.mlr.press/v202/refinetti23a.html) is a whole-network
regularity: neural networks often learn what can be read from low-order
statistics before they use more detailed structure in the data. I make an
internal cut in a network and treat the activations crossing it as a new
labelled dataset. I replace that dataset with simpler, class-conditional
surrogates, train identical copies of the remaining layers for a short fixed
budget, and evaluate every copy on held-out real activations. In the completed
runs, training on simplified activations falls farther behind training on real
activations after early cuts than after late cuts. The bounded claim is:

> Under warm-started, finite-budget, suffix-only training, low-order
> surrogates support less of the useful update after early cuts than after late
> cuts.

This does **not** yet show that representations become simpler with depth, that
training creates the depth pattern, or that non-Gaussian structure is the sole
cause of the gaps.

## Why look at this scale?

Most accounts of neural networks start either from small parts—weights,
neurons, heads—or from large outcomes such as loss, capability, and behaviour.
I am interested in the level between them: distributions of activations,
adaptation times, and other collective variables that might remain useful when
individual units change.

This is my tentative “middle-out” idea. The aim is not to declare a new
ontology in advance. A useful intermediate variable should earn its place by
making predictions and supporting interventions across changes of seed,
architecture, task, or scale. The activation distribution at a cut is one
candidate, and this experiment is a small test of how to work with it.

Two results helped push me in this direction. [*Are All Layers Created
Equal?*](https://jmlr.org/papers/v23/20-069.html) finds striking differences in
how much performance suffers when different trained layers are reinitialized
or rerandomized. [*Transformer Layers as
Painters*](https://doi.org/10.1609/aaai.v39i24.34708) finds that
lower and final transformer layers are special, while some tasks tolerate
surprising changes to middle layers, including skipping or reordering them.
Neither result supplies a general theory, but both make a simple
layer-by-layer story feel incomplete.

My hunch is that representations will often be better explained by the
optimization pressures that form and maintain them—especially the timescales
on which different parts must adapt—than by the static geometry of a final
activation cloud alone. This experiment does not establish that hunch. It
freezes one side of the dynamics so that one adaptation timescale can be
measured cleanly.

There is also a possible alignment connection. Steering vectors intervene
directly on intermediate activations, but their effects can vary substantially
across inputs and can be brittle to reasonable prompt changes ([Tan et al.,
2024](https://papers.nips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html)).
My colleague Brianna is considering a related unpublished question: how does
the region in which an activation-space intervention is approximately linear
change during training? If the relevant representation or its local geometry
moves, a direction found at one time or context may not remain the same
intervention elsewhere. That is a motivation, not a result of this study.

Representation drift makes the same point more starkly. In mice performing a
stable virtual-navigation task, the individual posterior-parietal neurons most
informative about maze features changed over days, even while useful
population-level information remained ([Driscoll et al.,
2017](https://doi.org/10.1016/j.cell.2017.07.021)). Related work studies the
geometry of drift in mouse visual cortex and in artificial networks ([Aitken et
al., 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1010716)).
This is evidence against identifying a representation too quickly with a fixed
list of units. I do **not** have evidence that representation drift causes the
known failures of steering vectors. The weaker lesson is that alignment tools
which act on representations may need an account of how those representations
are maintained and changed.

## From a macroscopic regularity to an internal cut

[Refinetti, Ingrosso, and
Goldt](https://proceedings.mlr.press/v202/refinetti23a.html) compare learning on
real images with learning on synthetic datasets that preserve progressively
more statistics. A class-conditional Gaussian, which preserves each class mean
and covariance but removes non-Gaussian structure, can reproduce the early
part of learning better than it reproduces the later part. [Belrose et
al.](https://proceedings.mlr.press/v235/belrose24a.html) approach the question
from the other direction: train on real data, then evaluate checkpoints on
maximum-entropy replacements that preserve chosen statistics. Their results
also support a broad progression from simple distributional structure to
richer structure.

I treat this as a macroscopic law-like regularity about networks. The analogy
to mechanics is organizational, not mathematical. Newton's \(F=ma\) describes
a whole body; a free-body diagram lets us isolate one part and apply the same
law at its boundary. Here, distributional simplicity bias is the whole-network
regularity. I cut the network after a block, call what crosses the cut “data,”
and ask the same question of the suffix.

![Distributional simplicity bias at the input, followed by an internal cut
that turns the same question onto a network suffix.](figures/conceptual_internal_cut.svg)

*Figure 1. A macroscopic regularity, then the internal-cut move. The free-body
analogy says where to draw the boundary; it does not imply a force law for
networks.*

## The measurement

The two testbeds are a small four-block residual CNN and a normalization-free,
pre-activation CIFAR ResNet-18—not the standard torchvision BatchNorm
model.[^protocol]

![Standard block diagrams of the CNN and ResNet-18, with the measured
interfaces marked as possible cuts.](figures/cnn_resnet_architectures.svg)

*Figure 2. The same operation is applied to two conventional residual
architectures. Later plots repeat a small version of the relevant network and
cut inside the axes.*

Write the checkpointed network as a prefix followed by a suffix. At training
time \(t\), a cut after block \(\ell\) gives a prefix
\(\phi_{t,\ell}\). For class \(c\), the distribution crossing the cut is the
pushforward

\[
P_{t,\ell}(z\mid y=c)
=
(\phi_{t,\ell})_{\#}P(x\mid y=c).
\]

In plain language: take, for example, all images from one CIFAR-10 class, run
them through the checkpointed prefix, and collect the activations at the cut.
Repeat for every class. The diagram below follows one class through this
process and shows where the fitted mean, PCA coordinates, and covariance come
from.

![One class-conditional image distribution pushed through a fixed prefix,
then summarized in one shared PCA space.](figures/class_conditioned_pushforward_pca.svg)

*Figure 3. The fit is made from activations, not pixels. PCA is fitted once to
the pooled activation set. Class means and covariances are then estimated in
that same coordinate system.*

The order matters. I do **not** fit a different PCA basis for each class. I
flatten the saved activations, fit one global PCA basis using examples pooled
across classes, and express every example in that basis. With the basis fixed,
I use the full training activation bank—not only the rows used to fit
PCA—to estimate a mean \(\mu_c\) and covariance \(\Sigma_c\) for each class.

From this fit I make four possible training datasets:

1. **Real:** the saved activations.
2. **PCA-projected real:** the same examples after projection into, and
   reconstruction from, the fitted PCA space.
3. **Gaussian:** new points drawn from
   \(\mathcal N(\mu_c,\Sigma_c)\) in PCA space, then reconstructed.
4. **Mean + isotropic noise:** new points
   \(u=\mu_c+\sqrt{v_{\mathrm{pool}}}\,\epsilon\), where
   \(\epsilon\sim\mathcal N(0,I)\).

The projected-real condition is an important control. Real versus
projected-real measures the cost of discarding PCA directions.
Projected-real versus Gaussian compares an empirical activation distribution
with a mean-and-covariance match inside the *same* retained subspace. Gaussian
versus mean-plus-noise asks whether class-specific covariance helps.
The fitted Gaussian covariance is slightly regularized toward a spherical
covariance, with a separate tiny jitter used only for numerical stability.

The noise in the final condition is not a tiny numerical epsilon. Its default
scale is trace-matched: \(v_{\mathrm{pool}}\) is the average retained
within-class variance per PCA coordinate. The isotropic cloud therefore has
roughly the same average total within-class variance as the fitted class
clouds, but no class-specific shape or direction. Adding it avoids replacing
each class by many identical copies of one point. More importantly, it changes
the question from “do ten exact centroids train well?” to “do class means plus
a generic cloud of comparable size train well?”

That choice could affect the result, so the confirmatory battery varies the
noise radius from zero through smaller, trace-matched, and larger values. It
also varies PCA rank. Using every native activation coordinate is not a clean
default: shallow activations can have tens of thousands of coordinates, while
the fitted covariance is sample-rank limited and a dense per-class estimate is
both expensive and poorly determined. Instead I fit one PCA basis at the
largest rank requested in the sweep, use nested prefixes of that basis for
lower-rank controls, and report held-out coverage of total, within-class, and
between-class variance.

For each dataset, I copy the suffix weights from the same end-to-end-training
checkpoint. I freeze the prefix, train each suffix copy for the same short
budget, and evaluate all copies on held-out **real** activations. Cross-entropy
and accuracy are both recorded. The figures use accuracy because it is easier
to scan; the loss values remain in the downloadable artifacts, since accuracy
can hide changes in confidence.

![Identical warm-started suffixes trained on alternative activation datasets
and evaluated on the same held-out real activations.](figures/method_train_compare.svg)

*Figure 4. This is a relaxation experiment. It asks what update each fixed
activation distribution supports from the current checkpoint; it is not a
one-pass corruption test.*

## What the current runs show

The focal CNN plot shows how to read the experiment. It uses one legacy
epoch-1 model from the separately scheduled pilot grid; its title names the
architecture, checkpoint epoch, and cut. The curves begin at the same
checkpointed suffix, then separate during suffix-only training.

![CNN suffix-relaxation trajectories from the prefix at one checkpoint, with
the cut drawn inside the plot.](figures/cnn_relaxation_time.gif)

*Figure 5. Legacy CNN pilot · prefix from a separately scheduled model at
epoch 1 · cut after block 3. Each curve is evaluated on held-out real
activations; only its training distribution differs.*

The endpoint summaries are just compressed versions of these curves. They show
the change from the common starting checkpoint, or the shortfall from the real
condition. The coloured marks are alternative runs, never additive parts of
one quantity.

![The final relaxation frame connected by an arrow to a common-baseline
summary of the same endpoints.](figures/curve_to_endpoint_bridge.png)

*Figure 6. How the trajectories become a common-baseline endpoint summary.
This bridge is included before the first compact endpoint plot so that the
summary does not look like a new measurement.*

![CNN endpoint changes drawn from a common baseline, ordered by their measured
values.](figures/cnn_relaxation_endpoint_bands.png)

*Figure 7. A compact view of the legacy CNN relaxation endpoints. The
conditions share a baseline but are not additive pieces of one bar, and their
vertical ordering follows the measured values. The nominal checkpoint epochs
came from separately scheduled runs; do not read the horizontal sequence as
one training trajectory.*

Across the completed CNN grid, the separation between real and simplified
training is generally larger after early blocks and smaller after late blocks.
The earlier CNN figures mixed checkpoints from separately scheduled runs, so I
do not use them to infer change over training. Those pilots also estimated
class moments from the PCA-fitting subset; the replacement protocol estimates
them from the full training bank after fixing the PCA basis. The replacement
battery takes prefixes from checkpoints of one uninterrupted end-to-end CNN
run. Its PCA-rank, projected-real, and noise-radius controls are in progress at
the time of this WIP. I will add the resulting across-checkpoint animation and
ablation panel only after those artifacts pass verification.

The ResNet-18 evidence is older but cleaner in one respect: its cuts and saved
times come from one uninterrupted training trajectory. It shows the same broad
early-cut/late-cut contrast. Repeated surrogate draws measure variation from
generated activations and minibatch order; they are not independent ResNet
seeds.

![ResNet-18 endpoint changes across cuts and checkpoints, with the active cut
shown inside each frame.](figures/resnet_over_training_time.gif)

*Figure 8. ResNet-18, one training trajectory. The depth contrast is visible,
but the change across checkpoint time is not monotonic.*

An important warning is that the depth pattern is already visible at random
initialization under this protocol. Moving the cut also shrinks the suffix.
Late cuts may look easier because little network remains, because the
checkpoint is already well adapted, or because the representation has changed.
The current experiment does not separate these explanations.

## Tracking versus resolving: a proposal

The frozen-prefix measurement isolates one job of the suffix:

- **Resolving:** learn more from the activation distribution currently at the
  cut.

During normal end-to-end training, that distribution moves because the prefix
also changes. The suffix has a second job:

- **Tracking:** remain adapted while its input distribution moves.

![A simple comparison between a fixed distribution that only needs resolving
and a moving distribution that also needs tracking.](figures/tracking_resolving.svg)

*Figure 9. Proposed decomposition, not a result. The present experiment studies
the fixed-distribution case. A fuller account would compare
representation-movement time with suffix-adaptation time at each cut.*

This is where the optimization view becomes useful. A static representation
plot says where points ended up. A tracking measurement asks how quickly the
input to a block changes, how quickly that block can adapt, and whether those
timescales predict failures or stable collective structure. I suspect those
dynamics may reveal better intermediate variables than a fixed inventory of
units. That remains the larger research programme, not a conclusion licensed
by these CIFAR runs.

## What would change my mind?

The main threats are straightforward:

- **PCA loss:** discarded directions can masquerade as a non-Gaussianity gap.
- **Receiver size:** later cuts leave a smaller and less expressive suffix.
- **Warm starts:** the result mixes retention, new learning, and forgetting.
- **Finite budget:** an endpoint mixes adaptation speed with final capability.
- **Synthetic mismatch:** a moment-matched sample can still lie in places the
  suffix never saw during ordinary training.
- **Statistics:** repeated surrogate draws are not variation across trained
  networks.

The immediate tests are therefore the projected-real control, nested PCA-rank
sweep, isotropic-noise-radius sweep, longer relaxation curves, independent
training seeds, and a capacity-matched receiver attached at every cut. Initial
gradient norms may also reveal whether a condition merely changes the scale of
the first update.

The cleaned [interactive data
appendix](../free-body-diagrams-for-neural-networks.html) is available now as
the canonical view of the completed pilot evidence. Its confirmatory CNN
update is still in progress.
The [repository README](../README.md) contains the replication commands and
artifact conventions. The public dashboard will include completed measurements
only; planned or unrun experiments will stay out of the result surface.

Questions I would especially value feedback on:

1. What is the cleanest capacity-matched receiver for comparing cuts?
2. Which notion of representation movement is most relevant to the suffix,
   rather than merely easy to compute?
3. What result would distinguish an optimization-timescale account from a
   static data-geometry account?
4. Is there a useful alignment intervention whose stability over training
   could serve as an external test of this framework?

[^ai]: Written with Claude (via Codex). Claude assisted with drafting, figure
    generation, and data checks. The experimental choices, interpretation, and
    final text are the author's responsibility.

[^protocol]: The confirmatory protocol records dataset splits, augmentation,
    ordered image/label hashes, checkpoint hashes, fit sizes, PCA ranks,
    covariance regularization, random seeds, and suffix-training budgets.
    Synthetic-data draws and held-out real evaluation use disjoint examples.
    The legacy artifacts do not contain the
    same source/checkpoint lineage; the dashboard labels that limitation and
    hashes the exact files used. The small CNN uses unnormalized \([0,1]\)
    CIFAR tensors without augmentation. The ResNet uses normalized images with
    random crops and horizontal flips for its training bank; its legacy
    analysis re-encoded fresh augmentation draws at each cut. I treat the two
    architectures as separate replications, not as a direct numerical
    comparison.
