# Free Body Diagrams for Neural Networks: Analysing training dynamics through the lens of internal distributional simplicity bias

*Work in progress. This is an exploratory CIFAR-10 analysis. I see a consistent
difference across depth, but I do not yet know what causes it, and several
controls and replications are still missing.[^ai]*

## Summary

- Previous work on **distributional simplicity bias** finds that neural
  networks tend to use simple statistics of their input data before using more
  complicated ones.
- I apply the same question inside a network. Pick a layer, hold everything
  before it fixed, and treat its outputs as the training data for everything
  after it.
- I replace those internal activations with simplified versions that preserve
  either differences between class averages or a fitted Gaussian for each
  class. I then continue training copies of the later part of the network and
  test them on real activations.
- In both a four-block CNN and a ResNet-18, extra training on simplified
  activations falls much further behind extra training on real activations
  after early blocks than after late blocks. Near the end of either network,
  the final classification accuracies are close.
- I do not yet know how much of this comes from changes in the internal data,
  how much comes from the later part of the network becoming smaller and
  simpler, and how much comes from details of the measurement.

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
network.](figures/fbd_materials_nn_analogy.svg)

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
replace a dataset with a Gaussian that has the same class averages and
covariances, we keep those low-order statistics but remove its non-Gaussian
structure. Comparing learning on the real and Gaussian data gives us a way to
ask when the missing structure matters.

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
useful inside a trained network.

## The experiment

I trained two image classifiers on CIFAR-10: a small four-block residual CNN and
a ResNet-18.[^protocol]

![Schematic architectures of the four-block residual CNN and normalization-free
pre-activation ResNet-18, with every measured cut
marked.](figures/cnn_resnet_architectures.svg)

*Figure 2. The small CNN has four residual blocks. The ResNet has eight
residual blocks arranged in four stages. I save activations after each marked
block and treat the rest of the network as the part to be trained.*

At several points during training, I repeated the following procedure after
different blocks. I first made three versions of the activation dataset:

1. Hold the earlier part of the network fixed.
2. Run labelled training images through it and save the resulting activations.
3. Use those activations to make three labelled datasets:

   - **Real:** the saved activations themselves.
   - **Gaussian:** samples from a fitted distribution using each class's mean
     and a regularized estimate of its covariance in PCA space.
   - **Mean + shared noise:** independent samples centred on each class mean.
     The added noise has the same variance in every direction and for every
     class. This keeps differences between class means but removes
     class-specific covariance.

![Labelled images pass through fixed earlier layers, then saved activations at
one cut become real, Gaussian, or mean-plus-shared-noise training
datasets.](figures/method_make_datasets.svg)

*Figure 3. The cut turns one internal layer into a labelled dataset. The real
version keeps the saved examples. The other two keep progressively less of
their class-conditional structure.*

I then compared what further learning each dataset supported:

1. Make three copies of the later part from the current checkpoint.
2. Give each copy a short period of additional training on one of the three
   activation datasets.
3. Test every copy on held-out real activations from the fixed earlier part.

![Three activation datasets train identical copies of the remaining layers,
which are all evaluated on the same held-out real
activations.](figures/method_train_compare.svg)

*Figure 4. The three copies start with the same weights and receive the same
training budget. Only the activation dataset used for further training
changes.*

I do not feed synthetic activations through the original network once and
measure the immediate damage. I use them for a short burst of further training,
then test on held-out real activations. This asks what further learning each
simplified dataset can support from the network's current state.

The real-activation copy is the reference. A small gap from the
Gaussian-trained copy means that the fitted class means and covariances
supported similar short-term change within this training budget. If the
mean-plus-noise copy does worse than the Gaussian copy, then class-specific
covariance helped under this setup.

The Gaussian is fitted in a lower-dimensional PCA representation of the
activations.[^pca] This makes the calculation possible, but it weakens the
interpretation. A gap between the real and Gaussian datasets could come from
non-Gaussian structure, or it could come from useful directions discarded by
PCA. I return to this below.

## What happens across depth

### Four-block CNN

The table compares two CNN runs after 30 epochs of ordinary training. One was
analysed after block 1 and the other after block 4. Each cell reports top-1
accuracy followed by cross-entropy loss after ten extra epochs of training the
later part. Higher accuracy and lower loss are better.

| Cut | Real activations | Gaussian | Mean + shared noise |
|---|---:|---:|---:|
| After block 1 | $77.0\%$ / $1.04$ | $62.8\%$ / $2.20$ | $43.2\%$ / $3.64$ |
| After block 4 | $76.7\%$ / $1.01$ | $76.6\%$ / $1.02$ | $76.8\%$ / $0.99$ |

![CNN accuracy during further training on real, Gaussian, or mean-plus-noise
activations after block 3, starting from six ordinary-training
checkpoints.](figures/cnn_relaxation_trajectories.png)

*Figure 5. At the cut after block 3, the three copies separate during further
training. Each panel starts from a different ordinary-training checkpoint.
These CNN checkpoints come from separate runs.*

After the first block, the choice of activation data makes a large difference.
After the final block, the three copies finish in almost the same place.

Other CNN runs show a similar early-to-late difference. But they came from
separate models with different training schedules, rather than snapshots of one
model along a single training trajectory.[^runs] I use them as a first pass
across depth, not as evidence about change over time.

### ResNet-18

For ResNet-18, all eight cuts come from one training run, measured at epochs 0,
1, 5, 20, and 100. I repeated each analysis three times with new generated
activations and minibatch orders. This measures variation in data generation
and short-term training, not variation across independently trained ResNets.

At epoch 100:

| Cut | Real activations | Gaussian | Mean + shared noise |
|---|---:|---:|---:|
| After block 2 | $91.0\%$ / $0.71$ | $81.0\%$ / $1.73$ | $70.9\%$ / $1.94$ |
| After block 8 | $91.3\%$ / $0.69$ | $90.7\%$ / $0.73$ | $89.7\%$ / $0.91$ |

The early cut again shows a large difference. After the final block, further
training on either simplified dataset leaves top-1 accuracy close to the
real-activation result. Cross-entropy still detects a cost for the
mean-plus-noise data, so "close" here refers mainly to classification accuracy.

Across all five saved training times, the gap between real and simplified
activations is larger at early cuts than late ones. For example, after one
epoch of ordinary training, the Gaussian result trails the real result by
$17.1$ accuracy points after the first block and $1.5$ points after the last.
The mean-plus-noise gaps are $20.6$ and $4.7$ points.

The early-to-late contrast is already present at random initialization under
this five-epoch downstream-training protocol. It changes non-monotonically
after that. The current results map a difference across depth, but do not yet
show when, or whether, each part acquires dependence on higher-order
statistics.

![Two ResNet heatmaps showing the accuracy gap between further training on real
and simplified activations across eight cuts and five training
times.](figures/resnet_shortfall_heatmaps.png)

*Figure 3. Extra training on simplified activations falls further behind extra
training on real activations at early cuts than late cuts. The change over
ordinary training is not monotonic. Each cell summarizes three analysis runs
from one trained ResNet.*

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

Each later part also begins from the current checkpoint rather than from
scratch. The experiment measures what it retains and how it changes over a
short period. It does not measure every piece of information present in the
activations, or everything a newly initialized network could eventually learn
from them.

The comparison uses a fixed training budget: ten extra epochs for the small CNN
and five for the ResNet. Some early-cut curves are still moving at the end, and
some copies trained on simplified data are getting worse on real activations.
The endpoint mixes adaptation speed, mismatch between real and generated data,
and forgetting during the extra training.

Finally, PCA removes part of the activation space before the synthetic data is
made. It retains between $78\%$ and $96\%$ of activation variance in the ResNet
measurements, and between $48\%$ and $99\%$ in the CNN measurements. Until I
separate PCA loss from non-Gaussian structure, I should not call the
real-versus-Gaussian difference a measurement of higher-order cumulants alone.

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
the time the remaining layers need to adjust.](figures/tracking_resolving.svg)

*Figure 4. In normal training, the data crossing a cut moves while the later
layers learn. The proposed analysis compares the speed of that movement with
the time the later layers need to adjust.*

## Next steps

The first priority is to make the statistical comparison cleaner.

1. Add real activations passed through the same PCA projection as a fourth
   dataset. This gives the sequence

   $$
   \text{real}
   \longrightarrow
   \text{PCA-projected real}
   \longrightarrow
   \text{Gaussian}
   \longrightarrow
   \text{mean + shared noise}.
   $$

   The first gap isolates the effect of the PCA projection. The second compares
   empirical non-Gaussian data with a matched Gaussian inside the same PCA
   space. The third isolates the effect of class-specific covariance.

2. Vary the number of PCA components and check whether the difference across
   depth survives.

3. Follow each additional-training curve for long enough to separate adaptation
   speed from its plateau. One simple summary is the number of minibatches
   needed to complete $90\%$ of the total change from its starting point.

4. Compare the existing later part with a newly initialized copy, and attach a
   small classifier of the same size at every cut. This should help separate the
   properties of the internal data from the capacity and training history of
   the network that receives it.

5. Repeat the measurements across independently trained networks. The current
   ResNet result uses one training run, and the current CNN runs need shared
   checkpoints and a shared training schedule.

6. Measure how quickly representations move on a fixed set of examples, then
   compare that movement with adaptation time. I also want to compare this with
   how much performance is lost when a layer is reset, how sensitive each layer
   is to added noise, and its Fisher spectrum, which summarizes parameter
   directions to which the model's predictions are most sensitive. Those may
   be useful comparisons, but the connection is still speculative.

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

[^protocol]: The ResNet is a normalization-free pre-activation CIFAR ResNet-18,
    not the standard torchvision BatchNorm model. It was trained on all 50,000
    CIFAR-10 training images. Each later-part analysis used 10,000 training
    activations, fitted PCA on 5,000, and evaluated on 2,000 held-out examples.
    The CNN analyses used 50,000 training and 10,000 held-out activations, with
    PCA fitted on 10,000. The CNN used raw images; the ResNet used normalized
    images with random crops and flips. ResNet activations were re-encoded with
    fresh augmentation at each cut, so the cuts did not see identical crop and
    flip draws.

[^pca]: Flattened activation tensors can have tens of thousands of coordinates.
    A dense covariance at that size is expensive to store and poorly estimated
    from the available examples. PCA provides a lower-dimensional space in
    which the covariance can be fitted and sampled. The Gaussian uses a
    covariance estimate shrunk $5\%$ toward a spherical covariance, so it only
    approximately matches the fitted covariance. A finite generated dataset
    adds further sampling error.

[^runs]: Each CNN row shown here comes from one trained model and one generated
    activation dataset of each type. The ResNet uses one trained network and
    three analysis runs per cut. Those runs vary the generated activations and
    minibatch order. These are exploratory measurements, not an estimate across
    independently trained networks.
