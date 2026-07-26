# Reproducing the LessWrong research note

This document describes the bounded experiment behind
`docs/lw_wip_post.md`. The note is exploratory rather than a paper: the goal of
this protocol is to make every plotted result traceable and to prevent smoke
tests, old pilots, or unrun proposals from entering the public data appendix.

## What is measured

The confirmatory CNN result uses one uninterrupted end-to-end CIFAR-10
training run.
Checkpoints are saved at epochs 0, 1, 5, 10, 20, and 30. At each checkpoint and
after each of the four residual blocks:

1. The prefix is frozen and used to encode train and held-out test images.
2. One PCA basis is fitted at the largest requested rank using 10,000 training
   activations. Lower-rank controls use leading coordinates from that same
   basis.
3. With that basis fixed, class means and covariances are estimated from all
   50,000 training activations in the retained coordinates.
4. Identical warm-started suffix copies receive ten extra training epochs on
   one of four activation datasets:

   - real activations;
   - real activations reconstructed through PCA;
   - class-conditional Gaussian samples in PCA space;
   - class means plus shared isotropic noise in PCA space.

5. Every copy is evaluated on the same held-out, unprojected real activations.

Within a cut and repeat, all four copies use the same minibatch-label order.
The repeat changes both that shared order and the random surrogate sample. In
the noise-radius ablation, every radius scales the same sampled
standard-normal cloud rather than drawing an unrelated cloud. In the PCA-rank
sweep, lower ranks use leading coordinates from the same maximal
standard-normal banks, so rank contrasts do not also change the random draw.

The default mean-only noise radius is not optimizer epsilon or Cholesky jitter.
At radius 1, its isotropic covariance has the same total trace as the pooled
average within-class covariance retained in PCA space. Radius 0 replays exact
class centroids; radius \(r\) multiplies the reference RMS radius by \(r\) and
the covariance trace by \(r^2\).

The Gaussian covariance is estimated separately for each class and shrunk 5%
toward a spherical covariance. A \(10^{-6}I\) term is used only to make the
Cholesky factor numerically safe.

## Why not use every PCA component?

At shallow cuts, a flattened activation has tens of thousands of coordinates.
The maximal empirical rank is limited by the number of PCA fitting examples,
and each class covariance is limited by the roughly 5,000 training examples in
that class. A dense covariance in the full native space would therefore be
both rank deficient and expensive to factor.

The protocol instead makes the approximation visible:

- PCA coverage is evaluated on held-out activations, separately for total
  variance, within-class variance, and between-class mean variance.
- PCA-projected real activations isolate reconstruction loss from Gaussian
  approximation error.
- A nested rank sweep at the focal checkpoint/cut tests whether the conclusion
  is stable as more directions are retained.

## Environment

The supported path uses `uv` and Python 3.10–3.13:

```bash
uv sync --extra dev
uv run pytest -q
```

CIFAR-10 is downloaded by `torchvision` into `data/` on first use. The measured
battery requires a CUDA machine with enough host memory to hold shallow-layer
activation banks. Tests and smoke runs work on CPU.
The runner checks for 90 GiB of visible host memory by default; this is a
conservative guard for the cut-1 activation bank and randomized PCA workspace.
Override it only after profiling the exact machine.

The post-facing runners explicitly select the torchvision CIFAR-10 ordering;
they do not silently switch to a local parquet mirror. Every measured artifact
records SHA-256 fingerprints of the exact ordered train/test image and label
tensors. The verifier requires every suffix run to match the checkpoint
trajectory's fingerprints.

The small CNN consumes unnormalized \([0,1]\) tensors without data
augmentation. The legacy ResNet was trained and encoded with normalized
images, random crops, and horizontal flips. Its cut banks therefore use fresh
augmentation draws rather than an identical image realization at every cut;
the dashboard marks the architectures as separate replications rather than
inviting direct numerical comparison.

## Smoke test

Smoke artifacts must stay outside the measured artifact tree:

```bash
PYTHONPATH=src uv run python -m tracking2.cnn_checkpoints \
  --output /tmp/tracking2-lw-smoke/checkpoints \
  --fake-data --train-size 32 --test-size 16 \
  --epochs 2 --checkpoints 0 1 2 --widths 4 8 \
  --batch-size 8 --device cpu

PYTHONPATH=src uv run python -m tracking2.post_statistics \
  --output /tmp/tracking2-lw-smoke/statistics \
  --checkpoint /tmp/tracking2-lw-smoke/checkpoints/checkpoint_epoch2.pt \
  --checkpoint-epoch 2 --fake-data \
  --train-size 32 --test-size 16 --cuts 1 2 --widths 4 8 \
  --batch-size 8 --pca-fit-size 24 --pca-ranks 4 8 \
  --mean-noise-radii 0 1 --surrogate-draws 1 \
  --relax-epochs 1 --device cpu
```

Both outputs are marked `MOCKUP / PIPELINE SMOKE TEST`; the public dashboard
rejects that status.

## Measured CNN battery

On a CUDA machine:

```bash
uv sync --extra dev
uv run bash scripts/run_lw_post_battery.sh
```

The script is resumable. It writes:

- `artifacts/lw_post/cnn_seed0/training.json` and six checkpoint files;
- one base statistics artifact per checkpoint under
  `artifacts/lw_post/cnn_statistics_seed0/base_epoch*/`;
- a rank sweep at epoch 30, cut 3 for 128, 512, and 1,024 components;
- a mean-noise sweep at epoch 30, cut 3 for radii 0, 0.1, 1, and 2.

The default uses three generated-dataset/minibatch-order repeats for both the
full grid and targeted controls. `DRAW_COUNT` can change the full-grid repeat
count, but three repeats still do not estimate variation across independently
trained CNNs.

`scripts/verify_lw_post_artifacts.py` checks measured status, checkpoint
identity, the intended checkpoint/cut grid, and held-out coverage bounds. The
checkpoint hashes also show that every prefix checkpoint belongs to the saved
end-to-end training trajectory.

## ResNet evidence and controls

The post's broad ResNet depth sweep is a legacy measured artifact from one
normalization-free CIFAR ResNet-18 trajectory, with three generated-dataset and
minibatch-order repeats per cell. Its old PCA coverage number was computed on
the fitting bank, so it must be labelled as legacy rather than held-out
coverage.

New PCA-projected-real and held-out-coverage controls can be generated from the
epoch-100 checkpoint with:

```bash
CHECKPOINT=/path/to/checkpoint_epoch100.pt \
  uv run bash scripts/run_lw_post_resnet_ablations.sh
```

These controls use post-block cuts 4 and 8 (zero-based indices 3 and 7), three
repeats, and PCA ranks 256, 512, and 1,024. They do not convert the one-ResNet
result into an across-seed estimate.

## Evidence gate for the public dashboard

Only an artifact with exact status `MEASURED` may appear as data. Every
dashboard panel must declare:

- architecture and checkpoint epoch;
- cut convention;
- training/evaluation distributions;
- PCA rank and whether coverage is held out or legacy;
- repeat unit and count;
- checkpoint/source hashes when available.

Planned experiments, smoke runs, and panels without a complete provenance
record belong in design notes, not in the canonical data appendix.
