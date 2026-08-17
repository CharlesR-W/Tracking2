# Reproducing the LessWrong research note

This document describes the current measured CNN evidence behind
`LW post/free-body-diagrams-for-neural-networks.md`. The committed JSON inputs are under
`artifacts/lw_post/measured/`; the self-contained dashboard reads only the
hashed files listed in `artifacts/lw_post/dashboard_manifest.json`.

The older rank-gate artifacts under
`artifacts/lw_post/deprecated/prior_diagnostic_different_checkpoint/` use a
different checkpoint hash and are intentionally excluded.

## Environment

The supported local path is:

```bash
uv sync --extra dev
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src uv run pytest -q
```

Measured runs require CUDA and substantial host RAM. Shallow cut-1
activations have shape `32 x 32 x 32` (32,768 float coordinates per example);
PCA fitting and generated 50,000-example activation banks are CPU/RAM heavy
even when suffix optimization uses the GPU.

Every measured process should record an immutable source identity:

```bash
export TRACKING2_SOURCE_REVISION=<full-clean-commit>
export TRACKING2_SOURCE_ARCHIVE_SHA256=<sha256-of-uploaded-source-archive>
```

The archive used for the final PCA gate and most controls had SHA-256
`80f3708d099ba0436d943ed5f9c0b6c3f55cf217954ea3ffcdd53ab209c7e98e`.
Training and analysis manifests retain the exact environment, command,
dataset, checkpoint, and source lineage.

## Dataset and model trajectory

The current experiment uses:

- torchvision CIFAR-10 in its ordered 50,000/10,000 train/test splits;
- unnormalized `[0,1]` images and no data augmentation;
- a four-block residual CNN with GroupNorm and widths `32, 64, 128, 128`;
- 30 ordinary training epochs; and
- three independently initialized model seeds (`0, 1, 2`).

Train a trajectory with:

```bash
PYTHONPATH=src uv run python -m tracking2.cnn_checkpoints \
  --output artifacts/lw_post/cnn_seed0 \
  --data-root data \
  --epochs 30 --checkpoints 0 1 5 10 20 30 \
  --seed 0 --device cuda
```

Repeat with seeds 1 and 2. The committed training manifests include the
checkpoint SHA-256 values and exact ordered dataset fingerprints. Checkpoint
files are not committed.

## PCA-only adequacy gate

The Gaussian replay runner is too expensive to use merely to test projection
rank. The dedicated gate fits no class moments or covariances:

```bash
PYTHONPATH=src uv run python -m tracking2.cnn_projection_adequacy \
  --output artifacts/lw_post/cnn_projection_seed0 \
  --checkpoint artifacts/lw_post/cnn_seed0/checkpoint_epoch30.pt \
  --training-manifest artifacts/lw_post/cnn_seed0/training.json \
  --checkpoint-epoch 30 \
  --data-root data --cuts 1 \
  --pca-ranks 2048 3072 4096 \
  --seed 0 --device cuda
```

For a measured run, the training manifest is required. The gate:

- fingerprints the full ordered train/test splits against the trajectory;
- activation-encodes only the declared first 10,000 training examples;
- fits one randomized maximal PCA basis and uses nested leading prefixes;
- rejects infeasible ranks rather than silently clamping them;
- streams held-out total/within/between coverage and update-0
  true-versus-projected CE, accuracy, and predictive KL; and
- records PCA basis hashes and the exact prefix image/label hashes.

The declared practical-equivalence thresholds are:

```text
absolute projected-minus-true CE <= 0.05 nat
true-to-projected predictive KL <= 0.02 nat
```

Seed-0 cut-1 ranks 2,048 and 3,072 fail the KL threshold; rank 4,096 passes.
Because rank 4,096 was chosen after inspecting this held-out test bank, that
result remains exploratory.

## Replay statistics runner

The canonical invocation is `tracking2.post_statistics`. For example:

```bash
PYTHONPATH=src uv run python -m tracking2.post_statistics \
  --output artifacts/lw_post/matched_seed0 \
  --checkpoint artifacts/lw_post/cnn_seed0/checkpoint_epoch30.pt \
  --checkpoint-epoch 30 \
  --data-root data --cuts 1 4 \
  --pca-ranks 2048 \
  --gaussian-covariance-shrinkages 0 \
  --mean-noise-radii 1 \
  --surrogate-draws 1 \
  --relax-epochs 5 \
  --learning-rate-regime match_true_initial_update \
  --seed 0 --device cuda
```

Unless `--true-eval-only` is passed, every trained suffix is evaluated on the
full Cartesian grid of true, projected-real, Gaussian, and mean-noise
deployment distributions.

### PCA and class moments

- The PCA basis is fitted on the first 10,000 ordered training activations.
- Class means/covariances are re-estimated from all 50,000 training
  activations after the basis is fixed.
- Lower ranks reuse leading coordinates of one maximal fit.
- Paired ranks/radii reuse leading standard-normal coordinates.
- Held-out coverage is computed on the 10,000-example test bank.

The primary Gaussian uses exact empirical class covariance in PCA space.
Positive-definite targets use zero-jitter Cholesky. Semidefinite targets use an
eigendecomposition; any negative-eigenvalue clipping is explicit provenance
and cannot be labelled exact. Five-percent spherical shrinkage is a separate
sensitivity, not a hidden default.

### Mean-noise radius

The mean control is not optimizer epsilon or Cholesky jitter. Radius 1 gives a
spherical covariance whose trace equals the pooled average fitted within-class
covariance trace. Radius 0 uses exact class centroids; covariance trace scales
as the squared radius.

The measured radius sweep is `0, 0.5, 1, 2`. It is scientifically essential:
at the shallow seed-0 cell, exact centroids outperform the Gaussian while
radius-1 mean replay underperforms it.

### Optimizer regimes

`fixed_lr` uses one learning rate for every replay distribution.

`match_true_initial_update` first measures the actual suffix update on one
paired optimizer batch, then rescales each condition's learning rate to match
the true-replay update norm. The base/effective LR, multiplier, gradient
norms, parameter count, weight norm, update norm, and batch-index hash are
recorded on every row.

These are different interventions. They must be reported separately and never
pooled as replicates.

### Other measured sensitivities

The committed artifact set also includes:

- a seed-0 rank-4,096 adequacy-passing replay cell;
- exact versus 5% shrunk covariance;
- four mean-noise radii;
- warm versus reinitialized suffixes;
- 5 versus 20 relaxation epochs; and
- full-matrix rank-2,048 matched-update hero cells for model seeds 1 and 2.

Reinitialized and warm suffixes are never pooled. Reinitialization removes
inherited suffix weights but does not capacity-match the receivers left by
different cuts.

## Smoke test

Smoke artifacts must remain outside the measured tree:

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
  --gaussian-covariance-shrinkages 0 0.05 \
  --mean-noise-radii 0 1 --surrogate-draws 1 \
  --relax-epochs 1 --device cpu
```

Smoke outputs are labelled `MOCKUP / PIPELINE SMOKE TEST`; the measured
dashboard rejects them.

## Rebuild the measured-only dashboard

The manifest generator supports `--cnn-source none`, allowing a selective
measured evidence surface without importing the legacy grid or requiring an
unrun battery. Add each current artifact explicitly:

```bash
PYTHONPATH=src uv run python scripts/make_lw_post_manifest.py \
  --artifact-root artifacts \
  --output artifacts/lw_post/dashboard_manifest.json \
  --source-commit <dashboard-source-commit> \
  --cnn-source none --resnet-source none \
  --projection-adequacy-path <projection-json> \
  --cnn-extra-path <statistics-json> \
  --cnn-extra-path <another-statistics-json>

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src uv run python -m tracking2.post_report \
  artifacts/lw_post/dashboard_manifest.json \
  --output "LW post/free-body-diagrams-for-neural-networks.html"
```

The manifest records SHA-256 hashes for every input. The report rejects
unmeasured status, wrong schemas, invalid grids, missing source/checkpoint
lineage, non-finite metrics, incompatible LR fields, and warm/reinitialized
pooling.

## Deliberate limitations

- Rank-2,048 shallow results are projected-subspace estimands.
- Rank 4,096 currently has one model seed and exploratory test-set selection.
- First-update matching does not match optimizer scale throughout training.
- Mean-only replay depends on a declared nuisance-noise distribution.
- Moving the cut changes suffix capacity and optimization difficulty.
- The legacy ResNet pilot is excluded from the canonical dashboard because it
  has one model seed, incomplete source lineage, and different stochastic
  augmentation draws across cuts.
