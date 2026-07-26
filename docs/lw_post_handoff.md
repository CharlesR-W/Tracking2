# Current handoff: Tracking2 LessWrong note

Date: 2026-07-26

## Current state

- Working branch: `codex/lw-post-revision`.
- Canonical source checkout: `/home/crw/Programming/Experiments/Tracking2`.
- Measured JSON inputs: `artifacts/lw_post/measured/`.
- Dashboard manifest: `artifacts/lw_post/dashboard_manifest.json`.
- Self-contained dashboard: `free-body-diagrams-for-neural-networks.html`.
- Revised WIP post: `docs/lw_wip_post.md`.
- Reproduction methodology: `docs/lw_post_reproduction.md`.

The public dashboard is measured-only. The older rank gates under
`artifacts/lw_post/prior_diagnostic_different_checkpoint/` use another
checkpoint hash and are excluded.

The final secure RunPod L4 was terminated after all JSON artifacts and training
manifests were downloaded. A subsequent API lookup returned HTTP 404.

## What was implemented

- uninterrupted four-block CNN trajectories;
- three independently initialized model seeds;
- exact empirical PCA-space covariance with explicit factorization provenance;
- optional 5% covariance shrinkage as a separate sensitivity;
- PCA-projected-real replay and the full train × evaluation matrix;
- held-out PCA total/within/between coverage;
- streamed update-0 true/projected CE, accuracy, and predictive KL;
- a covariance-free PCA adequacy runner through rank 4,096;
- paired first-batch gradient/update diagnostics;
- fixed-LR and matched-first-update optimizer regimes;
- mean-noise radii `0, 0.5, 1, 2`;
- warm and reinitialized suffix conditions;
- 5- and 20-epoch relaxation horizons;
- strict artifact/dashboard validation; and
- a measured-only interactive report with artifact-driven failure warnings.

The CPU test suite passes 95 tests in the current branch before the final
figure/blog integration changes.

## Claim-changing findings

### 1. Fixed LR was a large shallow optimizer confound

At the current seed-0 checkpoint, cut 1/rank 2,048, first updates were about:

- projected-real: 2.5× true;
- Gaussian: 184× true; and
- mean-$r1$: 287× true.

Matching first-update norm changed Gaussian endpoint CE from 1.805 to 1.045
and mean-$r1$ from 3.313 to 1.193. The ordering survived; most of the magnitude
did not.

### 2. The Gaussian/mean ordering depends on the mean-noise radius

At seed 0, cut 1/rank 2,048, matched-update CE is:

- Gaussian: 1.045;
- mean-$r0$: 0.991;
- mean-$r0.5$: 1.324;
- mean-$r1$: 1.193; and
- mean-$r2$: 1.041.

Exact centroids beat the Gaussian while radius-1 mean replay loses to it.
Therefore “covariance helps” is not a radius-robust conclusion.

### 3. PCA rank 2,048 is not adequate for a full shallow estimand

For current seed 0/cut 1:

- rank 2,048 predictive KL = 0.0373 (fails);
- rank 3,072 KL = 0.0211 (narrowly fails);
- rank 4,096 KL = 0.0146 (passes the declared 0.02 threshold).

Rank 4,096 retains 93.0% total and 92.8% within-class variance. Its selection
used the test population, so it is exploratory.

The three-seed rank-2,048 results are projected-subspace results. Only the
seed-0 rank-4,096 cell supports the declared practical-equivalence treatment
of shallow projection.

### 4. The three-seed depth contrast replicates under matched updates

At rank 2,048 after five epochs, mean ± sample SD in held-out CE:

| Cut | Gaussian − projected | Mean-$r1$ − Gaussian |
|---|---:|---:|
| After block 1 | +0.183 ± 0.055 | +0.144 ± 0.019 |
| After block 4 | −0.012 ± 0.002 | +0.023 ± 0.011 |

The late native suffix is insensitive under this finite protocol. This is not
evidence that “higher moments appear later.”

### 5. Rank 4,096 preserves the shallow seed-0 ordering

At cut 1/rank 4,096, matched-update endpoint CE:

- projected-real: 0.938;
- exact Gaussian: 1.065; and
- mean-$r1$: 1.233.

The projection gate passes at this rank.

### 6. Shrinkage is not driving the seed-0 result

Five-percent covariance shrinkage changes CE by:

- cut 1: −0.0084;
- cut 4: +0.0005.

### 7. Reinitialization strengthens the capacity/trainability warning

With fresh suffixes, cut-1 true/projected/Gaussian/mean accuracies are
68.7/64.5/43.4/37.8%; after cut 4 all are about 77.9%.

Inherited suffix knowledge is not required for the qualitative contrast, but
the fresh shallow true suffix itself is far from the late one after five
epochs. Reinitialization does not capacity-match cuts.

### 8. A longer horizon does not close the shallow gap

At cut 1, Gaussian-minus-projected CE grows from 0.120 at epoch 5 to 0.147 at
epoch 20; mean-minus-Gaussian grows from 0.148 to 0.462. Late accuracies remain
within 0.08 percentage points.

## Remaining blockers to a stronger claim

- capacity-matched receivers across cuts;
- independent model seeds for the rank-4,096 adequacy-passing cell;
- validation of PCA rank on a fresh held-out population;
- optimizer controls beyond the first update; and
- a less arbitrary mean-only nuisance distribution.

The older ResNet pilot remains useful historical context but is not canonical
evidence: it has one trained model, incomplete source lineage, and different
stochastic augmentation draws across cuts.
