# Canonical LW publication rerun specification

## Status and safety boundary

No paid rerun is required for the checked-in July 26 evidence. This is a
non-executable specification for a future clean replication; there is
intentionally no current-sounding publication runner.

Any future paid run must begin from one clean, committed source tree, retain a
source archive and its SHA-256, write into a new staging directory, and run
under `scripts/supervise_lw_publication_runpod.py`. The supervisor does not
create a pod. Pod creation and provider-side spend controls remain separate,
deliberate actions.

Hard limits are the earlier of six elapsed pod-hours or $9 accrued, with an
hourly price no greater than $1.50. The provider account must also have an
independent $10 maximum-spend guard. Use at least 96 GiB host RAM. Never upload
`.env` files, API keys, tokens, SSH private keys, or other secrets.

The staging root is an isolated clone or worktree of that clean commit, not the
current checked-in July 26 package. Before launching, create the retained
uncompressed source archive at `source/source.tar` with `git archive`; the
archive's embedded commit must equal every staged provenance revision.

Before any training or analysis command, run exactly:

```bash
mkdir -p source
git archive --format=tar --output=source/source.tar HEAD
export TRACKING2_SOURCE_REVISION
TRACKING2_SOURCE_REVISION="$(git rev-parse HEAD)"
export TRACKING2_SOURCE_ARCHIVE_SHA256
TRACKING2_SOURCE_ARCHIVE_SHA256="$(sha256sum source/source.tar | cut -d ' ' -f 1)"
```

Keep both variables exported for every checkpoint, statistics, projection, and
manifest command. The measured writers read these exact environment variables;
without them, the staged promotion gate rejects provenance as unrecorded.

## Frozen experiment contract

All cells use:

- torchvision CIFAR-10, ordered 50,000-example training and 10,000-example
  test splits;
- four-block CNN widths `[32, 64, 128, 128]`, batch size 256;
- epoch-30 checkpoints from model seeds 0, 1, and 2;
- PCA fitted on the first 10,000 ordered training examples, with moments fitted
  on all 50,000 training examples;
- cuts 1 and 4 and retained PCA rank 2,048 for the primary analysis;
- one surrogate draw per independently trained model;
- warm suffix initialization, matched first-update norm, and five replay
  epochs for the primary analysis;
- empirical class covariance (`shrinkage=0`), `mean_r1`, projected true, and
  true replay as the primary distributions; and
- held-out-real cross-entropy at replay epoch 5 as the endpoint.

The two primary contrasts are Gaussian minus projected-true loss and
`mean_r1` minus Gaussian loss. Replication uncertainty is the sample standard
deviation across the three independently trained CNN seeds. Within-model
surrogate-redraw uncertainty is not estimated.

First-update matching must use a relative error tolerance of 1%. A row whose
learning-rate multiplier reaches either configured bound (`0.001`, `1000`) is
clipped and cannot be reported as exactly matched even if its relative error is
below 1%.

## Phase A: train the three checkpoints

Run `python -m tracking2.cnn_checkpoints` once per seed from the same clean
source archive. Common arguments are:

```text
--data-backend torchvision --train-size 50000 --test-size 10000
--epochs 30 --widths 32 64 128 128 --batch-size 256
--learning-rate 0.05 --weight-decay 0.0005 --device cuda
```

Seed 0 records checkpoints `0 1 5 10 20 30`; seeds 1 and 2 record checkpoint
`30`. Each `training.json` must have exact status `MEASURED`,
`config.fake_data=false`, ordered dataset fingerprints, a resolvable full
source commit, the retained source-archive SHA-256, and an epoch-30 checkpoint
hash. Do not start Phase B until each checkpoint file matches that hash.
For each seed, also pass `--output checkpoints/cnn_seed<seed>` and
`--seed <seed>` inside the isolated staging root. Copy each completed
`training.json` byte-for-byte to
`artifacts/lw_post/measured/training_manifests/cnn_seed<seed>_training.json`;
never point output at the checked-in measured package.

## Phase B: build exactly the nine publication inputs

Every statistics command also uses:

```text
python -m tracking2.post_statistics
--checkpoint-epoch 30 --data-backend torchvision
--train-size 50000 --test-size 10000 --widths 32 64 128 128
--batch-size 256 --pca-fit-size 10000 --surrogate-draws 1
--device cuda
```

Write each result first under a unique run staging root. The canonical relative
destinations and cell-specific arguments are:

| Destination below `artifacts/lw_post/measured/` | Seed | Cell-specific arguments |
|---|---:|---|
| `cnn_fixed_gate_seed0/cut1_r2048_true_eval_sensitivity.json` | 0 | `--cuts 1 --pca-ranks 2048 --gaussian-covariance-shrinkages 0 --mean-noise-radii 1 --true-eval-only --relax-epochs 5 --learning-rate-regime fixed_lr --suffix-initialization warm` |
| `cnn_fixed_gate_seed0/cut4_r2048_true_eval_sensitivity.json` | 0 | As above, with `--cuts 4` |
| `cnn_matched_seed0/cuts1_4_covariance_noise_ablation.json` | 0 | `--cuts 1 4 --pca-ranks 2048 --gaussian-covariance-shrinkages 0 0.05 --mean-noise-radii 0 0.5 1 2 --true-eval-only --relax-epochs 5 --learning-rate-regime match_true_initial_update --suffix-initialization warm` |
| `cnn_matched_seed0/cut1_r4096_true_eval.json` | 0 | `--cuts 1 --pca-ranks 4096 --gaussian-covariance-shrinkages 0 --mean-noise-radii 1 --true-eval-only --relax-epochs 5 --learning-rate-regime match_true_initial_update --suffix-initialization warm` |
| `cnn_matched_seed0/cuts1_4_reinitialized_true_eval.json` | 0 | `--cuts 1 4 --pca-ranks 2048 --gaussian-covariance-shrinkages 0 --mean-noise-radii 1 --true-eval-only --relax-epochs 5 --learning-rate-regime match_true_initial_update --suffix-initialization reinitialized` |
| `cnn_matched_seed0/cuts1_4_horizon20_true_eval.json` | 0 | `--cuts 1 4 --pca-ranks 2048 --gaussian-covariance-shrinkages 0 --mean-noise-radii 1 --true-eval-only --relax-epochs 20 --learning-rate-regime match_true_initial_update --suffix-initialization warm` |
| `cnn_matched_seed1/cuts1_4_r2048_full_matrix.json` | 1 | `--cuts 1 4 --pca-ranks 2048 --gaussian-covariance-shrinkages 0 --mean-noise-radii 1 --relax-epochs 5 --learning-rate-regime match_true_initial_update --suffix-initialization warm` |
| `cnn_matched_seed2/cuts1_4_r2048_full_matrix.json` | 2 | As for seed 1 |

For every table row, pass `--output <new-staging-cell-directory>`,
`--checkpoint <that-seed-epoch30-checkpoint>`, and `--seed <seed>`. The named
destination receives the completed cell directory's `post_statistics.json`
only after validation.

The ninth input is produced with
`python -m tracking2.cnn_projection_adequacy`, seed 0's checkpoint and training
manifest, and:

```text
--checkpoint-epoch 30 --data-backend torchvision
--train-size 50000 --test-size 10000 --widths 32 64 128 128
--batch-size 256 --pca-fit-size 10000 --cuts 1
--pca-ranks 2048 3072 4096 --seed 0 --device cuda
```

Its destination is
`cnn_projection_seed0/cut1_pca_adequacy.json`. The exploratory rank-4,096 cell
does not expand the rank-2,048 primary claim beyond the retained PCA subspace.
Also pass `--output <new-staging-cell-directory>`,
`--checkpoint <seed0-epoch30-checkpoint>`, and
`--training-manifest <seed0-training.json>`.

## Resumption and promotion rules

A task is resumable only after its JSON is atomically complete and passes all
of these checks: exact schema/experiment/status, `fake_data=false`, finite
values, exact requested grid with no duplicate or missing rows, checkpoint and
ordered-dataset lineage, source commit/archive lineage, and matching training
manifest. File existence alone is never a completion marker. Store an
independent completion record containing the JSON SHA-256 after validation.

After all nine cells finish, construct the isolated promotion layout exactly:

```text
<stage-root>/
  source/source.tar
  checkpoints/cnn_seed{0,1,2}/checkpoint_epoch30.pt
  artifacts/lw_post/dashboard_manifest.json
  artifacts/lw_post/measured/...
  artifacts/waterfall_visuals/manifest.json
  LW post/figure_manifest.json
  LW post/figures/<every figure-manifest asset>
  LW post/free-body-diagrams-for-neural-networks.html
  LW post/waterfalls.html
```

Build the staged manifest with the dedicated clean-mode tool:

```text
.venv/bin/python scripts/make_lw_post_staged_manifest.py
  --stage-root <absolute-isolated-clone-root>
```

This writes schema 2 with the same nine IDs/paths and structural primary
contract, hashes the newly measured inputs, recomputes all four headline rows,
uses the exported full clean commit, verifies the retained source archive and
three checkpoint files, and omits `provenance_corrections`. The July sidecar
and headline values are never copied into the staged result. Copy the active
`figure_manifest.json`, `artifacts/waterfall_visuals/manifest.json`,
`waterfalls.html`, and every exact figure-manifest asset from the clean source
tree. The method illustrations are frozen talk copies; the waterfall exports
and viewer are deterministic derivatives of hash-pinned historical pilot
artifacts. None is an output of the new controlled battery. Build the appendix
from the staged evidence manifest.

From that isolated clone/worktree, run the actual promotion gate:

```text
.venv/bin/python scripts/verify_lw_post_publication.py
  --profile staged
  --stage-root <absolute-isolated-clone-root>
```

The staged profile does not inherit July 26 digests, corrections, headline
values, or the clipped radius-zero magnitude. It instead verifies new manifest
digests, exact protocol/config grids, all three retained epoch-30 checkpoint
files, the actual `git archive` digest and embedded commit, common source and
dataset lineage, recomputed headline assertions, primary update matching, the
hash-pinned active figure allowlist, byte-identical waterfall viewer/exports,
and a byte-identical appendix rebuild.
Promote no file until this command passes locally after download.

Stop immediately on source-revision or archive mismatch, host RAM below 96 GiB,
checkpoint/dataset lineage mismatch, missing or duplicate cells, non-finite
values, schema mismatch, OOM, worker failure, polling failure, hourly-rate
breach, time limit, or cost limit.

## Required RunPod supervision contract

Start the supervisor immediately after pod creation, using the provider's pod
creation timestamp and the remote orchestration command after `--`:

```text
.venv/bin/python scripts/supervise_lw_publication_runpod.py
  --pod-id <pod-id>
  --pod-started-at <provider-creation-unix-seconds>
  --starting-cost-usd <already-accrued-cost-not-covered-by-timestamp>
  --hourly-rate-cap-usd 1.50
  --cost-cap-usd 9
  --time-cap-hours 6
  --poll-seconds 60
  -- <remote-worker-command-and-arguments>
```

Before starting the worker, the supervisor polls the pod and rejects a price
above $1.50/hour. It then polls at least once per minute and reports a
conservative cost upper bound using the maximum observed rate, pod age, and
declared starting cost. It stops before another polling interval could cross
$9. Cleanup is fail-closed: successful completion, worker error, validation
error, price/cost/time breach, `SIGINT`, `SIGTERM`, or `SIGHUP` all request pod
termination. Polling failure also terminates rather than allowing an
unmonitored run. Failed automatic termination is printed as a critical manual
action and returns status 3. Once a recognizable `--pod-id` is present, cleanup
is armed before the rest of the CLI is validated; an invalid/missing worker,
manager path, cap, timestamp, or polling argument therefore still attempts
termination. Pure `--help` and invocations without a pod ID do not terminate
anything.

Because no process can trap `SIGKILL`, host loss, or provider control-plane
failure, the independent provider-side $10 spend limit is mandatory. Confirm
the pod is absent from the provider dashboard after every run.
