# Tracking2

Fresh CIFAR experiments on selected-statistics surrogates and layerwise tracking.
The research design is in `SPEC.md`; the current interactive result viewer is
`report.html`.

## Implemented

- Experiment A: train/test matrix across true CIFAR, a class-mean proxy, and a
  class-conditional Gaussian matching mean/covariance in a fitted PCA space.
- A residual CNN with GroupNorm and explicit residual-stream cuts. A plain-block
  implementation remains available only as an optional secondary control.
- Experiment B pilot: at epoch 5 and cut 3, freeze the prefix, build true,
  class-mean, and class-conditional Gaussian activation distributions, relax an
  identical warm-started suffix on each, and cross-evaluate the full 3x3 matrix.
- Existing moving-optimum refits and suffix update-direction dot products remain
  supporting diagnostics. The empirical-Fisher branch is on hold.
- A report-ready VGG-19 critical-module experiment with all 19 parametric layers,
  checkpoint transplantation, fresh re-randomization, and selected-layer
  downstream recovery curves. This is intentionally a separate artifact so it
  can be integrated into Panel B without rewriting the live dashboard.
- A single-file Plotly report embedding the full spec and run provenance.

## Reproduce the bounded pilot

```bash
uv sync --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m tracking2.experiment \
  --output artifacts/cifar_pilot_seed0 \
  --train-size 2000 --test-size 1000 \
  --pca-fit-size 2000 --pca-components 64 \
  --epochs 3 --batch-size 128 --widths 16 32 64 \
  --architectures residual \
  --train-distributions mean covariance true \
  --checkpoints 0 1 3 --refit-steps 20 --refit-batches 4 \
  --fisher-samples 8 --device cpu
.venv/bin/python -m tracking2.report \
  artifacts/cifar_pilot_seed0/results.json --output report.html
```

The pilot artifact remains available as a feasibility check. The report is built
from the five-seed confirmatory artifact once the battery has completed.

## Frozen-prefix suffix-statistics pilots

The redesigned Part B slices are implemented separately so the earlier artifact
cannot be mistaken for the new measurement. Run the command below with
`--checkpoint-epoch` set to 0, 1, or 5 and a matching output directory:

```bash
.venv/bin/python -m tracking2.suffix_statistics \
  --output artifacts/suffix_statistics/t5_cut3_seed0 \
  --checkpoint-epoch 5 --cut 3 --train-size 10000 --test-size 2000 \
  --pca-fit-size 5000 --pca-components 128 --relax-epochs 10 --device auto
```

Each run writes `suffix_statistics.json` plus the checkpoint weights. The artifact
contains every relaxation epoch in the 3x3 train/evaluation matrix, held-out
moment diagnostics, PCA coverage, and exact protocol configuration.

For a zoom within the first training epoch, set `--checkpoint-epoch 0` and use
`--checkpoint-batches 0`, `5`, `20`, or `98`. With the full 50,000-example
dataset and batch size 256 these are random initialization, a few batches, 10.2%,
and 50.0% of the 196-batch epoch. Pass those artifacts alongside the coarse
epoch artifacts; the report renders them as a separate shared-axis four-facet view.

Render the measured triptychs by passing the three artifacts in chronological
order to `--suffix-statistics`; the report also sorts them by checkpoint:

```bash
.venv/bin/python -m tracking2.report artifacts/confirmatory/results.json \
  --suffix-statistics \
  artifacts/suffix_statistics/t0_cut3_full_seed0/suffix_statistics.json \
  artifacts/suffix_statistics/t1_cut3_full_seed0/suffix_statistics.json \
  artifacts/suffix_statistics/t5_cut3_full_seed0/suffix_statistics.json \
  --output report.html
```

## Confirmatory battery

The confirmatory protocol uses full CIFAR-10, five independent seeds, 30 epochs,
and seven tracking checkpoints. Run `scripts/run_confirmatory_battery.sh` on a
CUDA machine, then combine and render the seed artifacts with:

```bash
python scripts/aggregate_results.py artifacts/confirmatory/seed*/results.json \
  --output artifacts/confirmatory/results.json
python -m tracking2.report artifacts/confirmatory/results.json --output report.html
```

GitHub Pages publishes `report.html` as the repository landing report on every
push to `main`.

## Critical-module Panel B

The bounded local smoke test is:

```bash
python -m tracking2.criticality \
  --output /tmp/tracking2-criticality-smoke --fake-data \
  --train-size 8 --test-size 8 --epochs 0 --batch-size 8 \
  --classifier-width 16 --checkpoints 0 \
  --recovery-sources none --recovery-steps 0 --device cpu
```

The RunPod entry point is `scripts/run_criticality_battery.sh`. It defaults to
one seed as a phenomenon gate; set `MAX_SEEDS=5` only after inspecting that run.
Each seed writes `artifacts/criticality/seedN/criticality.json`, and the aggregation
script writes `artifacts/criticality/results.json`.

The aggregate artifact is the Panel B integration contract:

- `module_names`: fixed forward ordering of the 16 convolutions and 3 classifiers;
- `baselines`: intact final-model loss, accuracy, and error by seed;
- `interventions`: one row per seed, module, and transplant source, including
  `delta_loss` and `delta_error` for the primary heatmap;
- `recoveries`: downstream-only recovery trajectories for selected modules;
- `training`: intact-model checkpoint performance for the heatmap reference.

Neither the experiment nor aggregation code modifies `report.html`.

The complete bounded smoke ladder is `bash scripts/run_criticality_smokes.sh`.
It runs unit invariants, a tiny learned-signal/classifier-reset positive control,
fake-data training and recovery, the real-CIFAR loader, aggregation, and schema
finiteness checks. Smoke artifacts go to `/tmp/tracking2-criticality-smokes` by
default and therefore cannot be mistaken for measured project results.

The isolated dashboard proposal is
`MOCKUP/criticality_panel_b_MOCKUP.html`. Every schematic figure is titled and
watermarked `MOCKUP` and has an interpretation guide. It is deliberately not
connected to `report.html` while Panel B is being edited concurrently.

Paid runs are separated from smoke tests:

- `scripts/run_criticality_gate.sh`: one full-width, 40-epoch, seed-0
  phenomenon gate with immediate interventions only;
- `scripts/run_criticality_battery.sh`: resumable 100-epoch battery, defaulting
  to one seed and expandable with `MAX_SEEDS=5` after the gate is accepted.

Neither paid entry point should be launched without explicit approval and a
fresh check of the remaining RunPod budget.

## VGG critical-module bridge (Part C)

`python -m tracking2.vgg_suffix_statistics` implements C2 and C3 without
touching Part B's artifacts. It evaluates native and checkpoint-0-transplanted
VGG interfaces at cuts 7, 8, 9, and 13, relaxes matched suffix copies on true,
class-Gaussian, and class-mean activations, and records the full cross-matrix.
On RunPod, use `scripts/run_vgg_suffix_statistics.sh`; set `SEED`,
`CHECKPOINT_EPOCH`, `DRAW_COUNT`, and `PCA_COMPONENTS` in the environment as
needed. Dashboard-bound artifacts require at least three surrogate draws and
at least 80% PCA coverage at every displayed cut; the component count must be
increased and the artifact rerun when the reported coverage misses that gate.

Load its result only into the C tab with `--vgg-suffix-statistics`; C4 (fresh
re-randomization) and C5 (matched activation/prediction perturbation
diagnostics) remain explicit TODO controls in `SPEC.md`.
