# Tracking2

Fresh CIFAR experiments on selected-statistics surrogates and layerwise tracking.
The research design is in `SPEC.md`.

## LessWrong research-note companion

The current public-facing thread of the project is
[*Free-body diagrams for neural networks*](LW%20post/free-body-diagrams-for-neural-networks.md),
an exploratory research note about applying distributional-simplicity probes
at internal network cuts.

- [`LW post/`](LW%20post/) collects the publication post, outline, handoff, reproduction
  notes, exactly five current figures, their measured-only generator, and the
  interactive appendix. Delivered-talk and preliminary assets are under
  [`LW post/deprecated/`](LW%20post/deprecated/).
- [`LW post/lw_post_reproduction.md`](LW%20post/lw_post_reproduction.md) gives the
  bounded protocol, controls, smoke test, measured run commands, and evidence
  gate.
- [`LW post/free-body-diagrams-for-neural-networks.html`](LW%20post/free-body-diagrams-for-neural-networks.html)
  is the canonical interactive ablation appendix, built from
  `artifacts/lw_post/dashboard_manifest.json`.
- [`deprecated/2026-07-omnibus/report.html`](deprecated/2026-07-omnibus/report.html)
  is the older omnibus research dashboard. It includes planned and unrelated
  surfaces, is not the note's evidence index, and is not published by Pages.

The post-facing dashboard deliberately includes only the measured July 26 CNN
control battery. ResNet, the older CNN depth-by-time sweep, smoke data, unrun
experiments, and mockups are outside its evidence manifest. The ResNet and older
CNN materials remain historical pilots, not support for the post's quantitative
claims.

Rebuild the appendix from the checked-in, hash-pinned manifest with:

```bash
PYTHONPATH=src .venv/bin/python -m tracking2.post_report \
  artifacts/lw_post/dashboard_manifest.json \
  --output "LW post/free-body-diagrams-for-neural-networks.html"
```

The uninterrupted three-seed CNN battery and its PCA, optimizer-scale,
mean-noise, covariance, reinitialization, and horizon controls are complete.
Loading or rebuilding the appendix validates every manifest-listed input hash
and schema. Run `scripts/verify_lw_post_publication.py` for the complete
checked-in publication gate. The old rank-512 runner and verifier are isolated
under `scripts/deprecated/lw_post/`; do not use them to verify or rerun the July
26 publication battery. No GPU rerun is required for the current bounded claim.
GitHub Pages rebuilds the canonical file from the committed manifest, checks it
against the tracked HTML, and publishes only that appendix plus the five current
figures.

## Implemented

- Experiment A: train/test matrix across true CIFAR, a class-mean proxy, and a
  class-conditional Gaussian matching mean/covariance in a fitted PCA space.
- A residual CNN with GroupNorm and explicit residual-stream cuts. A plain-block
  implementation remains available only as an optional secondary control.
- Experiment B: repeat the frozen-prefix true/mean/Gaussian suffix-relaxation
  matrix across all four residual-CNN cuts and training checkpoints.
- Experiment C runners: apply the same matrix to all eight ResNet-18 blocks and
  seven VGG-19 cuts spread across stages 2–5. Critical-module transplantation is
  supporting context at the bottom of C.
- Existing moving-optimum refits and suffix update-direction dot products remain
  supporting diagnostics. The observed-label empirical-Fisher branch remains an
  archived diagnostic and is not the proposed moment-sensitivity metric.
- A report-ready VGG-19 critical-module experiment with all 19 parametric layers,
  checkpoint transplantation, fresh re-randomization, and selected-layer
  downstream recovery curves. This is intentionally a separate artifact so it
  can be integrated into Panel B without rewriting the live dashboard.
- A single-file Plotly report embedding the full spec and run provenance.

## Planned extensions and measured diagnostics

- Part D freezes a tangent chart at a checkpoint and cut, compares eigenvalues of
  the full update map with the prefix-clamped suffix map, and derives the suffix
  frequency response to transport perturbations of the activation distribution.
  Fisher/GGN supplies the predictive metric; its Schur complement is the ideal
  static-compensation limit, not the stability operator.
- Part E has completed a one-seed GPT-2-small/TinyStories diagnostic. Its repaired
  schema-v2 analysis passes the mechanical replay, adaptive-PCA, parity, and
  leakage gates, but sequence Gaussian does not consistently beat mean replay.
  The result is scientifically negative; redesign and any multi-seed extension
  remain future work.

The relevant design documents are
[`docs/tangent_model_control_design.md`](docs/tangent_model_control_design.md)
and [`docs/part_e_gpt2_design.md`](docs/part_e_gpt2_design.md). Part D remains an
unrun theory/mockup surface; Part E's dashboard section reports the measured
negative diagnostic while retaining the original design for provenance.

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
  artifacts/cifar_pilot_seed0/results.json \
  --output /tmp/tracking2-omnibus-report.html
```

The original tracked pilot is archived at
`deprecated/2026-07-omnibus/artifacts/cifar_pilot_seed0/`. New runs should use a
fresh output directory and must not overwrite the archive.

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
Add `--relax-batch-zoom` to record the first suffix-relaxation epoch at batches
0, 1, 2, 5, 10, 20, 50, 100, and 196. The zoomed trajectories use this batch
axis while the run still completes all requested relaxation epochs for its final matrix.

Render the measured triptychs by passing the three artifacts in chronological
order to `--suffix-statistics`; the report also sorts them by checkpoint:

```bash
.venv/bin/python -m tracking2.report \
  deprecated/2026-07-omnibus/artifacts/confirmatory/results.json \
  --suffix-statistics \
  deprecated/2026-07-omnibus/artifacts/suffix_statistics/t0_cut3_full_seed0/suffix_statistics.json \
  deprecated/2026-07-omnibus/artifacts/suffix_statistics/t1_cut3_full_seed0/suffix_statistics.json \
  deprecated/2026-07-omnibus/artifacts/suffix_statistics/t5_cut3_full_seed0/suffix_statistics.json \
  --output /tmp/tracking2-omnibus-report.html
```

## Confirmatory battery

The confirmatory protocol uses full CIFAR-10, five independent seeds, 30 epochs,
and seven tracking checkpoints. Run `scripts/run_confirmatory_battery.sh` on a
CUDA machine, then combine and render the seed artifacts with:

```bash
python scripts/aggregate_results.py artifacts/confirmatory/seed*/results.json \
  --output artifacts/confirmatory/results.json
python -m tracking2.report artifacts/confirmatory/results.json \
  --output /tmp/tracking2-omnibus-report.html
```

GitHub Pages publishes the measured-only
`LW post/free-body-diagrams-for-neural-networks.html` as the landing report on every
push to `main`, together with exactly the five post figures. It does not publish
the archived omnibus dashboard.

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

Neither the experiment nor aggregation code modifies the archived omnibus
dashboard.

The complete bounded smoke ladder is `bash scripts/run_criticality_smokes.sh`.
It runs unit invariants, a tiny learned-signal/classifier-reset positive control,
fake-data training and recovery, the real-CIFAR loader, aggregation, and schema
finiteness checks. Smoke artifacts go to `/tmp/tracking2-criticality-smokes` by
default and therefore cannot be mistaken for measured project results.

The isolated dashboard proposal is
`MOCKUP/criticality_panel_b_MOCKUP.html`. Every schematic figure is titled and
watermarked `MOCKUP` and has an interpretation guide. It is deliberately not
connected to the archived omnibus dashboard while Panel B is being edited
concurrently.

Paid runs are separated from smoke tests:

- `scripts/run_criticality_gate.sh`: one full-width, 40-epoch, seed-0
  phenomenon gate with immediate interventions only;
- `scripts/run_criticality_battery.sh`: resumable 100-epoch battery, defaulting
  to one seed and expandable with `MAX_SEEDS=5` after the gate is accepted.

RunPod work is pre-approved up to the task's hard $10 cap. Before either paid
entry point, check active pods and projected cost, and terminate the pod as soon
as the bounded run finishes. Ask before any action that could exceed $10.

## Larger-architecture replication (Part C)

`scripts/run_resnet_statistics.sh` trains the paper's CIFAR ResNet-18 V2 in its
normalization-free, zero-weight-decay setting, then repeats Part B
measurement on intact native interfaces at checkpoint epochs 0, 1, 5, 20, and
100 and after all eight residual blocks. At every cell it relaxes matched suffix copies on true,
class-Gaussian, and class-mean activations and records the full cross-matrix.
Set `SEED`, `DRAW_COUNT`, and `PCA_COMPONENTS` as needed. The first gate uses one
ResNet seed, three surrogate draws, and 512 PCA components.

`scripts/run_vgg_suffix_statistics.sh` uses the same checkpoints and protocol on
VGG-19+BatchNorm. Its native cuts are `3 5 7 9 11 13 15`: enough coverage to
resolve middle depth while avoiding the exceptionally large stage-1 activation
in the first feasibility gate. Both scripts write under
`artifacts/c_architecture_statistics/` with `RUN_TAG=c_gate_v2` by default, so
older bridge artifacts cannot be mistaken for the new C experiment.

Run the one-seed gate on a CUDA machine:

```bash
bash scripts/run_resnet_statistics.sh
bash scripts/run_vgg_suffix_statistics.sh
```

Render Part B plus every available C checkpoint artifact together:

```bash
python -m tracking2.report \
  deprecated/2026-07-omnibus/artifacts/confirmatory/results.json \
  --suffix-statistics \
    deprecated/2026-07-omnibus/artifacts/suffix_statistics/t0_cut3_full_seed0/suffix_statistics.json \
    deprecated/2026-07-omnibus/artifacts/suffix_statistics/t{1,5,10,20,30}_cut{1,2,3,4}_full_seed0/suffix_statistics.json \
  --resnet-suffix-statistics artifacts/c_architecture_statistics/resnet/*/resnet_suffix_statistics.json \
  --vgg-suffix-statistics artifacts/c_architecture_statistics/vgg/*/vgg_suffix_statistics.json \
  --resnet-criticality artifacts/resnet_criticality/seed0/resnet_criticality.json \
  --criticality deprecated/2026-07-omnibus/artifacts/criticality/results.json \
  --output /tmp/tracking2-omnibus-report.html
```

The ResNet and VGG views lead with bars showing accuracy change from the common
unrelaxed checkpoint. Their compact secondary maps use one shared symmetric
true-activation shortfall scale, defined as true-relaxed minus surrogate-relaxed
accuracy, so positive means the surrogate endpoint is worse. The dashboard plots
accuracy only; loss remains in the saved artifacts rather than the visual surface.
Raw layer indices are not compared across architectures: ResNet cuts are
post-block and VGG cuts are post-convolution. Existing VGG and ResNet
critical-module plots appear only after the suffix-statistics results. Gaussian/true
gaps are not interpreted unless PCA coverage and held-out PCA-space moment
diagnostics are reported; three draws are required for a dashboard-bound artifact.

## Tangent stability and suffix response (Part D, planned)

Part D is the dynamic extension of the static Parts B/C observations. At a
checkpoint it anchors the refitted optimum suffix $\psi^*[\phi]$, compares the
full-system and suffix-only update eigenspectra, and treats a small movement of
the activation distribution as a driven input. The frequency-resolved operator
$\mathcal R(e^{i\omega})$ measures what the declared optimizer actually leaves
uncompensated; the Fisher/GGN Schur complement is its ideal geometric DC
reference.

No Part D result artifact exists. Its dashboard contains derivations,
interpretation guides, and a proposed measurement surface, all visibly labeled
`THEORY · PLANNED · UNRUN`. Multi-step Jacobian products and finite-time Lyapunov
analysis are explicitly deferred.

## Transformer extension (Part E, measured diagnostic)

> **Checkpointed gate run available (2026-07-22).** The complete seed-0 GPT-2
> small/TinyStories gate is stored locally at
> `artifacts/part_e/gate_seed0_20260722/` (4.9 GB). It includes resumable
> model+optimizer+RNG checkpoints at 0M, 4M, 16M, and 50M loss tokens, nine
> independently reusable checkpoint×cut analysis cells, immutable replay banks,
> source/config snapshots, and verified hash manifests. See
> `artifacts/part_e/README.md` before rerunning or deleting anything. The original
> gate missed its preregistered PCA target; the repaired schema-v2 endpoint
> diagnostic below passes its mechanical gates and reports a negative result.

Part E is a direct next-token-prediction replication of Part B. At selected
GPT-2-small scratch-training checkpoints and residual-stream cuts, it caches
complete masked sequences, fits mean-only and sequence-Gaussian activation
proxies, warm-starts three matched suffixes, and records the full 3x3
relax-by-evaluate matrix against true cached activations.

Only the changes needed for a valid sequence experiment are promoted into the
primary protocol: causal masks and positions travel with each cached sequence;
the Gaussian proxy models both channel and cross-position covariance; the tied
LM head is cloned before suffix-only optimization; uncertainty resamples stories;
and parity, PCA-coverage, projected-true, and future-label-leakage checks gate any
claim. IID-token Gaussian and projected-true replay remain secondary diagnostics.
Instruction/SFT/LoRA/RL branches and finite tracking/bandwidth studies are deferred
extensions rather than prerequisites for the Part E result.

The dashboard now loads the paired schema-v2 50M-checkpoint diagnostics: the
gradient-matched primary comparison and equal-nominal-LR control. It reports the
mechanically valid but scientifically negative result, while preserving the
original design in [`docs/part_e_gpt2_design.md`](docs/part_e_gpt2_design.md).
Pass them to the report builder with:

```bash
--part-e artifacts/part_e/diagnostic_v2_seed0_20260722/suffix_statistics.json \
--part-e-equal-lr artifacts/part_e/diagnostic_v2_equal_lr_seed0_20260722/suffix_statistics.json
```
