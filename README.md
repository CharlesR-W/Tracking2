# Tracking2

Fresh CIFAR experiments on selected-statistics surrogates and layerwise tracking.
The research design is in `SPEC.md`; the current interactive result viewer is
`report.html`.

## Implemented

- Experiment A: train/test matrix across true CIFAR, a class-mean proxy, and a
  class-conditional Gaussian matching mean/covariance in a fitted PCA space.
- A residual CNN with GroupNorm and explicit residual-stream cuts. A plain-block
  implementation remains available only as an optional secondary control.
- Experiment B: warm-started frozen-prefix suffix refits, signed held-out refit
  gain, prediction KL, refit path length, tracking demand across checkpoints,
  interface drift, and relative representation drift.
- Low-rank empirical-Fisher prefix/suffix traces and a ridge-regularized Schur
  estimate of suffix-compensable prefix sensitivity.
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
