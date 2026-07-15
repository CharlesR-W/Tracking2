#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/tracking2-criticality-smokes}"
mkdir -p "$SMOKE_ROOT"

echo "[1/7] unit and transplantation invariants"
.venv/bin/python -m pytest -q

echo "[2/7] tiny memorization and classifier-reset positive control"
.venv/bin/python scripts/criticality_overfit_smoke.py

echo "[3/7] fake-data training and all-module intervention pipeline"
.venv/bin/python -m tracking2.criticality \
  --output "$SMOKE_ROOT/fake-train" --fake-data \
  --train-size 32 --test-size 16 --epochs 1 --batch-size 8 \
  --classifier-width 16 --width-multiplier 0.0625 --checkpoints 0 1 \
  --recovery-modules 999 --recovery-sources none --recovery-steps 0 --device cpu

echo "[4/7] suffix-recovery pipeline"
.venv/bin/python -m tracking2.criticality \
  --output "$SMOKE_ROOT/recovery" --fake-data \
  --train-size 16 --test-size 8 --epochs 0 --batch-size 8 \
  --classifier-width 16 --width-multiplier 0.0625 --checkpoints 0 \
  --recovery-modules 12 --recovery-sources random \
  --recovery-steps 2 --recovery-eval-steps 0 1 2 --device cpu

echo "[5/7] real-CIFAR parquet loader and intervention pipeline"
.venv/bin/python -m tracking2.criticality \
  --output "$SMOKE_ROOT/real-loader" \
  --train-size 4 --test-size 4 --epochs 0 --batch-size 4 \
  --classifier-width 16 --width-multiplier 0.0625 --checkpoints 0 \
  --recovery-modules 999 --recovery-sources none --recovery-steps 0 --device cpu

echo "[6/7] multi-artifact aggregation contract"
.venv/bin/python scripts/aggregate_criticality.py \
  "$SMOKE_ROOT/fake-train/criticality.json" \
  "$SMOKE_ROOT/recovery/criticality.json" \
  --output "$SMOKE_ROOT/aggregate.json"

echo "[7/7] JSON finiteness and schema assertions"
.venv/bin/python -m scripts.verify_criticality_artifacts "$SMOKE_ROOT"

echo "[done] all criticality smoke tests passed: $SMOKE_ROOT"
