#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SMOKE_ROOT="${SMOKE_ROOT:-/tmp/tracking2-lw-smoke}"
CHECKPOINT_ROOT="$SMOKE_ROOT/checkpoints"
STATISTICS_ROOT="$SMOKE_ROOT/statistics"

python -m tracking2.cnn_checkpoints \
  --output "$CHECKPOINT_ROOT" \
  --fake-data --train-size 32 --test-size 16 \
  --epochs 2 --checkpoints 0 1 2 --widths 4 8 \
  --batch-size 8 --device cpu

python -m tracking2.post_statistics \
  --output "$STATISTICS_ROOT" \
  --checkpoint "$CHECKPOINT_ROOT/checkpoint_epoch2.pt" \
  --checkpoint-epoch 2 --fake-data \
  --train-size 32 --test-size 16 --cuts 1 2 --widths 4 8 \
  --batch-size 8 --pca-fit-size 24 --pca-ranks 4 8 \
  --mean-noise-radii 0 1 --surrogate-draws 1 \
  --relax-epochs 1 --device cpu

python scripts/verify_lw_post_smoke.py \
  "$CHECKPOINT_ROOT/training.json" \
  "$STATISTICS_ROOT/post_statistics.json"
