#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Tracking2

SEED="${SEED:-2}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-100}"
DRAW_COUNT="${DRAW_COUNT:-3}"
PCA_COMPONENTS="${PCA_COMPONENTS:-128}"
ROOT="artifacts/criticality/seed${SEED}"
OUTPUT="artifacts/vgg_suffix_statistics/seed${SEED}-epoch${CHECKPOINT_EPOCH}"

python -m tracking2.vgg_suffix_statistics \
  --output "$OUTPUT" \
  --checkpoint "$ROOT/checkpoint_epoch${CHECKPOINT_EPOCH}.pt" \
  --initialization-checkpoint "$ROOT/checkpoint_epoch0.pt" \
  --batch-norm --cuts 7 8 9 13 --conditions native reset0 \
  --train-size 10000 --test-size 2000 --batch-size 128 \
  --pca-fit-size 5000 --pca-components "$PCA_COMPONENTS" \
  --surrogate-draws "$DRAW_COUNT" --relax-epochs 5 \
  --seed "$SEED" --device cuda
