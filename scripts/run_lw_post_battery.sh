#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${TRACKING2_SOURCE_REVISION:-}" ]] \
  && git rev-parse HEAD >/dev/null 2>&1 \
  && [[ -z "$(git status --porcelain)" ]]; then
  TRACKING2_SOURCE_REVISION="$(git rev-parse HEAD)"
  export TRACKING2_SOURCE_REVISION
fi
if [[ ! "${TRACKING2_SOURCE_REVISION:-}" =~ ^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$ ]] \
  && [[ ! "${TRACKING2_SOURCE_ARCHIVE_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Measured runs require a clean full TRACKING2_SOURCE_REVISION or TRACKING2_SOURCE_ARCHIVE_SHA256." >&2
  exit 1
fi

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DRAW_COUNT="${DRAW_COUNT:-3}"
PCA_RANK="${PCA_RANK:-512}"
MIN_HOST_RAM_GIB="${MIN_HOST_RAM_GIB:-90}"
TRAINING_ROOT="artifacts/lw_post/cnn_seed${SEED}"
RESULT_ROOT="artifacts/lw_post/cnn_statistics_seed${SEED}"

if [[ "${SKIP_HOST_MEMORY_CHECK:-0}" != "1" ]]; then
  python scripts/check_host_memory.py "$MIN_HOST_RAM_GIB"
fi

if [[ ! -f "$TRAINING_ROOT/training.json" ]]; then
  python -m tracking2.cnn_checkpoints \
    --output "$TRAINING_ROOT" \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --epochs 30 --checkpoints 0 1 5 10 20 30 \
    --batch-size 256 --seed "$SEED" --device "$DEVICE"
fi

for checkpoint_epoch in 0 1 5 10 20 30; do
  output="$RESULT_ROOT/base_epoch${checkpoint_epoch}"
  if [[ -f "$output/post_statistics.json" ]]; then
    echo "[skip] $output"
    continue
  fi
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch${checkpoint_epoch}.pt" \
    --checkpoint-epoch "$checkpoint_epoch" \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts 1 2 3 4 --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks "$PCA_RANK" --gaussian-covariance-shrinkages 0 \
    --mean-noise-radii 1 \
    --surrogate-draws "$DRAW_COUNT" --relax-epochs 10 \
    --seed "$SEED" --device "$DEVICE"
done

output="$RESULT_ROOT/pca_ablation_epoch30_cut1"
if [[ ! -f "$output/post_statistics.json" ]]; then
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch30.pt" \
    --checkpoint-epoch 30 \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts 1 --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks 128 512 1024 --gaussian-covariance-shrinkages 0 \
    --mean-noise-radii 1 \
    --surrogate-draws 3 --relax-epochs 10 \
    --seed "$SEED" --device "$DEVICE"
fi

output="$RESULT_ROOT/pca_ablation_epoch30_cut4"
if [[ ! -f "$output/post_statistics.json" ]]; then
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch30.pt" \
    --checkpoint-epoch 30 \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts 4 --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks 128 512 1024 2048 --gaussian-covariance-shrinkages 0 \
    --mean-noise-radii 1 \
    --surrogate-draws 3 --relax-epochs 10 \
    --seed "$SEED" --device "$DEVICE"
fi

output="$RESULT_ROOT/noise_ablation_epoch30_cuts1_4"
if [[ ! -f "$output/post_statistics.json" ]]; then
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch30.pt" \
    --checkpoint-epoch 30 \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts 1 4 --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks "$PCA_RANK" --gaussian-covariance-shrinkages 0 \
    --mean-noise-radii 0 0.1 0.5 1 2 \
    --surrogate-draws 3 --true-eval-only --relax-epochs 10 \
    --seed "$SEED" --device "$DEVICE"
fi

output="$RESULT_ROOT/covariance_ablation_epoch30_cuts1_4"
if [[ ! -f "$output/post_statistics.json" ]]; then
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch30.pt" \
    --checkpoint-epoch 30 \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts 1 4 --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks "$PCA_RANK" --gaussian-covariance-shrinkages 0 0.01 0.05 \
    --mean-noise-radii 1 \
    --surrogate-draws 3 --true-eval-only --relax-epochs 10 \
    --seed "$SEED" --device "$DEVICE"
fi

python scripts/verify_lw_post_artifacts.py \
  "$TRAINING_ROOT" "$RESULT_ROOT" \
  --pca-rank "$PCA_RANK" --draw-count "$DRAW_COUNT" --seed "$SEED"
