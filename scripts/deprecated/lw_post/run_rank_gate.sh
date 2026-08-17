#!/usr/bin/env bash
set -euo pipefail

echo "DEPRECATED: running the historical exploratory PCA gate, not the publication protocol." >&2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
  echo "Measured runs require a clean source revision or source archive hash." >&2
  exit 1
fi

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
RELAX_EPOCHS="${RELAX_EPOCHS:-5}"
MIN_HOST_RAM_GIB="${MIN_HOST_RAM_GIB:-90}"
TRAINING_ROOT="artifacts/lw_post/cnn_seed${SEED}"
RESULT_ROOT="artifacts/lw_post/cnn_gate_seed${SEED}"

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

for cut in 1 4; do
  output="$RESULT_ROOT/rank_gate_cut${cut}"
  if [[ -f "$output/post_statistics.json" ]]; then
    echo "[skip] $output"
    continue
  fi
  ranks=(128 512 1024)
  if [[ "$cut" == "4" ]]; then
    ranks+=(2048)
  fi
  python -m tracking2.post_statistics \
    --output "$output" \
    --checkpoint "$TRAINING_ROOT/checkpoint_epoch30.pt" \
    --checkpoint-epoch 30 \
    --data-backend torchvision \
    --train-size 50000 --test-size 10000 \
    --cuts "$cut" --batch-size 256 --pca-fit-size 10000 \
    --pca-ranks "${ranks[@]}" --gaussian-covariance-shrinkages 0 \
    --mean-noise-radii 1 --surrogate-draws 1 \
    --relax-epochs "$RELAX_EPOCHS" \
    --seed "$SEED" --device "$DEVICE"
done

python scripts/deprecated/lw_post/verify_rank_gate.py \
  "$TRAINING_ROOT" "$RESULT_ROOT" \
  --seed "$SEED" --relax-epochs "$RELAX_EPOCHS"
