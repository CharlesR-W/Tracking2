#!/usr/bin/env bash
set -euo pipefail

# Paid scientific run: intentionally separate from run_criticality_smokes.sh.
# Do not launch without explicit approval and a live RunPod budget check.
cd /workspace/Tracking2
python -m pip install -e .
mkdir -p artifacts/criticality/gate logs

python -m tracking2.criticality \
  --output artifacts/criticality/gate/seed0 \
  --train-size 50000 --test-size 10000 \
  --epochs 40 --batch-size 128 \
  --checkpoints 0 1 2 5 10 20 40 \
  --lr-milestones 30 \
  --classifier-width 512 --width-multiplier 1.0 --batch-norm \
  --recovery-modules 999 --recovery-sources none --recovery-steps 0 \
  --seed 0 --device cuda \
  2>&1 | tee logs/criticality-gate-seed0.log
