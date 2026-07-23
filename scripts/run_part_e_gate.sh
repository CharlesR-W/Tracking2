#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Tracking2

output="artifacts/part_e/gate_seed0"
mkdir -p "$output" logs

python -m tracking2.part_e prepare --output "$output"
python -m tracking2.part_e train --output "$output"
python -m tracking2.part_e analyze --output "$output"
