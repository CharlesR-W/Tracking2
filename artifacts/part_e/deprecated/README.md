# Deprecated Part E artifacts

## Removed incomplete checkpoint staging directory (2026-08-16)

The former `artifacts/part_e/checkpoints/` directory was an incomplete 2.0 GB
download duplicate. Its 0M and 4M model/training-state files were byte-identical
to the verified files under
`artifacts/part_e/gate_seed0_20260722/checkpoints/`; its 16M model was an
89,825,280-byte prefix of the complete 495,414,912-byte canonical model.

The incomplete duplicate was removed after byte comparison. Preserve
`artifacts/part_e/gate_seed0_20260722/`: it is the sole canonical recovery
source for the 50M-token trajectory.
