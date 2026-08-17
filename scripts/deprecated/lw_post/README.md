# Deprecated LessWrong experiment entry points

These files preserve the pre-publication rank-512 omnibus and PCA-gate
workflows for historical reproducibility. They are not inputs to the July 26
publication package and must not be used to refresh its measured claims.

- `run_rank512_omnibus_battery.sh` runs the old all-checkpoint/all-cut,
  fixed-learning-rate, three-draw, ten-relaxation-epoch battery.
- `verify_rank512_omnibus_battery.py` verifies that old battery.
- `run_rank_gate.sh` and `verify_rank_gate.py` implement the superseded
  rank-128/512/1,024(/2,048) exploratory gate.

The current checked-in evidence is verified with
`scripts/verify_lw_post_publication.py`. No paid rerun is required for that
publication claim. A future clean rerun must follow
`scripts/lw_post_publication_runner_spec.md` and run under the fail-closed
RunPod supervisor described there.
