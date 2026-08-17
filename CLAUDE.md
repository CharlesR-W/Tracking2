# Tracking2 project conventions

This is a fresh project studying distributional-complexity learning, layerwise
tracking, plasticity, and frequency response in neural networks. Do not import
design or implementation choices from sibling `Tracking`, `SlidingDownStairs`,
or similarly named projects unless the user explicitly asks for a comparison.

The current research design is in `SPEC.md`. Primary-source PDFs are stored in
`papers/`. Results must be labeled as mock or real and should eventually be
integrated into one self-contained HTML report, following the parent experiment
workflow.

The LessWrong post's public-facing package lives in `LW post/`: the restored
pre-rewrite prose lineage, a hash-pinned active figure manifest, and the
interactive appendix. Active illustrations and animations are exact copies
from the delivered-talk visual family. Historical pilot plots remain clearly
labelled in the prose; the rejected July redesign is isolated under
`LW post/deprecated/rejected-2026-07-redesign/`. The older omnibus report and
inputs are under `deprecated/2026-07-omnibus/`. Experiment implementation and
current measured inputs remain in the standard `src/`, `scripts/`, `tests/`,
and `artifacts/lw_post/` locations.
