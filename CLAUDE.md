# Tracking2 project conventions

This is a fresh project studying distributional-complexity learning, layerwise
tracking, plasticity, and frequency response in neural networks. Do not import
design or implementation choices from sibling `Tracking`, `SlidingDownStairs`,
or similarly named projects unless the user explicitly asks for a comparison.

The current research design is in `SPEC.md`. Primary-source PDFs are stored in
`papers/`. Results must be labeled as mock or real and should eventually be
integrated into one self-contained HTML report, following the parent experiment
workflow.

The LessWrong post's public-facing package lives in `LW post/`: draft and
planning notes, figures, figure notebooks, interactive appendix, and talk
materials. Its experiment implementation and measured inputs remain in the
standard `src/`, `scripts/`, `tests/`, and `artifacts/lw_post/` locations.
