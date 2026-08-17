# LessWrong post package

This folder collects the public-facing materials for *Free-body diagrams for
neural networks*.

- `lw_wip_post.md` — current post draft
- `lw_wip_outline.md` — working structure and claim boundaries
- `lw_post_handoff.md` — current findings and open blockers
- `lw_post_reproduction.md` — reproduction protocol
- `measurement_supplement_proposal.md` — follow-up measurements with current status
- `figures/` — exactly five figures referenced by the current post
- `notebooks/lw_post_figures.py` — measured-only-by-default figure generator
- `free-body-diagrams-for-neural-networks.html` — interactive ablation appendix
- `deprecated/` — delivered talk, preliminary figures, and historical animator

The canonical post evidence is the July 26 three-seed CNN control battery under
`../artifacts/lw_post/measured/`. The hash-pinned CNN-only evidence index is
`../artifacts/lw_post/dashboard_manifest.json`, and the HTML appendix above is
built from it. The older CNN depth-by-time and one-model ResNet figures are pilot
or methods-demonstration material only. They and the delivered talk are under
`deprecated/`. The old omnibus dashboard is preserved at
`../deprecated/2026-07-omnibus/report.html`, not used as the post's evidence
source, and not published by GitHub Pages.

The five current figures are `class_conditioned_pushforward_pca.svg`,
`conceptual_internal_cut.svg`, `cnn_measured_controls.png`,
`cnn_measured_horizon.png`, and `tracking_resolving.svg`. Running
`notebooks/lw_post_figures.py` with no flags regenerates only the two measured
PNGs. Historical generation requires `--legacy-full-suite` and writes only
under `deprecated/figures/`.

The implementation stays in the normal project locations:

- measured inputs and manifest: `../artifacts/lw_post/`
- reusable experiment/report code: `../src/tracking2/`
- run and verification entry points: `../scripts/`
- tests: `../tests/`

Commands in the reproduction notes are run from the Tracking2 project root.
