# LessWrong post package

This folder collects the public-facing materials for *Free-body diagrams for
neural networks*.

- `lw_wip_post.md` — current post draft
- `lw_wip_outline.md` — working structure and claim boundaries
- `lw_post_handoff.md` — current findings and open blockers
- `lw_post_reproduction.md` — reproduction protocol
- `measurement_supplement_proposal.md` — follow-up measurements with current status
- `figures/` — canonical publication figures plus labelled historical pilots
- `notebooks/` — figure and animation generators
- `free-body-diagrams-for-neural-networks.html` — interactive ablation appendix
- `talk/` — archival slides and exports, preserved as delivered

The canonical post evidence is the July 26 three-seed CNN control battery under
`../artifacts/lw_post/measured/`. The hash-pinned CNN-only evidence index is
`../artifacts/lw_post/dashboard_manifest.json`, and the HTML appendix above is
built from it. The older CNN depth-by-time and one-model ResNet figures are pilot
or methods-demonstration material only. `../report.html` is an omnibus research
archive, not the post's evidence source.

The delivered talk predates the canonical control battery. Preserve it as an
archival record; its preliminary figures must not be used to support the post's
current quantitative claims.

The implementation stays in the normal project locations:

- measured inputs and manifest: `../artifacts/lw_post/`
- reusable experiment/report code: `../src/tracking2/`
- run and verification entry points: `../scripts/`
- tests: `../tests/`

Commands in the reproduction notes are run from the Tracking2 project root.
