# LessWrong post package

This folder collects the public-facing materials for *Free-body diagrams for
neural networks*.

- `free-body-diagrams-for-neural-networks.md` — publication post
- `lw_post_reproduction.md` — reproduction protocol
- `measurement_supplement_proposal.md` — follow-up measurements with current status
- `figure_manifest.json` — exact active asset set, hashes, lineage, and evidence status
- `figures/` — trusted talk-family schematics and animations used by the post
- `free-body-diagrams-for-neural-networks.html` — interactive ablation appendix
- `deprecated/` — delivered talk archive, unused pilots, and the rejected redesign

The post deliberately presents two evidence layers. The animated CNN
depth-by-checkpoint and one-model ResNet plots are historical pilots that make
the phenomenon visible; their captions state their lineage limitations. The
quantitative headline comes from the July 26 three-seed CNN control battery
under `../artifacts/lw_post/measured/`. Its hash-pinned evidence index is
`../artifacts/lw_post/dashboard_manifest.json`, and the HTML appendix above is
built from it. The old omnibus dashboard is preserved at
`../deprecated/2026-07-omnibus/report.html`, not used as the post's evidence
source, and not published by GitHub Pages.

The active figures are exact copies of the visual family used by the delivered
talk and are locked by `figure_manifest.json`. The delivered talk remains frozen
under `deprecated/talk/`, with its own copies in `deprecated/figures/` so its
relative links continue to resolve. The five figures and generator from the
rejected July redesign are retained only under
`deprecated/rejected-2026-07-redesign/` for auditability.

The implementation stays in the normal project locations:

- measured inputs and manifest: `../artifacts/lw_post/`
- reusable experiment/report code: `../src/tracking2/`
- run and verification entry points: `../scripts/`
- tests: `../tests/`

Commands in the reproduction notes are run from the Tracking2 project root.
