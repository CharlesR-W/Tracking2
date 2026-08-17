# Tracking2 results and evidence index

This file indexes trusted measured results and their interpretation boundaries.
Detailed plots and run-level provenance live in the linked artifacts and reports.

## LessWrong publication result

The current decision-relevant result supports
`LW post/free-body-diagrams-for-neural-networks.md`. Its canonical evidence is:

- interactive appendix: `LW post/free-body-diagrams-for-neural-networks.html`
- manifest: `artifacts/lw_post/dashboard_manifest.json`
- measured inputs: `artifacts/lw_post/measured/`
- audited provenance corrections: `artifacts/lw_post/provenance_corrections.json`
- active visual lineage: `LW post/figure_manifest.json`
- ResNet-first pilot explorer: `LW post/waterfalls.html`
- pinned pilot inputs: `artifacts/waterfall_visuals/manifest.json`
- reproduction protocol: `LW post/lw_post_reproduction.md`

The controlled follow-up uses three independently trained small-CNN seeds. The
primary paired held-out true cross-entropy differences are mean $\pm$ sample
standard deviation across seeds; positive means the first-named replay condition
finished worse:

| Cut | Gaussian $-$ PCA-projected empirical | Mean ($r=1$) $-$ Gaussian |
|---|---:|---:|
| After block 1 | $+0.183 \pm 0.055$ | $+0.144 \pm 0.019$ |
| After block 4 | $-0.012 \pm 0.002$ | $+0.023 \pm 0.011$ |

The supported claim is narrow: under this finite suffix-training protocol,
replay conditions separate more at the shallow cut than at the final cut. Moving
the cut also changes receiver size and trainability, so this is not identified as
a representation-only depth effect.

### Evidence limitations

- The CNN and ResNet GIF/waterfall plots are historical pilots used to illustrate
  the phenomenon. CNN frames came from separately scheduled models; the ResNet
  view used one trained model. They are not independent-seed replications.
- ResNet-18 is the canonical explanatory visualization because its eight native
  cuts make the depth profile legible. It remains one model; three redraws per
  cell measure generated-activation/minibatch variation, not model uncertainty.
- The shallow rank-2,048 projection has cross-entropy excess $0.0154$ nat and
  predictive KL $0.0373$ nat. It passes the $0.05$-nat cross-entropy threshold
  but fails the $0.02$-nat KL threshold, so it does not establish preservation of
  the full shallow interface.
- The mean-versus-Gaussian comparison is sensitive to nuisance-noise radius.
- The data do not identify mechanism, prove a universal depth law, or isolate
  representation geometry from receiver capacity and optimization.

## Publication lineage

- `4371d3c` is the trusted original post baseline.
- `9218f5d` restored that prose byte-for-byte after a rejected rewrite.
- `0d0cf06` and the later package ending at `9c791b4` belong to the rejected
  July/August presentation lineage.
- `6079743` restored the trusted post/talk asset family and isolated the rejected
  redesign under `LW post/deprecated/rejected-2026-07-redesign/`.
- `6c0bdd5` records the completed release and documentation state.

The rejected presentation included two plots of genuine measured data. Their
archival status rejects that post/figure lineage, not the underlying controlled
JSON evidence.

## Other experiment status

- Part D tangent stability and suffix response remains theory/planned/unrun.
- Part E is a measured one-seed GPT-2-small/TinyStories diagnostic. Its repaired
  schema-v2 mechanical gates pass, but sequence Gaussian replay does not
  consistently beat mean replay; this is a negative diagnostic, not a confirmed
  extension of the CNN result. See `docs/part_e_gpt2_design.md` and
  `artifacts/part_e/README.md`.
- Older omnibus, pilot, and unrelated surfaces are preserved under
  `deprecated/2026-07-omnibus/`; they are not the LessWrong post's evidence index.

## Verification

Last verified 2026-08-17 for publication content revision `6079743`: 106 tests
passed; the committed publication verifier passed all
measured inputs, provenance corrections, headline cells, active figure hashes,
and deterministic appendix identity. Independent reader, evidence, and final
verification reviews passed. The documentation-only GitHub Pages deployment at
`6c0bdd5` (run `32047025413`) succeeded, and the live appendix hash matched the
tracked file.
