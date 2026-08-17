# LessWrong post trusted-lineage cleanup

Status: Deprecated — completed 2026-08-17; superseded by `RESULTS.md`,
`AGENTS.md`, and corrective commit `6079743`

## Objective

Replace the rejected July/August post redesign with the restored original prose
and the figure family used in the delivered talk. Make the animated
depth-by-time/waterfall results prominent, while keeping historical pilot
evidence visibly separate from the later controlled CNN evidence.

## Authoritative decisions

- The user identifies the GIF/waterfall and delivered-talk visual family as
  trustworthy.
- The user explicitly rejects the SVG illustrations used by publication commit
  `9c791b4`.
- Commit `9218f5d` records that `docs/lw_wip_post.md` was restored byte-for-byte
  from `4371d3c`; later redesigned figures were left only for inspection and
  were never approved.
- The rejected `9c791b4` Markdown must not be published to LessWrong. The
  corrected Markdown at `6079743` is the publication candidate.

## Provenance found

- `4371d3c`: original WIP post.
- `0d0cf06`: rejected full prose/figure redesign.
- `9218f5d`: restored original prose and wrote the warning handoff.
- `2972040`: later large rewrite around the control battery.
- `68d8ecb` through `9c791b4`: packaged and published the later rewrite family.
- `LW post/deprecated/talk/slides.html` identifies the delivered visual assets,
  including the three talk GIFs.

## Evidence boundary

The GIF/waterfall plots are historical pilots. They are useful for explaining
the phenomenon but are not independent-seed evidence. The later three-seed CNN
control battery and appendix remain valid quantitative evidence, even though
their rejected presentation graphics must not remain active post assets.

## Verification already completed

- Git blob history confirms that the original post blobs at `4371d3c` and
  `9218f5d` are identical.
- Visual inspection confirms that the talk schematics and the deployed redesign
  SVGs are distinct families.
- The delivered talk directly references the trusted schematic and GIF names.

## Completed

- Restored the post structure and voice from `4371d3c`.
- Promoted nine exact talk-family assets into `LW post/figures/` and locked them
  in `LW post/figure_manifest.json`.
- Made the CNN and ResNet animations prominent, with explicit pilot captions.
- Kept the three-seed controlled result as the quantitative headline and the
  appendix as its evidence surface.
- Isolated the rejected figures, generator, outline, tests, and coordinator
  handoff under `LW post/deprecated/rejected-2026-07-redesign/`.
- Updated Pages, the verifier, documentation, and layout tests for the new
  allowlist. The full suite reports 106 passing tests; the strict
  verifier reports nine active trusted figures and a byte-identical appendix.
- Independent reader and evidence reviews passed after resolving their findings;
  a fresh verifier independently confirmed the lineage, active assets, and
  rejected-asset isolation.
- Corrective commit `60797435e4438931a7c983c47f64a148c0cff96f` was pushed to
  `main`. GitHub Pages run `32046726839` succeeded, and live HTTP checks found
  the appendix plus all nine assets at their locked hashes and expected MIME
  types.
- The original dirty experiment worktree was not modified by this cleanup.

## Remaining publication step

Paste `LW post/free-body-diagrams-for-neural-networks.md` into LessWrong and
inspect the platform preview. This cleanup did not create or publish a
LessWrong draft.
