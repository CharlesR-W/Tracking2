# Rejected July 2026 post redesign

This directory is a tombstone for the presentation family that began with the
stopped rewrite at `0d0cf06`, was revised again at `2972040`, and was
accidentally promoted into the August publication package ending at `9c791b4`.

The July handoff at `9218f5d` explicitly said that the redesigned figures were
left only for inspection and should not be treated as approved. The user has
now confirmed that these are not the visual or prose direction for the post.

`figures/` contains the five assets that were previously published by Pages.
`notebooks/lw_post_figures.py` is their generator. They are retained only so the
mistake remains auditable; no current post, workflow, verifier, or README should
name them as publication assets.

The two measured PNGs visualize real controlled results, so their rejection is
about the post/version lineage and presentation—not a claim that the underlying
JSON measurements are fabricated. The canonical measured evidence remains in
`artifacts/lw_post/` and the interactive appendix.

The rejected Markdown itself is not duplicated here because Git preserves it
exactly at `9c791b4`. The trusted prose baseline is `4371d3c`, restored by
`9218f5d`.
