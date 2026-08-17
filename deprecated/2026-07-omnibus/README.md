# July 2026 omnibus research archive

This directory preserves the tracked inputs and self-contained dashboard from
the pre-publication Tracking2 omnibus analysis. It is scientific history, not
evidence for *Free-body diagrams for neural networks*. The publication contract
is the CNN-only manifest and measured inputs under `../../artifacts/lw_post/`.

The archive was relocated without editing `report.html`, any result JSON, any
checkpoint, or the recorded paths inside those files. Embedded paths therefore
describe the original run locations rather than the files' new archive paths.

The tracked subtrees are:

- `artifacts/cifar_pilot_seed0/` — bounded feasibility pilot;
- `artifacts/confirmatory/` — older whole-network confirmatory battery;
- `artifacts/criticality/` — legacy critical-module aggregate and gate;
- `artifacts/suffix_statistics/` — older CNN depth-by-training-time sweep;
- `artifacts/resnet_suffix_statistics/` — one-model ResNet pilot;
- `artifacts/vgg_batchzoom/` — VGG batch-zoom pilot;
- `artifacts/vgg_suffix_statistics/` — VGG suffix-statistics preview; and
- `report.html` — the corresponding self-contained omnibus dashboard.

The recommended historical layout also named `artifacts/resnet_criticality/`
and `artifacts/vgg_checkpoints/`, but snapshot `68d8ecb` contained **zero
tracked files** under either path. Their local untracked checkpoints and result
files were deliberately not moved or added during publication cleanup. The
dashboard remains self-contained even though those raw local sources are not in
this tracked archive.

Do not add this archive to the LessWrong manifest, pool its rows with the July
26 matched-update CNN evidence, or silently regenerate `report.html` with newer
code.
