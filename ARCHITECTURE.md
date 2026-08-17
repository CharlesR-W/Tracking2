# Tracking2 architecture

Tracking2 is an experiment repository rather than a long-running service. Its
main dependency direction is:

```text
models + surrogate construction + measurements
                    ↓
         experiment-specific runners
                    ↓
       immutable JSON artifacts/checkpoints
                    ↓
        verifiers + manifest builders
                    ↓
       HTML reports and publication package
```

Reports consume artifacts; experiment code must not consume generated reports.
Publication manifests select and hash evidence rather than recomputing it.

## Components

| Area | Responsibility |
|---|---|
| `src/tracking2/models.py` | Small CNN model definitions and cut interfaces |
| `src/tracking2/surrogates.py`, `representation_surrogates.py` | Mean, Gaussian, PCA, and replay construction |
| `src/tracking2/measurements.py`, `provenance.py` | Shared metrics and run identity/provenance |
| `experiment.py`, `suffix_statistics.py` | Baseline CIFAR and frozen-prefix/suffix workflows |
| `cnn_checkpoints.py`, `cnn_projection_adequacy.py`, `post_statistics.py` | Controlled LessWrong CNN battery inputs |
| `criticality.py`, `resnet_criticality.py` | Module transplantation and downstream recovery |
| `resnet_suffix_statistics.py`, `vgg_suffix_statistics.py` | Larger-architecture suffix experiments |
| `part_e.py` | GPT-2-small/TinyStories diagnostic |
| `report.py` | Historical omnibus report builder |
| `post_report.py` | Deterministic measured-only LessWrong appendix builder |
| `scripts/` | Bounded orchestration, aggregation, manifest building, and verification |
| `artifacts/` | Run outputs, manifests, receipts, and local checkpoints |
| `LW post/` | Publication prose, active visual allowlist, and generated appendix |

Modules with `mockup` in their name generate planned explanatory surfaces, not
measured evidence.

## Main workflows

### Controlled LessWrong evidence

The CNN training, suffix-statistics, projection, and matched-control runners
write JSON under `artifacts/lw_post/measured/`. `scripts/make_lw_post_manifest.py`
selects the committed inputs and records hashes in
`artifacts/lw_post/dashboard_manifest.json`. The manifest plus audited
`provenance_corrections.json` feed `tracking2.post_report`. The strict verifier
checks input schemas and hashes, Git/source identities, headline values, figure
allowlist, and byte-identical HTML regeneration before Pages publishes anything.

Fresh paid reruns use the fail-closed protocol in
`scripts/lw_post_publication_runner_spec.md`; staged artifacts must pass the
staged verifier before promotion into committed paths.

### General CIFAR and architecture experiments

The baseline and suffix runners train a model, freeze a prefix at declared cuts,
construct replay distributions from its activations, train suffix copies, and
write cross-condition metrics. ResNet/VGG scripts adapt that contract to native
architecture cuts. Aggregators combine independent seeds only after validating
compatible schemas and configurations. `tracking2.report` renders the archived
omnibus surface; it is not the LessWrong evidence index.

### Criticality and transformer diagnostics

Criticality runners transplant or reset declared modules and optionally train
only downstream components. Part E builds reusable replay banks at GPT-2 cuts
and evaluates a schema-versioned endpoint diagnostic. These workflows have
their own artifact validators and must not be folded into the LessWrong manifest
without an explicit evidence-contract change.

## Schemas and invariants

- Artifact JSON records experiment/schema identity, configuration, seed, source
  revision/archive identity, and condition-specific metrics.
- Independent model seeds, surrogate redraws, checkpoints, and minibatches are
  different statistical units and must not be pooled interchangeably.
- Cut semantics are architecture-specific: small-CNN, post-block ResNet, and
  post-convolution VGG indices are not directly comparable labels.
- Generated reports are deterministic functions of pinned manifests and inputs.
- Provenance errors are corrected through an explicit audited sidecar, never by
  silently rewriting measured artifacts.
- Mock, smoke, pilot, measured, and confirmatory states remain explicit in names,
  manifests, documentation, and visual labels.

## Generated and archival boundaries

- `LW post/free-body-diagrams-for-neural-networks.html` is generated but tracked
  so the verifier can enforce byte identity.
- `LW post/figures/` is a manifest-controlled publication allowlist, not a general
  output directory.
- `deprecated/`, `LW post/deprecated/`, and `scripts/deprecated/` preserve frozen
  history. Current entry points must not import or call them.
- Checkpoints and large local artifact trees may be intentionally absent from
  Git. Consult the nearest artifact README and Git status before moving, deleting,
  or staging them.

## Tests and change map

| Change | Primary code/docs | Required checks |
|---|---|---|
| Model or cut semantics | `models.py`, architecture runner | focused model/architecture tests, full suite |
| Surrogate or PCA semantics | surrogate modules, statistics runner | surrogate/statistics tests and artifact schema checks |
| Artifact schema/provenance | producer, verifier, manifest builder | producer and report tests plus end-to-end verifier |
| LessWrong prose/figures | `LW post/`, figure manifest, Pages workflow | publication layout test, strict verifier, rendered visual inspection |
| Appendix presentation | `post_report.py` | `test_post_report.py`, strict verifier, byte-identical rebuild |
| Omnibus presentation | `report.py` | `test_report.py`; do not promote it into the post package |
| Trusted result or claim boundary | artifact/manifest plus `RESULTS.md` | source/hash audit and proportionate independent review |
