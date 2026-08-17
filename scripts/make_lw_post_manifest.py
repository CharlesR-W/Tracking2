from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence


TITLE = "Free-Body Diagrams for Neural Networks — Interactive Ablation Appendix (WIP)"
MANIFEST_SCHEMA_VERSION = 2
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
SOURCE_REVISION_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CNN_EPOCHS = (0, 1, 5, 10, 20, 30)
CNN_CUTS = (1, 2, 3, 4)
LEGACY_RESNET_PATHS = {
    0: "resnet_suffix_statistics/seed0-epoch0/resnet_suffix_statistics.json",
    1: "resnet_suffix_statistics/seed0-epoch1/resnet_suffix_statistics.json",
    5: "resnet_suffix_statistics/seed0-epoch5.json",
    20: "resnet_suffix_statistics/seed0-epoch20.json",
    100: "resnet_suffix_statistics/seed0-epoch100.json",
}
CANONICAL_RESNET_PATH = (
    "lw_post/resnet_ablations_seed0/"
    "nested_ranks_epoch100_cuts4_8/resnet_suffix_statistics.json"
)


def publication_primary_analysis() -> dict[str, object]:
    """Return the explicit July 26 publication estimand and its assertions."""

    return {
        "status": "MEASURED",
        "model_kind": "cnn",
        "checkpoint_epoch": 30,
        "cuts": [1, 4],
        "pca_rank": 2048,
        "projection_scope": "retained_pca_subspace_only",
        "projection_gate_input_id": "cnn-projection-0-s0-e30",
        "suffix_initialization": "warm",
        "learning_rate_regime": "match_true_initial_update",
        "relax_epochs": 5,
        "model_seeds": [0, 1, 2],
        "input_ids_by_model_seed": {
            "0": "cnn-extra-2-s0-e30",
            "1": "cnn-extra-6-s1-e30",
            "2": "cnn-extra-7-s2-e30",
        },
        "surrogate_draws_per_model": 1,
        "contrasts": [
            {
                "id": "gaussian_minus_projected_true",
                "minuend_distribution": "gaussian_empirical",
                "subtrahend_distribution": "projected_true",
            },
            {
                "id": "mean_r1_minus_gaussian",
                "minuend_distribution": "mean_r1",
                "subtrahend_distribution": "gaussian_empirical",
            },
        ],
        "uncertainty": {
            "statistic": "sample_standard_deviation",
            "independent_unit": "trained_model_seed",
            "n": 3,
            "within_model_surrogate_redraw_uncertainty": "not_measured",
        },
        "first_update_matching": {
            "relative_tolerance": 0.01,
            "learning_rate_multiplier_bounds": [0.001, 1000.0],
            "clipped_status": "approximately_matched_clipped",
            "outside_tolerance_policy": "reject_primary_display_secondary",
        },
        "headline_assertion_absolute_tolerance": 0.0000005,
        "headline_assertions": [
            {
                "cut": 1,
                "contrast_id": "gaussian_minus_projected_true",
                "mean": 0.182913,
                "sample_standard_deviation": 0.055069,
            },
            {
                "cut": 1,
                "contrast_id": "mean_r1_minus_gaussian",
                "mean": 0.144051,
                "sample_standard_deviation": 0.019072,
            },
            {
                "cut": 4,
                "contrast_id": "gaussian_minus_projected_true",
                "mean": -0.012495,
                "sample_standard_deviation": 0.001530,
            },
            {
                "cut": 4,
                "contrast_id": "mean_r1_minus_gaussian",
                "mean": 0.022739,
                "sample_standard_deviation": 0.010517,
            },
        ],
    }


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def measured_payload(path: Path, *, epoch: int, seed: int) -> dict:
    payload = json.loads(path.read_text())
    config = payload.get("config")
    if payload.get("status") != "MEASURED":
        raise RuntimeError(f"{path} does not have exact status MEASURED")
    if not isinstance(config, dict) or config.get("fake_data") is not False:
        raise RuntimeError(f"{path} is missing config.fake_data=false")
    if config.get("checkpoint_epoch") != epoch or config.get("seed") != seed:
        raise RuntimeError(f"{path} does not match epoch={epoch}, seed={seed}")
    return payload


def entry(
    *,
    manifest_dir: Path,
    artifact_path: Path,
    input_id: str,
    kind: str,
    artifact_format: str,
    label: str,
    epoch: int,
    seed: int,
) -> dict[str, object]:
    measured_payload(artifact_path, epoch=epoch, seed=seed)
    relative_path = artifact_path.relative_to(manifest_dir.parent)
    return {
        "id": input_id,
        "kind": kind,
        "format": artifact_format,
        "label": label,
        "path": f"../{relative_path.as_posix()}",
        "sha256": digest(artifact_path),
        "checkpoint_epoch": epoch,
        "model_seed": seed,
    }


def legacy_cnn_entries(artifact_root: Path, output: Path) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    for epoch in CNN_EPOCHS:
        for cut in CNN_CUTS:
            run_name = (
                f"t0_batch0_cut{cut}_full_seed0"
                if epoch == 0
                else f"t{epoch}_cut{cut}_full_seed0"
            )
            artifact_path = (
                artifact_root
                / "suffix_statistics"
                / run_name
                / "suffix_statistics.json"
            )
            payload = measured_payload(artifact_path, epoch=epoch, seed=0)
            if payload["config"].get("cut") != cut:
                raise RuntimeError(f"{artifact_path} does not match cut={cut}")
            inputs.append(
                entry(
                    manifest_dir=output.parent,
                    artifact_path=artifact_path,
                    input_id=f"cnn-legacy-e{epoch}-c{cut}",
                    kind="cnn",
                    artifact_format="legacy_cnn_suffix_statistics",
                    label=f"Four-block CNN · legacy epoch {epoch} · cut {cut}",
                    epoch=epoch,
                    seed=0,
                )
            )
    return inputs


def post_cnn_entries(
    artifact_root: Path,
    output: Path,
    seeds: Sequence[int] = (0,),
) -> list[dict[str, object]]:
    inputs = []
    for seed in seeds:
        result_root = artifact_root / "lw_post" / f"cnn_statistics_seed{seed}"
        paths = [
            (
                f"cnn-s{seed}-e{epoch}",
                f"Four-block CNN · seed {seed} · uninterrupted trajectory · epoch {epoch}",
                epoch,
                result_root / f"base_epoch{epoch}" / "post_statistics.json",
            )
            for epoch in CNN_EPOCHS
        ]
        paths.extend(
            [
                (
                    f"cnn-s{seed}-e30-pca-ablation-c1",
                    f"Four-block CNN · seed {seed} · epoch 30 · shallow-cut PCA ranks",
                    30,
                    result_root
                    / "pca_ablation_epoch30_cut1"
                    / "post_statistics.json",
                ),
                (
                    f"cnn-s{seed}-e30-pca-ablation-c4",
                    f"Four-block CNN · seed {seed} · epoch 30 · late-cut PCA ranks",
                    30,
                    result_root
                    / "pca_ablation_epoch30_cut4"
                    / "post_statistics.json",
                ),
                (
                    f"cnn-s{seed}-e30-noise-ablation-cuts1-4",
                    f"Four-block CNN · seed {seed} · epoch 30 · shallow/late noise radii",
                    30,
                    result_root
                    / "noise_ablation_epoch30_cuts1_4"
                    / "post_statistics.json",
                ),
                (
                    f"cnn-s{seed}-e30-covariance-ablation-cuts1-4",
                    f"Four-block CNN · seed {seed} · epoch 30 · empirical/shrunk covariance",
                    30,
                    result_root
                    / "covariance_ablation_epoch30_cuts1_4"
                    / "post_statistics.json",
                ),
            ]
        )
        for input_id, label, epoch, artifact_path in paths:
            payload = measured_payload(artifact_path, epoch=epoch, seed=seed)
            if (
                payload.get("schema_version") != 2
                or payload.get("experiment") != "lw_post_cnn_suffix_statistics"
            ):
                raise RuntimeError(
                    f"{artifact_path} is not a schema-v2 post-facing CNN artifact"
                )
            inputs.append(
                entry(
                    manifest_dir=output.parent,
                    artifact_path=artifact_path,
                    input_id=input_id,
                    kind="cnn",
                    artifact_format="post_statistics",
                    label=label,
                    epoch=epoch,
                    seed=seed,
                )
            )
    return inputs


def explicit_post_cnn_entries(
    output: Path,
    artifact_paths: Sequence[Path],
) -> list[dict[str, object]]:
    inputs = []
    for index, artifact_path in enumerate(artifact_paths):
        payload = json.loads(artifact_path.read_text())
        config = payload.get("config")
        if not isinstance(config, dict):
            raise RuntimeError(f"{artifact_path} config must be an object")
        epoch = config.get("checkpoint_epoch")
        seed = config.get("seed")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise RuntimeError(f"{artifact_path} lacks checkpoint_epoch")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError(f"{artifact_path} lacks seed")
        payload = measured_payload(artifact_path, epoch=epoch, seed=seed)
        if (
            payload.get("schema_version") != 2
            or payload.get("experiment") != "lw_post_cnn_suffix_statistics"
        ):
            raise RuntimeError(
                f"{artifact_path} is not a schema-v2 post-facing CNN artifact"
            )
        regime = config.get("learning_rate_regime", "fixed_lr")
        inputs.append(
            entry(
                manifest_dir=output.parent,
                artifact_path=artifact_path,
                input_id=f"cnn-extra-{index}-s{seed}-e{epoch}",
                kind="cnn",
                artifact_format="post_statistics",
                label=(
                    f"Four-block CNN · seed {seed} · epoch {epoch} · "
                    f"{regime} sensitivity"
                ),
                epoch=epoch,
                seed=seed,
            )
        )
    return inputs


def projection_adequacy_entries(
    output: Path,
    artifact_paths: Sequence[Path],
) -> list[dict[str, object]]:
    inputs = []
    for index, artifact_path in enumerate(artifact_paths):
        payload = json.loads(artifact_path.read_text())
        config = payload.get("config")
        if not isinstance(config, dict):
            raise RuntimeError(f"{artifact_path} config must be an object")
        epoch = config.get("checkpoint_epoch")
        seed = config.get("seed")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise RuntimeError(f"{artifact_path} lacks checkpoint_epoch")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError(f"{artifact_path} lacks seed")
        payload = measured_payload(artifact_path, epoch=epoch, seed=seed)
        if (
            payload.get("schema_version") != 1
            or payload.get("experiment")
            != "lw_post_cnn_projection_adequacy"
        ):
            raise RuntimeError(
                f"{artifact_path} is not a schema-v1 CNN projection gate"
            )
        inputs.append(
            entry(
                manifest_dir=output.parent,
                artifact_path=artifact_path,
                input_id=f"cnn-projection-{index}-s{seed}-e{epoch}",
                kind="cnn_projection",
                artifact_format="cnn_projection_adequacy",
                label=(
                    f"Four-block CNN · seed {seed} · epoch {epoch} · "
                    "PCA-only adequacy gate"
                ),
                epoch=epoch,
                seed=seed,
            )
        )
    return inputs


def canonical_resnet_entries(
    artifact_root: Path,
    output: Path,
    artifact_path: Path | None = None,
) -> list[dict[str, object]]:
    artifact_path = artifact_root / CANONICAL_RESNET_PATH if artifact_path is None else artifact_path
    payload = measured_payload(artifact_path, epoch=100, seed=0)
    schema = payload.get("schema_version")
    if schema not in {3, 4} or payload.get(
        "experiment"
    ) != "resnet18_suffix_statistics_sweep":
        raise RuntimeError(f"{artifact_path} is not a schema-v3/v4 ResNet artifact")
    config = payload["config"]
    expected_cuts = [3, 7] if schema == 3 else [0, 7]
    required = (
        config.get("pca_ranks") == [128, 256, 512]
        and config.get("cuts") == expected_cuts
        and config.get("mean_noise_radii") == [1.0]
        and config.get("include_projected_true") is True
    )
    if schema == 3:
        required = required and config.get("true_eval_only") is True
    else:
        required = (
            required
            and config.get("true_eval_only") is False
            and config.get("gaussian_covariance_shrinkages") == [0.0, 0.05]
        )
    if not required:
        raise RuntimeError(
            f"{artifact_path} does not match the canonical ResNet control grid"
        )
    provenance = payload.get("provenance")
    revision = provenance.get("source_revision") if isinstance(provenance, dict) else None
    archive = (
        provenance.get("source_archive_sha256")
        if isinstance(provenance, dict)
        else None
    )
    if not (
        isinstance(revision, str)
        and SOURCE_REVISION_RE.fullmatch(revision)
        or isinstance(archive, str)
        and DIGEST_RE.fullmatch(archive)
    ):
        raise RuntimeError(
            f"{artifact_path} lacks a reproducible source identity"
        )
    input_id = (
        "resnet-e100-pca-ablation-cuts4-8"
        if schema == 3
        else "resnet-e100-full-matrix-cuts1-8"
    )
    label = (
        "ResNet-18 · epoch 100 · nested PCA controls · cuts 4 and 8"
        if schema == 3
        else "ResNet-18 · epoch 100 · full matrix · shallow and late cuts"
    )
    return [
        entry(
            manifest_dir=output.parent,
            artifact_path=artifact_path,
            input_id=input_id,
            kind="resnet",
            artifact_format="resnet_suffix_statistics",
            label=label,
            epoch=100,
            seed=0,
        )
    ]


def resnet_entries(
    artifact_root: Path,
    output: Path,
    source: str,
    canonical_path: Path | None = None,
) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    if source in {"legacy", "both"}:
        raise RuntimeError(
            "The legacy ResNet discovery mode is deprecated and archive-only; "
            "it cannot be added to the publication manifest."
        )
    if source in {"canonical", "both"}:
        inputs.extend(canonical_resnet_entries(artifact_root, output, canonical_path))
    if source not in {"none", "legacy", "canonical", "both"}:
        raise ValueError(
            "resnet_source must be 'none', 'legacy', 'canonical', or 'both'"
        )
    return inputs


def build_manifest(
    artifact_root: Path,
    output: Path,
    source_commit: str,
    cnn_source: str = "post",
    resnet_source: str = "none",
    cnn_seeds: Sequence[int] = (0,),
    canonical_resnet_path: Path | None = None,
    projection_adequacy_paths: Sequence[Path] = (),
    extra_cnn_paths: Sequence[Path] = (),
    provenance_corrections_path: Path | None = None,
) -> dict:
    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a 7–64 character hexadecimal id")
    if cnn_source == "none":
        inputs = []
    elif cnn_source == "legacy":
        raise RuntimeError(
            "The legacy CNN discovery mode is deprecated and archive-only; "
            "publication manifests must use explicit measured input paths."
        )
    elif cnn_source == "post":
        inputs = post_cnn_entries(artifact_root, output, cnn_seeds)
    else:
        raise ValueError("cnn_source must be 'none', 'legacy', or 'post'")
    inputs.extend(explicit_post_cnn_entries(output, extra_cnn_paths))
    inputs.extend(
        projection_adequacy_entries(output, projection_adequacy_paths)
    )
    inputs.extend(
        resnet_entries(
            artifact_root,
            output,
            resnet_source,
            canonical_resnet_path,
        )
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "title": TITLE,
        "status": "MEASURED",
        "source_commit": source_commit,
        "primary_analysis": publication_primary_analysis(),
        "inputs": inputs,
    }
    if provenance_corrections_path is not None:
        relative_path = provenance_corrections_path.relative_to(output.parent)
        manifest["provenance_corrections"] = {
            "path": relative_path.as_posix(),
            "sha256": digest(provenance_corrections_path),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the explicit measured-artifact manifest for the LW appendix."
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lw_post/dashboard_manifest.json"),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--cnn-source",
        choices=("none", "post"),
        default="post",
        help=(
            "Select no implicit CNN inputs or the verified uninterrupted-run "
            "battery. Legacy pilots are archive-only."
        ),
    )
    parser.add_argument(
        "--cnn-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Model seeds whose complete schema-v2 canonical batteries are required.",
    )
    parser.add_argument(
        "--resnet-source",
        choices=("none", "canonical"),
        default="none",
        help="Omit ResNet or select an explicit canonical input; legacy sweeps are archive-only.",
    )
    parser.add_argument(
        "--canonical-resnet-path",
        type=Path,
        help="Explicit schema-v3/v4 ResNet artifact; required for new canonical runs.",
    )
    parser.add_argument(
        "--cnn-extra-path",
        type=Path,
        action="append",
        default=[],
        help="Additional measured schema-v2 CNN sensitivity artifact.",
    )
    parser.add_argument(
        "--projection-adequacy-path",
        type=Path,
        action="append",
        default=[],
        help="Optional measured schema-v1 CNN PCA-only adequacy artifact.",
    )
    parser.add_argument(
        "--provenance-corrections",
        type=Path,
        default=Path("artifacts/lw_post/provenance_corrections.json"),
        help="Audited correction sidecar for mistyped source revisions.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args.artifact_root.resolve(),
        args.output.resolve(),
        args.source_commit,
        args.cnn_source,
        args.resnet_source,
        args.cnn_seeds,
        (
            args.canonical_resnet_path.resolve()
            if args.canonical_resnet_path is not None
            else None
        ),
        [path.resolve() for path in args.projection_adequacy_path],
        [path.resolve() for path in args.cnn_extra_path],
        args.provenance_corrections.resolve(),
    )
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
