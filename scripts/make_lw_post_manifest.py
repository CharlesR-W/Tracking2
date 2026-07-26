from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence


TITLE = "Free-Body Diagrams for Neural Networks — Interactive Data Appendix (WIP)"
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
        for epoch, relative_path in LEGACY_RESNET_PATHS.items():
            inputs.append(
                entry(
                    manifest_dir=output.parent,
                    artifact_path=artifact_root / relative_path,
                    input_id=f"resnet-e{epoch}",
                    kind="resnet",
                    artifact_format="resnet_suffix_statistics",
                    label=f"ResNet-18 · legacy epoch {epoch}",
                    epoch=epoch,
                    seed=0,
                )
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
) -> dict:
    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a 7–64 character hexadecimal id")
    if cnn_source == "legacy":
        inputs = legacy_cnn_entries(artifact_root, output)
    elif cnn_source == "post":
        inputs = post_cnn_entries(artifact_root, output, cnn_seeds)
    else:
        raise ValueError("cnn_source must be 'legacy' or 'post'")
    inputs.extend(
        resnet_entries(
            artifact_root,
            output,
            resnet_source,
            canonical_resnet_path,
        )
    )
    return {
        "schema_version": 1,
        "title": TITLE,
        "status": "MEASURED",
        "source_commit": source_commit,
        "inputs": inputs,
    }


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
        choices=("legacy", "post"),
        default="post",
        help="Select the legacy pilot grid or the verified uninterrupted-run battery.",
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
        choices=("none", "legacy", "canonical", "both"),
        default="none",
        help="Omit ResNet, select legacy sweeps, an explicit canonical input, or both.",
    )
    parser.add_argument(
        "--canonical-resnet-path",
        type=Path,
        help="Explicit schema-v3/v4 ResNet artifact; required for new canonical runs.",
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
    )
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
