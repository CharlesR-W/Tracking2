from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


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


def post_cnn_entries(artifact_root: Path, output: Path) -> list[dict[str, object]]:
    result_root = artifact_root / "lw_post" / "cnn_statistics_seed0"
    paths = [
        (
            f"cnn-e{epoch}",
            f"Four-block CNN · uninterrupted trajectory · epoch {epoch}",
            epoch,
            result_root / f"base_epoch{epoch}" / "post_statistics.json",
        )
        for epoch in CNN_EPOCHS
    ]
    paths.extend(
        [
            (
                "cnn-e30-pca-ablation-c3",
                "Four-block CNN · epoch 30 · cut 3 · PCA-rank ablation",
                30,
                result_root
                / "pca_ablation_epoch30_cut3"
                / "post_statistics.json",
            ),
            (
                "cnn-e30-noise-ablation-c3",
                "Four-block CNN · epoch 30 · cut 3 · noise-radius ablation",
                30,
                result_root
                / "noise_ablation_epoch30_cut3"
                / "post_statistics.json",
            ),
        ]
    )
    inputs = []
    for input_id, label, epoch, artifact_path in paths:
        payload = measured_payload(artifact_path, epoch=epoch, seed=0)
        if payload.get("experiment") != "lw_post_cnn_suffix_statistics":
            raise RuntimeError(f"{artifact_path} is not a post-facing CNN artifact")
        inputs.append(
            entry(
                manifest_dir=output.parent,
                artifact_path=artifact_path,
                input_id=input_id,
                kind="cnn",
                artifact_format="post_statistics",
                label=label,
                epoch=epoch,
                seed=0,
            )
        )
    return inputs


def canonical_resnet_entries(
    artifact_root: Path, output: Path
) -> list[dict[str, object]]:
    artifact_path = artifact_root / CANONICAL_RESNET_PATH
    payload = measured_payload(artifact_path, epoch=100, seed=0)
    if (
        payload.get("schema_version") != 3
        or payload.get("experiment") != "resnet18_suffix_statistics_sweep"
    ):
        raise RuntimeError(f"{artifact_path} is not a schema-v3 ResNet artifact")
    config = payload["config"]
    if (
        config.get("pca_ranks") != [128, 256, 512]
        or config.get("cuts") != [3, 7]
        or config.get("mean_noise_radii") != [1.0]
        or config.get("include_projected_true") is not True
        or config.get("true_eval_only") is not True
    ):
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
    return [
        entry(
            manifest_dir=output.parent,
            artifact_path=artifact_path,
            input_id="resnet-e100-pca-ablation-cuts4-8",
            kind="resnet",
            artifact_format="resnet_suffix_statistics",
            label="ResNet-18 · epoch 100 · nested PCA controls · cuts 4 and 8",
            epoch=100,
            seed=0,
        )
    ]


def resnet_entries(
    artifact_root: Path, output: Path, source: str
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
        inputs.extend(canonical_resnet_entries(artifact_root, output))
    if source not in {"legacy", "canonical", "both"}:
        raise ValueError("resnet_source must be 'legacy', 'canonical', or 'both'")
    return inputs


def build_manifest(
    artifact_root: Path,
    output: Path,
    source_commit: str,
    cnn_source: str,
    resnet_source: str = "legacy",
) -> dict:
    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a 7–64 character hexadecimal id")
    if cnn_source == "legacy":
        inputs = legacy_cnn_entries(artifact_root, output)
    elif cnn_source == "post":
        inputs = post_cnn_entries(artifact_root, output)
    else:
        raise ValueError("cnn_source must be 'legacy' or 'post'")
    inputs.extend(resnet_entries(artifact_root, output, resnet_source))
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
        default="legacy",
        help="Select the legacy pilot grid or the verified uninterrupted-run battery.",
    )
    parser.add_argument(
        "--resnet-source",
        choices=("legacy", "canonical", "both"),
        default="legacy",
        help="Select legacy sweeps, the canonical schema-v3 control, or both.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args.artifact_root.resolve(),
        args.output.resolve(),
        args.source_commit,
        args.cnn_source,
        args.resnet_source,
    )
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
