"""Build the clean, sidecar-free manifest for an isolated LW rerun stage."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from verify_lw_post_publication import (
    COMMIT_RE,
    DEFAULT_MANIFEST,
    DIGEST_RE,
    EXPECTED_PRIMARY_ANALYSIS,
    INPUT_SPECS,
    PRIMARY_PATHS,
    PROJECT_ROOT,
    VerificationError,
    finite_float,
    git_archive_commit,
    load_json,
    rank_result,
    record,
    sha256,
    verify_manifest_inputs,
    verify_primary_analysis,
    verify_staged_provenance,
    verify_training_manifests,
)


def clean_source_identity(stage_root: Path) -> tuple[str, Path]:
    revision = os.environ.get("TRACKING2_SOURCE_REVISION", "")
    archive_digest = os.environ.get("TRACKING2_SOURCE_ARCHIVE_SHA256", "")
    if COMMIT_RE.fullmatch(revision) is None:
        raise VerificationError(
            "TRACKING2_SOURCE_REVISION must be the full clean 40-character commit"
        )
    if DIGEST_RE.fullmatch(archive_digest) is None:
        raise VerificationError(
            "TRACKING2_SOURCE_ARCHIVE_SHA256 must be a lowercase SHA-256"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=stage_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != revision:
        raise VerificationError(
            f"staging checkout HEAD {head} != TRACKING2_SOURCE_REVISION {revision}"
        )
    archive_path = stage_root / "source" / "source.tar"
    if sha256(archive_path) != archive_digest:
        raise VerificationError("source/source.tar does not match the exported digest")
    if git_archive_commit(archive_path) != revision:
        raise VerificationError("source/source.tar embeds the wrong Git commit")
    return revision, archive_path


def headline_assertions(
    payloads: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    values: dict[tuple[int, str], list[float]] = {
        (cut, contrast): []
        for cut in (1, 4)
        for contrast in (
            "gaussian_minus_projected_true",
            "mean_r1_minus_gaussian",
        )
    }
    for relative in PRIMARY_PATHS.values():
        artifact = payloads[relative]
        for cut in (1, 4):
            cell = rank_result(artifact, cut=cut, rank=2048, context=relative)
            rows = cell.get("records")
            losses = {}
            for distribution in (
                "projected_true",
                "gaussian_empirical",
                "mean_r1",
            ):
                endpoint = record(
                    rows,
                    distribution=distribution,
                    relax_epoch=5,
                    context=f"{relative} cut {cut}",
                )
                losses[distribution] = finite_float(
                    endpoint.get("loss"),
                    f"{relative} cut {cut} {distribution} loss",
                )
            values[(cut, "gaussian_minus_projected_true")].append(
                losses["gaussian_empirical"] - losses["projected_true"]
            )
            values[(cut, "mean_r1_minus_gaussian")].append(
                losses["mean_r1"] - losses["gaussian_empirical"]
            )
    assertions = []
    for template in EXPECTED_PRIMARY_ANALYSIS["headline_assertions"]:
        key = (int(template["cut"]), str(template["contrast_id"]))
        samples = values[key]
        assertions.append(
            {
                "cut": key[0],
                "contrast_id": key[1],
                "mean": statistics.mean(samples),
                "sample_standard_deviation": statistics.stdev(samples),
            }
        )
    return assertions


def build_staged_manifest(stage_root: Path) -> tuple[Path, dict[str, object]]:
    stage_root = stage_root.resolve()
    if stage_root != PROJECT_ROOT.resolve():
        raise VerificationError(
            "run this tool from the isolated staging clone and pass its root"
        )
    revision, archive_path = clean_source_identity(stage_root)
    output = stage_root / "artifacts" / "lw_post" / "dashboard_manifest.json"
    measured_root = output.parent / "measured"
    template = load_json(DEFAULT_MANIFEST)
    template_entries = template.get("inputs")
    if not isinstance(template_entries, list):
        raise VerificationError("committed manifest template lacks inputs")
    by_id = {
        entry.get("id"): entry
        for entry in template_entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if set(by_id) != {spec.input_id for spec in INPUT_SPECS.values()}:
        raise VerificationError("committed manifest template has the wrong input IDs")
    entries = []
    for relative, spec in INPUT_SPECS.items():
        entry = copy.deepcopy(by_id[spec.input_id])
        expected_path = f"../lw_post/measured/{relative}"
        entry["path"] = expected_path
        entry["sha256"] = sha256(measured_root / relative)
        entries.append(entry)
    primary = copy.deepcopy(EXPECTED_PRIMARY_ANALYSIS)
    provisional = {
        "schema_version": 2,
        "title": template["title"],
        "status": "MEASURED",
        "source_commit": revision,
        "primary_analysis": primary,
        "inputs": entries,
    }
    payloads = verify_manifest_inputs(
        output,
        provisional,
        measured_root=measured_root,
        locked_snapshot=False,
    )
    primary["headline_assertions"] = headline_assertions(payloads)
    training, _ = verify_training_manifests(
        payloads,
        measured_root=measured_root,
        locked_snapshot=False,
        checkpoint_root=stage_root / "checkpoints",
    )
    verify_staged_provenance(provisional, payloads, training, archive_path)
    verify_primary_analysis(provisional, payloads, locked_snapshot=False)
    return output, provisional


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output, manifest = build_staged_manifest(args.stage_root)
    except (OSError, subprocess.SubprocessError, VerificationError) as error:
        print(f"Staged manifest build failed: {error}")
        return 1
    temporary = output.with_name(".dashboard_manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
