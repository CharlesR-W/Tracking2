#!/usr/bin/env python3
"""Build the ResNet-first waterfall viewer and presentation exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking2.waterfall_visuals import build_waterfall_outputs, output_hashes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "waterfall_visuals" / "manifest.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "LW post" / "figures",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT_ROOT / "LW post" / "waterfalls.html",
    )
    args = parser.parse_args()
    outputs = build_waterfall_outputs(
        PROJECT_ROOT,
        args.manifest,
        args.figure_dir,
        args.html,
    )
    print(f"Wrote {display_path(outputs.html)}")
    hashes = output_hashes(outputs.figures)
    for path in outputs.figures:
        print(f"  {display_path(path)}  {hashes[path.name]}")
    print("Verified GIF metadata:")
    for report in outputs.gif_reports:
        path = Path(str(report["path"]))
        print(
            f"  {display_path(path)}: "
            f"{report['frame_count']} frames, {report['durations_ms']}, "
            f"{report['size']}"
        )


if __name__ == "__main__":
    main()
