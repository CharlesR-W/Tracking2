#!/usr/bin/env python3
"""Combine independently persisted VGG suffix-statistics slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = [json.loads(path.read_text()) for path in args.input]
    reference = artifacts[0]
    ignored = {"output", "cuts", "conditions", "checkpoint_epoch"}
    reference_config = {key: value for key, value in reference["config"].items() if key not in ignored}
    for path, artifact in zip(args.input[1:], artifacts[1:]):
        config = {key: value for key, value in artifact["config"].items() if key not in ignored}
        if config != reference_config:
            raise ValueError(f"incompatible config in {path}")
        if artifact["module_names"] != reference["module_names"]:
            raise ValueError(f"module ordering differs in {path}")

    slices = [item for artifact in artifacts for item in artifact["slices"]]
    keys = [(item["condition"], int(item["cut"])) for item in slices]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate condition/cut slices")

    combined = {key: value for key, value in reference.items() if key != "slices"}
    combined["config"] = dict(reference_config)
    combined["config"].update({
        "output": str(args.output.parent),
        "checkpoint_epoch": None,
        "cuts": sorted({cut for _condition, cut in keys}),
        "conditions": sorted({condition for condition, _cut in keys}),
    })
    combined["slices"] = sorted(slices, key=lambda item: (item["condition"], int(item["cut"])))
    combined["runtime_seconds"] = sum(float(artifact["runtime_seconds"]) for artifact in artifacts)
    combined["source_artifacts"] = [str(path) for path in args.input]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
