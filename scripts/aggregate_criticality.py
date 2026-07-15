from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Combine critical-module seed artifacts")
parser.add_argument("inputs", nargs="+")
parser.add_argument("--output", default="artifacts/criticality/results.json")
args = parser.parse_args()

artifacts = [json.loads(Path(path).read_text()) for path in args.inputs]
if not artifacts:
    raise SystemExit("No artifacts supplied")
reference = artifacts[0]
for artifact in artifacts[1:]:
    if artifact["module_names"] != reference["module_names"]:
        raise ValueError("Cannot aggregate artifacts with different module orderings")

combined = {
    "schema_version": 1,
    "experiment": "critical_module_tracking",
    "config": {**reference["config"], "seeds": [item["config"]["seed"] for item in artifacts]},
    "module_names": reference["module_names"],
    "source_artifacts": args.inputs,
    "baselines": [],
    "training": [],
    "interventions": [],
    "recoveries": [],
    "runtime_seconds": sum(item["runtime_seconds"] for item in artifacts),
}
for artifact in artifacts:
    seed = artifact["config"]["seed"]
    combined["baselines"].append({"seed": seed, **artifact["baseline"]})
    for key in ("training", "interventions", "recoveries"):
        combined[key].extend({"seed": seed, **row} for row in artifact[key])

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(combined, indent=2, allow_nan=False))
print(output)
