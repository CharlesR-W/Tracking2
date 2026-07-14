from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine independent Tracking2 seed artifacts.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    seeds = [payload["config"]["seed"] for payload in payloads]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Each input must represent a distinct independent seed")
    reference = payloads[0]["config"]
    invariant = {key: value for key, value in reference.items() if key not in {"seed", "output"}}
    combined = {
        "config": {**invariant, "seeds": seeds},
        "device": sorted({p["device"] for p in payloads}),
        "runtime_seconds": sum(p["runtime_seconds"] for p in payloads),
        "experiment_a": [],
        "experiment_b": [],
        "source_artifacts": [str(path) for path in args.inputs],
    }
    for payload in payloads:
        check = {key: value for key, value in payload["config"].items() if key not in {"seed", "output"}}
        if check != invariant:
            raise ValueError("Seed artifacts do not share one experimental configuration")
        seed = payload["config"]["seed"]
        for experiment in ("experiment_a", "experiment_b"):
            combined[experiment].extend({**row, "seed": seed} for row in payload[experiment])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2, allow_nan=False))
    print(args.output)


if __name__ == "__main__":
    main()
