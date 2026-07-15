from __future__ import annotations

import json
import math
import sys
from pathlib import Path


root = Path(sys.argv[1])
paths = sorted(root.glob("**/*.json"))
assert paths, f"no JSON artifacts below {root}"


def check_finite(value, location: str) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite value at {location}"
    elif isinstance(value, dict):
        for key, child in value.items():
            check_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_finite(child, f"{location}[{index}]")


for path in paths:
    payload = json.loads(path.read_text())
    check_finite(payload, str(path))
    if payload.get("experiment") == "critical_module_tracking":
        assert len(payload["module_names"]) == 19
        assert payload["module_names"][-1] == "final_linear"
        assert payload.get("interventions")
print({"artifacts_checked": len(paths), "paths": [str(path) for path in paths]})
