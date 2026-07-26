from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that LW post smoke artifacts cannot masquerade as data."
    )
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.artifacts:
        artifact = json.loads(path.read_text())
        if artifact["status"] != "MOCKUP / PIPELINE SMOKE TEST":
            raise RuntimeError(f"{path} was not marked as a smoke artifact")
    print("LW post end-to-end smoke passed.")


if __name__ == "__main__":
    main()
