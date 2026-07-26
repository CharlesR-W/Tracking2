from __future__ import annotations

import argparse
import json
import os
import urllib.request
import urllib.error


def key() -> str:
    for line in open(".env.local"):
        if line.startswith("RUNPOD_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("RUNPOD_API_KEY missing")


def request(method: str, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        "https://rest.runpod.io/v1" + path,
        method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["create", "get", "terminate"])
parser.add_argument("--pod-id")
parser.add_argument(
    "--min-ram-per-gpu",
    type=int,
    default=96,
    help="Minimum host RAM in GB per GPU for a new pod.",
)
parser.add_argument(
    "--image",
    default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    help="RunPod container image for a new pod.",
)
args = parser.parse_args()

if args.action == "create":
    result = request("POST", "/pods", {
        "name": "tracking2-confirmatory",
        "imageName": args.image,
        "cloudType": "COMMUNITY",
        "computeType": "GPU",
        "gpuTypeIds": [
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A5000",
            "NVIDIA RTX A4500",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA L4",
            "NVIDIA RTX 4000 Ada Generation",
        ],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "minRAMPerGPU": args.min_ram_per_gpu,
        "containerDiskInGb": 40,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()},
        "interruptible": False,
    })
elif args.action == "get":
    result = request("GET", f"/pods/{args.pod_id}")
else:
    result = request("DELETE", f"/pods/{args.pod_id}")

safe = {
    "id": result.get("id", args.pod_id),
    "name": result.get("name"),
    "status": result.get("desiredStatus"),
    "costPerHr": result.get("adjustedCostPerHr", result.get("costPerHr")),
    "gpu": (result.get("gpu") or {}).get("displayName") or (result.get("machine") or {}).get("gpuDisplayName"),
    "publicIp": result.get("publicIp"),
    "sshPort": (result.get("portMappings") or {}).get("22"),
    "minRamPerGpu": args.min_ram_per_gpu if args.action == "create" else None,
}
print(json.dumps(safe, indent=2))
