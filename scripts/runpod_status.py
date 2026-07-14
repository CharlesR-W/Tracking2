from __future__ import annotations

import json
import os
import urllib.request


env_file = {}
for line in open(".env.local") if os.path.exists(".env.local") else ():
    if "=" in line:
        key, value = line.strip().split("=", 1)
        env_file[key] = value
api_key = env_file.get("RUNPOD_API_KEY", os.environ.get("RUNPOD_API_KEY", ""))
request = urllib.request.Request(
    "https://rest.runpod.io/v1/pods?includeMachine=true",
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    pods = json.load(response)
safe = [
    {
        "id": pod.get("id"),
        "name": pod.get("name"),
        "status": pod.get("desiredStatus"),
        "costPerHr": pod.get("adjustedCostPerHr", pod.get("costPerHr")),
        "gpu": (pod.get("gpu") or {}).get("displayName") or (pod.get("machine") or {}).get("gpuDisplayName"),
    }
    for pod in pods
]
print(json.dumps(safe, indent=2))
