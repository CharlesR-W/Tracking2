from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def runtime_provenance() -> dict[str, object]:
    """Small, dependency-light environment record for measured artifacts."""

    packages: dict[str, str] = {}
    for name in ("numpy", "scikit-learn", "torch", "torchvision"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    import torch

    accelerator: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if torch.cuda.is_available():
        accelerator.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
        "command": sys.argv,
        "source_revision": os.environ.get(
            "TRACKING2_SOURCE_REVISION", "not recorded"
        ),
        "source_archive_sha256": os.environ.get(
            "TRACKING2_SOURCE_ARCHIVE_SHA256", "not recorded"
        ),
    }
