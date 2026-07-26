from __future__ import annotations

import argparse
from pathlib import Path


def host_memory_limit_bytes() -> int:
    """Return the smallest visible physical or cgroup memory limit."""

    mem_total_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    limits = [mem_total_kib * 1024]
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if not path.exists():
            continue
        value = path.read_text().strip()
        if value != "max":
            parsed = int(value)
            # Some cgroup-v1 hosts use a near-2**63 sentinel for no limit.
            if parsed < 2**60:
                limits.append(parsed)
    return min(limits)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse the shallow-cut CNN battery on a low-memory host."
    )
    parser.add_argument("minimum_gib", type=float)
    args = parser.parse_args()
    observed_gib = host_memory_limit_bytes() / 1024**3
    print(
        f"[host memory] visible={observed_gib:.1f} GiB; "
        f"required={args.minimum_gib:.1f} GiB"
    )
    if observed_gib < args.minimum_gib:
        print(
            "The cut-1 activation and randomized-PCA banks can exceed ordinary "
            "GPU-host memory. Choose a larger machine, lower the declared "
            "protocol, or set SKIP_HOST_MEMORY_CHECK=1 only after profiling."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
