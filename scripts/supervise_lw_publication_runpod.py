"""Fail-closed supervisor for a future paid LW publication rerun.

This program never creates a pod. It accepts an already-created RunPod pod ID,
checks its hourly price before starting a worker command, polls a conservative
accrued-cost bound, and terminates the pod on success, failure, signal, timeout,
price/cost-limit breach, or polling failure.

The supervisor must be started immediately after pod creation with the pod's
creation timestamp. SIGKILL and host power loss cannot be trapped, so an
independent provider-side spend limit remains required.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANAGER = PROJECT_ROOT / "scripts" / "runpod_manage.py"
ABSOLUTE_COST_CAP_USD = 9.0
ABSOLUTE_HOURLY_RATE_CAP_USD = 1.50
ABSOLUTE_TIME_CAP_HOURS = 6.0
TERMINATION_RETRIES = 3
TERMINATION_FAILURE_STATUS = 3


class SupervisorError(RuntimeError):
    """A condition that requires the worker and pod to stop."""


def finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SupervisorError(f"RunPod returned invalid {label}: {value!r}")
    return float(value)


class PodGuard:
    """Own termination of one pod and make repeated cleanup calls harmless."""

    def __init__(self, pod_id: str, manager: Path) -> None:
        self.pod_id = pod_id
        self.manager = manager
        self.terminated = False

    def _call(self, action: str) -> dict[str, object]:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.manager),
                    action,
                    "--pod-id",
                    self.pod_id,
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SupervisorError(
                f"RunPod {action} could not execute for {self.pod_id}: {error}"
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise SupervisorError(
                f"RunPod {action} failed for {self.pod_id}: {detail}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise SupervisorError(
                f"RunPod {action} returned non-JSON output"
            ) from error
        if not isinstance(payload, dict):
            raise SupervisorError(f"RunPod {action} returned a non-object")
        returned_id = payload.get("id")
        if returned_id not in (None, self.pod_id):
            raise SupervisorError(
                f"RunPod {action} returned pod {returned_id!r}, "
                f"expected {self.pod_id!r}"
            )
        return payload

    def inspect(self) -> dict[str, object]:
        return self._call("get")

    def terminate(self) -> bool:
        if self.terminated:
            return True
        errors: list[str] = []
        for attempt in range(1, TERMINATION_RETRIES + 1):
            try:
                self._call("terminate")
            except Exception as error:  # cleanup must report every failure
                errors.append(f"attempt {attempt}: {error}")
                if attempt < TERMINATION_RETRIES:
                    time.sleep(2.0)
                continue
            self.terminated = True
            print(f"RunPod pod {self.pod_id} termination requested.")
            return True
        print(
            "CRITICAL: automatic RunPod termination failed; terminate pod "
            f"{self.pod_id} manually. " + " | ".join(errors),
            file=sys.stderr,
        )
        return False


def stop_worker(worker: subprocess.Popen[bytes] | None) -> None:
    if worker is None or worker.poll() is not None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
        worker.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)


def _option_value(argv: Sequence[str], option: str) -> str | None:
    """Return the last pre-worker value for one long CLI option."""
    boundary = argv.index("--") if "--" in argv else len(argv)
    value: str | None = None
    for index, token in enumerate(argv[:boundary]):
        if token.startswith(f"{option}="):
            candidate = token.split("=", 1)[1]
            if candidate:
                value = candidate
        elif token == option and index + 1 < boundary:
            candidate = argv[index + 1]
            if candidate and not candidate.startswith("-"):
                value = candidate
    return value


def _recognizable_pod_id(argv: Sequence[str]) -> str | None:
    """Find an explicit pod ID without treating help as a destructive action."""
    boundary = argv.index("--") if "--" in argv else len(argv)
    if any(token in {"-h", "--help"} for token in argv[:boundary]):
        return None
    value = _option_value(argv, "--pod-id")
    if (
        value is None
        or len(value) > 256
        or value != value.strip()
        or any(ord(character) < 33 for character in value)
    ):
        return None
    return value


def arm_preparse_guard(argv: Sequence[str]) -> PodGuard | None:
    """Arm cleanup before full parsing once a pod target is recognizable.

    A missing or invalid custom manager cannot disable cleanup: validation will
    still fail, while termination falls back to the canonical manager.
    """
    pod_id = _recognizable_pod_id(argv)
    if pod_id is None:
        return None
    manager_value = _option_value(argv, "--manager")
    requested_manager = Path(manager_value) if manager_value else DEFAULT_MANAGER
    cleanup_manager = (
        requested_manager.resolve()
        if requested_manager.is_file()
        else DEFAULT_MANAGER.resolve()
    )
    return PodGuard(pod_id, cleanup_manager)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervise and always terminate an existing RunPod pod around one "
            "future publication-rerun worker command. This does not create pods."
        )
    )
    parser.add_argument("--pod-id", required=True)
    parser.add_argument(
        "--pod-started-at",
        required=True,
        type=float,
        help="Provider pod-creation time as Unix seconds (UTC).",
    )
    parser.add_argument(
        "--starting-cost-usd",
        type=float,
        default=0.0,
        help="Already accrued spend not represented by --pod-started-at.",
    )
    parser.add_argument("--cost-cap-usd", type=float, default=9.0)
    parser.add_argument("--hourly-rate-cap-usd", type=float, default=1.50)
    parser.add_argument("--time-cap-hours", type=float, default=6.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--manager", type=Path, default=DEFAULT_MANAGER)
    parser.add_argument(
        "worker",
        nargs=argparse.REMAINDER,
        help="Worker command, introduced by --.",
    )
    args = parser.parse_args(argv)
    if args.worker and args.worker[0] == "--":
        args.worker = args.worker[1:]
    if not args.worker:
        parser.error("a worker command is required after --")
    caps = (
        ("cost cap", args.cost_cap_usd, ABSOLUTE_COST_CAP_USD),
        (
            "hourly-rate cap",
            args.hourly_rate_cap_usd,
            ABSOLUTE_HOURLY_RATE_CAP_USD,
        ),
        ("time cap", args.time_cap_hours, ABSOLUTE_TIME_CAP_HOURS),
    )
    for label, value, maximum in caps:
        if not math.isfinite(value) or value <= 0 or value > maximum:
            parser.error(f"{label} must be in (0, {maximum}]")
    if (
        not math.isfinite(args.starting_cost_usd)
        or args.starting_cost_usd < 0
    ):
        parser.error("starting cost must be finite and non-negative")
    if (
        not math.isfinite(args.poll_seconds)
        or args.poll_seconds < 10
        or args.poll_seconds > 60
    ):
        parser.error("poll interval must be between 10 and 60 seconds")
    if not args.manager.is_file():
        parser.error(f"RunPod manager does not exist: {args.manager}")
    now = time.time()
    if args.pod_started_at > now + 60:
        parser.error("pod creation time is in the future")
    if args.pod_started_at <= 0:
        parser.error("pod creation time must be a positive Unix timestamp")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    guard = arm_preparse_guard(raw_argv)
    if guard is not None:
        atexit.register(guard.terminate)
    try:
        args = parse_args(raw_argv)
    except SystemExit as error:
        if guard is None:
            raise
        terminated = guard.terminate()
        atexit.unregister(guard.terminate)
        if not terminated:
            return TERMINATION_FAILURE_STATUS
        return int(error.code) if isinstance(error.code, int) else 2
    if guard is None:
        guard = PodGuard(args.pod_id, args.manager.resolve())
        atexit.register(guard.terminate)
    worker: subprocess.Popen[bytes] | None = None
    interrupted: list[int] = []

    def handle_signal(signum: int, _frame: object) -> None:
        interrupted.append(signum)
        stop_worker(worker)
        raise SupervisorError(f"received signal {signum}")

    prior_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    max_rate = 0.0
    exit_code = 1
    try:
        while True:
            pod = guard.inspect()
            rate = finite_number(pod.get("costPerHr"), "costPerHr")
            if rate <= 0 or rate > args.hourly_rate_cap_usd:
                raise SupervisorError(
                    f"hourly price ${rate:.4f} exceeds allowed "
                    f"${args.hourly_rate_cap_usd:.2f}/hour"
                )
            max_rate = max(max_rate, rate)
            now = time.time()
            elapsed_hours = max(0.0, now - args.pod_started_at) / 3600.0
            cost_bound = args.starting_cost_usd + max_rate * elapsed_hours
            next_cost_bound = cost_bound + (
                max_rate * args.poll_seconds / 3600.0
            )
            print(
                f"pod={args.pod_id} status={pod.get('status')} "
                f"rate=${rate:.4f}/h elapsed={elapsed_hours:.3f}h "
                f"cost_upper_bound=${cost_bound:.3f}",
                flush=True,
            )
            if elapsed_hours >= args.time_cap_hours:
                raise SupervisorError("six-hour-or-lower time cap reached")
            if cost_bound >= args.cost_cap_usd:
                raise SupervisorError("cost cap reached")
            if next_cost_bound >= args.cost_cap_usd:
                raise SupervisorError("next polling interval could exceed cost cap")
            if worker is None:
                try:
                    worker = subprocess.Popen(args.worker, start_new_session=True)
                except OSError as error:
                    raise SupervisorError(
                        f"worker could not start: {error}"
                    ) from error
            return_code = worker.poll()
            if return_code is not None:
                exit_code = return_code
                if return_code != 0:
                    raise SupervisorError(
                        f"worker exited with status {return_code}"
                    )
                break
            try:
                worker.wait(timeout=args.poll_seconds)
            except subprocess.TimeoutExpired:
                pass
    except SupervisorError as error:
        print(f"RunPod supervisor stopping: {error}", file=sys.stderr)
        stop_worker(worker)
        exit_code = 2
    finally:
        terminated = guard.terminate()
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
        atexit.unregister(guard.terminate)
        if not terminated:
            exit_code = TERMINATION_FAILURE_STATUS
    if exit_code == TERMINATION_FAILURE_STATUS:
        return exit_code
    if interrupted:
        return 128 + interrupted[-1]
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
