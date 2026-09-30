#!/usr/bin/env python3
"""Run one command and stop its process group when effective RAM stays low."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def proc_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def own_cgroup_memory_path() -> Path:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup") / line.partition("::")[2].lstrip("/") / "memory.current"
    return Path("/sys/fs/cgroup/memory.current")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--effective-total-mb", type=int, default=4096)
    parser.add_argument("--minimum-available-mb", type=int, default=768)
    parser.add_argument("--low-seconds", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("command is required")
    if args.command[0] == "--":
        args.command = args.command[1:]

    args.status.parent.mkdir(parents=True, exist_ok=True)
    unit_current = own_cgroup_memory_path()
    child = subprocess.Popen(args.command, start_new_session=True)
    peak_unit = 0
    minimum_available = 1 << 62
    low_since: float | None = None
    reason = "completed"
    started = time.time()
    while child.poll() is None:
        unit_bytes = read_int(unit_current) or 0
        # MemAvailable already subtracts unreclaimable use while accounting for
        # reclaimable page cache. Subtracting cgroup memory.current again would
        # double-count cache and can falsely stop a healthy 4 GiB host.
        effective_available = min(
            proc_available(), args.effective_total_mb * 1024 * 1024
        )
        peak_unit = max(peak_unit, unit_bytes)
        minimum_available = min(minimum_available, effective_available)
        now = time.monotonic()
        if effective_available < args.minimum_available_mb * 1024 * 1024:
            low_since = low_since or now
            if now - low_since >= args.low_seconds:
                reason = "stopped_low_memory"
                print(f"[guard] effective MemAvailable below {args.minimum_available_mb} MiB for {args.low_seconds}s", flush=True)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
        else:
            low_since = None
        status = {
            "state": "running",
            "pid": child.pid,
            "unit_memory_mib": round(unit_bytes / 1048576, 1),
            "peak_unit_memory_mib": round(peak_unit / 1048576, 1),
            "effective_mem_available_mib": round(effective_available / 1048576, 1),
            "started_at_unix": started,
        }
        args.status.write_text(json.dumps(status, indent=2) + "\n")
        time.sleep(5)
    return_code = child.wait()
    final = {
        "state": reason,
        "return_code": return_code,
        "peak_unit_memory_mib": round(peak_unit / 1048576, 1),
        "minimum_effective_mem_available_mib": round(minimum_available / 1048576, 1),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.status.write_text(json.dumps(final, indent=2) + "\n")
    print(f"[guard] {json.dumps(final)}", flush=True)
    raise SystemExit(return_code if reason == "completed" else 75)


if __name__ == "__main__":
    main()
