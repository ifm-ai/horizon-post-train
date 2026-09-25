# Modified for release: configurable deployment paths and public source references.
"""Read-only storage gate before container workloads start."""

import argparse
import json
import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


def assess_storage(path, minimum_free_gib=512, minimum_free_inodes=3_000_000):
    if any(
        not math.isfinite(value) or value <= 0
        for value in (minimum_free_gib, minimum_free_inodes)
    ):
        raise ValueError("Storage thresholds must be finite and positive")
    stats = os.statvfs(path)
    free_bytes = stats.f_bavail * stats.f_frsize
    return {
        "path": str(path),
        "total_gib": stats.f_blocks * stats.f_frsize / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "free_inodes": stats.f_favail,
        "minimum_free_gib": minimum_free_gib,
        "minimum_free_inodes": minimum_free_inodes,
        "passed": free_bytes >= minimum_free_gib * 1024**3
        and stats.f_favail >= minimum_free_inodes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, required=True, help="Filesystem used for container extraction"
    )
    parser.add_argument("--minimum-free-gib", type=float, default=512)
    parser.add_argument("--minimum-free-inodes", type=int, default=3_000_000)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    report = {
        "hostname": socket.gethostname().split(".")[0],
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        report.update(
            assess_storage(args.path, args.minimum_free_gib, args.minimum_free_inodes)
        )
    except (OSError, ValueError) as error:
        report.update(passed=False, error=str(error))
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{report['hostname']}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print("STORAGE_PREFLIGHT " + json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
