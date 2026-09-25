"""Read-only memory gate for the NUMA-bound, eight-GPU overfit allocation."""

import argparse
import json
import math
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path


def read_numa_memory(root):
    nodes = {}
    for path in sorted(Path(root).glob("node[0-9]*/meminfo")):
        fields = {}
        for line in path.read_text().splitlines():
            match = re.fullmatch(r"Node \d+ (\S+):\s+(\d+) kB", line.strip())
            if match:
                fields[match[1]] = int(match[2]) * 1024
        if "MemTotal" not in fields or "MemFree" not in fields:
            raise ValueError(f"Missing memory totals in {path}")
        nodes[path.parent.name] = fields
    if not nodes:
        raise ValueError(f"No NUMA memory information found under {root}")
    return nodes


def assess_memory(nodes, minimum_numa_gib=512, minimum_total_gib=1024):
    if not nodes:
        raise ValueError("No NUMA nodes to assess")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (minimum_numa_gib, minimum_total_gib)
    ):
        raise ValueError("Memory thresholds must be finite and positive")
    regions = {}
    total_available = 0
    for name, fields in nodes.items():
        clean_file_cache = max(
            0,
            fields.get("Active(file)", 0)
            + fields.get("Inactive(file)", 0)
            - fields.get("Dirty", 0)
            - fields.get("Writeback", 0),
        )
        available = min(fields["MemTotal"], fields["MemFree"] + clean_file_cache)
        total_available += available
        regions[name] = {
            "total_gib": fields["MemTotal"] / 1024**3,
            "free_gib": fields["MemFree"] / 1024**3,
            "shmem_gib": fields.get("Shmem", 0) / 1024**3,
            "clean_file_cache_gib": clean_file_cache / 1024**3,
            "estimated_available_gib": available / 1024**3,
            "passed": available >= minimum_numa_gib * 1024**3,
        }
    return {
        "minimum_numa_gib": minimum_numa_gib,
        "minimum_total_gib": minimum_total_gib,
        "estimate": "MemFree + clean file LRU; excludes shared memory and slab",
        "numa_regions": regions,
        "total_estimated_available_gib": total_available / 1024**3,
        "passed": all(region["passed"] for region in regions.values())
        and total_available >= minimum_total_gib * 1024**3,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numa-root", type=Path, default=Path("/sys/devices/system/node")
    )
    parser.add_argument("--minimum-numa-gib", type=float, default=512)
    parser.add_argument("--minimum-total-gib", type=float, default=1024)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    report = {
        "hostname": socket.gethostname().split(".")[0],
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        report.update(
            assess_memory(
                read_numa_memory(args.numa_root),
                args.minimum_numa_gib,
                args.minimum_total_gib,
            )
        )
    except (OSError, ValueError) as error:
        report.update(passed=False, error=str(error))
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{report['hostname']}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print("NUMA_MEMORY_PREFLIGHT " + json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
