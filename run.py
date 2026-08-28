#!/usr/bin/env python
"""Run the Cflows workflow."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from common import sha256  # noqa: E402


def check_inputs() -> None:
    with (DATA / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures = []
    for row in rows:
        path = DATA / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size mismatch: {row['path']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"checksum mismatch: {row['path']}")
    if failures:
        raise SystemExit("input bundle check failed:\n  " + "\n  ".join(failures))
    print(f"input bundle: {len(rows)} files verified", flush=True)


def run(script: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script)]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=("all", "in-vitro", "in-vivo"), nargs="?", default="all")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="CPU workers for the Granger tests (default: all cores)")
    args = parser.parse_args()

    cache = RESULTS / "cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "numba").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ["CFLOWS_N_JOBS"] = str(args.n_jobs)
    check_inputs()

    if args.branch in ("all", "in-vitro"):
        run("in_vitro.py")
    if args.branch in ("all", "in-vivo"):
        run("trajectories.py")
        run("growth_filter.py")
        run("in_vivo.py")


if __name__ == "__main__":
    main()
