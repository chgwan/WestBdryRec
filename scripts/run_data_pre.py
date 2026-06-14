# -*- coding: utf-8 -*-
"""Project data-preparation entry point (run from the project root).

Drives the per-step data-prep pipeline. Builds the per-shot status table
(``shot_status.csv``) and per-signal frequency scan (``shot_freq_stats.csv``) via
``src.data.shot_status``; derives the per-signal frequency summary
(``signal_freq_summary.csv``) plus a DCS .mat time-axis check via
``src.data.scan_frequencies``; then merges boundary and DCS h5 files for the
selected shots via ``src.data.merge_dcs_bdry``.

Usage:
    python scripts/run_data_pre.py
    python scripts/run_data_pre.py --workers 16
"""
import argparse
import os
import pathlib
import sys

# allow importing src when run as `python scripts/run_data_pre.py`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.flat_top import run as run_flat_top  # noqa: E402
from src.data.merge_dcs_bdry import run as run_merge  # noqa: E402
from src.data.scan_frequencies import run as run_scan_frequencies  # noqa: E402
from src.data.shot_status import run as run_shot_status  # noqa: E402
from src.data.build_npz import run as run_build_npz  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int,
                        default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    print("== shot_status: boundary + DCS status & per-signal freq scan ==")
    run_shot_status(workers=args.workers)

    print("\n== scan_frequencies: per-signal freq summary + DCS .mat time axis ==")
    run_scan_frequencies()

    print("\n== flat_top: per-shot Ip flat-top detection (selection criterion) ==")
    run_flat_top(workers=args.workers)

    print("\n== merge: resample DataOrg onto Ip grid -> new Merged h5 ==")
    run_merge(workers=args.workers)

    print("\n== build_npz: Merged h5 -> ProjDB/Npz/<shot>.npz (polar r(theta)) ==")
    run_build_npz(workers=args.workers)


if __name__ == "__main__":
    main()