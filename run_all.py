#!/usr/bin/env python
"""
run_all.py — Full Ablation Run
================================
Runs all 7 configs x 7 stations = 49 experiments sequentially.
Logs each run individually and prints a final summary.

Usage (from the project root, srip-transformer/):
    python run_all.py                          # run all 49
    python run_all.py --start-from 15          # resume from run #15
    python run_all.py --config seq_hier        # only one config, all 7 stations
    python run_all.py --dry-run                # print commands without running

Configs (ablation order):
    1  baseline       — Vanilla LSTM
    2  seq            — Sequential LSTM
    3  seq_hier       — Sequential + Hierarchical
    4  seq_fitter     — Sequential + Hierarchical + Fitter
    5  full           — Sequential + Hierarchical + Fitter + MHA
    6  baseline_mha   — Baseline + MHA
    7  bidirectional  — Bidirectional LSTM
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
CONFIGS_DIR = SRC_DIR / "configs"
STATIONS_DIR = ROOT / "hierarchial"
OUTPUT_DIR = ROOT / "runs"
LOG_DIR = ROOT / "run_logs"
TRAIN_PY = SRC_DIR / "train.py"

# Ablation order — maps to 7 yaml files in src/configs/
CONFIGS = [
    "baseline",  # 1  Vanilla LSTM
    "seq",  # 2  Sequential
    "seq_hier",  # 3  Sequential + Hierarchical
    "seq_fitter",  # 4  Sequential + Hierarchical + Fitter
    "full",  # 5  Sequential + Hierarchical + Fitter + MHA
    "baseline_mha",  # 6  Baseline + MHA
    "bidirectional",  # 7  Bidirectional
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def station_name_from_path(p: Path) -> str:
    """'new_Barmanghat_with_hierarchical_quantiles.csv' -> 'Barmanghat'"""
    name = p.stem  # drop .csv
    name = re.sub(r"^new_", "", name)  # drop leading 'new_'
    name = re.sub(r"_with_hierarchical_quantiles$", "", name)
    name = re.sub(r"_with_hierarchical$", "", name)
    return name


def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if td.days or h:
        return f"{td.days * 24 + h}h {m}m {s}s"
    return f"{m}m {s}s"


def print_sep(char="-", width=65, color=""):
    RESET = "\033[0m"
    print(f"{color}{char * width}{RESET}")


# ANSI colours (work on Windows 10+ with VT mode, and all Unix)
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_GRAY = "\033[90m"
C_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run all ablation experiments")
    p.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="1-indexed run number to start from (to resume)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="NAME",
        help="Run only this config name across all stations",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build config list
    configs = CONFIGS
    if args.config:
        if args.config not in CONFIGS:
            print(
                f"{C_RED}Unknown config '{args.config}'. Choose from: {CONFIGS}{C_RESET}"
            )
            sys.exit(1)
        configs = [args.config]

    # Collect station CSVs
    station_csvs = sorted(STATIONS_DIR.glob("*.csv"))
    if not station_csvs:
        print(f"{C_RED}No CSVs found in {STATIONS_DIR}{C_RESET}")
        sys.exit(1)

    # Validate config files exist
    for cfg in configs:
        cfg_path = CONFIGS_DIR / f"{cfg}.yaml"
        if not cfg_path.exists():
            print(f"{C_RED}Missing config file: {cfg_path}{C_RESET}")
            sys.exit(1)

    # Validate train.py exists
    if not TRAIN_PY.exists():
        print(f"{C_RED}train.py not found at {TRAIN_PY}{C_RESET}")
        sys.exit(1)

    # Ensure output dirs exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(configs) * len(station_csvs)
    current_run = 0
    failed = []
    start_time = time.time()

    # Header
    print()
    print_sep("=", color=C_CYAN)
    print(f"{C_CYAN}  SRIP Full Ablation Suite{C_RESET}")
    print(
        f"{C_CYAN}  {len(configs)} configs x {len(station_csvs)} stations = {total_runs} runs{C_RESET}"
    )
    print(f"{C_CYAN}  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C_RESET}")
    if args.dry_run:
        print(
            f"{C_YELLOW}  *** DRY-RUN: commands will be printed but not executed ***{C_RESET}"
        )
    print_sep("=", color=C_CYAN)
    print()

    for cfg in configs:
        cfg_path = CONFIGS_DIR / f"{cfg}.yaml"

        for station_csv in station_csvs:
            current_run += 1
            station = station_name_from_path(station_csv)

            # Skip if before resume point
            if current_run < args.start_from:
                print(
                    f"{C_GRAY}  [SKIP {current_run}/{total_runs}] {cfg} @ {station} "
                    f"(before --start-from={args.start_from}){C_RESET}"
                )
                continue

            # Banner
            print_sep("-", color=C_CYAN)
            print(f"  RUN {current_run} / {total_runs}")
            print(f"  Config  : {cfg}")
            print(f"  Station : {station}")
            print(f"{C_GRAY}  CSV     : {station_csv.name}{C_RESET}")
            print_sep("-", color=C_CYAN)

            cmd = [
                sys.executable,
                str(TRAIN_PY),
                "--config",
                str(cfg_path),
                "--station",
                str(station_csv),
                "--station_name",
                station,
                "--output_dir",
                str(OUTPUT_DIR),
            ]

            if args.dry_run:
                print(f"{C_YELLOW}  [DRY-RUN] {' '.join(cmd)}{C_RESET}\n")
                continue

            # Log file
            log_path = LOG_DIR / f"{current_run:02d}_{cfg}_{station}.log"
            print(f"  Log     : {log_path}")
            print(f"  Started : {datetime.now().strftime('%H:%M:%S')}\n")

            run_start = time.time()
            try:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    for line in proc.stdout:
                        print(line, end="")  # live console output
                        log_file.write(line)  # capture to log
                    proc.wait()

                elapsed = time.time() - run_start
                if proc.returncode != 0:
                    raise RuntimeError(f"Exit code {proc.returncode}")

                print(f"\n{C_GREEN}  DONE  ({fmt_duration(elapsed)}){C_RESET}\n")

            except Exception as exc:
                elapsed = time.time() - run_start
                print(
                    f"\n{C_RED}  FAILED after {fmt_duration(elapsed)} — {exc}{C_RESET}"
                )
                print(f"{C_RED}  Log: {log_path}{C_RESET}\n")
                failed.append(
                    {
                        "run": current_run,
                        "config": cfg,
                        "station": station,
                        "error": str(exc),
                        "log": str(log_path),
                    }
                )

    # Final summary
    total_elapsed = time.time() - start_time
    n_success = total_runs - len(failed)

    print()
    print_sep("=", color=C_CYAN)
    print(f"{C_CYAN}  ABLATION SUITE COMPLETE{C_RESET}")
    print_sep("=", color=C_CYAN)
    print(f"  Total runs  : {total_runs}")
    print(f"{C_GREEN}  Successful  : {n_success}{C_RESET}")
    fc = C_RED if failed else C_GREEN
    print(f"{fc}  Failed      : {len(failed)}{C_RESET}")
    print(f"  Total time  : {fmt_duration(total_elapsed)}")
    print(f"  Outputs in  : {OUTPUT_DIR}")
    print(f"  Logs in     : {LOG_DIR}")

    if failed:
        print()
        print(f"{C_RED}  Failed runs:{C_RESET}")
        for f in failed:
            print(
                f"{C_RED}    Run {f['run']:02d}: {f['config']} @ {f['station']}{C_RESET}"
            )
            print(f"{C_RED}      Error : {f['error']}{C_RESET}")
            print(f"{C_RED}      Log   : {f['log']}{C_RESET}")
        print()
        print(
            f"{C_YELLOW}  To resume, run:  python run_all.py --start-from <run_number>{C_RESET}"
        )

    print_sep("=", color=C_CYAN)
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
