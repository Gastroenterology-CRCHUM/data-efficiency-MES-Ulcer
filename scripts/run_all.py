"""
scripts/run_all.py
==================
Run ulcer then MES data-efficiency experiments sequentially.

All CLI arguments are forwarded identically to both sub-scripts.

Usage
-----
    python -m scripts.run_all
    python -m scripts.run_all --subset-ratios 0.1 0.5 1.0 --seeds 42 84
    python -m scripts.run_all --epochs 5 --max-runs 2 --dry-run
    python -m scripts.run_all --plan configs/experiments/data_efficiency.yaml
"""

import subprocess
import sys


def main() -> None:
    forwarded = sys.argv[1:]

    for task in ("ulcer", "mes"):
        module = f"scripts.{task}.run_data_efficiency"
        cmd = [sys.executable, "-m", module, *forwarded]

        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  TASK : {task.upper()}")
        print(f"  CMD  : {' '.join(cmd)}")
        print(sep)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[!] {task} exited with code {result.returncode} — stopping.")
            sys.exit(result.returncode)

    print("\nBoth tasks completed successfully.")


if __name__ == "__main__":
    main()
