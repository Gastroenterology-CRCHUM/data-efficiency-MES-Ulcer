"""
scripts/ulcer/generate_confusion_matrices.py
--------------------------------------------
Generate binary confusion matrices for all ulcer data-efficiency runs.

Reads saved predictions (test_labels.npy + test_probs.npy) and the tuned
threshold from results_per_seed.csv, then saves one PNG per run following
the same convention as the MES pipeline: cm_{run_name}.png.

Usage
-----
    python -m scripts.ulcer.generate_confusion_matrices
    python -m scripts.ulcer.generate_confusion_matrices --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.evaluation.plots import plot_confusion_matrix

RESULTS_DIR = Path("results/ulcer/data_efficiency")
PREDS_DIR   = RESULTS_DIR / "predictions"
FIGS_DIR    = RESULTS_DIR / "figures"
CSV_PATH    = RESULTS_DIR / "results_per_seed.csv"

CLASS_NAMES = ("No ulcer", "Ulcer")


def _run_name(row: pd.Series) -> str:
    return f"{row['model']}_{row['freeze']}_{row['head_type']}_{int(row['pct_data'])}pct_seed{int(row['seed'])}"


def generate(fixed_threshold: float | None = None) -> None:
    df = pd.read_csv(CSV_PATH)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    generated, skipped = 0, 0

    for _, row in df.iterrows():
        run_name  = _run_name(row)
        pred_dir  = PREDS_DIR / run_name
        labels_fp = pred_dir / "test_labels.npy"
        probs_fp  = pred_dir / "test_probs.npy"

        if not labels_fp.exists() or not probs_fp.exists():
            print(f"  [SKIP] {run_name} — predictions not found")
            skipped += 1
            continue

        labels = np.load(labels_fp)
        probs  = np.load(probs_fp)
        if probs.ndim == 2:
            probs = probs[:, 1]

        threshold = fixed_threshold if fixed_threshold is not None else float(row["tuned_threshold"])
        preds     = (probs >= threshold).astype(int)
        cm        = confusion_matrix(labels, preds)

        fig = plot_confusion_matrix(
            cm,
            threshold=threshold,
            class_names=CLASS_NAMES,
            title=f"Confusion matrix — Ulcer ({run_name})",
        )

        out_path = FIGS_DIR / f"cm_{run_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close(fig)

        print(f"  [OK]   {out_path.name}  (thr={threshold:.2f})")
        generated += 1

    print(f"\nDone — {generated} matrices generated, {skipped} skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ulcer confusion matrices")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold for all runs (default: use tuned_threshold from CSV)",
    )
    args = parser.parse_args()
    generate(fixed_threshold=args.threshold)


if __name__ == "__main__":
    main()
