"""Metrics, thresholds and per-run figures used by the training runners.

Importing this package never changes the matplotlib backend. The post-hoc statistics
of the notebooks are in ``src.analysis``.
"""

from src.evaluation.aggregation import aggregate_frame_to_clip
from src.evaluation.metrics import (
    METRIC_FNS,
    bootstrap_ci,
    bootstrap_indices,
    compute_metrics_with_ci,
    fmt,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_confusion_matrix_multiclass,
    plot_learning_curves,
    plot_roc_curve,
    plot_roc_curves,
)
from src.evaluation.threshold import find_best_threshold, sweep_thresholds

__all__ = [
    "METRIC_FNS",
    "aggregate_frame_to_clip",
    "bootstrap_ci",
    "bootstrap_indices",
    "compute_metrics_with_ci",
    "find_best_threshold",
    "fmt",
    "plot_confusion_matrix",
    "plot_confusion_matrix_multiclass",
    "plot_learning_curves",
    "plot_roc_curve",
    "plot_roc_curves",
    "sweep_thresholds",
]
