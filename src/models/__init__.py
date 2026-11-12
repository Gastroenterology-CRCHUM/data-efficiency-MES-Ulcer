"""Model module: registry-driven classifier wrapper."""

from .checkpoint import load_run_model
from .classifier import ClassifierModel, FitResult

__all__ = ["ClassifierModel", "FitResult", "load_run_model"]
