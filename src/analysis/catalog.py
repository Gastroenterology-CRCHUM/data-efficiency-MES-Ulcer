"""Result folders, model order, labels and colours shared by the analysis notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config.models import MODEL_REGISTRY
from src.utils.results import REPO_ROOT

RESULTS = {
    "ulcer": REPO_ROOT / "results" / "ulcer" / "data_efficiency",
    "mes": REPO_ROOT / "results" / "mes" / "data_efficiency",
    "limuc": REPO_ROOT / "results" / "mes" / "data_efficiency_LIMUC",
}
RESULTS_VITB = {task: path.with_name(path.name + "_vitb") for task, path in RESULTS.items()}
"""Results of the exploratory ViT-B runs (``run_all --vitb``): SECONDARY_MODELS only."""
MANIFESTS = {
    "ulcer": REPO_ROOT / "data" / "ulcer" / "splits" / "dataset_manifest.csv",
    "mes": REPO_ROOT / "data" / "mes" / "splits" / "dataset_manifest.csv",
}

MODEL_ORDER = [
    "resnet50_imagenet",
    "resnet50_200K",
    "resnet50_1M",
    "resnet50_5M",
    "efficientnetb0",
    "vits16_imagenet",
    "vits16_gastronet",
    "vits14_dinov2",
    "vits16_dinov3",
]
"""Row order of the tables and legend order of the figures."""
SECONDARY_MODELS = ["vitb14_dinov2", "vitb14_gastronet", "vitb16_dinov3"]
"""Exploratory ViT-Base models (336 px)."""
FAMILY_ORDER = ["CNN", "ViT", "ViT-B"]
"""Figure panels, as returned by ``model_family``."""

FULL_PCT = 100
ENDPOINT_METRIC = {"ulcer": "auroc", "mes": "clin_auroc", "limuc": "clin_auroc"}
"""Per-seed column of the AUROC endpoint: ulcer binary; MES / LIMUC active disease
(MES 2-3 vs MES 0-1, score p2 + p3)."""
MES_NUM_CLASSES = 4
MES_CLASS_NAMES = [f"MES {i}" for i in range(MES_NUM_CLASSES)]
ULCER_CLASS_NAMES = ["No ulcer", "Ulcer"]

# Colour = architecture (ResNet-50 blue, ViT-S purple, EfficientNet-B0 orange, generic
# DINOv2/v3 green, ViT-B crimson), darker for a larger GastroNet corpus.
COLORS = {
    "resnet50_imagenet": "#7F93AD",
    "resnet50_200K": "#9CC7F2",
    "resnet50_1M": "#3D86D6",
    "resnet50_5M": "#0B3D91",
    "vits16_imagenet": "#B7A4D3",
    "vits16_gastronet": "#5B2A86",
    "vits14_dinov2": "#4C9F70",
    "vits16_dinov3": "#1B5E20",
    "efficientnetb0": "#E4572E",
    "vitb14_gastronet": "#A0154A",
    "vitb14_dinov2": "#D2608A",
    "vitb16_dinov3": "#3A8A4A",
}
MARKERS = {"ResNet-50": "o", "ViT-S/16": "s", "ViT-S/14": "D", "EfficientNet-B0": "^",
           "ViT-B/14": "P", "ViT-B/16": "X"}
"""Marker of each architecture (``short_arch``)."""


def short_arch(arch: str) -> str:
    return arch.replace("ViT-Small/", "ViT-S/").replace("ViT-Base/", "ViT-B/")


def _short_method(method: str) -> str:
    """'Self-sup. (DINOv1)' -> 'DINOv1'; 'Supervised' -> 'supervised'."""
    m = re.search(r"\(([^)]+)\)", method)
    return m.group(1) if m else method.lower()


def model_label(key: str) -> str:
    """'<architecture> / <pretraining data> (<method>)' from MODEL_REGISTRY."""
    e = MODEL_REGISTRY[key]
    return f"{short_arch(e.architecture)} / {e.pretrain_data} ({_short_method(e.pretrain_method)})"


def model_family(key: str) -> str:
    """'CNN', 'ViT' (ViT-Small) or 'ViT-B' (ViT-Base), from the registry architecture."""
    arch = MODEL_REGISTRY[key].architecture
    return "ViT-B" if arch.startswith("ViT-Base") else "ViT" if arch.startswith("ViT") else "CNN"


def pm(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def results_dir(task: str, model: str) -> Path:
    """Results folder of a model's runs: RESULTS_VITB for SECONDARY_MODELS, else RESULTS."""
    return (RESULTS_VITB if model in SECONDARY_MODELS else RESULTS)[task]


def predictions_dir(task: str, model: str, seed: int, pct: int = FULL_PCT) -> Path:
    """Saved test / validation predictions of one run."""
    return results_dir(task, model) / "predictions" / f"{model}_finetune_linear_{pct}pct_seed{seed}"


def read_meta(task: str) -> dict:
    """experiment_meta.json of a task's results folder ({} when absent)."""
    path = RESULTS[task] / "experiment_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
