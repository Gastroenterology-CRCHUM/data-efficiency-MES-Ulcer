"""Reload the model of a finished run from its run folder."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config.models import get_norm_stats
from src.models.classifier import ClassifierModel


def load_run_model(run_dir: Path | str) -> ClassifierModel:
    """Model of a run folder (``run_config.json``, ``best.pt``), in eval mode on the CPU.

    ``img_size``, ``equalize``, ``norm_mean`` and ``norm_std`` are set to the run's
    preprocessing, and ``threshold`` to its val-tuned threshold (binary runs).

    Raises:
        FileNotFoundError: ``best.pt`` is missing (checkpoints are local only).
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"best.pt of {cfg['run_name']} not found (local checkpoints only)")
    thr = cfg.get("tuned_threshold")  # binary runs only
    model = ClassifierModel(
        base_model=cfg["model"],
        num_classes=int(cfg["num_classes"]),
        optimizer="Adam",  # training-only
        learning_rate=1e-4,  # training-only
        dropout_rate=float(cfg["dropout_rate"]),
        freeze_layers=int(cfg["freeze_layers"]),
        head_type=cfg["head_type"],
        threshold=0.5 if thr is None else float(thr),
        gastronet_path=None,  # the weights come from best.pt
    )
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.base_model.load_state_dict(state)
    model.eval()
    mean, std = (cfg["mean"], cfg["std"]) if cfg.get("mean") else get_norm_stats(cfg["model"])
    model.img_size, model.equalize = int(cfg["img_size"]), bool(cfg["equalize"])
    model.norm_mean, model.norm_std = tuple(mean), tuple(std)
    return model
