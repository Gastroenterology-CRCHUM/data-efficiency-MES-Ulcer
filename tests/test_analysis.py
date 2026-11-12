"""Notebook helpers (src/analysis): catalog, plan monitoring, run reloading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis import catalog, monitoring
from src.config.models import MODEL_REGISTRY
from src.models.checkpoint import load_run_model
from src.utils.results import run_dir_of


def test_model_family_splits_cnn_vit_and_vit_base():
    vits = [k for k in catalog.MODEL_ORDER if catalog.model_family(k) == "ViT"]
    assert vits == ["vits16_imagenet", "vits16_gastronet", "vits14_dinov2", "vits16_dinov3"]
    assert {catalog.model_family(k) for k in catalog.MODEL_ORDER} == {"CNN", "ViT"}
    assert {catalog.model_family(k) for k in catalog.SECONDARY_MODELS} == {"ViT-B"}
    assert catalog.FAMILY_ORDER == ["CNN", "ViT", "ViT-B"]
    limuc = catalog.RESULTS["limuc"]
    assert catalog.results_dir("limuc", "vitb14_gastronet") == limuc.with_name("data_efficiency_LIMUC_vitb")
    assert catalog.results_dir("limuc", "vits16_gastronet") == limuc


def test_every_catalog_model_is_registered_with_a_colour_and_a_label():
    models = [*catalog.MODEL_ORDER, *catalog.SECONDARY_MODELS]
    assert set(models) <= set(MODEL_REGISTRY)
    assert set(models) == set(catalog.COLORS)
    assert len({catalog.model_label(m) for m in models}) == len(models)


def test_load_task_counts_planned_and_finished_runs(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text("subset_ratios: [0.1, 1.0]\nseeds: [1, 2]\nmodels:\n  - model: resnet50_5M\n",
                    encoding="utf-8")
    res = tmp_path / "results"
    res.mkdir()
    pd.DataFrame([{"model": "resnet50_5M", "head_type": "linear", "pct_data": p, "seed": s,
                   "stop_reason": "early_stopping", "checkpoint_dir": f"<models-dir>/x/{p}_{s}",
                   "run_name": f"r{p}_{s}"} for p, s in [(10, 1), (10, 2), (100, 1)]]
                 ).to_csv(res / "results_per_seed.csv", index=False)
    (res / "experiment_meta.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    out = monitoring.load_task({"results": res, "plan": plan, "models": tmp_path / "models"}, root=tmp_path)
    summary = out["summary"]
    assert (summary["planned"], summary["done"], summary["missing"]) == (4, 3, 1)
    assert out["plan"]["pcts"] == [10, 100]
    assert out["issues"] == []


def test_resolve_run_dir(tmp_path):
    assert monitoring.resolve_run_dir("output/a/b", None, tmp_path) == tmp_path / "output/a/b"
    assert monitoring.resolve_run_dir("<models-dir>/a/b", tmp_path / "m", tmp_path) == tmp_path / "m" / "a/b"
    assert monitoring.resolve_run_dir(float("nan"), None, tmp_path) is None


def test_run_dir_of_and_missing_checkpoint(tmp_path):
    per_seed = pd.DataFrame({"model": ["m", "m"], "pct_data": [100, 50], "seed": [42, 42],
                             "checkpoint_dir": [str(tmp_path / "run"), "other"]})
    assert run_dir_of(per_seed, "m", 100, 42) == tmp_path / "run"
    with pytest.raises(LookupError, match="0 runs of m at 10 %"):
        run_dir_of(per_seed, "m", 10, 42)

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run_config.json").write_text(json.dumps({"run_name": "m_100pct"}))
    with pytest.raises(FileNotFoundError, match="best.pt of m_100pct"):
        load_run_model(tmp_path / "run")


def test_model_style_marks_architecture_and_imagenet():
    from src.analysis.figures import model_style

    models = [*catalog.MODEL_ORDER, *catalog.SECONDARY_MODELS]
    assert {catalog.short_arch(MODEL_REGISTRY[m].architecture) for m in models} == set(catalog.MARKERS)
    assert model_style("vits14_dinov2")["marker"] == "D" and model_style("vits16_dinov3")["marker"] == "s"
    imagenet, gastronet = model_style("resnet50_imagenet"), model_style("resnet50_5M")
    assert (imagenet["linestyle"], imagenet["markerfacecolor"]) == ("--", "white")
    assert (gastronet["linestyle"], gastronet["markerfacecolor"]) == ("-", catalog.COLORS["resnet50_5M"])
