"""scripts/mes/evaluate_external.py: checks, resume and tables, with a stand-in for the models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.mes import evaluate_external as ext
from tests.conftest import write_frames

ARGS = ["--direction", "internal_to_limuc", "--device-id", "-1", "--num-workers", "0"]


def _dataset(root: Path, seed: int) -> dict[str, Path]:
    """Test split of 8 patients (grades 0-3 twice, 3 frames each) as PNGs, and its results folder."""
    rows = [
        {"filepath": f"{p % 4}/p{p}/c0/f{k}.png", "label": p % 4, "split": "test",
         "patient_id": f"p{p}", "video_id": f"p{p}__c0"}
        for p in range(8)
        for k in range(3)
    ]
    write_frames(root / "processed", rows, seed=seed)
    manifest = root / "splits" / "dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {"manifest": manifest, "data_dir": root / "processed", "results": root / "results"}


def _write_runs(results: Path, specs: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Source results_per_seed.csv and run folders; stored test metrics are perfect (as the stand-in)."""
    rows = []
    for model, pct, seed in specs:
        run_name = f"{model}_finetune_linear_{pct}pct_seed{seed}"
        run_dir = results.parent / "models" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"")
        (run_dir / "run_config.json").write_text(json.dumps({"run_name": run_name}))
        rows.append({"model": model, "freeze": "finetune", "head_type": "linear",
                     "learning_rate": 1e-5, "subset_ratio": pct / 100, "pct_data": pct,
                     "seed": seed, "f1": 0.3, "auroc": 1.0, "clin_auroc": 1.0, "qwk": 1.0,
                     "val_auroc": 0.8, "checkpoint_dir": str(run_dir), "run_name": run_name})
    results.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(results / "results_per_seed.csv", index=False)
    return df


class _StandIn:
    """Loaded run: probability 0.7 on the true grade of every loader frame."""

    img_size, equalize, norm_mean, norm_std = 32, True, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    n_bootstrap = 1000

    def test_evaluation(self, loader, device, threshold=None, aggregate_by_clip=True):
        labels = np.concatenate([batch[1].numpy() for batch in loader]).astype(int)
        probs = np.full((labels.size, 4), 0.1, dtype=np.float32)
        probs[np.arange(labels.size), labels] = 0.7
        return {"labels": labels, "probabilities": probs,
                "frame_level": {"f1": 1.0, "roc_auc": 1.0}, "clip_level": None}


@pytest.fixture
def setup(tmp_path, monkeypatch, restore_torch_flags):
    """Internal source with 3 runs, LIMUC target with in-domain predictions; counts the model loads."""
    internal, limuc = _dataset(tmp_path / "internal", seed=0), _dataset(tmp_path / "limuc", seed=1)
    _write_runs(internal["results"], [("m", 10, 1), ("m", 100, 1), ("m", 10, 2)])
    preds = limuc["results"] / "predictions" / "m_finetune_linear_100pct_seed1"
    preds.mkdir(parents=True)
    np.save(preds / "test_labels.npy", pd.read_csv(limuc["manifest"])["label"].to_numpy())
    monkeypatch.setattr(ext, "_datasets", lambda vitb=False: {"internal": internal, "LIMUC": limuc})
    loads: list[Path] = []
    monkeypatch.setattr(ext, "load_run_model", lambda run_dir: loads.append(run_dir) or _StandIn())
    return internal, limuc, loads


def test_dry_run_lists_the_runs_and_writes_nothing(setup, capsys):
    internal, _, loads = setup
    assert ext.main([*ARGS, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "To evaluate    : 3 (0 already done)" in out
    assert "Checkpoints    : all found" in out
    assert not (internal["results"] / "external_LIMUC").exists()
    assert loads == []


def test_max_runs_then_resume_writes_the_tables(setup, capsys):
    internal, _, loads = setup
    out_dir = internal["results"] / "external_LIMUC"

    ext.main([*ARGS, "--max-runs", "1"])  # reproduction check + 1 run
    assert len(loads) == 2
    assert len(pd.read_csv(out_dir / "results_per_seed.partial.csv")) == 1
    assert not (out_dir / "results_per_seed.csv").exists()

    ext.main(ARGS)  # reproduction check + the 2 runs left
    assert len(loads) == 5
    assert not (out_dir / "results_per_seed.partial.csv").exists()
    per_seed = pd.read_csv(out_dir / "results_per_seed.csv")
    source = pd.read_csv(internal["results"] / "results_per_seed.csv")
    assert list(per_seed["run_name"]) == list(source["run_name"])
    assert (per_seed["f1"] == 1.0).all() and (per_seed["val_auroc"] == 0.8).all()
    assert (per_seed[["source", "target"]] == ["internal", "LIMUC"]).all().all()
    for run in per_seed["run_name"]:
        assert np.load(out_dir / "predictions" / run / "test_probs.npy").shape == (24, 4)
    agg = pd.read_csv(out_dir / "results.csv")
    assert len(agg) == 2 and {"auroc_mean", "clin_auroc_mean", "qwk_mean"} <= set(agg.columns)
    meta = json.loads((out_dir / "experiment_meta.json").read_text())
    assert meta["checks"]["target_labels"] == "match"
    assert meta["target_test"]["n_frames"] == 24

    ext.main(ARGS)
    assert "Up to date" in capsys.readouterr().out and len(loads) == 5

    # A retrained run (new checkpoint folder) is the only one evaluated again.
    retrained = _write_runs(internal["results"].parent / "retrain" / "results", [("m", 10, 2)])
    source.loc[2, "checkpoint_dir"] = retrained.loc[0, "checkpoint_dir"]
    source.to_csv(internal["results"] / "results_per_seed.csv", index=False)
    ext.main(ARGS)
    assert loads[-1] == Path(retrained.loc[0, "checkpoint_dir"]) and len(loads) == 7


@pytest.mark.parametrize("fault", ["missing_checkpoint", "misaligned_labels", "not_reproduced"])
def test_checks_fail_before_any_external_prediction(setup, fault):
    internal, limuc, loads = setup
    source = pd.read_csv(internal["results"] / "results_per_seed.csv")
    if fault == "missing_checkpoint":
        (Path(source.loc[1, "checkpoint_dir"]) / "best.pt").unlink()
        error = FileNotFoundError
    elif fault == "misaligned_labels":
        labels_path = next((limuc["results"] / "predictions").glob("*/test_labels.npy"))
        np.save(labels_path, np.load(labels_path)[::-1])
        error = ValueError
    else:
        source.loc[0, "qwk"] = 0.5
        source.to_csv(internal["results"] / "results_per_seed.csv", index=False)
        error = RuntimeError
    with pytest.raises(error):
        ext.main(ARGS)
    assert len(loads) == (1 if fault == "not_reproduced" else 0)  # the reproduction check's run
    assert not (internal["results"] / "external_LIMUC").exists()


def test_vitb_runs_are_read_from_their_own_folders():
    main, vitb = ext._datasets(), ext._datasets(vitb=True)
    for name in main:
        assert vitb[name]["manifest"] == main[name]["manifest"]
        assert vitb[name]["results"] == main[name]["results"].with_name(main[name]["results"].name + "_vitb")
