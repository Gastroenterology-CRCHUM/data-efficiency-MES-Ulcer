"""ClassifierModel on CPU without downloads (src.models.classifier).

Only GastroNet keys are built: ResNet-50 (a plain torchvision ResNet-50,
``weights=None``) and the ViT-B/14 DINOv2 entry (``timm.create_model`` replaced by a
small local ViT). The external weights come from fake checkpoints in ``tmp_path``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.config.models import MODEL_REGISTRY, dino_resnet50
from src.data.dataloader import get_split_loaders, get_test_loader
from src.models import ClassifierModel, FitResult
from src.models.classifier import (
    _ActiveDiseaseLoss,
    _extract_backbone_state_dict,
    _fix_dino_pos_embed,
    _resample_dino_pos_embed,
)
from src.utils import ConfigurationError

MODEL = "resnet50_5M"


def _model(**kw) -> ClassifierModel:
    kw.setdefault("num_classes", 1)
    return ClassifierModel(MODEL, optimizer="AdamW", learning_rate=1e-4, **kw)


@pytest.fixture(scope="module")
def backbone_state():
    torch.manual_seed(0)
    state = dino_resnet50(weights=None).state_dict()
    state["conv1.weight"] = torch.full_like(state["conv1.weight"], 0.125)
    return state


@pytest.fixture(scope="module")
def gastronet_file(tmp_path_factory, backbone_state):
    """DINO-style checkpoint: teacher/student wrappers, module.backbone. prefixes, head keys."""
    path = tmp_path_factory.mktemp("pretrained") / "RN50_fake_DINOv1.pth"
    teacher = {f"module.backbone.{k}": v for k, v in backbone_state.items()}
    teacher["module.head.mlp.0.weight"] = torch.zeros(4, 4)
    student = {f"module.backbone.{k}": torch.zeros_like(v) for k, v in backbone_state.items()}
    torch.save({"teacher": teacher, "student": student, "epoch": 3}, path)
    return path


def _bn_buffers(module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in module.state_dict().items() if "running" in k}


# ---------------------------------------------------------------------------
# External weights
# ---------------------------------------------------------------------------


def test_extract_backbone_state_dict_unwraps_and_strips():
    w = torch.ones(2)
    raw = {
        "state_dict": {
            "teacher": {
                "module.backbone.layer1.0.conv1.weight": w,
                "backbone.fc.weight": w,  # classifier of the pretraining model
                "module.head.last_layer.weight": w,
                "dino_head.mlp.weight": w,
                "step": 10,  # not a tensor
            }
        }
    }
    assert _extract_backbone_state_dict(raw) == {"layer1.0.conv1.weight": w}
    assert _extract_backbone_state_dict({"a.weight": w, "b.x": w}, drop_prefixes=("b.",)) == {
        "a.weight": w
    }
    with pytest.raises(TypeError):
        _extract_backbone_state_dict([w])


def test_gastronet_weights_are_loaded_from_the_teacher(gastronet_file, backbone_state):
    model = _model(gastronet_path=gastronet_file)
    loaded = model.unwrapped_backbone.state_dict()
    for key in ("conv1.weight", "layer4.2.bn3.running_var", "layer3.0.downsample.0.weight"):
        assert torch.equal(loaded[key], backbone_state[key]), key
    assert model.gastronet_file == gastronet_file.name  # file name only, never a local path


def test_partial_checkpoint_raises(tmp_path, backbone_state):
    path = tmp_path / "partial.pth"
    torch.save({k: v for k, v in backbone_state.items() if not k.startswith("layer4.")}, path)
    with pytest.raises(RuntimeError, match="missing"):
        _model(gastronet_path=path)


def test_mismatched_checkpoint_raises(tmp_path, backbone_state):
    path = tmp_path / "mismatch.pth"
    bad = dict(backbone_state)
    bad["conv1.weight"] = torch.zeros(8, 3, 3, 3)
    torch.save(bad, path)
    with pytest.raises(RuntimeError, match="does not match"):
        _model(gastronet_path=path)


def test_missing_weight_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _model(gastronet_path=tmp_path / "nope.pth")


def test_dinov2_checkpoint_with_registers_loads_into_the_timm_vit(tmp_path, monkeypatch):
    """Original DINOv2 teacher (backbone. prefix, register_tokens, mask_token, position grid of
    another size, dino_head) -> the timm ViT with registers, through timm's converter."""
    timm = pytest.importorskip("timm")
    real, calls = timm.create_model, []

    def create_model(name, **kwargs):  # the hub model, replaced by a small one of the same family
        calls.append(kwargs)
        return real("vit_small_patch14_reg4_dinov2", **{**kwargs, "pretrained": False})

    monkeypatch.setattr(timm, "create_model", create_model)
    torch.manual_seed(0)
    sd = real("vit_small_patch14_reg4_dinov2", pretrained=False, num_classes=0, img_size=168).state_dict()
    sd["cls_token"] = torch.randn_like(sd["cls_token"])  # timm initialises it near 0
    cls_pos = torch.randn(1, 1, sd["pos_embed"].shape[-1])  # DINOv2 keeps a class position embedding
    teacher = {f"backbone.{k}": v for k, v in sd.items() if k not in ("cls_token", "reg_token", "pos_embed")}
    teacher |= {
        "backbone.cls_token": sd["cls_token"] - cls_pos,
        "backbone.pos_embed": torch.cat([cls_pos, sd["pos_embed"]], dim=1),  # 12 x 12 grid
        "backbone.register_tokens": sd["reg_token"],
        "backbone.mask_token": torch.zeros(1, sd["pos_embed"].shape[-1]),
        "dino_head.mlp.0.weight": torch.zeros(4, 4),
    }
    path = tmp_path / "DINOv2.pth"
    torch.save({"teacher": teacher}, path)

    def build():
        return ClassifierModel("vitb14_gastronet", num_classes=4, optimizer="Adam", learning_rate=1e-4,
                               gastronet_path=path)

    vit = build().unwrapped_backbone.vit
    assert calls[0]["pretrained"] is False  # no hub weights for a GastroNet entry
    assert torch.equal(vit.reg_token, sd["reg_token"])
    assert torch.allclose(vit.cls_token, sd["cls_token"], atol=1e-6)  # DINOv2 cls + its position
    assert vit.pos_embed.shape[1] == (MODEL_REGISTRY["vitb14_gastronet"].img_size // 14) ** 2
    assert torch.equal(vit.blocks[3].attn.qkv.weight, sd["blocks.3.attn.qkv.weight"])

    del teacher["backbone.blocks.3.mlp.fc2.weight"]
    torch.save({"teacher": teacher}, path)
    with pytest.raises(RuntimeError, match="missing"):
        build()


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_linear_probe_keeps_the_backbone_fixed(gastronet_file):
    model = _model(gastronet_path=gastronet_file, freeze_layers=-1, head_lr_scale=10.0)
    backbone = model.unwrapped_backbone
    before = _bn_buffers(backbone)

    model.train()
    assert model.training and not backbone.training
    model.base_model(torch.randn(4, 3, 64, 64))  # a training-mode forward pass

    after = _bn_buffers(backbone)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in model._get_layer(model.head_prefix).parameters())
    groups = model._create_optimizer().param_groups
    assert [g["name"] for g in groups] == ["head"]
    assert groups[0]["lr"] == pytest.approx(1e-3)


def test_full_finetuning_updates_batchnorm(gastronet_file):
    model = _model(gastronet_path=gastronet_file, freeze_layers=0)
    before = _bn_buffers(model.unwrapped_backbone)
    model.train()
    model.base_model(torch.randn(4, 3, 64, 64))
    after = _bn_buffers(model.unwrapped_backbone)
    assert not torch.equal(before["bn1.running_mean"], after["bn1.running_mean"])
    names = [g["name"] for g in model._create_optimizer().param_groups]
    assert names == ["backbone", "head"]


def test_partial_freezing_of_resnet_groups(gastronet_file):
    model = _model(gastronet_path=gastronet_file, freeze_layers=2)  # stem + layer1
    arch = model.unwrapped_backbone
    assert not arch.conv1.weight.requires_grad
    assert not any(p.requires_grad for p in arch.layer1.parameters())
    assert all(p.requires_grad for p in arch.layer2.parameters())

    before = _bn_buffers(arch)
    model.train()
    model.base_model(torch.randn(4, 3, 64, 64))
    after = _bn_buffers(arch)
    assert torch.equal(before["bn1.running_mean"], after["bn1.running_mean"])
    assert torch.equal(before["layer1.0.bn1.running_mean"], after["layer1.0.bn1.running_mean"])
    assert not torch.equal(before["layer2.0.bn1.running_mean"], after["layer2.0.bn1.running_mean"])


@pytest.mark.parametrize("freeze", [6, -2])
def test_invalid_freezing_raises(gastronet_file, freeze):
    with pytest.raises(ConfigurationError):
        _model(gastronet_path=gastronet_file, freeze_layers=freeze)


def test_constructor_validation(gastronet_file):
    model = _model(gastronet_path=gastronet_file, ci=95)
    assert model.ci == pytest.approx(0.95)
    assert model.norm_mean == (0.485, 0.456, 0.406)
    with pytest.raises(ValueError, match="head_lr_scale"):
        _model(gastronet_path=gastronet_file, head_lr_scale=0)
    with pytest.raises(ValueError, match="drop_path_rate"):
        _model(gastronet_path=gastronet_file, drop_path_rate=1.0)
    with pytest.raises(ValueError, match="layer_decay"):
        _model(gastronet_path=gastronet_file, layer_decay=0.0)


# ---------------------------------------------------------------------------
# Learning rates: schedule, ViT stochastic depth and layer-wise decay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup", [0, 3, 10])
def test_schedule_is_warmup_then_cosine_with_fixed_group_ratios(gastronet_file, warmup):
    epochs, min_lr = 10, 1e-6
    model = _model(gastronet_path=gastronet_file, warmup_epochs=warmup, min_lr=min_lr, head_lr_scale=10.0)
    opt = model._create_optimizer()
    scheduler = model._create_scheduler(opt, epochs)
    # Reference: LinearLR warm-up then CosineAnnealingLR, on one group at the backbone LR.
    ref_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=model.lr)
    if 0 < warmup < epochs:
        ref = SequentialLR(
            ref_opt,
            [LinearLR(ref_opt, start_factor=1e-3, total_iters=warmup),
             CosineAnnealingLR(ref_opt, T_max=epochs - warmup, eta_min=min_lr)],
            milestones=[warmup],
        )
    else:
        ref = CosineAnnealingLR(ref_opt, T_max=epochs, eta_min=min_lr)
    for _ in range(epochs):
        backbone, head = (g["lr"] for g in opt.param_groups)
        assert backbone == pytest.approx(ref_opt.param_groups[0]["lr"], rel=1e-9)
        assert head == pytest.approx(10 * backbone)
        opt.step()
        scheduler.step()
        ref_opt.step()
        ref.step()


class _TinyDinoViT(torch.nn.Module):
    """DINO-like ViT: patch embedding, cls / position tokens, blocks, final norm, no head."""

    def __init__(self, depth: int = 3, dim: int = 8):
        super().__init__()
        self.patch_embed = torch.nn.Linear(3, dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 2, dim))
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(depth))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x.mean(dim=(2, 3))) + self.cls_token[:, 0] + self.pos_embed[:, 0]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def test_vit_drop_path_and_layer_decay(monkeypatch):
    built: dict = {}

    def hub_load(*args, **kwargs):  # stands in for torch.hub.load
        built.update(kwargs)
        return _TinyDinoViT()

    entry = replace(MODEL_REGISTRY["vits16_imagenet"], builder=hub_load)
    monkeypatch.setitem(MODEL_REGISTRY, "vits16_imagenet", entry)

    model = ClassifierModel("vits16_imagenet", num_classes=1, optimizer="Adam", learning_rate=1e-4,
                            drop_path_rate=0.1, layer_decay=0.5, head_lr_scale=2.0)
    assert built["drop_path_rate"] == 0.1
    groups = model._create_optimizer().param_groups
    assert [g["name"] for g in groups] == [
        "backbone.0", "backbone.1", "backbone.2", "backbone.3", "backbone", "head"
    ]
    # Depth 3: layer k at 1e-4 * 0.5 ** (4 - k); the final norm at 1e-4; the head at 2e-4.
    assert [g["lr"] for g in groups] == pytest.approx([1e-4 * 0.5**k for k in (4, 3, 2, 1, 0)] + [2e-4])
    vit = model.unwrapped_backbone
    embeddings = [*vit.patch_embed.parameters(), vit.cls_token, vit.pos_embed]
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in embeddings}
    assert {id(p) for p in groups[4]["params"]} == {id(p) for p in vit.norm.parameters()}

    built.clear()
    plain = ClassifierModel("vits16_imagenet", num_classes=1, optimizer="Adam", learning_rate=1e-4)
    assert "drop_path_rate" not in built
    assert [g["name"] for g in plain._create_optimizer().param_groups] == ["backbone", "head"]


def test_vit_only_settings_are_off_for_cnns(gastronet_file):
    model = _model(gastronet_path=gastronet_file, drop_path_rate=0.1, layer_decay=0.5)
    assert (model.drop_path_rate, model.layer_decay) == (0.0, 1.0)
    groups = model._create_optimizer().param_groups
    assert [g["name"] for g in groups] == ["backbone", "head"]
    assert groups[0]["lr"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# MES loss
# ---------------------------------------------------------------------------


def test_active_disease_loss_adds_the_weighted_binary_term():
    torch.manual_seed(0)
    logits = torch.randn(6, 4, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 2, 0])
    w = torch.tensor([1.0, 2.0, 0.5, 3.0])
    p_active = logits.softmax(1)[:, 2:].sum(1)
    nll = -torch.where(y >= 2, p_active.log(), (1 - p_active).log())
    expected = torch.nn.functional.cross_entropy(logits, y, weight=w) + 0.7 * (w[y] * nll).sum() / w[y].sum()

    loss = _ActiveDiseaseLoss(w, 0.0, 0.7)(logits, y)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_criterion_by_active_loss_weight(gastronet_file):
    weights, cpu = [1.0, 2.0, 0.5, 3.0], torch.device("cpu")
    mes = _model(num_classes=4, gastronet_path=gastronet_file, active_loss_weight=0.7)
    assert isinstance(mes._create_criterion(weights, cpu), _ActiveDiseaseLoss)
    plain = _model(num_classes=4, gastronet_path=gastronet_file, active_loss_weight=0.0)
    assert type(plain._create_criterion(weights, cpu)) is torch.nn.CrossEntropyLoss
    binary = _model(gastronet_path=gastronet_file, active_loss_weight=0.7)  # ignored
    assert binary.active_loss_weight == 0.0
    assert isinstance(binary._create_criterion([1.0, 2.0], cpu), torch.nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError, match="active_loss_weight"):
        _model(num_classes=4, gastronet_path=gastronet_file, active_loss_weight=-1.0)


# ---------------------------------------------------------------------------
# fit / evaluation smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_and_evaluate_smoke(image_manifest, gastronet_file, tmp_path):
    manifest_path, data_dir, _ = image_manifest
    lkw = dict(
        manifest_path=manifest_path,
        data_dir=data_dir,
        batch_size=8,
        img_size=64,
        num_workers=0,
        equalize=True,
        seed=0,
    )
    train_loader, val_loader = get_split_loaders(
        subset_ratio=1.0, random_seed=0, verbose=False, **lkw
    )
    test_loader = get_test_loader(**lkw)

    model = _model(
        gastronet_path=gastronet_file,
        freeze_layers=-1,
        n_bootstrap=20,
        warmup_epochs=1,
        use_amp=False,
    )
    root = tmp_path / "models"
    res = model.fit(
        train_loader,
        val_loader,
        epochs=2,
        device=torch.device("cpu"),
        checkpoint_root=root,
        run_name="smoke run/1",
        run_metadata={"seed": 0, "task": "ulcer"},
    )

    assert isinstance(res, FitResult)
    probs, labels, ckpt = res  # historical 3-tuple unpacking
    assert ckpt == res.checkpoint_dir == model.checkpoint_dir
    assert ckpt.parent == root / MODEL / "-1" / "linear"
    assert ckpt.name.startswith("smoke_run_1_")
    assert {p.name for p in ckpt.iterdir()} >= {"best.pt", "history.json", "run_config.json"}
    assert probs.shape == labels.shape == (len(val_loader.dataset),)
    assert res.stop_reason in ("max_epochs", "early_stopping")
    assert 1 <= res.best_epoch <= res.stopped_epoch <= 2

    history = json.loads((ckpt / "history.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in history] == list(range(1, res.stopped_epoch + 1))
    assert sum(h["is_best"] for h in history) >= 1

    text = (ckpt / "run_config.json").read_text(encoding="utf-8")
    run_config = json.loads(text)
    assert run_config["model"] == MODEL and run_config["seed"] == 0
    assert run_config["best_epoch"] == res.best_epoch
    assert run_config["pos_weight"] == pytest.approx(model.pos_weight)
    assert str(tmp_path) not in text and str(data_dir) not in text  # no local paths
    model.update_run_config({"tuned_threshold": 0.3})
    assert json.loads((ckpt / "run_config.json").read_text())["tuned_threshold"] == 0.3

    y, p = model.get_labels_and_probs(val_loader, torch.device("cpu"))
    assert y.shape == p.shape == (len(val_loader.dataset),)
    assert np.all((p >= 0) & (p <= 1))

    results = model.test_evaluation(test_loader, torch.device("cpu"), threshold=0.5)
    n_clips = test_loader.dataset.df["clip_key"].nunique()
    assert len(results["clip_ids"]) == len(test_loader.dataset)
    n_patients = test_loader.dataset.df["patient_id"].nunique()
    assert results["metrics_with_ci"]["_bootstrap"]["n_groups"] == n_patients  # patient bootstrap
    assert results["clip_level"]["n_clips"] == n_clips
    assert results["frame_level"]["confusion_matrix"].shape == (2, 2)


class _LabelsOnly:
    """Dataset stand-in for fit() with scripted epochs: labels only."""

    df = pd.DataFrame({"label": [0, 1, 0, 1]})

    def __len__(self) -> int:
        return len(self.df)


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step")
def test_patience_counts_only_after_es_start_epoch(gastronet_file, tmp_path, monkeypatch):
    model = _model(gastronet_path=gastronet_file, es_patience=2, es_start_epoch=3)
    aurocs = iter([0.9, 0.5, 0.5, 0.5, 0.5, 0.5])  # best at epoch 1, no gain afterwards
    zeros = np.zeros(4)
    monkeypatch.setattr(model, "_train_one_epoch", lambda *a: (0.1, 0.5))
    monkeypatch.setattr(model, "validate", lambda *a: (0.1, 0.5, 0.5, 0.5, 0.5, next(aurocs), zeros, zeros))
    loader = SimpleNamespace(dataset=_LabelsOnly())

    res = model.fit(loader, loader, epochs=10, device=torch.device("cpu"), checkpoint_root=tmp_path)
    # Epochs 2-3 are not counted: patience 2 runs out at epoch 5 instead of 3.
    assert (res.best_epoch, res.stopped_epoch, res.stop_reason) == (1, 5, "early_stopping")
    run_config = json.loads((res.checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["es_start_epoch"] == 3


# ---------------------------------------------------------------------------
# DINOv1 position embedding
# ---------------------------------------------------------------------------


class _DinoPosStub(torch.nn.Module):
    """The position-embedding part of DINO's VisionTransformer (pretrained 14 x 14 grid)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.patch_size = 16
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 14 * 14 + 1, dim))

    def interpolate_pos_encoding(self, x, w, h):  # as in facebookresearch/dino
        npatch, n = x.shape[1] - 1, self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        return _resample_dino_pos_embed(self.pos_embed, w // self.patch_embed.patch_size)


def test_dino_pos_embed_is_resampled_once():
    torch.manual_seed(0)
    vit = _DinoPosStub()
    tokens = torch.zeros(1, 16 * 16 + 1, 8)
    expected = vit.interpolate_pos_encoding(tokens, 256, 256).detach().clone()
    pretrained = {"pos_embed": vit.pos_embed.detach().clone()}

    _fix_dino_pos_embed(vit, 256)
    assert vit.pos_embed.shape == (1, 16 * 16 + 1, 8)
    assert vit.interpolate_pos_encoding(tokens, 256, 256) is vit.pos_embed  # no per-forward resampling
    torch.testing.assert_close(vit.pos_embed.detach(), expected)

    # 224-px weights (GastroNet, older checkpoints) are resampled on load.
    vit.pos_embed.data.zero_()
    vit.load_state_dict(pretrained)
    torch.testing.assert_close(vit.pos_embed.detach(), expected)
