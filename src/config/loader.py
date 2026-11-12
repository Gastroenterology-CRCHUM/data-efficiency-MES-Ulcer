"""Configuration loading and management.

Fallback defaults: the dataclasses ``ModelConfig``, ``TrainingConfig``, ``EvaluationConfig``
and ``PathConfig``. ``configs/base.yaml`` (the runners' default config) overrides them.

    cfg = load_config("configs/base.yaml")             # file -> Config (validated)
    run_cfg = legacy_dict_to_config(plan_run, base=cfg) # overlay one plan run
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path

from src.utils import ConfigurationError, get_logger

from .models import ModelConfig
from .paths import PathConfig, get_default_paths
from .training import EvaluationConfig, TrainingConfig

logger = get_logger(__name__)

_SECTIONS = ("model", "training", "evaluation", "paths")


class Config:
    """Global configuration container."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        training: TrainingConfig | None = None,
        evaluation: EvaluationConfig | None = None,
        paths: PathConfig | None = None,
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.paths = paths or get_default_paths()

    def to_dict(self) -> dict:
        """Nested, JSON-serialisable dict (paths as POSIX strings), e.g. for experiment_meta.json."""
        return {
            "model": _jsonable(asdict(self.model)),
            "training": _jsonable(asdict(self.training)),
            "evaluation": _jsonable(asdict(self.evaluation)),
            "paths": _jsonable(asdict(self.paths)),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  model={self.model},\n"
            f"  training={self.training},\n"
            f"  evaluation={self.evaluation},\n"
            f"  paths={self.paths}\n"
            f")"
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from a JSON/YAML file, or return the defaults.

    The file holds up to four sections (``model``, ``training``, ``evaluation``,
    ``paths``) whose keys are the dataclass fields; missing keys keep their
    defaults and unknown keys raise ``ConfigurationError``.

    Args:
        config_path: Path to a ``.json``, ``.yaml`` or ``.yml`` file (optional).

    Returns:
        Validated Config instance.
    """
    if config_path is None:
        cfg = Config()
        _validate_loaded_config(cfg)
        return cfg

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        cfg = _load_json_config(config_path)
    elif suffix in (".yaml", ".yml"):
        cfg = _load_yaml_config(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    _validate_loaded_config(cfg)
    return cfg


def _validate_loaded_config(config: Config) -> None:
    from .validation import validate_config

    validate_config(config)


def _build_config_from_dict(data: dict) -> Config:
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file must contain a mapping, got {type(data).__name__}")
    unknown = sorted(set(data) - set(_SECTIONS))
    if unknown:
        raise ConfigurationError(
            f"Unknown config section(s) {unknown}; expected a subset of {list(_SECTIONS)}"
        )
    for name, section in data.items():
        if section is not None and not isinstance(section, dict):
            raise ConfigurationError(f"Config section '{name}' must be a mapping")
    return apply_overrides(Config(), {k: v or {} for k, v in data.items()}, strict=True)


def _load_json_config(path: Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return _build_config_from_dict(json.load(f))


def _load_yaml_config(path: Path) -> Config:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        ) from None
    with open(path, encoding="utf-8") as f:
        return _build_config_from_dict(yaml.safe_load(f) or {})


def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


def apply_overrides(cfg: Config, overrides: dict, *, strict: bool = False) -> Config:
    """Return a copy of ``cfg`` with ``overrides`` applied (``cfg`` is not modified).

    ``overrides`` may use flat keys (a plan run: ``{"model": "efficientnetb0",
    "freeze_layers": 0, "learning_rate": 2e-5}``), which are routed to every
    section that has a field of that name, and/or section mappings
    (``{"training": {"epochs": 50}, "paths": {"mes": {"root": ...}}}``).

    Every overridden section is rebuilt with ``dataclasses.replace``, so its
    ``__post_init__`` validation runs again.

    Args:
        cfg:       Base configuration (e.g. ``load_config(path)`` or ``Config()``).
        overrides: Keys to override.
        strict:    Raise ``ConfigurationError`` on unknown keys instead of
                   logging a warning and ignoring them.

    Returns:
        New Config.
    """
    names = {
        "model": _field_names(ModelConfig),
        "training": _field_names(TrainingConfig),
        "evaluation": _field_names(EvaluationConfig),
        "paths": _field_names(PathConfig),
    }
    kw: dict[str, dict] = {s: {} for s in _SECTIONS}
    unknown: list[str] = []

    for key, value in overrides.items():
        if key in _SECTIONS and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key in names[key]:
                    kw[key][sub_key] = sub_value
                else:
                    unknown.append(f"{key}.{sub_key}")
            continue
        targets = [s for s in ("model", "training", "evaluation") if key in names[s]]
        if not targets:
            unknown.append(key)
        for section in targets:
            kw[section][key] = value

    if unknown:
        msg = f"Unknown config key(s) {sorted(unknown)}"
        if strict:
            raise ConfigurationError(msg)
        logger.warning(f"{msg}: ignored")

    new = deepcopy(cfg)
    if kw["model"]:
        new.model = replace(new.model, **kw["model"])
    if kw["training"]:
        new.training = replace(new.training, **kw["training"])
    if kw["evaluation"]:
        new.evaluation = replace(new.evaluation, **kw["evaluation"])
    if kw["paths"]:
        new.paths = replace(new.paths, **kw["paths"])
    return new


def legacy_dict_to_config(
    config_dict: dict, base: Config | None = None, *, strict: bool = False
) -> Config:
    """Convert a flat plan-run dict to a Config.

    Starts from ``base`` (default: the dataclass defaults, ``Config()``) and
    overrides only the keys present in ``config_dict`` (see ``apply_overrides``
    for the accepted keys). Every ModelConfig / TrainingConfig /
    EvaluationConfig field can be set, including ``warmup_epochs``, ``min_lr``,
    ``use_amp``, ``device_id``, ``grad_clip_norm``, ``use_randaugment``,
    ``randaugment_m``, ``use_random_erasing``, ``random_erasing_p``,
    ``bootstrap_samples`` and ``confidence_interval``.

    Args:
        config_dict: Plan-run keys, e.g. ``{"model": ..., "freeze_layers": ...,
                     "learning_rate": ...}``.
        base:        Config to start from, e.g. ``load_config(args.config)``.
        strict:      Raise on unknown keys instead of warning.
    """
    return apply_overrides(base if base is not None else Config(), config_dict, strict=strict)
