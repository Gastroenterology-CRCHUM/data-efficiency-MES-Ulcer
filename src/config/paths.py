"""Path configuration for all project directories.

Data layout
-----------
data/
├── ulcer/
│   ├── processed/   ← pre-cropped frames (YOU PROVIDE)
│   └── splits/      ← train/val/test manifest (auto-generated)
├── mes/
│   ├── processed/
│   └── splits/
└── assets/
    └── pretrained/  ← GastroNet weight files (optional)

output/
├── ulcer/models/
└── mes/models/

results/
├── ulcer/data_efficiency/
└── mes/data_efficiency/
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from src.utils import ConfigurationError

# ============================================================================
# Per-task path groups
# ============================================================================


@dataclass
class MesPaths:
    """Data directories for the MES pipeline."""

    root: Path = field(default_factory=lambda: Path("data/mes"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def splits(self) -> Path:
        return self.root / "splits"


@dataclass
class UlcerPaths:
    """Data directories for the ulcer detection pipeline."""

    root: Path = field(default_factory=lambda: Path("data/ulcer"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def splits(self) -> Path:
        return self.root / "splits"


# ============================================================================
# Central config
# ============================================================================


@dataclass
class PathConfig:
    """Central path configuration for all project directories.

    Usage::

        cfg.paths.mes.processed  # data/mes/processed  (frames the manifest points to)
        cfg.paths.ulcer.splits   # data/ulcer/splits
        cfg.paths.mes_models_dir # output/mes/models

    Values loaded from YAML/JSON are coerced in ``__post_init__``: every
    directory field becomes a ``Path``, and ``mes`` / ``ulcer`` accept a
    ``MesPaths`` / ``UlcerPaths``, a ``{"root": ...}`` mapping or a root path.
    """

    # ── Per-task data paths ───────────────────────────────────────────────────
    mes: MesPaths = field(default_factory=MesPaths)
    ulcer: UlcerPaths = field(default_factory=UlcerPaths)

    # ── Flat aliases ──────────────────────────────────────────────────────────
    @property
    def mes_processed_dir(self) -> Path:  return self.mes.processed
    @property
    def mes_splits_dir(self) -> Path:     return self.mes.splits
    @property
    def ulcer_processed_dir(self) -> Path: return self.ulcer.processed
    @property
    def ulcer_splits_dir(self) -> Path:   return self.ulcer.splits

    # ── Shared assets ─────────────────────────────────────────────────────────
    pretrained_dir: Path = field(default_factory=lambda: Path("data/assets/pretrained"))

    # ── Output directories ────────────────────────────────────────────────────
    output_ulcer_dir: Path = field(default_factory=lambda: Path("output/ulcer"))
    output_mes_dir: Path = field(default_factory=lambda: Path("output/mes"))
    ulcer_models_dir: Path = field(default_factory=lambda: Path("output/ulcer/models"))
    mes_models_dir: Path = field(default_factory=lambda: Path("output/mes/models"))

    # ── Results directories ───────────────────────────────────────────────────
    results_ulcer_dir: Path = field(default_factory=lambda: Path("results/ulcer"))
    results_mes_dir: Path = field(default_factory=lambda: Path("results/mes"))
    results_ulcer_data_efficiency_dir: Path = field(
        default_factory=lambda: Path("results/ulcer/data_efficiency")
    )
    results_mes_data_efficiency_dir: Path = field(
        default_factory=lambda: Path("results/mes/data_efficiency")
    )

    def __post_init__(self) -> None:
        self.mes = _coerce_task_paths(self.mes, MesPaths, "mes")
        self.ulcer = _coerce_task_paths(self.ulcer, UlcerPaths, "ulcer")
        for f in fields(self):
            if f.name in ("mes", "ulcer"):
                continue
            value = getattr(self, f.name)
            if not isinstance(value, (str, Path)):
                raise ConfigurationError(
                    f"paths.{f.name} must be a path string, got {type(value).__name__}: {value!r}"
                )
            setattr(self, f.name, Path(value))



def _coerce_task_paths(value, cls: type, name: str):
    """Build ``cls`` (MesPaths / UlcerPaths) from an instance, a mapping or a root path."""
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigurationError(
                f"paths.{name}: unknown key(s) {unknown}; allowed: {sorted(allowed)}"
            )
        return cls(**value)
    if isinstance(value, (str, Path)):
        return cls(root=Path(value))
    raise ConfigurationError(
        f"paths.{name} must be a mapping or a path, got {type(value).__name__}: {value!r}"
    )


def get_default_paths() -> PathConfig:
    return PathConfig()
