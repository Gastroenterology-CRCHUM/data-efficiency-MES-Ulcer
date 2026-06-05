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

from dataclasses import dataclass, field
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

        cfg.paths.mes.raw        # data/mes/raw
        cfg.paths.ulcer.splits   # data/ulcer/splits
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
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_ulcer_dir: Path = field(default_factory=lambda: Path("output/ulcer"))
    output_mes_dir: Path = field(default_factory=lambda: Path("output/mes"))
    ulcer_models_dir: Path = field(default_factory=lambda: Path("output/ulcer/models"))
    mes_models_dir: Path = field(default_factory=lambda: Path("output/mes/models"))

    # ── Results directories ───────────────────────────────────────────────────
    results_root_dir: Path = field(default_factory=lambda: Path("results"))
    results_ulcer_dir: Path = field(default_factory=lambda: Path("results/ulcer"))
    results_mes_dir: Path = field(default_factory=lambda: Path("results/mes"))
    results_ulcer_data_efficiency_dir: Path = field(
        default_factory=lambda: Path("results/ulcer/data_efficiency")
    )
    results_mes_data_efficiency_dir: Path = field(
        default_factory=lambda: Path("results/mes/data_efficiency")
    )

    def get_task_output_config(self, task: str) -> dict:
        task_key = task.strip().lower()
        mapping = {
            "ulcer_detection": {
                "output_dir": self.output_ulcer_dir,
                "models_dir": self.ulcer_models_dir,
                "results_dir": self.results_ulcer_dir,
            },
            "mes": {
                "output_dir": self.output_mes_dir,
                "models_dir": self.mes_models_dir,
                "results_dir": self.results_mes_dir,
            },
            "mes_data_efficiency": {
                "output_dir": self.output_mes_dir,
                "models_dir": self.mes_models_dir,
                "results_dir": self.results_mes_data_efficiency_dir,
            },
            "ulcer_data_efficiency": {
                "output_dir": self.output_ulcer_dir,
                "models_dir": self.ulcer_models_dir,
                "results_dir": self.results_ulcer_data_efficiency_dir,
            },
        }
        if task_key not in mapping:
            raise ConfigurationError(
                f"Unknown task '{task}'. Supported: {', '.join(sorted(mapping))}"
            )
        return mapping[task_key]


def get_default_paths() -> PathConfig:
    return PathConfig()
