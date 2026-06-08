# Data Efficiency — Ulcer Detection & MES Scoring

Learning-curve experiments for two colonoscopy classification tasks:

- **Ulcer detection** — binary classification (ulcer / no ulcer) on still frames
- **MES scoring** — Mayo Endoscopic Score (0–3) multiclass classification

The goal is to measure how model performance varies with training set size, enabling comparison of architectures and pre-training strategies in low-data regimes.

---

## Table of Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Data Structure](#data-structure)
4. [Quick Start](#quick-start)
5. [Experiment Configuration](#experiment-configuration)
6. [Model Registry](#model-registry)
7. [Outputs](#outputs)
8. [Notebooks](#notebooks)

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU strongly recommended (CPU training is supported but very slow)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/data-efficiency-MES-Ulcer.git
cd data-efficiency-MES-Ulcer

# 2. Create a conda environment (recommended — avoids DLL issues on managed Windows machines)
conda create -n data-efficiency python=3.10 -y
conda activate data-efficiency

# 3. Install PyTorch with CUDA
# Pick the line that matches your CUDA version (check with: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU only
# Full list: https://pytorch.org/get-started/locally/

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install the project in editable mode (makes `src.` imports work from any directory)
pip install -e .
```

> **venv alternative (Linux / macOS / unmanaged Windows)**
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Linux/macOS
> .venv\Scripts\activate      # Windows
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt && pip install -e .
> ```

---

## Repository Structure

```
data-efficiency-MES-Ulcer/
│
├── src/                         # Core library
│   ├── config/                  # Configuration dataclasses, loader, validation
│   ├── data/                    # Dataset, DataLoader, transforms, manifest utilities
│   ├── models/                  # ClassifierModel (registry-driven backbone + head)
│   ├── evaluation/              # Metrics, bootstrap CI, plots, DeLong test, threshold sweep
│   ├── utils/                   # Logging, device helpers
│   └── visualization/           # GradCAM (CNN) and CLS attention map (ViT) explainability
│
├── scripts/
│   ├── run_all.py                   # Run ulcer then MES experiments sequentially
│   ├── ulcer/
│   │   ├── create_manifest.py       # Generate ulcer train/val/test splits
│   │   └── run_data_efficiency.py   # Ulcer detection learning curves
│   └── mes/
│       ├── create_manifest.py       # Generate MES train/val/test splits
│       └── run_data_efficiency.py   # MES scoring learning curves
│
├── configs/
│   ├── example.yaml                 # Reference config (all fields + defaults)
│   └── experiments/
│       └── data_efficiency.yaml     # Experiment plan (models, ratios, seeds)
│
├── notebooks/
│   ├── 01_data_overview.ipynb       # Dataset statistics and class distributions
│   ├── 02_training_monitoring.ipynb # Training curves (loss, AUROC per epoch)
│   ├── 03_learning_curves.ipynb     # Data efficiency learning curves
│   ├── 04_results_analysis.ipynb    # Statistical comparison, DeLong test
│   └── 05_explainability.ipynb      # GradCAM (CNN) / attention maps (ViT) — correct preds & misclassifications
│
├── data/                        # YOU PROVIDE THIS (see Data Structure below)
├── output/                      # Model checkpoints (auto-created)
├── results/                     # Metrics and figures (auto-created)
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Data Structure

### Image requirements

**Frames must be pre-cropped** to contain only the endoscopic field of view (no
scope border, no overlaid text, no black margins). The model receives the image
as-is — if the frame still contains the endoscope border or HUD overlays, those
artefacts will be part of the input and may degrade or bias the results.

### Folder layout

Both tasks use the **same convention**: `processed/{label}/{video_id}/{clip_id}/*.jpg`

```
data/
├── ulcer/
│   └── processed/
│       ├── 0/                        # label 0 = no ulcer
│       │   └── vid_03_XXXX/          # patient video  (= patient_id)
│       │       └── normal_N/         # annotated clip (= clip_id)
│       │           └── frame_XXXXXX.jpg
│       └── 1/                        # label 1 = ulcer
│           └── vid_03_XXXX/
│               └── ulcer_N/
│                   └── frame_XXXXXX.jpg
│
├── mes/
│   └── processed/
│       ├── 0/                        # Mayo score 0
│       │   └── vid_03_XXXX/
│       │       └── mayo_NNN/
│       │           └── frame_XXXXXX.jpg
│       ├── 1/                        # Mayo score 1
│       ├── 2/                        # Mayo score 2
│       └── 3/                        # Mayo score 3
│
└── assets/
    └── pretrained/              # GastroNet weight files (optional — see Model Registry)
        ├── RN50_GastroNet-5M_DINOv1.pth
        ├── VITS_GastroNet-5M_DINOv1.pth
        └── DINOv2.pth
```

### Generate the manifest

Scan the raw folder and generate the train/val/test split CSVs (patient-level split):

```bash
python -m scripts.ulcer.create_manifest
python -m scripts.mes.create_manifest
```

This produces `data/{task}/splits/dataset_manifest.csv`.

Optional flags: `--val-ratio 0.15 --test-ratio 0.15 --seed 42`

### Manifest format

| Column | Type | Description |
|--------|------|-------------|
| `filepath` | str | Relative path to the frame from `data/{task}/processed/` |
| `label` | int | Class label (0/1 for ulcer ; 0/1/2/3 for MES) |
| `split` | str | `train`, `val`, or `test` |
| `patient_id` | str | Patient identifier — splits are done at patient level to avoid leakage |
| `video_id` | str | Clip identifier — used for clip-level metric aggregation |

```csv
filepath,label,split,patient_id,video_id
1/vid_03_1448/ulcer_1/frame_000060.jpg,1,train,vid_03_1448,vid_03_1448__ulcer_1
0/vid_03_1239/normal_1/frame_000120.jpg,0,val,vid_03_1239,vid_03_1239__normal_1
```

---

## Quick Start

### 1. Generate manifests

```bash
python -m scripts.ulcer.create_manifest
python -m scripts.mes.create_manifest
```

### 2. Dry-run to verify the experiment plan

```bash
python -m scripts.ulcer.run_data_efficiency --dry-run
python -m scripts.mes.run_data_efficiency --dry-run
```

### 3. Smoke test (2 runs, 5 epochs)

```bash
python -m scripts.ulcer.run_data_efficiency --subset-ratios 0.1 0.5 --seeds 42 --max-runs 2 --epochs 5
python -m scripts.mes.run_data_efficiency   --subset-ratios 0.1 0.5 --seeds 42 --max-runs 2 --epochs 5
```

### 4. Run both tasks sequentially

```bash
# Forwards all CLI flags to both ulcer and MES experiments in order
python -m scripts.run_all --plan configs/experiments/data_efficiency.yaml
python -m scripts.run_all --subset-ratios 0.1 0.5 1.0 --seeds 42 84 --epochs 50
```

### 5. Full experiment from plan file (single task)

```bash
python -m scripts.ulcer.run_data_efficiency --plan configs/experiments/data_efficiency.yaml
python -m scripts.mes.run_data_efficiency   --plan configs/experiments/data_efficiency.yaml
```

### 7. Single model, custom fractions

```bash
python -m scripts.ulcer.run_data_efficiency --model vits16_imagenet --subset-ratios 0.25 0.5 1.0 --seeds 42 84 128
```

### 8. Parallel execution

```bash
python -m scripts.ulcer.run_data_efficiency --num-workers 2 --batch-size 32
```

---

## Experiment Configuration

Edit `configs/experiments/data_efficiency.yaml`:

```yaml
subset_ratios: [0.10, 0.25, 0.50, 0.75, 1.00]
head_types: ["linear"]          # linear | mlp1 | mlp2
seeds: [42, 84, 128]            # multiple seeds → confidence bands

head_lr_scales:
  linear: 1.0
  mlp1:   1.0
  mlp2:   1.0

runs:
  - model: vits16_imagenet
    freeze_layers: 0             # 0 = full fine-tune ; -1 = freeze backbone
    learning_rate: 1.0e-6

  - model: resnet50_imagenet
    freeze_layers: 0
    learning_rate: 1.0e-5

  - model: vits16_gastronet      # requires GastroNet weights in data/assets/pretrained/
    freeze_layers: 0
    learning_rate: 1.0e-6
```

**CLI flags (override the plan file):**

| Flag | Description |
|------|-------------|
| `--plan PATH` | Custom plan YAML |
| `--model NAME` | Run a single model from the plan |
| `--manifest PATH` | Override manifest CSV path |
| `--subset-ratios F [F ...]` | Override training fractions |
| `--seeds N [N ...]` | Override random seeds |
| `--epochs N` | Override number of epochs |
| `--max-runs N` | Stop after N runs (smoke test) |
| `--num-workers N` | Parallel jobs (default: 1 = sequential) |
| `--batch-size N` | Override batch size |
| `--dry-run` | Print plan and exit without training |

---

## Model Registry

| Key | Architecture | Pre-training | Method |
|-----|-------------|--------------|--------|
| `resnet18` | ResNet-18 | ImageNet | Supervised |
| `resnet50_imagenet_sup` | ResNet-50 | ImageNet | Supervised |
| `resnet50_imagenet` | ResNet-50 | ImageNet | DINOv1 |
| `resnet50_gastronet` | ResNet-50 | GastroNet-5M | DINOv1 |
| `resnet50_1M` | ResNet-50 | GastroNet-1M | DINOv1 |
| `resnet50_5M` | ResNet-50 | GastroNet-5M | DINOv1 |
| `resnet50_200K` | ResNet-50 | GastroNet-200K | DINOv1 |
| `efficientnetb0` | EfficientNet-B0 | ImageNet | Supervised |
| `efficientnetb1` | EfficientNet-B1 | ImageNet | Supervised |
| `efficientnetb4` | EfficientNet-B4 | ImageNet | Supervised |
| `vitb16_imagenet_sup` | ViT-Base/16 | ImageNet | Supervised |
| `vitb16_imagenet` | ViT-Base/16 | ImageNet | DINOv1 |
| `vitb16_gastronet` | ViT-Base/16 | GastroNet-5M | DINOv2 |
| `vits16_imagenet_hf` | ViT-Small/16 | ImageNet | Supervised (timm) |
| `vits16_imagenet` | ViT-Small/16 | ImageNet | DINOv1 |
| `vits16_gastronet` | ViT-Small/16 | GastroNet-5M | DINOv1 |

### GastroNet weights

Place weight files in `data/assets/pretrained/`:

| File | Model keys |
|------|-----------|
| `RN50_GastroNet-5M_DINOv1.pth` | `resnet50_gastronet`, `resnet50_5M` |
| `RN50_GastroNet-1M_DINOv1.pth` | `resnet50_1M` |
| `RN50_GastroNet-200K_DINOv1.pth` | `resnet50_200K` |
| `VITS_GastroNet-5M_DINOv1.pth` | `vits16_gastronet` |
| `DINOv2.pth` | `vitb16_gastronet` |

ImageNet-pretrained backbones download automatically via `torch.hub` or `timm`.

---

## Outputs

```
results/
├── ulcer/data_efficiency/
│   ├── results.csv              # Aggregated: mean±std per model×fraction
│   ├── results_per_seed.csv     # Raw: one row per model×fraction×seed
│   ├── experiment_meta.json
│   ├── figures/
│   │   ├── learning_curve_f1.png
│   │   ├── learning_curve_auroc.png
│   │   └── roc_<run_name>.png
│   └── predictions/
│       └── <run_name>/
│           ├── test_labels.npy
│           └── test_probs.npy
│
└── mes/data_efficiency/
    ├── results.csv
    ├── results_per_seed.csv
    ├── experiment_meta.json
    ├── figures/
    │   ├── learning_curve_f1.png
    │   ├── learning_curve_auroc.png
    │   ├── roc_<run_name>.png
    │   └── cm_<run_name>.png    # confusion matrix
    └── predictions/
        └── <run_name>/
            ├── test_labels.npy
            └── test_probs.npy   # shape (N, num_classes)
```

**Key columns — `results.csv` (ulcer)**

| Column | Description |
|--------|-------------|
| `model` | Backbone key |
| `pct_data` | Training fraction (%) |
| `f1_mean` / `f1_std` | Frame-level F1 across seeds |
| `auroc_mean` / `auroc_std` | Frame-level AUROC across seeds |
| `tuned_sensitivity_mean` | Sensitivity at threshold tuned on val set |
| `tuned_specificity_mean` | Specificity at threshold tuned on val set |
| `clip_f1_mean` / `clip_auroc_mean` | Clip-level metrics |

**Additional columns — `results.csv` (MES)**

| Column | Description |
|--------|-------------|
| `micro_f1_mean` / `macro_f1_mean` | Micro / macro F1 |
| `cli_sens_mean` / `cli_spec_mean` | Clinical binary (Active = Mayo ≥ 2 vs Remission) |
| `sens_mayo0` … `sens_mayo3` | Per-class sensitivity |

---

## Notebooks

```bash
jupyter lab notebooks/
```

| Notebook | What it shows |
|----------|--------------|
| `01_data_overview.ipynb` | Frame counts, class distributions, sample images |
| `02_training_monitoring.ipynb` | Loss and AUROC curves per epoch |
| `03_learning_curves.ipynb` | AUROC and F1 vs training fraction per model |
| `04_results_analysis.ipynb` | Model ranking, DeLong test, per-class sensitivity |
| `05_explainability.ipynb` | GradCAM (CNN) and CLS attention maps (ViT) — correct predictions and misclassifications.  Auto-selects the most recent checkpoint for `TASK` / `MODEL`. |
