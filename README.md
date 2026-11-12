# Annotation efficiency of endoscopy-specific pretraining in IBD

Learning curves (10–100 % of the labelled frames, 10 seeds) on two internal tasks: Mayo endoscopic
subscore (MES) grading in ulcerative colitis (4 classes) and ulcer detection in Crohn's disease
(binary). Backbones pretrained on GastroNet (endoscopy) are compared with ImageNet pretraining of
the same architecture (DINOv1) and with generic DINOv2 / DINOv3 models. LIMUC is used as a
within-dataset replication and as an external test in both directions.

## Install

```bash
conda env create -f environment.yml     # exact training environment (Python 3.10, torch 2.14 + CUDA 13.2)
conda activate data-efficiency
pip install -e ".[notebooks,dev]"
```

Without conda: install torch/torchvision for your CUDA from `https://download.pytorch.org/whl/`,
then `pip install -e ".[notebooks,dev]"`.

## Data

Internal frames are not shared. Expected layout (`data/` and `data_LIMUC/` are git-ignored):

```
data/
├── {mes,ulcer}/processed/{label}/{procedure}/{clip}/<frame>.jpg   # label: 0-3 (MES), 0-1 (ulcer)
├── {mes,ulcer}/splits/            # dataset_manifest.csv, split_info.json (create_manifest)
├── video_data.csv                 # optional patient linkage
└── assets/pretrained/             # GastroNet weights
```

`video_data.csv` has one row per procedure: `video_id` (procedure folder), `patient_id` (keyed
pseudonym; raw hospital numbers are rejected), `dataset` (`mes` or `ulcer`) and optionally
`date_of_procedure`, `age`, `gender`. With it, splits and statistics are patient-level; without it,
each procedure folder counts as a patient. Never commit it.

## Pipeline

```bash
# Manifests: patient-level stratified split (seed 42), 40 / 20 / 40 % train / val / test
python -m scripts.mes.create_manifest   --split-strategy stratified --val-ratio 0.2 --test-ratio 0.4
python -m scripts.ulcer.create_manifest --split-strategy stratified --val-ratio 0.2 --test-ratio 0.4

# LIMUC (Zenodo, CC BY 4.0): convert, then split 70 / 15 / 15 %
python -m scripts.mes.prepare_limuc --input-dir data_LIMUC/patient_based_classified_images --output-dir data_LIMUC/mes/processed
python -m scripts.mes.create_manifest --input-dir data_LIMUC/mes/processed --output-dir data_LIMUC/mes/splits --split-strategy stratified

# Cohort tables (results/inventory/, local only) and frame cache (used by the loaders when complete)
python -m scripts.tools.cohort_inventory
python -m scripts.tools.build_frame_cache --task mes      # also: ulcer, limuc; --img-size 336 for ViT-B

# Training: ulcer, MES, then LIMUC (results/<task>/data_efficiency*, checkpoints in output/)
python -m scripts.run_all --limuc --dry-run
python -m scripts.run_all --limuc --parallel-jobs 8       # Ctrl+C stops, --resume continues

# Exploratory ViT-B runs (results/<task>/data_efficiency*_vitb)
python -m scripts.run_all --limuc --vitb-only --parallel-jobs 3

# External test, inference only: internal runs on the LIMUC test split and the reverse
python -m scripts.mes.evaluate_external                   # --vitb: the ViT-B runs
```

`--split-strategy temporal --test-cutoff YYYY-MM-DD` holds out the most recent patients instead.
`--parallel-jobs` also sets the data-loader workers per job (8 jobs: 1 worker, 3 jobs: 2, as in the
published runs); the augmentation streams depend on it, so keep these values to reproduce the
tables exactly.

## Models and training

| Key | Backbone | Pretraining | Source |
|---|---|---|---|
| `resnet50_imagenet`, `resnet50_5M` | ResNet-50 | ImageNet / GastroNet-5M, DINOv1 | Meta DINO; GastroNet |
| `vits16_imagenet`, `vits16_gastronet` | ViT-S/16 | ImageNet / GastroNet-5M, DINOv1 | Meta DINO (torch.hub); GastroNet |
| `resnet50_200K`, `resnet50_1M` | ResNet-50 | GastroNet subsets, DINOv1 | GastroNet |
| `efficientnetb0` | EfficientNet-B0 | ImageNet, supervised | torchvision |
| `vits14_dinov2` | ViT-S/14 | LVD-142M, DINOv2 | timm (Apache-2.0) |
| `vits16_dinov3` | ViT-S/16 | LVD-1689M, DINOv3 | timm (DINOv3 licence) |
| `vitb14_gastronet`, `vitb14_dinov2` | ViT-B/14, 4 registers | GastroNet-5M / LVD-142M, DINOv2 | GastroNet; timm (Apache-2.0) |
| `vitb16_dinov3` | ViT-B/16 | LVD-1689M, DINOv3 | timm (DINOv3 licence) |

The ViT-B models are exploratory (added after the main analysis, 336 px, own plans and results
folders). GastroNet weights are available on request from the
[GastroNet provider](https://cortex.thetavision.nl/dataset-provider/listing/2/); place
`RN50_GastroNet-{5M,1M,200K}_DINOv1.pth`, `VITS_GastroNet-5M_DINOv1.pth` and `DINOv2.pth`
(GastroNet-5M, Gastroenterology 2025, doi:10.1053/j.gastro.2025.07.030) in
`data/assets/pretrained/`. Other weights download automatically.

Every model uses the GastroNet-5M fine-tuning protocol (`configs/base.yaml`): 256 px, Adam
(lr 1e-5, weight decay 1e-3), 10 warm-up epochs then cosine, ≤ 100 epochs, early stopping after 20
(validation AUROC), batch 32, gradient clip 10, mixed precision, full fine-tuning with a linear
head, class-balanced loss (MES: plus an active vs remission term), CLAHE and flip / rotation /
shift / shear / colour-jitter augmentation. ViTs add stochastic depth (0.1) and layer-wise
learning-rate decay (0.75 per block).

## Analysis

| Notebook | Content |
|---|---|
| `01_dataset_overview` | patients, procedures, clips, frames per split and grade; demographics |
| `02_training_monitoring` | plan progress, failures, effective protocol, training curves |
| `03_statistics` | paired seed × patient bootstrap, Holm families, AULC, label efficiency; external test |
| `04_results_tables` | descriptive results, operating points, per-grade recall, confusion matrices |

The notebooks need the local data, predictions or run folders; they are published with their
aggregate outputs. Shared helpers are in `src/analysis/`.

## Results

`results/` holds aggregate tables only: `results.csv` (mean ± SD over the seeds),
`results_per_seed.csv`, `experiment_meta.json` and `statistics/*.csv` (notebook 03). They were
produced before this public release, so their `git_sha` refers to the private development history.

The LIMUC split has 565 patient folders (395 / 85 / 85): the 564 patients of the release plus one
stray duplicate image counted as a patient; rebuilding `data_LIMUC/mes/processed` from the release
removes it. Ten pairs of identical LIMUC training images carry different labels.

## Tests

```bash
python -m pytest                     # CPU, synthetic data
python -m ruff check src scripts tests
```

## Data availability and licence

Internal frames, manifests, linkage, checkpoints and predictions are not shared. LIMUC is available
from its authors on Zenodo (CC BY 4.0). Pretrained weights keep their own licences. Code and
aggregate results: Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).
