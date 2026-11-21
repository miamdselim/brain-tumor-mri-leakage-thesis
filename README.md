# Leakage-Free Brain Tumor MRI Classification

Master's thesis project, School of Artificial Intelligence, Nanjing University of Information Science & Technology (NUIST).
Author: **Md. Selim Mia**, supervised by **Prof. Cao Ben**.

Public brain tumor MRI benchmarks contain exact and near-duplicate images. When duplicates fall on both sides of a
train/test split, a model can score highly by memorising rather than by learning tumour appearance. This project
measures that leakage, removes it with a group-aware split, and reports honest performance for a multi-backbone
CNN ensemble.

## Key results so far (v2, group-aware split, 954 held-out test images)

| Model | Test accuracy | 95% CI | Macro-F1 | MCC |
|---|---|---|---|---|
| **Ensemble (selected on validation)** | **97.80%** | 96.86–98.64 | 0.9781 | 0.9706 |
| Fusion (EfficientNetV2-B0 + ResNet50V2) | 98.32% | 97.48–99.06 | 0.9835 | 0.9776 |
| ResNet50V2 | 97.59% | 96.65–98.53 | 0.9761 | 0.9679 |
| EfficientNetV2-B0 | 96.96% | 95.70–98.01 | 0.9696 | 0.9596 |

- 36.25% of the official test images (580 of 1,600) have an exact or near duplicate in the official training set.
- 7,200 images as distributed; after removing 186 exact copies and 11 label-conflict images: 7,003 images in 5,364 duplicate groups; split 5,046 / 1,003 / 954 (train / val / test).
- Only 1 of 740 tumour test images was predicted as "no tumour" by the selected model.

Full numbers are in [`results/v2/results.json`](results/v2/results.json).

## Repository layout

```
notebooks/   Kaggle notebooks (run these)
  01_v2_leakage_free_ensemble.ipynb   de-duplication audit + 3 models + ensemble (paper results)
  02_wp1_leaky_vs_clean_split.ipynb   WP1: same model under official / random / group-aware splits
scripts/     the same notebooks as plain Python (# %% cells) + to_ipynb.py converter
results/     outputs of each finished run (results.json, tables, figures)
data/        how to obtain the dataset (images are not stored in git)
docs/        experiment log and thesis plan
references/  references.bib and references.ris (import into Zotero)
```

## Thesis work packages

| WP | Question | Status |
|---|---|---|
| WP1 | How much does leakage inflate accuracy? (official vs random vs group-aware split) | done — see [`results/wp1`](results/wp1/README.md) |
| WP3 | Does duplicate grouping approximate real patient-level splitting? (figshare dataset with patient IDs and tumour masks) | planned |
| WP2 | Do newer backbones (ConvNeXt V2, ViT) help under a leakage-free protocol? Multi-seed group CV and calibration | planned |

## How to reproduce

1. Open a notebook on Kaggle, attach the dataset `masoudnickparvar/brain-tumor-mri-dataset`.
2. Settings → Accelerator **GPU T4**, Internet **ON**.
3. *Save Version → Save & Run All (Commit)*. Download the output zip when it finishes.

Every notebook uses seed 42 and the identical duplicate-grouping settings, so the group-aware split is the same in
every experiment. `SMOKE_TEST=1` runs a fast end-to-end check on CPU.

## Data

Brain Tumor MRI Dataset, Version 2, by Masoud Nickparvar (Kaggle), licensed CC BY 4.0. See [`data/README.md`](data/README.md).

## Licence

Code: MIT (see `LICENSE`). The dataset keeps its own licence.
