# Data

Images are **not** stored in this repository.

## Main dataset (v2, WP1, WP2)

- **Brain Tumor MRI Dataset, Version 2** — Masoud Nickparvar, Kaggle
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 7,200 images (Training 5,600 / Testing 1,600; 1,800 per class: glioma, meningioma, notumor, pituitary)
- Licence: CC BY 4.0
- On Kaggle the notebooks find it automatically at `/kaggle/input/...`. Elsewhere, set the `DATA_ROOT` environment variable.

## Patient-level validation dataset (WP3, planned)

- **figshare brain tumor dataset** — Jun Cheng et al., 3,064 T1-weighted contrast-enhanced slices from 233 patients
  (glioma, meningioma, pituitary), with patient IDs and tumour masks.
- Used to test whether duplicate grouping approximates true patient-level splitting, and to score Grad-CAM against tumour masks.

## Split manifests

The exact group-aware split used for all reported results is in
`results/v2/tables/split_manifest.csv` (columns: src_path, label, official_split, md5, group, split).
To reproduce it, keep `SEED = 42`, `FINGERPRINT_SIZE = 224`, `PHASH_MAX_DIST = 8`, `NCC_MIN = 0.90`, 7 test folds and 6 validation folds.
