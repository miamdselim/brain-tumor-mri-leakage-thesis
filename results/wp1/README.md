# WP1 results — how much does leakage inflate accuracy?

Run: 2026-11-22, Kaggle Tesla T4, interactive session. Model: Fusion (EfficientNetV2-B0 + ResNet50V2), v2 recipe, seed 42, one run per protocol.
Executed notebook: [`wp1_executed.ipynb`](wp1_executed.ipynb).

## Splits

| Protocol | Train | Val | Test | Test images with a near duplicate in train/val |
|---|---|---|---|---|
| A. Official split | 4,760 | 840 | 1,600 | 580 (36.25%) |
| B. Random image-level split (de-duplicated pool) | 5,046 | 1,003 | 954 | 325 (34.07%) |
| C. Group-aware split (ours) | 5,046 | 1,003 | 954 | 0 (0.00%) |

No official test image is an exact copy of a training image; all 580 leaks are near duplicates (pHash + NCC).
Protocol C reproduces the v2 split exactly (7,003 images, 5,364 groups, 5,046 / 1,003 / 954).

## Main result

| Protocol | CNN test accuracy (95% CI) | Macro-F1 | MCC | AUC | Tumours missed | 1-NN memoriser |
|---|---|---|---|---|---|---|
| A. Official | 96.12% (95.12–97.06) | 0.9608 | 0.9493 | 0.9928 | 17 | 88.44% |
| B. Random image-level | 98.64% (97.80–99.27) | 0.9865 | 0.9819 | 0.9989 | 0 | 91.51% |
| C. Group-aware (ours) | 97.69% (96.75–98.53) | 0.9769 | 0.9693 | 0.9985 | 1 | 87.84% |

## Leaked vs unseen test images (same model, same test set)

| Protocol | Subset | Images | CNN accuracy | 1-NN accuracy |
|---|---|---|---|---|
| A | leaked | 580 | 99.14% | 99.31% |
| A | unseen | 1,020 | 94.41% | 82.25% |
| B | leaked | 325 | 100.00% | 99.69% |
| B | unseen | 629 | 97.93% | 87.28% |
| C | unseen | 954 | 97.69% | 87.84% |

## Class-controlled check (CELL 12, 1-NN memoriser)

CNN per-image predictions were lost with the Kaggle session, so this check uses the 1-NN memoriser only.

| Protocol | Leaked acc | Unseen acc | Unseen acc at leaked class mix | Gap | Class-adjusted gap | CMH p (stratified by class) |
|---|---|---|---|---|---|---|
| A | 99.31% | 82.25% | 91.33% | 17.06 pp | **7.99 pp** | 7.1e-11 |
| B | 99.69% | 87.28% | 88.52% | 12.41 pp | **11.17 pp** | 1.8e-7 |

Leaked images are not spread evenly over classes. In the official test set, 314 of 400 no-tumour images (78.5%) are leaked, but only 35 of 400 gliomas.

1-NN accuracy per class (%):

| Protocol | Glioma | Meningioma | No tumour | Pituitary |
|---|---|---|---|---|
| A | 75.25 | 88.25 | 99.75 | 90.50 |
| B | 82.30 | 90.91 | 96.51 | 96.67 |
| C | 84.92 | 83.13 | 91.12 | 92.65 |

## Findings

1. **Leaked images are answered by memory, and this survives control for class.** A pixel-matching 1-NN that learns nothing scores 99.3–99.7% on leaked test images but 82–88% on unseen ones; after re-weighting to the same class mix the gap is still 8.0 pp (A) and 11.2 pp (B), p < 1e-6. The CNN shows the same raw pattern (99.1–100% vs 94.4–97.9%), but its raw gap is partly a class-mix effect (leaked images are mostly no-tumour, unseen ones mostly glioma); a class-controlled CNN comparison is pending.
2. **A random image-level split leaks even after exact duplicates are removed.** 34.07% of its test images still have a near duplicate in training, and the reported accuracy rises from 97.69% (C) to 98.64% (B).
3. **The official split is not easier overall.** Its unseen part is harder (94.41%), mainly glioma: 36 gliomas predicted as meningioma and 16 as no tumour. The official test folder appears to differ in distribution from the training folder, so its 96.12% mixes leakage (+) with distribution shift (−).
4. **Leaky validation is over-optimistic.** Under Protocol A, validation accuracy was 99.29% while test accuracy was 96.12%.
5. **A strong 1-NN baseline on clean data (87.84%)** shows that whole-image appearance already carries much class information. It is high in every class (83–93%), not only in no-tumour, so this check alone does not confirm a single-source bias; WP3 (patient-level figshare data) will test it further.

## Limitations

- One run per protocol. The same group-aware split gave 98.32% in v2 and 97.69% here, so run-to-run variation is about ±0.6 pp; the B − C gap (0.94 pp) needs multiple seeds. The leaked-vs-unseen comparison is within one model and is the stronger evidence.
- Leaked and unseen subsets differ strongly in class mix. The class-controlled test is done for the 1-NN; for the CNN it will be done in the WP2 multi-seed runs, which save every per-image prediction.
