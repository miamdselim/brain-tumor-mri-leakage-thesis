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

## Findings

1. **Leaked images are answered by memory.** A pixel-matching 1-NN that learns nothing scores 99.3–99.7% on leaked test images but 82–88% on unseen ones. The CNN shows the same pattern (99.1–100% vs 94.4–97.9%).
2. **A random image-level split leaks even after exact duplicates are removed.** 34.07% of its test images still have a near duplicate in training, and the reported accuracy rises from 97.69% (C) to 98.64% (B).
3. **The official split is not easier overall.** Its unseen part is harder (94.41%), mainly glioma: 36 gliomas predicted as meningioma and 16 as no tumour. The official test folder appears to differ in distribution from the training folder, so its 96.12% mixes leakage (+) with distribution shift (−).
4. **Leaky validation is over-optimistic.** Under Protocol A, validation accuracy was 99.29% while test accuracy was 96.12%.
5. **A strong 1-NN baseline on clean data (87.84%)** suggests image appearance carries class information beyond tumour morphology (possible source bias); to be tested in WP3.

## Limitations

- One run per protocol. The same group-aware split gave 98.32% in v2 and 97.69% here, so run-to-run variation is about ±0.6 pp; the B − C gap (0.94 pp) needs multiple seeds. The leaked-vs-unseen comparison is within one model and is the stronger evidence.
- Leaked and unseen subsets may differ in class mix; a class-stratified comparison is planned.
