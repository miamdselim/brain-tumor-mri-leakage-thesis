# Experiment log

One row per finished run. Never delete a row; if a run was wrong, add a new row and explain in Notes.
Save each run's output zip under `results/<run-id>/`.

| Run ID | Date | Notebook | Split | Model(s) | Key settings | Test acc. (%) | Macro-F1 | Runtime | Output folder | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 2026-10 | brain_tumor_leakage_free | group-aware (seed 42) | Fusion, ResNet50V2, EffNetV2-B0 | 224 px, FT LR 3e-5, 25 epochs, no TTA | Fusion 96.75 · ResNet 97.48 · EffNet 93.92 | – | – | – | First leakage-free run; most errors meningioma ↔ glioma |
| v2 | 2026-11 | 01_v2_leakage_free_ensemble | group-aware (seed 42) | Ensemble of 3 | 288 px, warm-up + cosine 5e-5, 35 FT epochs, label smoothing 0.1, TTA flip | Ensemble 97.80 (CI 96.86–98.64) · Fusion 98.32 | 0.9781 | ≈2–2.5 h, T4 | results/v2 | Paper results; selected on validation (98.80%) |
| wp1 | 2026-11-22 | 02_wp1_leaky_vs_clean_split | A official · B random · C group-aware | Fusion | v2 recipe, seed 42, 1 run each | A 96.12 · B 98.64 · C 97.69 | A 0.9608 · B 0.9865 · C 0.9769 | ≈3.8 h, T4 | results/wp1 | Leaked test images: CNN 99.1–100% vs unseen 94.4–97.9%; 1-NN 99.3–99.7% vs 82–88% |

| wp1-final | | 03_wp1_final_seed42/43/44 + 04 | A · B · C | Fusion | v2 recipe, training seeds 42/43/44, split seed 42 | | | ≈4 h per seed | results/wp1_final | Fill in after the runs |

## Template for a new row

| Run ID | Date | Notebook | Split | Model(s) | Key settings | Test acc. (%) | Macro-F1 | Runtime | Output folder | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| | YYYY-MM-DD | | | | | | | | results/ | |
