# Thesis plan (Sep 2025 – Jul 2027)

Order of work: easiest first — WP1 → WP3 → WP2 — and write each chapter as its experiment finishes.

| Period | Work | Milestone |
|---|---|---|
| Nov 2026 | Setup: GitHub repo, experiment log, Zotero library, NUIST thesis template | |
| Nov–Dec 2026 | WP1: official vs random vs group-aware split; start Ch. 2 Literature Review | Proposal defense |
| Jan–Feb 2027 | WP3: figshare patient-level validation; image vs group vs patient split; Grad-CAM vs tumour masks; write Ch. 3 | |
| Feb–Mar 2027 | WP2: ConvNeXt V2 (ViT optional); 5-fold group CV × 3 seeds; calibration (ECE, temperature scaling); submit journal paper | Mid-term check |
| Mar–Apr 2027 | Write Ch. 4 Method and Ch. 5 Results, then Ch. 1, 6, 7 | |
| May 2027 | Plagiarism check, blind review | Blind review |
| Jun–Jul 2027 | Final presentation, defense, graduation | Final defense |

## Rules for every experiment

1. Same split manifest and seed 42 for every model, so results are comparable.
2. Save every `results.json` and log the run in `docs/experiment_log.md`.
3. Run long jobs with Kaggle *Save & Run All (Commit)* so they finish even if the browser closes.
4. Model choices are made on validation only; the test set is used once at the end.
5. Meet the supervisor every two weeks with the latest results.

## Thesis chapters

1. Introduction
2. Literature Review
3. Dataset, Duplicate Audit and Leakage-Free Protocol
4. Methodology
5. Experiments and Results (WP1, WP2, WP3)
6. Discussion
7. Conclusion and Future Work
