# %% [markdown]
# # WP1 (final) — Analysis of the three seed runs
#
# **Author:** Md. Selim Mia — Department of Artificial Intelligence, Nanjing University of Information Science & Technology
#
# Combines the outputs of notebooks `03_wp1_final_seed42`, `…seed43` and `…seed44`. No GPU is needed (CPU, < 5 min).
#
# **How to run:** *Add Input → Your Work → Notebooks*, add the three finished seed notebooks, then *Run All*.
#
# **What it answers**
# 1. Accuracy of the same model under the three split protocols, as mean ± SD over three training seeds.
# 2. Is the B − C difference (random vs group-aware split) larger than run-to-run noise? (bootstrap CI over images and seeds)
# 3. Does a leaky validation set over-estimate test accuracy? (validation − test gap)
# 4. Are leaked test images answered better than unseen ones **after controlling for class**? (CNN and both memorisers;
#    class-reweighted gap with bootstrap CI, and Cochran–Mantel–Haenszel tests stratified by class)
# 5. Per-class accuracy of the memorisers and the CNN.

# %%
# ============================================================
# CELL 1 — Load every saved prediction
# ============================================================
import os, glob, json, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
                             roc_auc_score)
from statsmodels.stats.contingency_tables import StratifiedTable

INPUT_ROOT = os.environ.get("ANALYSIS_INPUT", "/kaggle/input")
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
OUT_DIR = os.path.join(WORK_DIR, "wp1_final_analysis")
FIG_DIR, TAB_DIR = os.path.join(OUT_DIR, "figures"), os.path.join(OUT_DIR, "tables")
for d in (FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)
BOOT = int(os.environ.get("BOOT", 2000))
RNG_SEED = 42

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TITLES = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}
PROTOCOLS = {"A": "A. Official split", "B": "B. Random image-level split", "C": "C. Group-aware split (ours)"}
PROB_COLS = [f"p_{c}" for c in CLASSES]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.family": "serif",
                     "font.size": 11, "axes.titleweight": "bold"})


def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close()
    print(f"   saved -> {path}")


files = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "predictions_protocol_*_seed*.csv"), recursive=True))
frames, seen = [], set()
for f in files:
    key = os.path.basename(f)
    if key in seen:            # the same run attached twice
        continue
    seen.add(key)
    frames.append(pd.read_csv(f))
assert frames, "No prediction files found. Attach the three finished seed notebooks with Add Input."
P = pd.concat(frames, ignore_index=True)
SEEDS = sorted(P.run_seed.unique())
print(f"Loaded {len(seen)} prediction files | seeds {SEEDS} | protocols {sorted(P.protocol.unique())}")
missing = [(k, s) for k in PROTOCOLS for s in SEEDS if not ((P.protocol == k) & (P.run_seed == s)).any()]
if missing:
    print("⚠️ Missing runs:", missing)
P["cnn_ok"] = P.cnn_pred == P.y
T = P[P.split == "test"].copy()
T["nn_pixel_ok"] = T.nn_pixel_pred == T.y
T["nn_feat_ok"] = T.nn_feat_pred == T.y
RESULTS = {"seeds": [int(s) for s in SEEDS], "n_files": len(seen)}

# %%
# ============================================================
# CELL 2 — Accuracy per protocol: every run, then mean ± SD over seeds
# ============================================================
run_rows = []
for (k, s), g in T.groupby(["protocol", "run_seed"]):
    v = P[(P.protocol == k) & (P.run_seed == s) & (P.split == "val")]
    probs = g[PROB_COLS].to_numpy()
    leaked = g.dup_in_dev.to_numpy()
    run_rows.append({
        "protocol": k, "seed": int(s), "test_n": len(g),
        "val_acc": v.cnn_ok.mean(), "test_acc": g.cnn_ok.mean(),
        "macro_f1": precision_recall_fscore_support(g.y, g.cnn_pred, average="macro", zero_division=0)[2],
        "mcc": matthews_corrcoef(g.y, g.cnn_pred),
        "auc": roc_auc_score(np.eye(4)[g.y], probs, average="macro", multi_class="ovr"),
        "tumours_missed": int(((g.y != CLASSES.index("notumor")) & (g.cnn_pred == CLASSES.index("notumor"))).sum()),
        "leaked_share": leaked.mean(),
        "cnn_leaked_acc": g.cnn_ok[leaked].mean() if leaked.any() else np.nan,
        "cnn_unseen_acc": g.cnn_ok[~leaked].mean(),
        "nn_pixel_acc": g.nn_pixel_ok.mean(), "nn_feat_acc": g.nn_feat_ok.mean(),
    })
RUNS = pd.DataFrame(run_rows)
RUNS.to_csv(os.path.join(TAB_DIR, "table_runs_all_seeds.csv"), index=False)

agg = []
for k, g in RUNS.groupby("protocol"):
    row = {"Protocol": PROTOCOLS[k], "Runs": len(g), "Test images": int(g.test_n.iloc[0]),
           "Leaked test (%)": round(100 * g.leaked_share.iloc[0], 2)}
    for col, name, scale, nd in [("test_acc", "CNN test acc (%)", 100, 2), ("val_acc", "CNN val acc (%)", 100, 2),
                                 ("macro_f1", "Macro-F1", 1, 4), ("mcc", "MCC", 1, 4), ("auc", "AUC", 1, 4),
                                 ("tumours_missed", "Tumours missed", 1, 1),
                                 ("nn_pixel_acc", "1-NN pixel (%)", 100, 2), ("nn_feat_acc", "1-NN ImageNet (%)", 100, 2)]:
        m, s = scale * g[col].mean(), scale * g[col].std(ddof=1) if len(g) > 1 else 0.0
        row[name] = f"{m:.{nd}f} ± {s:.{nd}f}"
    agg.append(row)
MAIN = pd.DataFrame(agg)
MAIN.to_csv(os.path.join(TAB_DIR, "table_main_mean_sd.csv"), index=False)
pd.set_option("display.width", 260)
print("Per run:\n", RUNS.round(4).to_string(index=False))
print("\nMean ± SD over seeds:\n", MAIN.to_string(index=False))
RESULTS["runs"] = RUNS.to_dict("records")
RESULTS["main_mean_sd"] = MAIN.to_dict("records")

# %%
# ============================================================
# CELL 3 — Is the difference between protocols larger than run-to-run noise?
#   Hierarchical bootstrap: resample test images within every run, average over seeds.
# ============================================================
rng = np.random.default_rng(RNG_SEED)
acc_runs = {(k, s): g.cnn_ok.to_numpy() for (k, s), g in T.groupby(["protocol", "run_seed"])}
unseen_runs = {(k, s): g.cnn_ok.to_numpy()[~g.dup_in_dev.to_numpy()] for (k, s), g in T.groupby(["protocol", "run_seed"])}


def boot_mean(arrs):
    """Bootstrap distribution of the seed-averaged accuracy."""
    out = np.empty(BOOT)
    for b in range(BOOT):
        out[b] = np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrs])
    return out


dist = {k: boot_mean([acc_runs[(k, s)] for s in SEEDS if (k, s) in acc_runs]) for k in PROTOCOLS}
dist["A_unseen"] = boot_mean([unseen_runs[("A", s)] for s in SEEDS if ("A", s) in unseen_runs])
cmp_rows = []
for a, b, label in [("B", "C", "B − C (random vs group-aware, same images)"),
                    ("A", "C", "A − C (official vs group-aware)"),
                    ("A_unseen", "C", "A unseen part − C")]:
    d = dist[a] - dist[b]
    point = (np.mean([acc_runs[(a, s)].mean() for s in SEEDS]) if a in PROTOCOLS
             else np.mean([unseen_runs[("A", s)].mean() for s in SEEDS])) - np.mean([acc_runs[(b, s)].mean() for s in SEEDS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    cmp_rows.append({"Comparison": label, "Difference (pp)": round(100 * point, 2),
                     "95% CI (pp)": f"{100*lo:.2f} to {100*hi:.2f}",
                     "Per-seed differences (pp)": ", ".join(
                         f"{100*((acc_runs[(a, s)] if a in PROTOCOLS else unseen_runs[('A', s)]).mean() - acc_runs[(b, s)].mean()):+.2f}"
                         for s in SEEDS),
                     "CI excludes 0": bool(lo > 0 or hi < 0)})
COMP = pd.DataFrame(cmp_rows)
COMP.to_csv(os.path.join(TAB_DIR, "table_protocol_differences.csv"), index=False)
print(COMP.to_string(index=False))
RESULTS["protocol_differences"] = COMP.to_dict("records")

# Validation optimism: validation accuracy minus test accuracy
vg = RUNS.assign(gap=100 * (RUNS.val_acc - RUNS.test_acc)).groupby("protocol").gap.agg(["mean", "std"]).round(2)
vg.index = [PROTOCOLS[k] for k in vg.index]
vg.columns = ["Val − test gap, mean (pp)", "SD (pp)"]
vg.to_csv(os.path.join(TAB_DIR, "table_validation_optimism.csv"))
print("\nValidation optimism:\n", vg.to_string())
RESULTS["validation_optimism"] = vg.reset_index().to_dict("records")

# %%
# ============================================================
# CELL 4 — Leaked vs unseen test images, controlling for class
#   Per image, CNN correctness is averaged over seeds (the same image is not counted three times).
#   Class-adjusted gap = leaked accuracy − unseen accuracy re-weighted to the class mix of the leaked images.
#   CI: bootstrap within each class × leaked/unseen cell. CMH test: 2×2 tables stratified by class.
# ============================================================
def class_adjusted(df, ok_col):
    leaked = df[df.dup_in_dev]
    w = leaked.label.value_counts()
    un = df[~df.dup_in_dev].groupby("label")[ok_col].mean()
    common = [c for c in w.index if c in un.index]
    return leaked[ok_col].mean(), df[~df.dup_in_dev][ok_col].mean(), float(np.average(un[common], weights=w[common]))


def boot_gap(df, ok_col):
    cells = {key: g[ok_col].to_numpy() for key, g in df.groupby(["label", "dup_in_dev"])}
    w = df[df.dup_in_dev].label.value_counts()
    out = np.empty(BOOT)
    for b in range(BOOT):
        res = {key: a[rng.integers(0, len(a), len(a))].mean() for key, a in cells.items()}
        lk = np.average([res[(c, True)] for c in w.index], weights=w.values)
        un = np.average([res[(c, False)] for c in w.index if (c, False) in res],
                        weights=[w[c] for c in w.index if (c, False) in res])
        out[b] = lk - un
    return np.percentile(out, [2.5, 97.5])


def cmh_p(df, ok_col):
    tabs = []
    for _, g in df.groupby("label"):
        ok = g[ok_col].astype(bool).to_numpy()
        lk = g.dup_in_dev.to_numpy()
        t = np.array([[(ok & lk).sum(), (~ok & lk).sum()], [(ok & ~lk).sum(), (~ok & ~lk).sum()]], float)
        if t[0].sum() and t[1].sum():
            tabs.append(t + 0.5)          # Haldane correction: some cells are 0 (every leaked image correct)
    return float(StratifiedTable(tabs).test_null_odds(correction=True).pvalue) if tabs else float("nan")


leak_rows, per_class_rows = [], []
for k in ("A", "B"):
    g = T[T.protocol == k]
    img = (g.groupby("src_path").agg(label=("label", "first"), dup_in_dev=("dup_in_dev", "first"),
                                     cnn_ok=("cnn_ok", "mean"), nn_pixel_ok=("nn_pixel_ok", "first"),
                                     nn_feat_ok=("nn_feat_ok", "first")).reset_index())
    for col, model in [("cnn_ok", f"Fusion CNN (mean of {len(SEEDS)} seeds)"),
                       ("nn_pixel_ok", "1-NN pixel"), ("nn_feat_ok", "1-NN ImageNet features")]:
        lk, un, un_adj = class_adjusted(img, col)
        lo, hi = boot_gap(img, col)
        p_seed = [cmh_p(T[(T.protocol == k) & (T.run_seed == s)], "cnn_ok") for s in SEEDS] if col == "cnn_ok" \
            else [cmh_p(img.assign(**{col: img[col].astype(bool)}), col)]
        leak_rows.append({"Protocol": PROTOCOLS[k], "Model": model,
                          "Leaked n": int(img.dup_in_dev.sum()), "Unseen n": int((~img.dup_in_dev).sum()),
                          "Leaked acc (%)": round(100 * lk, 2), "Unseen acc (%)": round(100 * un, 2),
                          "Raw gap (pp)": round(100 * (lk - un), 2),
                          "Class-adjusted gap (pp)": round(100 * (lk - un_adj), 2),
                          "95% CI (pp)": f"{100*lo:.2f} to {100*hi:.2f}",
                          "CMH p (by class)": ", ".join(f"{p:.1e}" for p in p_seed)})
        for c in CLASSES:
            m = img.label == c
            per_class_rows.append({"Protocol": k, "Model": model, "Class": CLASS_TITLES[c],
                                   "Leaked n": int((m & img.dup_in_dev).sum()),
                                   "Leaked acc (%)": round(100 * img[col][m & img.dup_in_dev].mean(), 2) if (m & img.dup_in_dev).any() else None,
                                   "Unseen n": int((m & ~img.dup_in_dev).sum()),
                                   "Unseen acc (%)": round(100 * img[col][m & ~img.dup_in_dev].mean(), 2)})
LEAK = pd.DataFrame(leak_rows)
LEAK_CLASS = pd.DataFrame(per_class_rows)
LEAK.to_csv(os.path.join(TAB_DIR, "table_leaked_vs_unseen_class_adjusted.csv"), index=False)
LEAK_CLASS.to_csv(os.path.join(TAB_DIR, "table_leaked_vs_unseen_per_class.csv"), index=False)
print(LEAK.to_string(index=False))
print("\nPer class:\n", LEAK_CLASS.to_string(index=False))
RESULTS["leaked_vs_unseen"] = LEAK.to_dict("records")
RESULTS["leaked_vs_unseen_per_class"] = LEAK_CLASS.to_dict("records")

# %%
# ============================================================
# CELL 5 — Per-class test accuracy (CNN mean over seeds, and both memorisers)
# ============================================================
pc_rows = []
for k in PROTOCOLS:
    g = T[T.protocol == k]
    for col, model in [("cnn_ok", "Fusion CNN"), ("nn_pixel_ok", "1-NN pixel"), ("nn_feat_ok", "1-NN ImageNet")]:
        row = {"Protocol": PROTOCOLS[k], "Model": model}
        for c in CLASSES:
            row[CLASS_TITLES[c]] = round(100 * g[g.label == c][col].mean(), 2)
        row["Overall"] = round(100 * g[col].mean(), 2)
        pc_rows.append(row)
PER_CLASS = pd.DataFrame(pc_rows)
PER_CLASS.to_csv(os.path.join(TAB_DIR, "table_per_class_accuracy.csv"), index=False)
print(PER_CLASS.to_string(index=False))
RESULTS["per_class_accuracy"] = PER_CLASS.to_dict("records")

# %%
# ============================================================
# CELL 6 — Figures
# ============================================================
keys = list(PROTOCOLS)
xl = [PROTOCOLS[k].replace(" split", "\nsplit") for k in keys]

# (1) accuracy by protocol: CNN (mean ± SD, every seed shown) and both memorisers
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.26
for j, (col, nm, colr) in enumerate([("test_acc", "Fusion CNN", "#4C72B0"),
                                     ("nn_feat_acc", "1-NN ImageNet features", "#8172B2"),
                                     ("nn_pixel_acc", "1-NN pixels", "#DD8452")]):
    m = [100 * RUNS[RUNS.protocol == k][col].mean() for k in keys]
    s = [100 * RUNS[RUNS.protocol == k][col].std(ddof=1) if len(SEEDS) > 1 else 0 for k in keys]
    bars = ax.bar(np.arange(3) + (j - 1) * w, m, w, yerr=s, capsize=4, color=colr, edgecolor="black", label=nm)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    if col == "test_acc":
        for i, k in enumerate(keys):
            v = 100 * RUNS[RUNS.protocol == k][col].to_numpy()
            ax.scatter(np.full(len(v), i + (j - 1) * w), v, color="black", s=12, zorder=3)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(xl)
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 118)
ax.set_title(f"Same model, three split protocols (mean ± SD over {len(SEEDS)} seeds)")
ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fig_wp1_final_accuracy_by_protocol")

# (2) leaked vs unseen, class-adjusted
fig, ax = plt.subplots(figsize=(10, 4.8))
labels, gaps, los, his = [], [], [], []
for _, r in LEAK.iterrows():
    lo, hi = [float(x) for x in r["95% CI (pp)"].split(" to ")]
    g_ = r["Class-adjusted gap (pp)"]
    labels.append(f"{r['Protocol'][:2]} {r['Model'].split(' (')[0]}")
    gaps.append(g_); los.append(max(g_ - lo, 0)); his.append(max(hi - g_, 0))
cols = ["#4C72B0", "#DD8452", "#8172B2"] * 2
b = ax.barh(np.arange(len(gaps)), gaps, xerr=[los, his], capsize=4, color=cols, edgecolor="black")
ax.bar_label(b, fmt="%.2f pp", padding=4, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_yticks(np.arange(len(gaps))); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("Leaked − unseen accuracy, class-adjusted (percentage points, 95% CI)")
ax.set_title("Leaked vs unseen test accuracy at equal class mix")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
save_fig("fig_wp1_final_leaked_vs_unseen_adjusted")

# (3) validation vs test accuracy
fig, ax = plt.subplots(figsize=(8, 4.6))
for j, (col, nm, colr) in enumerate([("val_acc", "Validation", "#55A868"), ("test_acc", "Test", "#4C72B0")]):
    m = [100 * RUNS[RUNS.protocol == k][col].mean() for k in keys]
    s = [100 * RUNS[RUNS.protocol == k][col].std(ddof=1) if len(SEEDS) > 1 else 0 for k in keys]
    bars = ax.bar(np.arange(3) + (j - 0.5) * 0.38, m, 0.38, yerr=s, capsize=4, color=colr, edgecolor="black", label=nm)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(xl)
ax.set_ylim(max(0, 100 * RUNS[["val_acc", "test_acc"]].min().min() - 4), 101.5)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Validation vs test accuracy")
ax.legend(loc="best"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fig_wp1_final_validation_vs_test")

# (4) per-class accuracy of the three models
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for a, model in zip(ax, ["Fusion CNN", "1-NN ImageNet", "1-NN pixel"]):
    sub = PER_CLASS[PER_CLASS.Model == model]
    for j, k in enumerate(keys):
        v = sub[sub.Protocol == PROTOCOLS[k]][[CLASS_TITLES[c] for c in CLASSES]].to_numpy().ravel()
        bb = a.bar(np.arange(4) + (j - 1) * 0.26, v, 0.26, label=PROTOCOLS[k], edgecolor="black")
        a.bar_label(bb, fmt="%.0f", fontsize=7, padding=1)
    a.set_xticks(np.arange(4)); a.set_xticklabels([CLASS_TITLES[c] for c in CLASSES], rotation=15)
    a.set_title(model); a.set_ylim(0, 112); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("Test accuracy (%)"); ax[0].legend(loc="lower left", fontsize=8)
plt.tight_layout()
save_fig("fig_wp1_final_per_class")

# %%
# ============================================================
# CELL 7 — Save everything
# ============================================================
with open(os.path.join(OUT_DIR, "results_wp1_final.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
zip_path = shutil.make_archive(os.path.join(WORK_DIR, "wp1_final_analysis"), "zip", OUT_DIR)
print("=" * 90)
print(MAIN[["Protocol", "CNN test acc (%)", "CNN val acc (%)", "1-NN pixel (%)", "1-NN ImageNet (%)"]].to_string(index=False))
print()
print(COMP[["Comparison", "Difference (pp)", "95% CI (pp)", "CI excludes 0"]].to_string(index=False))
print("=" * 90)
print(f"📦 {zip_path}")
