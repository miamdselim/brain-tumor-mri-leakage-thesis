# ============================================================
# CELL 12 — Extra analysis (CPU, < 2 min): is the leakage effect real after controlling for class?
#   (a) leaked vs unseen accuracy per class + Cochran–Mantel–Haenszel test stratified by class
#   (b) unseen accuracy re-weighted to the class mix of the leaked images
#   (c) 1-NN memoriser accuracy per class (source-bias check)
#   (d) official-test glioma errors
# Works in two ways:
#   • same session, right after CELL 11; or
#   • after a restart: run CELLS 1–6 only (≈ 3 min, no training), then this cell.
#     CNN predictions are then loaded from wp1_outputs/tables/test_probs_protocol_*.npy if they still exist.
# ============================================================
if "CNN" not in globals() or not all(k in CNN for k in SPLITS):
    CNN = {}
    for k in SPLITS:
        f = os.path.join(TAB_DIR, f"test_probs_protocol_{k}.npy")
        if os.path.exists(f):
            CNN[k] = {"probs": np.load(f)}
HAVE_CNN = all(k in CNN for k in SPLITS)
print("CNN predictions available:", HAVE_CNN,
      "" if HAVE_CNN else "→ running the 1-NN parts only (CNN files were deleted with the old session)")
from scipy.stats import fisher_exact
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
except Exception:
    StratifiedTable = None

EXTRA = {"leaked_vs_unseen": {}, "nn_per_class": {}, "official_glioma": {}}
per_class_rows, test_rows = [], []

for k in ("A", "B"):
    te = SPLITS[k][SPLITS[k].split == "test"]
    y, leaked = label_ids(te), te["dup_in_dev"].to_numpy()
    models = ([("CNN", CNN[k]["probs"].argmax(1))] if HAVE_CNN else []) + [("1-NN", NN[k]["pred"])]
    for model, pred in models:
        ok = pred == y
        tables, w, acc_u = [], [], []
        for c, cn in enumerate(CLASSES):
            m = y == c
            a, b = int((ok & leaked & m).sum()), int((~ok & leaked & m).sum())
            cc, d = int((ok & ~leaked & m).sum()), int((~ok & ~leaked & m).sum())
            per_class_rows.append({
                "Protocol": k, "Model": model, "Class": CLASS_TITLES[cn],
                "Leaked n": a + b, "Leaked acc (%)": round(100 * a / (a + b), 2) if a + b else None,
                "Unseen n": cc + d, "Unseen acc (%)": round(100 * cc / (cc + d), 2) if cc + d else None})
            if a + b and cc + d:
                tables.append(np.array([[a, b], [cc, d]], dtype=float))
                w.append(a + b); acc_u.append(cc / (cc + d))
        acc_l = ok[leaked].mean()
        acc_un = ok[~leaked].mean()
        acc_un_adj = float(np.average(acc_u, weights=w))          # unseen accuracy at the leaked class mix
        _, p_fisher = fisher_exact([[int((ok & leaked).sum()), int((~ok & leaked).sum())],
                                    [int((ok & ~leaked).sum()), int((~ok & ~leaked).sum())]])
        p_cmh = float("nan")
        if StratifiedTable is not None and tables:
            try:
                p_cmh = float(StratifiedTable(tables).test_null_odds(correction=True).pvalue)
            except Exception as e:
                print("CMH failed:", e)
        r = {"leaked_acc": float(acc_l), "unseen_acc": float(acc_un),
             "unseen_acc_at_leaked_class_mix": acc_un_adj,
             "gap_pp": round(100 * (acc_l - acc_un), 2),
             "gap_class_adjusted_pp": round(100 * (acc_l - acc_un_adj), 2),
             "fisher_p": float(p_fisher), "cmh_p_stratified_by_class": p_cmh}
        EXTRA["leaked_vs_unseen"][f"{k}_{model}"] = r
        test_rows.append({"Protocol": PROTOCOLS[k], "Model": model,
                          "Leaked acc (%)": round(100 * acc_l, 2), "Unseen acc (%)": round(100 * acc_un, 2),
                          "Unseen acc, leaked class mix (%)": round(100 * acc_un_adj, 2),
                          "Gap (pp)": r["gap_pp"], "Class-adjusted gap (pp)": r["gap_class_adjusted_pp"],
                          "Fisher p": f"{p_fisher:.2e}", "CMH p (by class)": f"{p_cmh:.2e}"})

PER_CLASS = pd.DataFrame(per_class_rows)
LEAK_TESTS = pd.DataFrame(test_rows)
PER_CLASS.to_csv(os.path.join(TAB_DIR, "table_wp1_leaked_vs_unseen_per_class.csv"), index=False)
LEAK_TESTS.to_csv(os.path.join(TAB_DIR, "table_wp1_leaked_vs_unseen_tests.csv"), index=False)
print("(a)+(b) Leaked vs unseen, controlling for class\n")
print(LEAK_TESTS.to_string(index=False))
print("\nPer class:\n")
print(PER_CLASS.to_string(index=False))

# ---- (c) 1-NN memoriser accuracy per class ----
nn_rows = []
for k in SPLITS:
    te = SPLITS[k][SPLITS[k].split == "test"]
    y = label_ids(te)
    row = {"Protocol": PROTOCOLS[k]}
    for c, cn in enumerate(CLASSES):
        m = y == c
        row[CLASS_TITLES[cn]] = round(100 * (NN[k]["pred"][m] == c).mean(), 2)
        EXTRA["nn_per_class"].setdefault(k, {})[cn] = row[CLASS_TITLES[cn]]
    row["Overall"] = round(100 * (NN[k]["pred"] == y).mean(), 2)
    nn_rows.append(row)
NN_CLASS = pd.DataFrame(nn_rows)
NN_CLASS.to_csv(os.path.join(TAB_DIR, "table_wp1_nn_per_class.csv"), index=False)
print("\n(c) 1-NN memoriser accuracy per class (%)\n")
print(NN_CLASS.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.6))
wd = 0.26
for j, k in enumerate(SPLITS):
    vals = [NN_CLASS.iloc[j][CLASS_TITLES[c]] for c in CLASSES]
    b = ax.bar(np.arange(4) + (j - 1) * wd, vals, wd, label=PROTOCOLS[k], edgecolor="black")
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
ax.set_xticks(np.arange(4)); ax.set_xticklabels([CLASS_TITLES[c] for c in CLASSES])
ax.set_ylabel("1-NN accuracy (%)"); ax.set_ylim(0, 112)
ax.set_title("Pixel-matching 1-NN (no learning): accuracy per class")
ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fig_wp1_nn_per_class")

# ---- class-stratified leaked vs unseen figure ----
FIG_MODEL = "CNN" if HAVE_CNN else "1-NN"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for a, k in zip(ax, ("A", "B")):
    sub = PER_CLASS[(PER_CLASS.Protocol == k) & (PER_CLASS.Model == FIG_MODEL)]
    x = np.arange(len(sub))
    for j, (col, nm, colr) in enumerate([("Leaked acc (%)", "leaked", "#C44E52"),
                                          ("Unseen acc (%)", "unseen", "#55A868")]):
        v = sub[col].astype(float).fillna(0).to_numpy()
        n = sub["Leaked n" if nm == "leaked" else "Unseen n"].to_numpy()
        b = a.bar(x + (j - 0.5) * 0.38, v, 0.38, color=colr, edgecolor="black", label=nm)
        a.bar_label(b, labels=[f"{vv:.1f}\nn={nn}" if nn else "–" for vv, nn in zip(v, n)], fontsize=7, padding=2)
    a.set_xticks(x); a.set_xticklabels(sub["Class"]); a.set_ylim(0, 115)
    a.set_title(f"{PROTOCOLS[k]}: {FIG_MODEL} accuracy by class"); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("Test accuracy (%)"); ax[0].legend(loc="lower left")
plt.tight_layout()
save_fig("fig_wp1_leaked_vs_unseen_per_class")

# ---- (d) official-test glioma errors (CNN only) ----
if HAVE_CNN:
    teA = SPLITS["A"][SPLITS["A"].split == "test"].reset_index(drop=True)
    yA, pA = label_ids(teA), CNN["A"]["probs"].argmax(1)
    g = CIDX["glioma"]
    gm = yA == g
    wrong_g = np.where(gm & (pA != g))[0]
    simA = NN["A"]["sim"]
    EXTRA["official_glioma"] = {
        "test_gliomas": int(gm.sum()), "wrong": int(len(wrong_g)),
        "predicted_as": {CLASS_TITLES[CLASSES[c]]: int((pA[wrong_g] == c).sum()) for c in range(4) if c != g},
        "wrong_that_are_leaked": int(teA.loc[wrong_g, "dup_in_dev"].sum()),
        "median_nearest_sim_wrong": float(np.median(simA[wrong_g])) if len(wrong_g) else None,
        "median_nearest_sim_correct": float(np.median(simA[gm & (pA == g)])),
    }
    print("\n(d) Official-test gliomas:", json.dumps(EXTRA["official_glioma"], indent=2))

    if len(wrong_g):
        show = wrong_g[np.argsort(simA[wrong_g])][:8]
        cols = min(4, len(show)); rws = int(np.ceil(len(show) / cols))
        fig, ax = plt.subplots(rws, cols, figsize=(3.4 * cols, 3.9 * rws))
        ax = np.array(ax).reshape(-1)
        for a in ax:
            a.axis("off")
        for a, i in zip(ax, show):
            a.imshow(cv2.cvtColor(cv2.imread(teA.loc[i, "path"]), cv2.COLOR_BGR2RGB))
            a.set_title(f"Glioma → {CLASS_TITLES[CLASSES[pA[i]]]}\nnearest-train r = {simA[i]:.2f}", fontsize=9, color="red")
        plt.suptitle("Official test: misclassified gliomas (least similar to training first)", fontweight="bold")
        plt.tight_layout()
        save_fig("fig_wp1_official_glioma_errors")


if "RESULTS" not in globals():
    RESULTS = {}
RESULTS["extra_analysis"] = EXTRA
save_results()
zip_path = shutil.make_archive(os.path.join(CFG["WORK_DIR"], "wp1_outputs"), "zip", OUT_DIR)
print(f"\n📦 Updated: {zip_path}")
