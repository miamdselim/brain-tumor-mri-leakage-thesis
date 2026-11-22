# %% [markdown]
# # WP1 (final) — Leaky vs clean splits, seed 44
#
# **Author:** Md. Selim Mia — Department of Artificial Intelligence, Nanjing University of Information Science & Technology
#
# This is one of **three identical notebooks** (training seeds 42, 43 and 44). The data splits are the same in all three
# (split seed 42); only the random initialisation, shuffling and augmentation change. Running the experiment three times
# shows which differences are real and which are run-to-run noise.
#
# | Protocol | How train / val / test are formed | Leakage |
# |---|---|---|
# | **A. Official split** | the dataset's own Training / Testing folders (7,200 images as distributed); val = random 15 % of Training | near duplicates cross into test (36.25 %) |
# | **B. Random image-level split** | de-duplicated pool (7,003 images), stratified random split by image, same sizes as C | near duplicates cross into test (34.07 %) |
# | **C. Group-aware split (ours)** | de-duplicated pool, StratifiedGroupKFold on duplicate groups | none, verified |
#
# **Models compared under every protocol**
# 1. **Fusion CNN** (EfficientNetV2-B0 + ResNet50V2) with the unchanged v2 recipe.
# 2. **1-NN memoriser on pixels**: copies the label of the most similar training image (NCC, 64×64).
# 3. **1-NN memoriser on ImageNet features**: the same, using frozen ResNet50V2 features (no training on MRI).
#
# Every test and validation prediction is saved per image in `predictions/`, so the full analysis
# (notebook 04: mean ± SD over seeds, class-controlled leaked-vs-unseen tests, confidence intervals) can be run on CPU later.
#
# **How to run:** GPU T4, Internet ON, *Save Version → Save & Run All (Commit)*. About 4 hours. Nothing needs to be edited.
# %%
# ============================================================
# CELL 1 — Configuration and imports
# ============================================================
import os, sys, gc, json, time, random, hashlib, shutil, warnings, platform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = os.environ.get("RUN_SEED", "44")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dct
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef, roc_auc_score, log_loss,
)

import tensorflow as tf
import keras
from keras import layers

CFG = {
    # ---- paths ----
    "DATA_ROOT": os.environ.get(
        "DATA_ROOT", "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"),
    "WORK_DIR": os.environ.get("WORK_DIR", "/kaggle/working"),
    # ---- reproducibility ----
    "SEED": 42,          # data split seed — fixed, so every run uses identical splits
    "RUN_SEED": int(os.environ.get("RUN_SEED", 44)),  # training seed — the ONLY difference between the three notebooks
    # ---- images ----
    "IMG_SIZE": 288,
    "FINGERPRINT_SIZE": 224,  # same as v1/v2 -> identical duplicate groups and identical Protocol C split
    "BATCH_SIZE": 32,
    # ---- de-duplication ----
    "PHASH_MAX_DIST": 8,
    "NCC_MIN": 0.90,
    # ---- splits ----
    "TEST_FOLDS": 7,
    "VAL_FOLDS": 6,
    "OFFICIAL_VAL_FRACTION": 0.15,
    # ---- model: the v2 fusion model ----
    "BACKBONES": ("effnetv2b0", "resnet50v2"),
    # ---- training (identical to v2) ----
    "HEAD_EPOCHS": 12,
    "HEAD_LR": 1e-3,
    "FT_EPOCHS": 35,
    "FT_LR": 5e-5,
    "WARMUP_EPOCHS": 1,
    "LABEL_SMOOTHING": 0.1,
    "HEAD_PATIENCE": 4,
    "FT_PATIENCE": 8,
    # ---- inference ----
    "TTA": True,
    "BOOTSTRAP_ROUNDS": 1000,
    # ---- misc ----
    "MIXED_PRECISION": True,
    "N_JOBS": -1,
    "SMOKE_TEST": os.environ.get("SMOKE_TEST", "0") == "1",
}
if CFG["SMOKE_TEST"]:
    CFG.update(IMG_SIZE=96, BATCH_SIZE=8, HEAD_EPOCHS=1, FT_EPOCHS=1, BOOTSTRAP_ROUNDS=50)
    if os.environ.get("SMOKE_SMALL_MODEL") == "1":   # low-RAM test machines only
        CFG.update(BACKBONES=("effnetv2b0",))

SEED, IMG, FP = CFG["SEED"], CFG["IMG_SIZE"], CFG["FINGERPRINT_SIZE"]
RUN_SEED = CFG["RUN_SEED"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TITLES = {"glioma": "Glioma", "meningioma": "Meningioma",
                "notumor": "No tumor", "pituitary": "Pituitary"}
CLASS_MAP = {
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "notumor": "notumor", "no_tumor": "notumor", "no": "notumor",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
}
CIDX = {c: i for i, c in enumerate(CLASSES)}
PROTOCOLS = {
    "A": "A. Official split",
    "B": "B. Random image-level split",
    "C": "C. Group-aware split (ours)",
}

RUN_NAME = f"wp1_final_seed{RUN_SEED}"
OUT_DIR = os.path.join(CFG["WORK_DIR"], RUN_NAME)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
PROC_DIR = os.path.join(CFG["WORK_DIR"], "processed_images")
MODEL_DIR = os.path.join(CFG["WORK_DIR"], "models_wp1")
for d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)
if os.path.exists(PROC_DIR):
    shutil.rmtree(PROC_DIR)
os.makedirs(PROC_DIR)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seed()

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

GPU_NAME, CC = "CPU", (0, 0)
if gpus:
    try:
        det = tf.config.experimental.get_device_details(gpus[0])
        GPU_NAME, CC = det.get("device_name", "GPU"), det.get("compute_capability", (0, 0))
    except Exception:
        GPU_NAME = "GPU"
USE_MP = CFG["MIXED_PRECISION"] and gpus and CC[0] >= 7
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

ENV_INFO = {"python": platform.python_version(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "device": GPU_NAME, "mixed_precision": bool(USE_MP)}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.family": "serif",
                     "font.size": 11, "axes.titleweight": "bold"})


def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()
    print(f"   saved -> {path}")


RESULTS = {"run_name": RUN_NAME, "config": CFG, "environment": ENV_INFO}


def save_results():
    with open(os.path.join(OUT_DIR, "run_results.json"), "w") as fj:
        json.dump(RESULTS, fj, indent=2, default=str)


print("Environment:", json.dumps(ENV_INFO, indent=2))
print(f"\n▶ Run: {RUN_NAME}  (split seed {SEED}, training seed {RUN_SEED})")

# %%
# ============================================================
# CELL 2 — Locate the dataset and list every image
# ============================================================
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_root(preferred):
    if os.path.isdir(preferred):
        return preferred

    def search(base):
        for r, d, _ in os.walk(base):
            if any(x.lower() == "training" for x in d) and any(x.lower() == "testing" for x in d):
                return r
        return None

    root = search("/kaggle/input")
    if root:
        return root
    # Dataset not attached: download the public dataset directly (needs Internet ON)
    try:
        import kagglehub
        path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print("Dataset downloaded with kagglehub ->", path)
        root = search(path)
    except Exception as e:
        print("kagglehub download failed:", e)
    if root:
        return root
    raise FileNotFoundError("Dataset not found. Attach it with Add Input, or turn Internet ON.")


DATA_ROOT = find_root(CFG["DATA_ROOT"])
records = []
for r, _, files in os.walk(DATA_ROOT):
    cls = CLASS_MAP.get(os.path.basename(r).lower())
    if cls is None:
        continue
    official = "test" if "test" in r.lower() else "train"
    for f in sorted(files):
        if f.lower().endswith(VALID_EXT):
            records.append({"src_path": os.path.join(r, f), "label": cls, "official_split": official})

df = pd.DataFrame(records).sort_values("src_path").reset_index(drop=True)
if CFG["SMOKE_TEST"]:
    df = (df.sample(frac=1.0, random_state=SEED).groupby("label").head(60)
            .sort_values("src_path").reset_index(drop=True))

print(f"Dataset root : {DATA_ROOT}\nTotal images : {len(df)}\n")
raw_counts = pd.crosstab(df["label"], df["official_split"], margins=True, margins_name="Total")
print(raw_counts)
raw_counts.to_csv(os.path.join(TAB_DIR, "table_raw_dataset_counts.csv"))

# %%
# ============================================================
# CELL 3 — Preprocessing (parallel): brain cropping, resizing, fingerprints
# ============================================================
def crop_brain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    th = cv2.erode(th, None, iterations=2)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w < 0.2 * img.shape[1] or h < 0.2 * img.shape[0]:
        return img
    return img[y:y + h, x:x + w]


def phash64(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return np.packbits((d > np.median(d)).flatten()).view(">u8")[0]


def ncc_vector(gray):
    v = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def process(i, src, label):
    raw = open(src, "rb").read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    crop = crop_brain(img)
    gray_fp = cv2.cvtColor(cv2.resize(crop, (FP, FP), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    out = os.path.join(PROC_DIR, f"{i:05d}_{label}.png")
    cv2.imwrite(out, cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA))
    return hashlib.md5(raw).hexdigest(), phash64(gray_fp), ncc_vector(gray_fp), out


t0 = time.time()
cv2.setNumThreads(1)
outs = Parallel(n_jobs=CFG["N_JOBS"], batch_size=32)(
    delayed(process)(i, r.src_path, r.label) for i, r in enumerate(df.itertuples()))
ok = [o is not None for o in outs]
n_unreadable = int(len(ok) - sum(ok))
df = df[ok].reset_index(drop=True)
outs = [o for o in outs if o is not None]
df["md5"] = [o[0] for o in outs]
df["phash"] = [o[1] for o in outs]
df["path"] = [o[3] for o in outs]
df["ridx"] = np.arange(len(df))     # row index into the NCC matrix
NCC = np.stack([o[2] for o in outs])
print(f"Preprocessed {len(df)} images in {time.time() - t0:.1f}s (unreadable: {n_unreadable})")

# %%
# ============================================================
# CELL 4 — Exact + near-duplicate grouping and leakage audit
# ============================================================
N = len(df)
H = df["phash"].to_numpy(dtype=np.uint64)
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
parent = np.arange(N)


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


exact_pairs = 0
for _, idx in df.groupby("md5").indices.items():
    for j in idx[1:]:
        union(idx[0], j)
        exact_pairs += 1

near_pairs = []
for s in range(0, N, 512):
    x = H[s:s + 512, None] ^ H[None, :]
    dist = POP[x.view(np.uint8)].reshape(x.shape[0], N, 8).sum(-1)
    I, J = np.where(dist <= CFG["PHASH_MAX_DIST"])
    I = I + s
    m = I < J
    I, J = I[m], J[m]
    if len(I):
        corr = (NCC[I] * NCC[J]).sum(1)
        for a, b, c in zip(I[corr >= CFG["NCC_MIN"]], J[corr >= CFG["NCC_MIN"]], corr[corr >= CFG["NCC_MIN"]]):
            near_pairs.append((int(a), int(b), float(c)))
            union(a, b)

df["group"] = [find(i) for i in range(N)]
PRE_PATHS = df["path"].tolist()

off = df["official_split"].to_numpy()
md5_train = set(df.loc[off == "train", "md5"])
test_idx_off = np.where(off == "test")[0]
group_has_train = df.groupby("group")["official_split"].apply(lambda s: (s == "train").any())
audit = {
    "official_test_images": int(len(test_idx_off)),
    "official_test_with_exact_copy_in_train": int(sum(df.loc[i, "md5"] in md5_train for i in test_idx_off)),
    "official_test_with_exact_or_near_duplicate_in_train":
        int(sum(bool(group_has_train[df.loc[i, "group"]]) for i in test_idx_off)),
}
audit["percent_official_test_leaked"] = round(
    100 * audit["official_test_with_exact_or_near_duplicate_in_train"] / max(len(test_idx_off), 1), 2)

# Keep the raw, un-cleaned table: Protocol A (official split) uses the data exactly as distributed.
DF_ALL = df.copy()

labels_per_group = df.groupby("group")["label"].nunique()
conflict_groups = set(labels_per_group[labels_per_group > 1].index)
n_conflict = int(df["group"].isin(conflict_groups).sum())
df = df[~df["group"].isin(conflict_groups)]
n_before = len(df)
df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)

dedup_report = {
    "images_found": int(N + n_unreadable), "unreadable_removed": n_unreadable,
    "exact_duplicate_links": exact_pairs, "near_duplicate_links": len(near_pairs),
    "exact_duplicates_removed": int(n_before - len(df)),
    "conflicting_label_images_removed": n_conflict,
    "final_images": int(len(df)), "final_groups": int(df["group"].nunique()),
    "largest_group": int(df["group"].value_counts().max()), **audit,
}
print("De-duplication report:", json.dumps(dedup_report, indent=2))
pd.Series(dedup_report).to_csv(os.path.join(TAB_DIR, "table_dedup_report.csv"), header=["value"])

if near_pairs:
    show = sorted(near_pairs, key=lambda t: t[2])[:4]
    fig, ax = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.6))
    ax = np.array(ax).reshape(2, -1)
    for k, (a, b, c) in enumerate(show):
        for r, i in enumerate((a, b)):
            ax[r, k].imshow(cv2.cvtColor(cv2.imread(PRE_PATHS[i]), cv2.COLOR_BGR2RGB)); ax[r, k].axis("off")
        ax[0, k].set_title(f"r = {c:.3f}", fontsize=10)
    plt.suptitle("Near-duplicate pairs grouped together (top vs bottom)", fontweight="bold")
    plt.tight_layout()
    save_fig("fig_near_duplicate_examples")

# %%
# ============================================================
# CELL 5 — Build the three split protocols
# ============================================================
def label_ids(part):
    return part["label"].map(CIDX).to_numpy()


# ---- Protocol C: group-aware split, identical code and seed to v2 ----
dfC = df.copy()
y_all, g_all = label_ids(dfC), dfC["group"].to_numpy()
trval_idx, test_idx = next(StratifiedGroupKFold(CFG["TEST_FOLDS"], shuffle=True, random_state=SEED)
                           .split(dfC, y_all, g_all))
tr_rel, va_rel = next(StratifiedGroupKFold(CFG["VAL_FOLDS"], shuffle=True, random_state=SEED)
                      .split(trval_idx, y_all[trval_idx], g_all[trval_idx]))
dfC["split"] = ""
dfC.loc[trval_idx[tr_rel], "split"] = "train"
dfC.loc[trval_idx[va_rel], "split"] = "val"
dfC.loc[test_idx, "split"] = "test"
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not (set(dfC.loc[dfC.split == a, "group"]) & set(dfC.loc[dfC.split == b, "group"])), f"leak {a}-{b}"
n_val_C, n_test_C = int((dfC.split == "val").sum()), int((dfC.split == "test").sum())

# ---- Protocol B: random image-level split of the SAME de-duplicated images, same sizes as C ----
dfB = df.copy()
idx = np.arange(len(dfB))
dev_i, te_i = train_test_split(idx, test_size=n_test_C, stratify=label_ids(dfB), random_state=SEED)
tr_i, va_i = train_test_split(dev_i, test_size=n_val_C, stratify=label_ids(dfB)[dev_i], random_state=SEED)
dfB["split"] = ""
dfB.loc[tr_i, "split"], dfB.loc[va_i, "split"], dfB.loc[te_i, "split"] = "train", "val", "test"

# ---- Protocol A: the official Training / Testing folders, data as distributed ----
dfA = DF_ALL.copy().reset_index(drop=True)
dfA["split"] = np.where(dfA["official_split"] == "test", "test", "train")
tr_off = np.where(dfA.split == "train")[0]
_, va_off = train_test_split(tr_off, test_size=CFG["OFFICIAL_VAL_FRACTION"],
                             stratify=label_ids(dfA)[tr_off], random_state=SEED)
dfA.loc[va_off, "split"] = "val"

SPLITS = {"A": dfA, "B": dfB, "C": dfC}
RESULTS["dedup"] = dedup_report

# ---- Leakage flags for every test image: does an exact copy / near duplicate sit in train or val? ----
for k, d in SPLITS.items():
    dev = d[d.split != "test"]
    dev_md5, dev_groups = set(dev["md5"]), set(dev["group"])
    d["exact_in_dev"] = d["md5"].isin(dev_md5) & (d.split == "test")
    d["dup_in_dev"] = d["group"].isin(dev_groups) & (d.split == "test")

rows = []
for k, d in SPLITS.items():
    t = d[d.split == "test"]
    rows.append({"Protocol": PROTOCOLS[k], "Train": int((d.split == "train").sum()),
                 "Val": int((d.split == "val").sum()), "Test": len(t),
                 "Test with exact copy in train/val": int(t.exact_in_dev.sum()),
                 "Test with exact or near duplicate in train/val": int(t.dup_in_dev.sum()),
                 "Leaked test (%)": round(100 * t.dup_in_dev.mean(), 2)})
SPLIT_TABLE = pd.DataFrame(rows)
SPLIT_TABLE.to_csv(os.path.join(TAB_DIR, "table_wp1_split_sizes.csv"), index=False)
print(SPLIT_TABLE.to_string(index=False))
assert SPLIT_TABLE.iloc[2]["Test with exact or near duplicate in train/val"] == 0
RESULTS["splits"] = SPLIT_TABLE.to_dict("records")

# ---- Check that Protocol C reproduces the v2 split (only if the v2 manifest is attached as input) ----
v2_manifest = None
for r, _, files in os.walk("/kaggle/input"):
    if "split_manifest.csv" in files:
        v2_manifest = os.path.join(r, "split_manifest.csv")
        break
if v2_manifest:
    m = pd.read_csv(v2_manifest)
    joined = dfC[["md5", "split"]].merge(m[["md5", "split"]], on="md5", suffixes=("", "_v2"))
    agree = float((joined.split == joined.split_v2).mean()) if len(joined) else 0.0
    RESULTS["protocol_C_matches_v2_manifest"] = round(100 * agree, 2)
    print(f"\nProtocol C vs v2 split_manifest.csv: {100*agree:.2f}% of images in the same split")
else:
    print("\n(v2 split_manifest.csv not attached — skipping the reproducibility check)")
for k, d in SPLITS.items():
    d.to_csv(os.path.join(TAB_DIR, f"split_manifest_protocol_{k}.csv"),
                              columns=["src_path", "label", "official_split", "md5", "group", "split", "dup_in_dev"],
                              index=False)

# %%
# ============================================================
# CELL 6 — 1-nearest-neighbour memoriser (no learning; shows what can be answered by memory)
# ============================================================
NN = {}
for k, d in SPLITS.items():
    dev = d[d.split != "test"]
    te = d[d.split == "test"]
    A_dev, A_te = NCC[dev["ridx"].to_numpy()], NCC[te["ridx"].to_numpy()]
    best_sim, best_j = np.empty(len(te), np.float32), np.empty(len(te), np.int64)
    for s in range(0, len(te), 256):
        sim = A_te[s:s + 256] @ A_dev.T
        best_j[s:s + 256] = sim.argmax(1)
        best_sim[s:s + 256] = sim.max(1)
    dev_labels = label_ids(dev)
    NN[k] = {"pred": dev_labels[best_j], "sim": best_sim,
             "nn_path": dev["path"].to_numpy()[best_j], "nn_label": dev_labels[best_j]}
    acc = accuracy_score(label_ids(te), NN[k]["pred"])
    print(f"{PROTOCOLS[k]:32s} 1-NN memoriser accuracy = {100*acc:.2f}% | "
          f"median nearest-train similarity = {np.median(best_sim):.3f}")

# %%
# ============================================================
# CELL 7 — Data pipelines and the fusion model (v2 recipe; training seed = RUN_SEED)
# ============================================================
AUTOTUNE, BS = tf.data.AUTOTUNE, CFG["BATCH_SIZE"]
augment = keras.Sequential([
    layers.RandomFlip("horizontal", seed=RUN_SEED),
    layers.RandomRotation(0.05, fill_mode="constant", seed=RUN_SEED),
    layers.RandomZoom(0.10, fill_mode="constant", seed=RUN_SEED),
    layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=RUN_SEED),
    layers.RandomContrast(0.10, seed=RUN_SEED),
    layers.RandomBrightness(0.10, seed=RUN_SEED),
], name="augmentation")


def load_image(path, label):
    img = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)
    return img, tf.one_hot(label, len(CLASSES))


def make_ds(d, split, training=False, flip=False):
    part = d[d.split == split]
    ds = tf.data.Dataset.from_tensor_slices((part["path"].values, label_ids(part)))
    if training:
        ds = ds.shuffle(len(part), seed=RUN_SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if flip:
        ds = ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BS)
    if training:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0., 255.), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


WEIGHT_FILES = {"effnetv2b0": "efficientnetv2-b0_notop.h5",
                "resnet50v2": "resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"}


def resolve_weights(key):
    if CFG["SMOKE_TEST"]:
        return None
    for root in ("/kaggle/input", os.path.expanduser("~/.keras/models")):
        for r, _, files in os.walk(root):
            if WEIGHT_FILES[key] in files:
                return os.path.join(r, WEIGHT_FILES[key])
    import socket
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=5).close()
    except OSError:
        raise RuntimeError("\n❌ No Internet: turn ON Settings -> Internet, then Run All again.") from None
    return "imagenet"


def build_model(backbones, name):
    inp = keras.Input((IMG, IMG, 3), name="image")
    feats, bases = [], {}
    if "effnetv2b0" in backbones:
        b = keras.applications.EfficientNetV2B0(include_top=False, weights=resolve_weights("effnetv2b0"),
                                                input_shape=(IMG, IMG, 3), include_preprocessing=True)
        b.trainable = False
        bases["effnetv2b0"] = b
        feats.append(layers.GlobalAveragePooling2D(name="gap_effnet")(b(inp, training=False)))
    if "resnet50v2" in backbones:
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="resnet_preprocess")(inp)
        b = keras.applications.ResNet50V2(include_top=False, weights=resolve_weights("resnet50v2"),
                                          input_shape=(IMG, IMG, 3))
        b.trainable = False
        bases["resnet50v2"] = b
        feats.append(layers.GlobalAveragePooling2D(name="gap_resnet")(b(x, training=False)))
    x = feats[0] if len(feats) == 1 else layers.Concatenate(name="feature_fusion")(feats)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32", name="classifier")(x)
    return keras.Model(inp, out, name=name), bases


LOSS = keras.losses.CategoricalCrossentropy(label_smoothing=CFG["LABEL_SMOOTHING"])
MODEL_NAME = " + ".join({"effnetv2b0": "EfficientNetV2-B0", "resnet50v2": "ResNet50V2"}[b]
                        for b in CFG["BACKBONES"])
print("Model used under every protocol:", MODEL_NAME)


def train_and_predict(k):
    d = SPLITS[k]
    keras.backend.clear_session(); gc.collect()
    set_seed(RUN_SEED)
    train_ds, val_ds = make_ds(d, "train", training=True), make_ds(d, "val")
    model, bases = build_model(CFG["BACKBONES"], f"protocol_{k}")
    t0 = time.time()

    print(f"\n🚀 [{PROTOCOLS[k]}] Phase 1 — classifier head")
    model.compile(optimizer=keras.optimizers.Adam(CFG["HEAD_LR"]), loss=LOSS, metrics=["accuracy"])
    es1 = keras.callbacks.EarlyStopping("val_loss", patience=CFG["HEAD_PATIENCE"], restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-6)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=CFG["HEAD_EPOCHS"], callbacks=[es1, rlr], verbose=2)
    if es1.best_weights is not None:
        model.set_weights(es1.best_weights)

    print(f"\n🔥 [{PROTOCOLS[k]}] Phase 2 — full fine-tuning")
    for b in bases.values():
        b.trainable = True
    spe = int(tf.data.experimental.cardinality(train_ds).numpy())
    warm = CFG["WARMUP_EPOCHS"] * spe
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CFG["FT_LR"] / 100, decay_steps=max(1, CFG["FT_EPOCHS"] * spe - warm),
        alpha=0.01, warmup_target=CFG["FT_LR"], warmup_steps=warm)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=LOSS, metrics=["accuracy"])
    es2 = keras.callbacks.EarlyStopping("val_loss", patience=CFG["FT_PATIENCE"], restore_best_weights=True)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=CFG["FT_EPOCHS"], callbacks=[es2], verbose=2)
    if es2.best_weights is not None:
        model.set_weights(es2.best_weights)
    minutes = round((time.time() - t0) / 60, 2)

    views = [make_ds(d, "test")] + ([make_ds(d, "test", flip=True)] if CFG["TTA"] else [])
    probs = np.mean([model.predict(v, verbose=0) for v in views], axis=0)
    vviews = [make_ds(d, "val")] + ([make_ds(d, "val", flip=True)] if CFG["TTA"] else [])
    vprobs = np.mean([model.predict(v, verbose=0) for v in vviews], axis=0)

    hist = pd.DataFrame({m: h1.history[m] + h2.history[m] for m in ["accuracy", "val_accuracy", "loss", "val_loss"]})
    hist["phase"] = [1] * len(h1.history["loss"]) + [2] * len(h2.history["loss"])
    hist.index = np.arange(1, len(hist) + 1); hist.index.name = "epoch"
    hist.to_csv(os.path.join(TAB_DIR, f"history_protocol_{k}.csv"))
    del model, bases
    return probs, vprobs, hist, minutes


def plot_history(hist, k):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    sp = (hist["phase"] == 1).sum() + 0.5
    for a, (tr, va, lab) in zip(ax, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
        a.plot(hist.index, hist[tr], "-o", ms=3, lw=2, label=f"Train {lab.lower()}")
        a.plot(hist.index, hist[va], "-s", ms=3, lw=2, label=f"Validation {lab.lower()}")
        a.axvline(sp, color="red", ls="--", lw=1.5, label="Fine-tuning starts")
        a.set_xlabel("Epoch"); a.set_ylabel(lab); a.grid(alpha=0.3); a.legend()
    ax[0].set_title(f"{PROTOCOLS[k]}: accuracy"); ax[1].set_title(f"{PROTOCOLS[k]}: loss")
    plt.tight_layout()
    save_fig(f"fig_wp1_training_curves_protocol_{k}")


# %%
# ============================================================
# CELL 7b — Second memoriser: 1-NN on frozen ImageNet features (no training on MRI)
#   Pixel matching (CELL 6) only finds near-identical images; ImageNet features also find
#   images that look alike after small shifts, crops or contrast changes.
# ============================================================
def imagenet_features(paths):
    b = keras.applications.ResNet50V2(include_top=False, weights=resolve_weights("resnet50v2"),
                                      input_shape=(IMG, IMG, 3), pooling="avg")
    inp = keras.Input((IMG, IMG, 3))
    fx = keras.Model(inp, b(layers.Rescaling(1 / 127.5, offset=-1.0)(inp), training=False))
    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(lambda p: tf.cast(tf.io.decode_png(tf.io.read_file(p), channels=3), tf.float32),
               num_parallel_calls=AUTOTUNE).batch(64).prefetch(AUTOTUNE))
    f = fx.predict(ds, verbose=0).astype(np.float32)
    del fx, b
    keras.backend.clear_session(); gc.collect()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


t0 = time.time()
FEAT = imagenet_features(DF_ALL.sort_values("ridx")["path"].values)
print(f"ImageNet features: {FEAT.shape} in {time.time() - t0:.0f}s")

NNF = {}
for k, d in SPLITS.items():
    dev, te = d[d.split != "test"], d[d.split == "test"]
    A_dev, A_te = FEAT[dev["ridx"].to_numpy()], FEAT[te["ridx"].to_numpy()]
    sims = A_te @ A_dev.T
    j = sims.argmax(1)
    NNF[k] = {"pred": label_ids(dev)[j], "sim": sims.max(1)}
    print(f"{PROTOCOLS[k]:32s} 1-NN (ImageNet features) accuracy = "
          f"{100*accuracy_score(label_ids(te), NNF[k]['pred']):.2f}%")


# %%
# ============================================================
# CELL 8 — Train the same model under each protocol (≈ 75 min per protocol on a T4)
#   Every prediction is written to disk as soon as a protocol finishes, so nothing is lost.
# ============================================================
PRED_DIR = os.path.join(OUT_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)
PROB_COLS = [f"p_{c}" for c in CLASSES]


def prediction_table(k, split, probs):
    d = SPLITS[k]
    part = d[d.split == split].reset_index(drop=True)
    out = pd.DataFrame({
        "run_seed": RUN_SEED, "protocol": k, "split": split,
        "src_path": part["src_path"], "md5": part["md5"], "group": part["group"],
        "label": part["label"], "y": label_ids(part),
        "dup_in_dev": part["dup_in_dev"].astype(bool), "exact_in_dev": part["exact_in_dev"].astype(bool),
    })
    out[PROB_COLS] = probs
    out["cnn_pred"] = probs.argmax(1)
    if split == "test":
        out["nn_pixel_pred"], out["nn_pixel_sim"] = NN[k]["pred"], NN[k]["sim"]
        out["nn_feat_pred"], out["nn_feat_sim"] = NNF[k]["pred"], NNF[k]["sim"]
    return out


CNN = {}
for k in SPLITS:
    probs, vprobs, hist, minutes = train_and_predict(k)
    CNN[k] = {"probs": probs, "val_probs": vprobs, "minutes": minutes}
    pd.concat([prediction_table(k, "test", probs), prediction_table(k, "val", vprobs)]).to_csv(
        os.path.join(PRED_DIR, f"predictions_protocol_{k}_seed{RUN_SEED}.csv"), index=False)
    plot_history(hist, k)
    yt = label_ids(SPLITS[k][SPLITS[k].split == "test"])
    yv = label_ids(SPLITS[k][SPLITS[k].split == "val"])
    print(f"   [{PROTOCOLS[k]}] val {100*accuracy_score(yv, vprobs.argmax(1)):.2f}% | "
          f"test {100*accuracy_score(yt, probs.argmax(1)):.2f}% ({minutes} min) — predictions saved")

# %%
# ============================================================
# CELL 9 — Per-run summary (the full multi-seed analysis is in notebook 04)
# ============================================================
rows = []
for k, d in SPLITS.items():
    te, va = d[d.split == "test"], d[d.split == "val"]
    y, leaked = label_ids(te), te["dup_in_dev"].to_numpy()
    p = CNN[k]["probs"].argmax(1)
    rows.append({
        "Protocol": PROTOCOLS[k], "Seed": RUN_SEED,
        "Val acc (%)": round(100 * accuracy_score(label_ids(va), CNN[k]["val_probs"].argmax(1)), 2),
        "Test acc (%)": round(100 * accuracy_score(y, p), 2),
        "Macro-F1": round(precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2], 4),
        "MCC": round(matthews_corrcoef(y, p), 4),
        "Leaked test (%)": round(100 * leaked.mean(), 2),
        "CNN leaked (%)": round(100 * (p[leaked] == y[leaked]).mean(), 2) if leaked.any() else None,
        "CNN unseen (%)": round(100 * (p[~leaked] == y[~leaked]).mean(), 2),
        "1-NN pixel (%)": round(100 * accuracy_score(y, NN[k]["pred"]), 2),
        "1-NN ImageNet (%)": round(100 * accuracy_score(y, NNF[k]["pred"]), 2),
        "Train (min)": CNN[k]["minutes"],
    })
SUMMARY = pd.DataFrame(rows)
SUMMARY.to_csv(os.path.join(TAB_DIR, f"run_summary_seed{RUN_SEED}.csv"), index=False)
pd.set_option("display.width", 250)
print(SUMMARY.to_string(index=False))
RESULTS["summary"] = SUMMARY.to_dict("records")
RESULTS["splits"] = SPLIT_TABLE.to_dict("records")
RESULTS["dedup"] = dedup_report
save_results()

# %%
# ============================================================
# CELL 10 — Confusion matrices for this run
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
names_c = [CLASS_TITLES[c] for c in CLASSES]
for a, k in zip(ax, SPLITS):
    y = label_ids(SPLITS[k][SPLITS[k].split == "test"])
    cm = confusion_matrix(y, CNN[k]["probs"].argmax(1), labels=range(4))
    a.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(4)); a.set_xticklabels(names_c, rotation=30, ha="right")
    a.set_yticks(range(4)); a.set_yticklabels(names_c)
    a.set_title(f"{PROTOCOLS[k]} (seed {RUN_SEED})\nacc {100*accuracy_score(y, CNN[k]['probs'].argmax(1)):.2f}%")
    a.set_xlabel("Predicted"); a.set_ylabel("True")
plt.tight_layout()
save_fig(f"fig_confusion_matrices_seed{RUN_SEED}")

# %%
# ============================================================
# CELL 11 — Package this run
# ============================================================
save_results()
zip_path = shutil.make_archive(os.path.join(CFG["WORK_DIR"], RUN_NAME), "zip", OUT_DIR)
print("=" * 80)
print(f"✅ {RUN_NAME} finished. Output folder: {OUT_DIR}")
print(f"   Zip: {zip_path}")
print("   Next: after all three seeds have finished, run notebook 04 (analysis) with the three outputs attached.")
print("=" * 80)
