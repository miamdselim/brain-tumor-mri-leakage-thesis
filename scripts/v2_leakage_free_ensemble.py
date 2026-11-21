# %% [markdown]
# # Leakage-Free Brain Tumor MRI Classification: Multi-Backbone Ensemble (v2)
#
# **Author:** Md. Selim Mia — Department of Artificial Intelligence, Nanjing University of Information Science & Technology
#
# **Dataset:** Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle, Version 2) — 7,200 MRI images, 4 classes
#
# **What changed from v1, and why (each change answers a finding from the v1 run):**
#
# | v1 finding | v2 change |
# |---|---|
# | Most errors were meningioma ↔ glioma (fine texture) | Input resolution 224 → **288** |
# | EfficientNetV2-B0 was still improving when training ended (best epoch 24/25) | Fine-tuning LR 3e-5 → **5e-5 with 1-epoch warm-up**, up to **35** epochs |
# | ResNet50V2 alone (97.48%) ≈ fusion (96.75%); the branches make different errors | **Ensemble** of three diverse models (ResNet50V2, EfficientNetV2-B0, fusion) |
# | Borderline predictions | **Test-time augmentation** (original + horizontal flip) |
# | Preprocessing took 33 min | Parallel preprocessing; **mixed precision** on T4 |
#
# **Honest protocol (unchanged):** the *identical* leakage-free, group-aware split as v1 (seed 42).
# The final model is chosen **only on the validation set**; the test set is used once, at the end.
#
# **Runtime:** GPU T4 (or P100), Internet ON. Total ≈ 2–2.5 h. Use *Save Version → Save & Run All*.

# %%
# ============================================================
# CELL 1 — Configuration and imports
# ============================================================
import os, sys, json, time, random, hashlib, shutil, warnings, platform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.stats import binomtest
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, roc_curve, auc, log_loss,
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
    "SEED": 42,
    # ---- images ----
    "IMG_SIZE": 288,          # training / inference resolution
    "FINGERPRINT_SIZE": 224,  # duplicate detection resolution (same as v1 -> identical split)
    "BATCH_SIZE": 32,
    # ---- de-duplication ----
    "PHASH_MAX_DIST": 8,
    "NCC_MIN": 0.90,
    # ---- split ----
    "TEST_FOLDS": 7,
    "VAL_FOLDS": 6,
    # ---- training ----
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
    "MIXED_PRECISION": True,  # used only on GPUs with compute capability >= 7 (T4, V100, A100)
    "N_JOBS": -1,
    "SMOKE_TEST": os.environ.get("SMOKE_TEST", "0") == "1",
}
if CFG["SMOKE_TEST"]:
    CFG.update(IMG_SIZE=96, BATCH_SIZE=8, HEAD_EPOCHS=1, FT_EPOCHS=1, BOOTSTRAP_ROUNDS=50)

SEED, IMG, FP = CFG["SEED"], CFG["IMG_SIZE"], CFG["FINGERPRINT_SIZE"]
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

OUT_DIR = os.path.join(CFG["WORK_DIR"], "paper_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
PROC_DIR = os.path.join(CFG["WORK_DIR"], "processed_images")
MODEL_DIR = os.path.join(CFG["WORK_DIR"], "models")
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


def save_results():
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fj:
        json.dump({"config": CFG, "environment": ENV_INFO, "dedup": dedup_report,
                   "selection": SELECTION, "results": RESULTS}, fj, indent=2, default=str)


RESULTS, SELECTION = {}, {}
print("Environment:", json.dumps(ENV_INFO, indent=2))

# %%
# ============================================================
# CELL 2 — Locate the dataset and list every image
# ============================================================
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_root(preferred):
    if os.path.isdir(preferred):
        return preferred
    for r, d, _ in os.walk("/kaggle/input"):
        if any(x.lower() == "training" for x in d):
            return r
    raise FileNotFoundError(f"Dataset not found at {preferred}")


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
# CELL 5 — Group-aware stratified split (identical to v1)
# ============================================================
y_all = df["label"].map(CIDX).to_numpy()
g_all = df["group"].to_numpy()
trval_idx, test_idx = next(StratifiedGroupKFold(CFG["TEST_FOLDS"], shuffle=True, random_state=SEED)
                           .split(df, y_all, g_all))
tr_rel, va_rel = next(StratifiedGroupKFold(CFG["VAL_FOLDS"], shuffle=True, random_state=SEED)
                      .split(trval_idx, y_all[trval_idx], g_all[trval_idx]))
df["split"] = ""
df.loc[trval_idx[tr_rel], "split"] = "train"
df.loc[trval_idx[va_rel], "split"] = "val"
df.loc[test_idx, "split"] = "test"

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not (set(df.loc[df.split == a, "group"]) & set(df.loc[df.split == b, "group"])), f"leak {a}-{b}"
    assert not (set(df.loc[df.split == a, "md5"]) & set(df.loc[df.split == b, "md5"]))
print("✅ Verified: no exact or near-duplicate is shared between train / val / test.\n")

split_counts = pd.crosstab(df["label"], df["split"])[["train", "val", "test"]]
split_counts.loc["Total"] = split_counts.sum()
split_counts["Total"] = split_counts.sum(axis=1)
print(split_counts)
split_counts.to_csv(os.path.join(TAB_DIR, "table_split_counts.csv"))
df[["src_path", "label", "official_split", "md5", "group", "split"]].to_csv(
    os.path.join(TAB_DIR, "split_manifest.csv"), index=False)

sc = split_counts.drop("Total").drop(columns="Total")
ax = sc.rename(index=CLASS_TITLES).plot(kind="bar", figsize=(8, 4.5), width=0.75,
                                        color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
for cont in ax.containers:
    ax.bar_label(cont, fontsize=8)
ax.set_xlabel(""); ax.set_ylabel("Number of images")
ax.set_title("Class distribution after de-duplication (group-aware split)")
plt.xticks(rotation=0); plt.tight_layout()
save_fig("fig_class_distribution")

fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))
for k, c in enumerate(CLASSES):
    r = df[df.label == c].iloc[0]
    ax[0, k].imshow(cv2.cvtColor(cv2.imread(r["src_path"]), cv2.COLOR_BGR2RGB)); ax[0, k].axis("off")
    ax[0, k].set_title(CLASS_TITLES[c])
    ax[1, k].imshow(cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)); ax[1, k].axis("off")
ax[0, 0].text(-0.15, 0.5, "Original", transform=ax[0, 0].transAxes, rotation=90, va="center", fontsize=12)
ax[1, 0].text(-0.15, 0.5, f"Cropped {IMG}×{IMG}", transform=ax[1, 0].transAxes, rotation=90, va="center", fontsize=12)
plt.tight_layout()
save_fig("fig_preprocessing_examples")

# %%
# ============================================================
# CELL 6 — tf.data pipelines, augmentation, TTA
# ============================================================
AUTOTUNE, BS = tf.data.AUTOTUNE, CFG["BATCH_SIZE"]
augment = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.05, fill_mode="constant", seed=SEED),
    layers.RandomZoom(0.10, fill_mode="constant", seed=SEED),
    layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=SEED),
    layers.RandomContrast(0.10, seed=SEED),
    layers.RandomBrightness(0.10, seed=SEED),
], name="augmentation")


def load_image(path, label):
    img = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)
    return img, tf.one_hot(label, len(CLASSES))


def make_ds(split, training=False, flip=False):
    part = df[df.split == split]
    ds = tf.data.Dataset.from_tensor_slices((part["path"].values, part["label"].map(CIDX).values))
    if training:
        ds = ds.shuffle(len(part), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if flip:
        ds = ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BS)
    if training:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0., 255.), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


train_ds, val_ds = make_ds("train", training=True), make_ds("val")
EVAL_DS = {s: [make_ds(s)] + ([make_ds(s, flip=True)] if CFG["TTA"] else []) for s in ("val", "test")}
Y = {s: df[df.split == s]["label"].map(CIDX).to_numpy() for s in ("val", "test")}
test_paths = df[df.split == "test"]["path"].to_numpy()
print(f"Train {sum(df.split=='train')} | Val {sum(df.split=='val')} | Test {sum(df.split=='test')} | "
      f"TTA views: {len(EVAL_DS['test'])}")

# %%
# ============================================================
# CELL 7 — Models (each backbone receives its own correct preprocessing)
# ============================================================
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
    inp = keras.Input((IMG, IMG, 3), name="image")          # raw pixels [0, 255]
    feats, maps, bases = [], [], {}
    if "effnetv2b0" in backbones:                           # built-in rescaling: expects [0, 255]
        b = keras.applications.EfficientNetV2B0(include_top=False, weights=resolve_weights("effnetv2b0"),
                                                input_shape=(IMG, IMG, 3), include_preprocessing=True)
        b.trainable = False
        m = b(inp, training=False)
        maps.append(m); bases["effnetv2b0"] = b
        feats.append(layers.GlobalAveragePooling2D(name="gap_effnet")(m))
    if "resnet50v2" in backbones:                           # expects [-1, 1]
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="resnet_preprocess")(inp)
        b = keras.applications.ResNet50V2(include_top=False, weights=resolve_weights("resnet50v2"),
                                          input_shape=(IMG, IMG, 3))
        b.trainable = False
        m = b(x, training=False)
        maps.append(m); bases["resnet50v2"] = b
        feats.append(layers.GlobalAveragePooling2D(name="gap_resnet")(m))
    x = feats[0] if len(feats) == 1 else layers.Concatenate(name="feature_fusion")(feats)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="fc512")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32", name="classifier")(x)
    return keras.Model(inp, out, name=name), bases, keras.Model(inp, maps + [out], name=name + "_cam")


def count_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


EXPERIMENTS = {
    "ResNet50V2": ("resnet50v2",),
    "EfficientNetV2-B0": ("effnetv2b0",),
    "Fusion (EfficientNetV2-B0 + ResNet50V2)": ("effnetv2b0", "resnet50v2"),
}
_m, _, _ = build_model(EXPERIMENTS["Fusion (EfficientNetV2-B0 + ResNet50V2)"], "fusion")
_m.summary(line_length=110)
del _m

# %%
# ============================================================
# CELL 8 — Training (phase 1: head, phase 2: full fine-tuning with warm-up + cosine)
# ============================================================
LOSS = keras.losses.CategoricalCrossentropy(label_smoothing=CFG["LABEL_SMOOTHING"])


def train_model(title, backbones):
    set_seed()
    tag = "_".join(backbones)
    model, bases, cam = build_model(backbones, tag)
    t0 = time.time()

    print(f"\n🚀 [{title}] Phase 1 — classifier head (backbones frozen)")
    model.compile(optimizer=keras.optimizers.Adam(CFG["HEAD_LR"]), loss=LOSS, metrics=["accuracy"])
    p1 = count_trainable(model)
    es1 = keras.callbacks.EarlyStopping("val_loss", patience=CFG["HEAD_PATIENCE"], restore_best_weights=True, verbose=1)
    rlr = keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=CFG["HEAD_EPOCHS"], callbacks=[es1, rlr], verbose=2)
    if es1.best_weights is not None:
        model.set_weights(es1.best_weights)

    print(f"\n🔥 [{title}] Phase 2 — full fine-tuning (warm-up + cosine decay, BatchNorm frozen)")
    for b in bases.values():
        b.trainable = True
    spe = int(tf.data.experimental.cardinality(train_ds).numpy())
    warm = CFG["WARMUP_EPOCHS"] * spe
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CFG["FT_LR"] / 100, decay_steps=max(1, CFG["FT_EPOCHS"] * spe - warm),
        alpha=0.01, warmup_target=CFG["FT_LR"], warmup_steps=warm)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=LOSS, metrics=["accuracy"])
    p2 = count_trainable(model)
    es2 = keras.callbacks.EarlyStopping("val_loss", patience=CFG["FT_PATIENCE"], restore_best_weights=True, verbose=1)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=CFG["FT_EPOCHS"], callbacks=[es2], verbose=2)
    if es2.best_weights is not None:
        model.set_weights(es2.best_weights)

    model.save(os.path.join(MODEL_DIR, f"{tag}.keras"))
    hist = pd.DataFrame({k: h1.history[k] + h2.history[k] for k in ["accuracy", "val_accuracy", "loss", "val_loss"]})
    hist["phase"] = [1] * len(h1.history["loss"]) + [2] * len(h2.history["loss"])
    hist.index = np.arange(1, len(hist) + 1); hist.index.name = "epoch"
    hist.to_csv(os.path.join(TAB_DIR, f"history_{tag}.csv"))
    info = {"params_total": int(model.count_params()), "params_trainable_phase1": p1,
            "params_trainable_phase2": p2, "epochs_phase1": len(h1.history["loss"]),
            "epochs_phase2": len(h2.history["loss"]),
            "best_epoch_phase1": int(np.argmin(h1.history["val_loss"]) + 1),
            "best_epoch_phase2": int(np.argmin(h2.history["val_loss"]) + 1),
            "train_minutes": round((time.time() - t0) / 60, 2)}
    return model, cam, hist, info


def predict_tta(model, split):
    """Average softmax over the TTA views; returns probs and ms/image (all views)."""
    _ = model.predict(EVAL_DS[split][0].take(1), verbose=0)
    t0 = time.time()
    probs = np.mean([model.predict(d, verbose=0) for d in EVAL_DS[split]], axis=0)
    return probs, 1000 * (time.time() - t0) / len(probs)


def plot_history(hist, title, tag):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    sp = (hist["phase"] == 1).sum() + 0.5
    for a, (tr, va, lab) in zip(ax, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
        a.plot(hist.index, hist[tr], "-o", ms=3, lw=2, label=f"Train {lab.lower()}")
        a.plot(hist.index, hist[va], "-s", ms=3, lw=2, label=f"Validation {lab.lower()}")
        a.axvline(sp, color="red", ls="--", lw=1.5, label="Fine-tuning starts")
        a.set_xlabel("Epoch"); a.set_ylabel(lab); a.grid(alpha=0.3); a.legend()
    ax[0].set_title(f"{title}: accuracy"); ax[1].set_title(f"{title}: loss")
    plt.tight_layout()
    save_fig(f"fig_training_curves_{tag}")


# %%
# ============================================================
# CELL 9 — Train the three models (val + test probabilities are stored; nothing is selected yet)
# ============================================================
PROBS, MODELS, CAMS, INFO, MS = {}, {}, {}, {}, {}
for title, bb in EXPERIMENTS.items():
    model, cam, hist, info = train_model(title, bb)
    MODELS[title], CAMS[title], INFO[title] = model, cam, info
    pv, _ = predict_tta(model, "val")
    pt, ms = predict_tta(model, "test")
    PROBS[title] = {"val": pv, "test": pt}
    MS[title] = ms
    plot_history(hist, title, "_".join(bb))
    print(f"   [{title}] validation accuracy (TTA) = {accuracy_score(Y['val'], pv.argmax(1)):.4f}")

ENS = "Ensemble (ResNet50V2 + EfficientNetV2-B0 + Fusion)"
PROBS[ENS] = {s: np.mean([PROBS[t][s] for t in EXPERIMENTS], axis=0) for s in ("val", "test")}
MS[ENS] = sum(MS[t] for t in EXPERIMENTS)
INFO[ENS] = {"params_total": sum(INFO[t]["params_total"] for t in EXPERIMENTS),
             "train_minutes": round(sum(INFO[t]["train_minutes"] for t in EXPERIMENTS), 2)}

# %%
# ============================================================
# CELL 10 — Model selection on the VALIDATION set only
# ============================================================
val_table = []
for t, p in PROBS.items():
    val_table.append({"Model": t, "Val accuracy": accuracy_score(Y["val"], p["val"].argmax(1)),
                      "Val macro-F1": precision_recall_fscore_support(Y["val"], p["val"].argmax(1), average="macro", zero_division=0)[2],
                      "Val log-loss": log_loss(Y["val"], np.clip(p["val"], 1e-7, 1), labels=range(len(CLASSES)))})
val_table = pd.DataFrame(val_table).sort_values(["Val accuracy", "Val log-loss"], ascending=[False, True])
val_table.to_csv(os.path.join(TAB_DIR, "table_validation_selection.csv"), index=False)
print(val_table.round(4).to_string(index=False))

FINAL = val_table.iloc[0]["Model"]
SELECTION = {"rule": "highest validation accuracy (TTA); ties broken by lower validation log-loss",
             "selected_model": FINAL, "validation_table": val_table.round(6).to_dict("records")}
print(f"\n🏁 Selected on validation (before looking at test): {FINAL}")

# %%
# ============================================================
# CELL 11 — Test-set evaluation (used once) for every model
# ============================================================
def bootstrap_ci(y, p, fn, rounds, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(rounds))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics(title, probs, ms):
    y, yp = Y["test"], probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=range(len(CLASSES)), zero_division=0)
    cm = confusion_matrix(y, yp, labels=range(len(CLASSES)))
    spec = [(cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]) /
            max(cm.sum() - cm[k].sum(), 1) for k in range(len(CLASSES))]
    try:
        auc_m = roc_auc_score(np.eye(len(CLASSES))[y], probs, average="macro", multi_class="ovr")
    except ValueError:
        auc_m = float("nan")
    mf1 = lambda a, b: precision_recall_fscore_support(a, b, average="macro", zero_division=0)[2]
    tumor_missed = int(sum(cm[k, CIDX["notumor"]] for k in range(len(CLASSES)) if k != CIDX["notumor"]))
    return {
        "model": title, "test_images": int(len(y)),
        "accuracy": float(accuracy_score(y, yp)),
        "accuracy_95ci": bootstrap_ci(y, yp, accuracy_score, CFG["BOOTSTRAP_ROUNDS"]),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)), "macro_f1_95ci": bootstrap_ci(y, yp, mf1, CFG["BOOTSTRAP_ROUNDS"]),
        "weighted_f1": float(precision_recall_fscore_support(y, yp, average="weighted", zero_division=0)[2]),
        "macro_specificity": float(np.mean(spec)),
        "cohen_kappa": float(cohen_kappa_score(y, yp)), "mcc": float(matthews_corrcoef(y, yp)),
        "macro_roc_auc": float(auc_m),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1), labels=range(len(CLASSES)))),
        "misclassified": int((yp != y).sum()),
        "tumours_predicted_as_no_tumour": tumor_missed,
        "inference_ms_per_image": round(ms, 3),
        "confusion_matrix": cm.tolist(),
        "per_class": {CLASSES[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                                   "specificity": float(spec[k]), "support": int(s[k])}
                      for k in range(len(CLASSES))},
        **INFO[title],
    }


for t in PROBS:
    RESULTS[t] = metrics(t, PROBS[t]["test"], MS[t])
    r = RESULTS[t]
    print(f"{t:52s} acc {100*r['accuracy']:.2f}% (95% CI {100*r['accuracy_95ci'][0]:.2f}–"
          f"{100*r['accuracy_95ci'][1]:.2f}) | F1 {r['macro_f1']:.4f} | MCC {r['mcc']:.4f} | AUC {r['macro_roc_auc']:.4f}")

r = RESULTS[FINAL]
print(f"\n📊 Selected model — {FINAL} — classification report (test set)")
print(classification_report(Y["test"], PROBS[FINAL]["test"].argmax(1),
                            target_names=[CLASS_TITLES[c] for c in CLASSES], digits=4))
print("Confusion matrix:\n", np.array(r["confusion_matrix"]))

# McNemar exact test: selected model vs every other model
y = Y["test"]
mcn = []
for t in PROBS:
    if t == FINAL:
        continue
    a = PROBS[FINAL]["test"].argmax(1) == y
    b = PROBS[t]["test"].argmax(1) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    pv = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    mcn.append({"Comparison": f"{FINAL} vs {t}", "only selected correct": n01,
                "only other correct": n10, "McNemar exact p": round(pv, 4)})
mcn = pd.DataFrame(mcn)
mcn.to_csv(os.path.join(TAB_DIR, "table_mcnemar.csv"), index=False)
print("\nMcNemar tests:\n", mcn.to_string(index=False))
save_results()

# %%
# ============================================================
# CELL 12 — Comparison table + figures for the selected model
# ============================================================
rows = []
for t, r in RESULTS.items():
    rows.append({"Model": t, "Accuracy (%)": round(100 * r["accuracy"], 2),
                 "95% CI (%)": f"{100*r['accuracy_95ci'][0]:.2f}–{100*r['accuracy_95ci'][1]:.2f}",
                 "Precision": round(r["macro_precision"], 4), "Recall": round(r["macro_recall"], 4),
                 "F1": round(r["macro_f1"], 4), "Specificity": round(r["macro_specificity"], 4),
                 "Kappa": round(r["cohen_kappa"], 4), "MCC": round(r["mcc"], 4),
                 "AUC": round(r["macro_roc_auc"], 4), "Params (M)": round(r["params_total"] / 1e6, 2),
                 "Train (min)": r["train_minutes"], "Inference (ms/img)": r["inference_ms_per_image"],
                 "Selected": "✔" if t == FINAL else ""})
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(TAB_DIR, "table_model_comparison.csv"), index=False)
pd.set_option("display.width", 250)
print(comp.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.8))
mets, names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc"], ["Accuracy", "Precision", "Recall", "F1", "MCC"]
w = 0.8 / len(RESULTS)
for i, (t, r) in enumerate(RESULTS.items()):
    bars = ax.bar(np.arange(len(mets)) + i * w, [r[m] for m in mets], w, label=t, edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
lo = min(r[m] for r in RESULTS.values() for m in mets)
ax.set_ylim(max(0, lo - 0.04), 1.01)
ax.set_xticks(np.arange(len(mets)) + w * (len(RESULTS) - 1) / 2); ax.set_xticklabels(names)
ax.set_ylabel("Score"); ax.set_title("Test-set comparison of all models (TTA)")
ax.legend(loc="lower left", fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("fig_model_comparison")

cm = np.array(RESULTS[FINAL]["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True)
names_c = [CLASS_TITLES[c] for c in CLASSES]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a, mat, fmt, t in [(ax[0], cm, "d", "Counts"), (ax[1], cmn, ".2%", "Row-normalised")]:
    im = a.imshow(mat, cmap="Blues")
    a.set_xticks(range(4)); a.set_xticklabels(names_c, rotation=30, ha="right")
    a.set_yticks(range(4)); a.set_yticklabels(names_c)
    for i in range(4):
        for j in range(4):
            a.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                   color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=11)
    a.set_xlabel("Predicted label"); a.set_ylabel("True label"); a.set_title(t)
    plt.colorbar(im, ax=a, fraction=0.046)
plt.suptitle(f"Confusion matrix — {FINAL}", fontweight="bold"); plt.tight_layout()
save_fig("fig_confusion_matrix_final")

Yb = np.eye(len(CLASSES))[Y["test"]]
plt.figure(figsize=(6.4, 5.6))
for k, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(Yb[:, k], PROBS[FINAL]["test"][:, k])
    plt.plot(fpr, tpr, lw=2, label=f"{CLASS_TITLES[c]} (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("One-vs-rest ROC — selected model"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("fig_roc_curves_final")

# %%
# ============================================================
# CELL 13 — Grad-CAM (fusion model, both branches)
# ============================================================
FUS = "Fusion (EfficientNetV2-B0 + ResNet50V2)"
cam_model, fus_probs = CAMS[FUS], PROBS[FUS]["test"]


def grad_cams(img):
    x = tf.convert_to_tensor(img[None].astype("float32"))
    with tf.GradientTape() as tape:
        outs = cam_model(x, training=False)
        maps, pred = outs[:-1], outs[-1]
        cls = int(tf.argmax(pred[0]))
        score = pred[:, cls]
    grads = tape.gradient(score, maps)
    heat = []
    for m, g in zip(maps, grads):
        m, g = tf.cast(m, tf.float32), tf.cast(g, tf.float32)
        h = tf.nn.relu(tf.reduce_sum(m[0] * tf.reduce_mean(g, axis=(0, 1, 2)), axis=-1)).numpy()
        heat.append(cv2.resize(h / (h.max() + 1e-8), (IMG, IMG)))
    return heat, cls, float(pred[0, cls])


def overlay(img, h):
    return np.uint8(0.6 * img + 0.4 * cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)[:, :, ::-1])


fig, ax = plt.subplots(4, 3, figsize=(10, 13.4))
for k, c in enumerate(CLASSES):
    cand = np.where((Y["test"] == k) & (fus_probs.argmax(1) == k))[0]
    i = cand[0] if len(cand) else np.where(Y["test"] == k)[0][0]
    img = cv2.cvtColor(cv2.imread(test_paths[i]), cv2.COLOR_BGR2RGB)
    heats, cls, conf = grad_cams(img)
    ax[k, 0].imshow(img); ax[k, 0].set_title(f"{CLASS_TITLES[c]} (pred: {CLASS_TITLES[CLASSES[cls]]}, {conf:.1%})", fontsize=10)
    for j, (h, nm) in enumerate(zip(heats, ["EfficientNetV2-B0 branch", "ResNet50V2 branch"])):
        ax[k, j + 1].imshow(overlay(img, h)); ax[k, j + 1].set_title(f"Grad-CAM: {nm}", fontsize=10)
    for a in ax[k]:
        a.axis("off")
plt.tight_layout()
save_fig("fig_gradcam")

# %%
# ============================================================
# CELL 14 — Sample predictions and misclassified cases (selected model)
# ============================================================
fp = PROBS[FINAL]["test"]
rng = np.random.default_rng(SEED)
pick = np.concatenate([rng.choice(np.where(Y["test"] == k)[0], 3, replace=False) for k in range(len(CLASSES))])
fig, ax = plt.subplots(3, 4, figsize=(13, 11.5))
for a, i in zip(ax.T.flatten(), pick):
    pr = fp[i].argmax()
    a.imshow(cv2.cvtColor(cv2.imread(test_paths[i]), cv2.COLOR_BGR2RGB)); a.axis("off")
    a.set_title(f"True: {CLASS_TITLES[CLASSES[Y['test'][i]]]}\nPred: {CLASS_TITLES[CLASSES[pr]]} ({fp[i, pr]:.1%})",
                color="green" if pr == Y["test"][i] else "red", fontsize=10)
plt.tight_layout(h_pad=2.0)
save_fig("fig_sample_predictions")

wrong = np.where(fp.argmax(1) != Y["test"])[0]
if len(wrong):
    n = min(8, len(wrong)); cols = min(4, n); rws = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(3.4 * cols, 4.0 * rws))
    ax = np.array(ax).reshape(-1)
    for a in ax:
        a.axis("off")
    for a, i in zip(ax, wrong[:n]):
        pr = fp[i].argmax()
        a.imshow(cv2.cvtColor(cv2.imread(test_paths[i]), cv2.COLOR_BGR2RGB))
        a.set_title(f"True: {CLASS_TITLES[CLASSES[Y['test'][i]]]}\nPred: {CLASS_TITLES[CLASSES[pr]]} ({fp[i, pr]:.1%})",
                    color="red", fontsize=10)
    plt.suptitle("Misclassified test images (selected model)", fontweight="bold")
    plt.tight_layout(h_pad=2.5)
    save_fig("fig_misclassified")

# %%
# ============================================================
# CELL 15 — Final summary + package outputs
# ============================================================
save_results()
r = RESULTS[FINAL]
print("=" * 84)
print(f"🏆 FINAL (selected on validation): {FINAL}")
print("=" * 84)
print(f"Test images          : {r['test_images']}")
print(f"Accuracy             : {r['accuracy']*100:.2f}%  (95% CI {r['accuracy_95ci'][0]*100:.2f}–{r['accuracy_95ci'][1]*100:.2f}%)")
print(f"Macro F1             : {r['macro_f1']:.4f}")
print(f"Cohen's kappa / MCC  : {r['cohen_kappa']:.4f} / {r['mcc']:.4f}")
print(f"Macro ROC-AUC        : {r['macro_roc_auc']:.4f}")
print(f"Tumours missed (→ no tumour): {r['tumours_predicted_as_no_tumour']}")
print(f"Official split leak  : {dedup_report['percent_official_test_leaked']}% of official test images")
print("=" * 84)
zip_path = shutil.make_archive(os.path.join(CFG["WORK_DIR"], "paper_outputs_v2"), "zip", OUT_DIR)
print(f"\n📦 Saved: {zip_path}  — download it from the Output panel and send it for the paper.")
