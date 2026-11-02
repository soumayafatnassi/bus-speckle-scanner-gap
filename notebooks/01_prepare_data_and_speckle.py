# %% [markdown]
# # 01 · Data preparation and speckle measurement
#
# Downloads BUSI (with the Curated BUSI mapping), BUS-BRA and BUS-UCLM, builds one metadata table with patient identifiers, resizes images and masks to a working resolution (longest side at most 512 px), crops the lesion regions and measures speckle level and grain size in every image.
#
# **Run on Google Colab; a CPU runtime is enough.** Outputs are written to `PROJECT_DIR` (default `MyDrive/bus_crossscanner`).

# %%
# ---------------- 0. Setup ----------------
import os, sys, re, time, glob, subprocess, zipfile, urllib.request
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bm3d", "kagglehub"], check=True)

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
from joblib import Parallel, delayed

PROJECT_DIR = os.environ.get(
    "BUS_PROJECT_DIR", "/content/drive/MyDrive/bus_crossscanner" if IN_COLAB else "./bus_crossscanner")
LOCAL_DATA = os.environ.get("BUS_LOCAL_DATA", "/content/bus_data" if IN_COLAB else "./bus_data")
CURATED_CSV_SRC = os.environ.get(
    "BUSI_CURATED_CSV",
    "https://raw.githubusercontent.com/caumente/multi_task_breast_cancer/main/data/mapping_curated_BUSI.csv")
CHECK_COUNTS = os.environ.get("BUS_SKIP_COUNT_CHECK") is None

SOURCES = {   # dataset: (environment override, Kaggle handle, direct URL, zip name expected in PROJECT_DIR)
    "BUSI":     ("BUSI_DATA_ROOT", "aryashah2k/breast-ultrasound-images-dataset", None, "Dataset_BUSI_with_GT.zip"),
    "BUS-BRA":  ("BUSBRA_DATA_ROOT", None, "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1", "BUSBRA.zip"),
    "BUS-UCLM": ("BUSUCLM_DATA_ROOT", "orvile/bus-uclm-breast-ultrasound-dataset", None, "BUS-UCLM.zip"),
}
LABELS = ["benign", "malignant"]
DEVICE_BUSI, DEVICE_UCLM = "GE LOGIQ E9", "Siemens ACUSON S2000"

MAX_SIDE = 512        # working resolution: longest side in pixels (downscaling only)
ROI_MARGIN = 0.30     # lesion crop = bounding box enlarged by 30% of its size on each side
ROI_MIN_SIDE = 64     # minimum crop side in pixels
FILTERS = ["raw", "median", "lee", "srad", "bm3d"]
MEDIAN_K = 5                       # median: fixed 5x5, the common naive preprocessing in the literature
GRAIN_HALF, GRAIN_SMOOTH = 24, 10.0   # speckle-grain measurement: max lag and high-pass scale (pixels)
WIN_MIN, WIN_MAX = 5, 31           # Lee / local-statistics windows adapt to the measured grain within these bounds
SRAD_BASE_ITERS, SRAD_MAX_ITERS, SRAD_DT = 50, 400, 0.5   # iterations grow with grain size (dt <= 1 is stable)
STAT_WIN = 5                       # fixed window for the speckle index metric (comparable across filters)
DISPLAY_DR_DB = 50.0               # assumed log-compression dynamic range (results stable for 40-60 dB)
OUT_DIR = os.path.join(PROJECT_DIR, "filtered_v2")   # v2 = grain-adaptive filters
TISSUE_THR = 0.02                  # background threshold on a 0-1 scale
N_JOBS = max(1, os.cpu_count() or 1)

WORK_DIR = os.path.join(PROJECT_DIR, "work")
for d in ["work", "filtered", "tables", "figures"]:
    os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)
print("Project folder:", PROJECT_DIR, "| CPU cores:", N_JOBS)

# %% [markdown]
# ## 1. Download the three datasets
# Kaggle and Zenodo downloads need no login. If a download fails, put the dataset zip in `PROJECT_DIR` or point the matching `*_DATA_ROOT` environment variable to an extracted copy.

# %%
def find_root(base, dirs_needed, file_pred=None):
    """Folder under `base` that contains all `dirs_needed` (case-insensitive) and a file matching `file_pred`."""
    if not base or not os.path.isdir(base):
        return None
    for root, dirs, files in os.walk(base):
        low = {d.lower() for d in dirs}
        if all(d.lower() in low for d in dirs_needed) and (file_pred is None or any(file_pred(f) for f in files)):
            return root
    return None

def subdir(root, name):
    for d in os.listdir(root):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(root, d)):
            return os.path.join(root, d)
    raise FileNotFoundError(f"No '{name}' folder in {root}")

def get_dataset(name, dirs_needed, file_pred=None):
    env, handle, url, zip_name = SOURCES[name]
    target = os.path.join(LOCAL_DATA, name)
    root = find_root(os.environ.get(env), dirs_needed, file_pred) or find_root(target, dirs_needed, file_pred)
    if root is None and handle:
        try:
            import kagglehub
            root = find_root(kagglehub.dataset_download(handle), dirs_needed, file_pred)
        except Exception as e:
            print(f"{name}: Kaggle download failed ({e!r})")
    if root is None and url:
        try:
            os.makedirs(target, exist_ok=True)
            zp = os.path.join(LOCAL_DATA, zip_name)
            if not os.path.exists(zp):
                print(f"{name}: downloading {url}")
                urllib.request.urlretrieve(url, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(target)
            root = find_root(target, dirs_needed, file_pred)
        except Exception as e:
            print(f"{name}: direct download failed ({e!r})")
    if root is None:
        zp = os.path.join(PROJECT_DIR, zip_name)
        if os.path.exists(zp):
            with zipfile.ZipFile(zp) as z:
                z.extractall(target)
            root = find_root(target, dirs_needed, file_pred)
    if root is None:
        raise FileNotFoundError(f"{name} not found: download it manually and put {zip_name} in {PROJECT_DIR}")
    print(f"{name}: {root}")
    return root

busi_root = get_dataset("BUSI", ["benign", "malignant", "normal"])
bra_root = get_dataset("BUS-BRA", ["Images", "Masks"], lambda f: f.lower() == "bus_data.csv")
uclm_root = get_dataset("BUS-UCLM", ["images", "masks"], lambda f: f.lower().endswith(".csv"))

# %% [markdown]
# ## 2. Unified metadata table
# Only benign and malignant images with a lesion mask are kept. In BUS-UCLM, Doppler images and images that contain both lesion types are excluded.

# %%
def truthy(v):
    return str(v).strip().lower() in {"1", "yes", "y", "true", "si", "sí", "x"}

def first_col(df, *keys):
    """Column whose name equals one of `keys` (case-insensitive), else the first one containing a key."""
    for c in df.columns:
        if c.strip().lower() in keys:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None

def label_from_rgb_mask(path):
    m = cv2.imread(path, cv2.IMREAD_COLOR)            # BGR
    if m is None:
        return None
    b, g, r = [m[..., i].astype(int) for i in range(3)]
    green, red = int(((g > 128) & (r < 128)).sum()), int(((r > 128) & (g < 128)).sum())
    if green == 0 and red == 0:
        return "normal"
    return "benign" if green >= red else "malignant"

# ---- Curated BUSI ----
busi_pat = re.compile(r"^(benign|malignant|normal) \((\d+)\)\.png$")
cur = pd.read_csv(CURATED_CSV_SRC, sep=";")
cur.to_csv(os.path.join(PROJECT_DIR, "tables", "mapping_curated_BUSI.csv"), sep=";", index=False)
cur_keys = set(cur["class"] + "_" + cur["id"].astype(int).astype(str))
orig_counts, cur_counts, rows = {}, {}, []
for c in ["benign", "malignant", "normal"]:
    for f in sorted(os.listdir(os.path.join(busi_root, c))):
        m = busi_pat.match(f)
        if m is None:
            continue
        i = int(m.group(2))
        orig_counts[c] = orig_counts.get(c, 0) + 1
        if f"{c}_{i}" not in cur_keys:
            continue
        cur_counts[c] = cur_counts.get(c, 0) + 1
        if c in LABELS:
            masks = sorted(glob.glob(os.path.join(busi_root, c, f"{c} ({i})_mask*.png")))
            rows.append(dict(dataset="BUSI", image_id=f"busi_{c}_{i}", patient_id=f"busi_{c}_{i}", label=c,
                             device=DEVICE_BUSI, src_path=os.path.join(busi_root, c, f), mask_paths="|".join(masks)))
print("BUSI original:", orig_counts, "| curated:", cur_counts)
if CHECK_COUNTS:
    assert orig_counts == {"benign": 437, "malignant": 210, "normal": 133}, "Original BUSI counts differ"
    assert cur_counts == {"benign": 222, "malignant": 164, "normal": 64}, "Curated mapping does not match"

# ---- BUS-BRA ----
bra = pd.read_csv(os.path.join(bra_root, "bus_data.csv"))
id_col, path_col = first_col(bra, "id"), first_col(bra, "pathology")
case_col, dev_col = first_col(bra, "case", "patient"), first_col(bra, "device", "scanner", "equipment")
print("BUS-BRA columns:", list(bra.columns), "| patient column:", case_col, "| device column:", dev_col)
img_dir, msk_dir = subdir(bra_root, "Images"), subdir(bra_root, "Masks")
for _, r in bra.iterrows():
    iid = str(r[id_col]).strip()
    lab = str(r[path_col]).strip().lower()
    if lab not in LABELS:
        continue
    rows.append(dict(dataset="BUS-BRA", image_id=iid,
                     patient_id="bra_" + (str(r[case_col]) if case_col else iid.split("-")[0]),
                     label=lab, device=("BUS-BRA " + str(r[dev_col])) if dev_col else "BUS-BRA (unknown)",
                     src_path=os.path.join(img_dir, iid + ".png"),
                     mask_paths=os.path.join(msk_dir, "mask" + iid[3:] + ".png")))   # bus_0001-l -> mask_0001-l

# ---- BUS-UCLM ----
csv_name = [f for f in os.listdir(uclm_root) if f.lower().endswith(".csv")][0]
info = pd.read_csv(os.path.join(uclm_root, csv_name), sep=None, engine="python", encoding="utf-8-sig")
name_col = first_col(info, "image", "name", "file")
lab_col = first_col(info, "label", "class")
dop_col, comb_col = first_col(info, "doppler"), first_col(info, "combin")
print("BUS-UCLM columns:", list(info.columns), "| label column:", lab_col)
img_dir, msk_dir = subdir(uclm_root, "images"), subdir(uclm_root, "masks")
excluded = {"doppler": 0, "combined": 0}
for _, r in info.iterrows():
    fname = str(r[name_col]).strip()
    fname = fname if fname.lower().endswith(".png") else fname + ".png"
    mpath = os.path.join(msk_dir, fname)
    lab = str(r[lab_col]).strip().lower() if lab_col else label_from_rgb_mask(mpath)
    if lab not in LABELS:
        continue
    if dop_col and truthy(r[dop_col]):
        excluded["doppler"] += 1
        continue
    if comb_col and truthy(r[comb_col]):
        excluded["combined"] += 1
        continue
    rows.append(dict(dataset="BUS-UCLM", image_id=os.path.splitext(fname)[0],
                     patient_id="uclm_" + fname.split("_")[0], label=lab, device=DEVICE_UCLM,
                     src_path=os.path.join(img_dir, fname), mask_paths=mpath))
print("BUS-UCLM excluded:", excluded)

meta = pd.DataFrame(rows)
meta["exists"] = meta["src_path"].map(os.path.exists)
if (~meta.exists).any():
    print("WARNING: missing image files:", int((~meta.exists).sum()))
    meta = meta[meta.exists].reset_index(drop=True)
summary = meta.pivot_table(index="dataset", columns="label", values="image_id", aggfunc="count", fill_value=0)
summary["patients"] = meta.groupby("dataset")["patient_id"].nunique()
print(summary)
print(meta.groupby(["dataset", "device"]).size().rename("images"))

# %% [markdown]
# ## 3. Working copies and lesion crops
# Images are downscaled only when the longest side exceeds 512 pixels. A lesion crop is the mask bounding box enlarged by 30% on each side (minimum side 64 pixels).

# %%
def load_mask_union(mask_paths, shape):
    out = np.zeros(shape, np.uint8)
    for p in [p for p in str(mask_paths).split("|") if p]:
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m.max(axis=2)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out |= (m > 0).astype(np.uint8)
    return out * 255

def prepare_one(dataset, image_id, src_path, mask_paths):
    img_out = os.path.join(WORK_DIR, dataset, "images", image_id + ".png")
    msk_out = os.path.join(WORK_DIR, dataset, "masks", image_id + ".png")
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    h0, w0 = img.shape
    if not (os.path.exists(img_out) and os.path.exists(msk_out)):
        msk = load_mask_union(mask_paths, img.shape)
        s = min(1.0, MAX_SIDE / max(h0, w0))
        if s < 1.0:
            size = (int(round(w0 * s)), int(round(h0 * s)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, size, interpolation=cv2.INTER_NEAREST)
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        os.makedirs(os.path.dirname(msk_out), exist_ok=True)
        cv2.imwrite(img_out, img)
        cv2.imwrite(msk_out, msk)
    h, w = cv2.imread(img_out, cv2.IMREAD_GRAYSCALE).shape
    lesion_frac = float((cv2.imread(msk_out, cv2.IMREAD_GRAYSCALE) > 0).mean())
    return dict(orig_h=h0, orig_w=w0, work_h=h, work_w=w, work_path=img_out, mask_path=msk_out, lesion_frac=lesion_frac)

def crop_roi(dataset, image_id, work_path, mask_path):
    """Lesion-centred crop: bounding box of the mask enlarged by ROI_MARGIN on each side."""
    out = os.path.join(WORK_DIR, dataset, "roi", image_id + ".png")
    img = cv2.imread(work_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    H, W = img.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my = max(int(round(ROI_MARGIN * (y1 - y0))), (ROI_MIN_SIDE - (y1 - y0) + 1) // 2, 0)
    mx = max(int(round(ROI_MARGIN * (x1 - x0))), (ROI_MIN_SIDE - (x1 - x0) + 1) // 2, 0)
    y0, y1, x0, x1 = max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, img[y0:y1, x0:x1])
    return dict(roi_path=out, roi_box=f"{y0},{y1},{x0},{x1}", roi_h=y1 - y0, roi_w=x1 - x0)

t0 = time.time()
prep = Parallel(n_jobs=N_JOBS)(delayed(prepare_one)(d, i, s, m)
                               for d, i, s, m in zip(meta.dataset, meta.image_id, meta.src_path, meta.mask_paths))
meta = pd.concat([meta, pd.DataFrame(prep)], axis=1)
empty = meta.lesion_frac == 0
print(f"Prepared {len(meta)} images in {time.time() - t0:.0f}s | excluded (empty lesion mask): {int(empty.sum())}")
meta = meta[~empty].reset_index(drop=True)
rois = Parallel(n_jobs=N_JOBS)(delayed(crop_roi)(d, i, w, m)
                               for d, i, w, m in zip(meta.dataset, meta.image_id, meta.work_path, meta.mask_path))
meta = pd.concat([meta, pd.DataFrame(rois)], axis=1)
print(meta.groupby("dataset")[["roi_h", "roi_w"]].median().rename(columns=lambda c: "median " + c))
meta.to_csv(os.path.join(PROJECT_DIR, "tables", "metadata_all.csv"), index=False)

# %% [markdown]
# ## 4. Speckle measurement utilities
# Display values are mapped back to intensity assuming 50 dB log compression. Grain size is the lag at which the autocorrelation of the high-pass residual falls to 0.5.

# %%
def local_stats(x, win):
    m = uniform_filter(x, win, mode="reflect")
    v = np.maximum(uniform_filter(x * x, win, mode="reflect") - m * m, 0.0)
    return m, v

def to_intensity(x, dr=DISPLAY_DR_DB):
    return 10.0 ** ((x.astype(np.float64) - 1.0) * dr / 10.0)

def to_display(I, dr=DISPLAY_DR_DB):
    return np.clip(1.0 + 10.0 * np.log10(np.maximum(I, 1e-12)) / dr, 0.0, 1.0).astype(np.float32)

def odd_win(v, lo=WIN_MIN, hi=WIN_MAX):
    return int(np.clip(2 * int(np.ceil(v)) + 1, lo, hi))

def grain_windows(gy, gx):
    """Anisotropic window of about four grain half-widths in each direction (axial, lateral)."""
    return (odd_win(2 * gy), odd_win(2 * gx))

def speckle_level(I, mask, win=(STAT_WIN, STAT_WIN)):
    """Median squared local coefficient of variation over tissue pixels (intensity domain)."""
    m, v = local_stats(I, win)
    return float(np.median(v[mask] / (m[mask] ** 2))) if mask.any() else 0.0

def residual_acf(x, mask, half=6, smooth=3.0):
    """Normalised autocorrelation (lags -half..half) of the high-pass residual inside tissue."""
    x = x.astype(np.float64)
    M, N = x.shape
    r = (x - gaussian_filter(x, smooth)) * mask
    num = np.real(np.fft.ifft2(np.abs(np.fft.fft2(r, (2 * M, 2 * N))) ** 2))
    den = np.real(np.fft.ifft2(np.abs(np.fft.fft2(mask.astype(np.float64), (2 * M, 2 * N))) ** 2))
    k = np.arange(-half, half + 1)
    acf = (num / np.maximum(den, 1.0))[np.ix_(k % (2 * M), k % (2 * N))]
    return acf / acf[half, half], k

def speckle_index(x):
    m, v = local_stats(x, STAT_WIN)
    t = m > TISSUE_THR
    return float(np.mean(np.sqrt(v[t]) / m[t]))

def grain_size(x, mask, half=GRAIN_HALF):
    """Speckle grain half-width (pixels) along rows (axial) and columns (lateral): lag where ACF falls to 0.5."""
    acf, _ = residual_acf(x, mask, half, smooth=GRAIN_SMOOTH)
    out = []
    for prof in (acf[half:, half], acf[half, half:]):
        below = np.where(prof < 0.5)[0]
        if len(below) == 0:
            out.append(float(half))
        else:
            j = below[0]
            out.append(float(j - 1 + (prof[j - 1] - 0.5) / (prof[j - 1] - prof[j] + 1e-12)))
    return out

# %% [markdown]
# ## 5. Speckle fingerprint per scanner
# Speckle level and grain size of the full images, summarised by scanner (Table 2 of the paper). The grain sizes also tune the filters in notebook 02.

# %%
def raw_stats(work_path):
    x = cv2.imread(work_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mask = x > TISSUE_THR
    gy, gx = grain_size(x, mask)
    return dict(speckle_level=speckle_level(to_intensity(x), mask, grain_windows(gy, gx)), SI=speckle_index(x),
                grain_axial=gy, grain_lateral=gx, mean_gray=float(x[mask].mean()))

st = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(raw_stats)(p) for p in meta.work_path))
speckle = pd.concat([meta[["dataset", "device", "image_id", "label"]], st], axis=1)
meta["grain_axial"], meta["grain_lateral"] = st["grain_axial"].values, st["grain_lateral"].values
print("Images whose lateral grain reached the measurement limit:", int((st.grain_lateral >= GRAIN_HALF).sum()))
speckle.to_csv(os.path.join(PROJECT_DIR, "tables", "speckle_stats_raw.csv"), index=False)
print(speckle.groupby("device")[["speckle_level", "SI", "grain_axial", "grain_lateral", "mean_gray"]].median().round(3))

cols = ["speckle_level", "SI", "grain_axial", "grain_lateral"]
devices = sorted(speckle.device.unique())
fig, ax = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4))
for a, c in zip(ax, cols):
    a.boxplot([speckle.loc[speckle.device == d, c] for d in devices], showfliers=False)
    a.set_xticks(range(1, len(devices) + 1))
    a.set_xticklabels(devices, rotation=60, ha="right", fontsize=7)
    a.set_title(c)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "figures", "speckle_by_device.png"), dpi=300)
plt.show()
