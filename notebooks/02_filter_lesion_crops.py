# %% [markdown]
# # 02 · Filtering the lesion crops
#
# Applies the four speckle filters (median 5 × 5, Lee, SRAD and BM3D for correlated noise) to every lesion crop. Windows and noise spectra are adapted to the grain size measured in notebook 01. Results are saved in chunks and the run resumes automatically.
#
# **Colab CPU runtime; about one to two hours.**

# %%
# ---------------- 0. Setup ----------------
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bm3d"], check=True)

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
from joblib import Parallel, delayed

PROJECT_DIR = os.environ.get(
    "BUS_PROJECT_DIR", "/content/drive/MyDrive/bus_crossscanner" if IN_COLAB else "./bus_crossscanner")
OUT_DIR = os.path.join(PROJECT_DIR, "filtered_v2")
PART_DIR = os.path.join(OUT_DIR, "parts")
os.makedirs(PART_DIR, exist_ok=True)

FILTERS = ["raw", "median", "lee", "srad", "bm3d"]
CHUNK = 250                        # images per saved batch
MEDIAN_K = 5
WIN_MIN, WIN_MAX = 5, 31
SRAD_ITERS, SRAD_DT = 50, 0.5     # fixed: more iterations over-smoothed real tissue texture
SRAD_Q0_QUANTILE = 0.25            # q0 from the lower quartile of local CoV^2 = homogeneous speckle regions
SRAD_Q0_EVERY = 5                  # re-estimate q0 every 5 iterations (same result, 1.5x faster)
# Version of each filter's settings. When a version changes, its old outputs (and week-2 caches and
# predictions that used them) are deleted automatically and recomputed.
FILTER_VERSIONS = {"raw": "1", "median": "1", "lee": "1", "srad": "2", "bm3d": "1"}
STAT_WIN = 5
DISPLAY_DR_DB = 50.0
TISSUE_THR = 0.02
N_JOBS = max(1, os.cpu_count() or 1)
print("Project folder:", PROJECT_DIR, "| CPU cores:", N_JOBS)

# %% [markdown]
# ## 1. Load the lesion crops and grain sizes

# %%
meta = pd.read_csv(os.path.join(PROJECT_DIR, "tables", "metadata_all.csv"))
st = pd.read_csv(os.path.join(PROJECT_DIR, "tables", "speckle_stats_raw.csv"))
meta = meta.drop(columns=[c for c in ["grain_axial", "grain_lateral"] if c in meta.columns])
meta = meta.merge(st[["dataset", "image_id", "grain_axial", "grain_lateral"]], on=["dataset", "image_id"], how="left")
assert meta.grain_lateral.notna().all(), "grain sizes missing: run section 5 of the week-1 notebook first"
meta["roi_file"] = [os.path.join(PROJECT_DIR, "work", d, "roi", f"{i}.png") for d, i in zip(meta.dataset, meta.image_id)]
keys = [f"{d}__{i}" for d, i in zip(meta.dataset, meta.image_id)]

t0 = time.time()
with ThreadPoolExecutor(16) as ex:                       # many parallel reads are much faster on Drive
    crops = list(ex.map(lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE), meta.roi_file))
missing = [p for p, c in zip(meta.roi_file, crops) if c is None]
assert not missing, f"{len(missing)} lesion crops missing, e.g. {missing[0]}"
crops = [c.astype(np.float32) / 255.0 for c in crops]
GY, GX = meta.grain_axial.values, meta.grain_lateral.values
print(f"Loaded {len(crops)} lesion crops in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Filters (Lee and SRAD in the intensity domain; BM3D with an estimated noise spectrum)

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
    return (odd_win(2 * gy), odd_win(2 * gx))

def speckle_level(I, mask, win=(STAT_WIN, STAT_WIN)):
    m, v = local_stats(I, win)
    return float(np.median(v[mask] / (m[mask] ** 2))) if mask.any() else 0.0

def residual_acf(x, mask, half=6, smooth=3.0):
    x = x.astype(np.float64)
    M, N = x.shape
    r = (x - gaussian_filter(x, smooth)) * mask
    num = np.real(np.fft.ifft2(np.abs(np.fft.fft2(r, (2 * M, 2 * N))) ** 2))
    den = np.real(np.fft.ifft2(np.abs(np.fft.fft2(mask.astype(np.float64), (2 * M, 2 * N))) ** 2))
    k = np.arange(-half, half + 1)
    acf = (num / np.maximum(den, 1.0))[np.ix_(k % (2 * M), k % (2 * N))]
    return acf / acf[half, half], k

def f_raw(x, gy, gx):
    return x

def f_median(x, gy, gx):
    return cv2.medianBlur(np.round(x * 255).astype(np.uint8), MEDIAN_K).astype(np.float32) / 255.0

def f_lee(x, gy, gx):
    win = grain_windows(gy, gx)
    mask = x > TISSUE_THR
    I = to_intensity(x)
    m, v = local_stats(I, win)
    w = np.clip(1.0 - speckle_level(I, mask, win) / np.maximum(v / (m * m + 1e-30), 1e-30), 0.0, 1.0)
    return to_display(m + w * (I - m))

def homogeneous_speckle_level(I, mask, win, q=SRAD_Q0_QUANTILE):
    """Lower quantile of the local squared coefficient of variation: speckle level of homogeneous tissue."""
    m, v = local_stats(I, win)
    return float(np.quantile(v[mask] / (m[mask] ** 2), q)) if mask.any() else 0.0

def f_srad(x, gy, gx, dt=SRAD_DT):
    win = grain_windows(gy, gx)
    mask = x > TISSUE_THR
    I = np.maximum(to_intensity(x), 1e-8)
    for it in range(SRAD_ITERS):
        if it % SRAD_Q0_EVERY == 0:
            q0 = homogeneous_speckle_level(I, mask, win)
        P = np.pad(I, 1, mode="edge")
        dN, dS = P[:-2, 1:-1] - I, P[2:, 1:-1] - I
        dW, dE = P[1:-1, :-2] - I, P[1:-1, 2:] - I
        g2 = (dN ** 2 + dS ** 2 + dW ** 2 + dE ** 2) / (I * I)
        lap = (dN + dS + dW + dE) / I
        q2 = (0.5 * g2 - lap * lap / 16.0) / ((1.0 + 0.25 * lap) ** 2 + 1e-12)
        c = np.clip(1.0 / (1.0 + (q2 - q0) / (q0 * (1.0 + q0) + 1e-12)), 0.0, 1.0)
        C = np.pad(c, 1, mode="edge")
        I = np.maximum(I + (dt / 4.0) * (c * dN + C[2:, 1:-1] * dS + c * dW + C[1:-1, 2:] * dE), 1e-8)
    return to_display(I)

def estimate_noise_psd(x, mask, gy, gx):
    gm = max(gy, gx)
    half = int(np.clip(np.ceil(3 * gm), 6, 24))
    acf, k = residual_acf(x, mask, half, smooth=float(np.clip(2 * gm, 3, 12)))
    taper = np.outer(np.hanning(2 * half + 3)[1:-1], np.hanning(2 * half + 3)[1:-1])
    _, v = local_stats(x.astype(np.float64), odd_win(3 * gm, 15, 41))
    acf = acf * taper * float(np.median(v[mask]))
    M, N = x.shape
    circ = np.zeros((M, N))
    circ[np.ix_(k % M, k % N)] = acf
    return np.maximum(np.real(np.fft.fft2(circ)), 0.0) * M * N

def f_bm3d(x, gy, gx):
    if not hasattr(np, "trapz"):          # bm3d/bm4d still call np.trapz (removed in recent NumPy)
        np.trapz = np.trapezoid
    import bm3d
    return np.asarray(bm3d.bm3d(x, sigma_psd=estimate_noise_psd(x, x > TISSUE_THR, gy, gx)), dtype=np.float32)

FILTER_FUNCS = {"raw": f_raw, "median": f_median, "lee": f_lee, "srad": f_srad, "bm3d": f_bm3d}

def run_one(name, x, gy, gx):
    return np.round(np.clip(FILTER_FUNCS[name](x, gy, gx), 0, 1) * 255).astype(np.uint8)

# %% [markdown]
# ## 3. Filtering in chunks (saved after each chunk, resumable)

# %%
n = len(crops)
chunks = [list(range(i, min(i + CHUNK, n))) for i in range(0, n, CHUNK)]

def invalidate_if_outdated(name):
    """Delete outputs made with older settings of this filter, including week-2 caches and predictions."""
    vfile = os.path.join(OUT_DIR, f"{name}.version")
    old = open(vfile).read().strip() if os.path.exists(vfile) else "1"
    if old == FILTER_VERSIONS[name]:
        return
    stale = [os.path.join(OUT_DIR, f"{name}.npz")] + glob.glob(os.path.join(PART_DIR, f"{name}_*.npz"))
    stale += glob.glob(os.path.join(PROJECT_DIR, "cache_week2", f"{name}_*.npy"))
    stale += glob.glob(os.path.join(PROJECT_DIR, "cache_week2", f"features_{name}_*.npy"))
    stale += glob.glob(os.path.join(PROJECT_DIR, "results_week2", "predictions", f"{name}__*.csv"))
    removed = [p for p in stale if os.path.exists(p)]
    for p in removed:
        os.remove(p)
    print(f"{name:6s}: settings changed (v{old} -> v{FILTER_VERSIONS[name]}), removed {len(removed)} old files")

for name in FILTERS:
    invalidate_if_outdated(name)
    final = os.path.join(OUT_DIR, f"{name}.npz")
    if os.path.exists(final):
        print(f"{name:6s}: already finished")
        continue
    t0, done_now = time.time(), 0
    for ci, idx in enumerate(chunks):
        part = os.path.join(PART_DIR, f"{name}_{ci:03d}.npz")
        if os.path.exists(part):
            continue
        res = Parallel(n_jobs=N_JOBS)(delayed(run_one)(name, crops[i], GY[i], GX[i]) for i in idx)
        np.savez(part, **{keys[i]: r for i, r in zip(idx, res)})
        done_now += len(idx)
        left = sum(len(c) for cj, c in enumerate(chunks)
                   if not os.path.exists(os.path.join(PART_DIR, f"{name}_{cj:03d}.npz")))
        rate = (time.time() - t0) / done_now
        print(f"{name:6s}: batch {ci + 1}/{len(chunks)} saved | {rate:.2f}s per image | "
              f"remaining for this filter ≈ {rate * left / 60:.0f} min")
    merged = {}
    for ci in range(len(chunks)):
        with np.load(os.path.join(PART_DIR, f"{name}_{ci:03d}.npz")) as z:
            merged.update({k: z[k] for k in z.files})
    assert len(merged) == n, f"{name}: {len(merged)} of {n} images"
    np.savez(final, **merged)
    for ci in range(len(chunks)):
        os.remove(os.path.join(PART_DIR, f"{name}_{ci:03d}.npz"))
    with open(os.path.join(OUT_DIR, f"{name}.version"), "w") as fh:
        fh.write(FILTER_VERSIONS[name])
    print(f"{name:6s}: finished, saved to {final}")

# %% [markdown]
# ## 4. No-reference quality measures (Table 3) and example crops (Figure 1)

# %%
def speckle_index(x):
    m, v = local_stats(x, STAT_WIN)
    t = m > TISSUE_THR
    return float(np.mean(np.sqrt(v[t]) / m[t]))

def edge_preservation(out, raw):
    d_out = np.abs(np.diff(out, axis=0)).sum() + np.abs(np.diff(out, axis=1)).sum()
    d_raw = np.abs(np.diff(raw, axis=0)).sum() + np.abs(np.diff(raw, axis=1)).sum()
    return float(d_out / (d_raw + 1e-12))

def mean_preservation(out, raw):
    t = raw > TISSUE_THR
    return float(out[t].mean() / (raw[t].mean() + 1e-12))

rows = []
for name in FILTERS:
    with np.load(os.path.join(OUT_DIR, f"{name}.npz")) as z:
        for i, k in enumerate(keys):
            out = z[k].astype(np.float32) / 255.0
            rows.append(dict(dataset=meta.dataset[i], image_id=meta.image_id[i], device=meta.device[i], filter=name,
                             SI=speckle_index(out), EPI=edge_preservation(out, crops[i]),
                             MPR=mean_preservation(out, crops[i])))
metrics = pd.DataFrame(rows)
metrics.to_csv(os.path.join(PROJECT_DIR, "tables", "nr_metrics_per_image.csv"), index=False)
table = metrics.groupby(["dataset", "filter"])[["SI", "EPI", "MPR"]].mean().round(3)
table.to_csv(os.path.join(PROJECT_DIR, "tables", "nr_metrics_summary.csv"))
print(table)

examples = [meta.index[meta.dataset == d][0] for d in ["BUSI", "BUS-BRA", "BUS-UCLM"]]
fig, ax = plt.subplots(len(examples), len(FILTERS), figsize=(3 * len(FILTERS), 3 * len(examples)), squeeze=False)
for c_, name in enumerate(FILTERS):
    with np.load(os.path.join(OUT_DIR, f"{name}.npz")) as z:
        for r, i in enumerate(examples):
            ax[r, c_].imshow(z[keys[i]], cmap="gray", vmin=0, vmax=255)
            ax[r, c_].set_title(f"{meta.dataset[i]} | {name}", fontsize=8)
            ax[r, c_].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "figures", "filters_examples.png"), dpi=300)
plt.show()
print("Done. Filtered images:", OUT_DIR)
