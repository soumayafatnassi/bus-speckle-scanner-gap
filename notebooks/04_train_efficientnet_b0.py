# %% [markdown]
# # 04 · Second network (EfficientNet-B0)
#
# Repeats notebook 03 with EfficientNet-B0, using the same splits, filtered crops and settings. Results are written to `results_week2_efficientnet_b0`.
#
# **Colab with a T4 GPU; about two hours.**

# %%
# ---------------- 0. Setup ----------------
import os, sys, time, json, copy, random
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2 as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_DIR = os.environ.get(
    "BUS_PROJECT_DIR", "/content/drive/MyDrive/bus_crossscanner" if IN_COLAB else "./bus_crossscanner")
FILT_DIR = os.path.join(PROJECT_DIR, "filtered_v2")
ARCH = os.environ.get("BUS_ARCH", "efficientnet_b0")              # resnet18 (main) or efficientnet_b0 (second network)
RES_DIR = os.path.join(PROJECT_DIR, "results_week2" if ARCH == "resnet18" else f"results_week2_{ARCH}")
SPLIT_PATH = os.path.join(PROJECT_DIR, "results_week2", "splits.csv")   # one set of splits shared by both networks
PRED_DIR = os.path.join(RES_DIR, "predictions")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache_week2")
for d in [RES_DIR, PRED_DIR, CACHE_DIR, os.path.join(RES_DIR, "figures"), os.path.dirname(SPLIT_PATH)]:
    os.makedirs(d, exist_ok=True)

FAST = os.environ.get("BUS_TEST_FAST") is not None          # only for local testing of the code
FILTERS = ["raw", "median", "lee", "srad", "bm3d"]
IMG = 64 if FAST else 224
EPOCHS = 2 if FAST else 20
BATCH, LR, WD = 32, 1e-4, 1e-4
N_FOLDS = 3 if FAST else 5
VAL_FOLDS = 7                     # one of 7 patient groups of the training part (~14%) is used for model selection
SEED = 2026
PRETRAINED = not FAST
MIN_TEST = 20                     # smallest target domain we evaluate on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if IN_COLAB else 0
assert ARCH in ("resnet18", "efficientnet_b0"), f"unknown network {ARCH}"
print("Network:", ARCH, "| device:", DEVICE, "| image size:", IMG, "| epochs:", EPOCHS, "| folds:", N_FOLDS)
if DEVICE == "cpu" and not FAST:
    print("WARNING: no GPU. Choose Runtime > Change runtime type > T4 GPU, otherwise training will be very slow.")

# %% [markdown]
# ## 1. Data and domains

# %%
meta = pd.read_csv(os.path.join(PROJECT_DIR, "tables", "metadata_all.csv"))
meta["key"] = meta["dataset"] + "/" + meta["image_id"].astype(str)
meta["y"] = (meta["label"] == "malignant").astype(int)

def scanner_name(row):
    if row.dataset == "BUS-BRA":
        d = str(row.device).replace("BUS-BRA", "").split("@")[0].strip()
        return "BRA-" + "".join(ch for ch in d if ch.isalnum())
    return {"BUSI": "BUSI-LogiqE9", "BUS-UCLM": "UCLM-SiemensS2000"}[row.dataset]

meta["site"] = meta["dataset"]
meta["scanner"] = meta.apply(scanner_name, axis=1)
bra_counts = meta[meta.dataset == "BUS-BRA"]["scanner"].value_counts()
print(meta.groupby(["site", "scanner"]).agg(images=("key", "size"), malignant=("y", "sum"),
                                              patients=("patient_id", "nunique")))

EXPERIMENTS = {
    "cross_site": dict(column="site", pool=meta.index,
                       sources=["BUSI", "BUS-BRA", "BUS-UCLM"]),
    "cross_scanner": dict(column="scanner", pool=meta.index[meta.dataset == "BUS-BRA"],
                          sources=list(bra_counts.index[:2])),      # the two largest BUS-BRA scanners
}
for name, e in EXPERIMENTS.items():
    doms = meta.loc[e["pool"], e["column"]].value_counts()
    e["targets_all"] = [d for d in doms.index if doms[d] >= MIN_TEST]
    print(f"{name}: sources {e['sources']} | evaluated domains {e['targets_all']}")

# %% [markdown]
# ## 2. Patient-level splits (shared by all filters and both networks)

# %%
def make_splits():
    rows = []
    for exp, e in EXPERIMENTS.items():
        sub = meta.loc[e["pool"]]
        for src in e["sources"]:
            d = sub[sub[e["column"]] == src]
            outer = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
            for k, (tv, te) in enumerate(outer.split(d, d.y, d.patient_id)):
                dtv = d.iloc[tv]
                # small sources get fewer groups: never more groups than patients of the rarer class
                n_inner = max(2, min(VAL_FOLDS, dtv.groupby("y").patient_id.nunique().min()))
                inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=SEED + k)
                tr, va = next(inner.split(dtv, dtv.y, dtv.patient_id))
                for idx, role in [(dtv.index[tr], "train"), (dtv.index[va], "val"), (d.index[te], "test")]:
                    rows += [dict(experiment=exp, source=src, fold=k, key=meta.key[i], role=role) for i in idx]
    return pd.DataFrame(rows)

if os.path.exists(SPLIT_PATH):
    splits = pd.read_csv(SPLIT_PATH)
else:
    splits = make_splits()
    splits.to_csv(SPLIT_PATH, index=False)
leak = splits.pivot_table(index=["experiment", "source", "fold", "key"], columns="role", aggfunc="size", fill_value=0)
assert (leak.sum(axis=1) == 1).all(), "an image appears in two roles of the same fold"
pid = meta.set_index("key").patient_id
for (exp, src, k), g in splits.groupby(["experiment", "source", "fold"]):
    roles = {r: set(pid[g.key[g.role == r]]) for r in ["train", "val", "test"]}
    assert not (roles["train"] & roles["test"]) and not (roles["val"] & roles["test"]), f"patient leak in {exp}/{src}/{k}"
print(splits.groupby(["experiment", "source", "role"]).size().unstack())

# %% [markdown]
# ## 3. Load lesion crops into memory
# Crops are zero-padded to a square and resized to 224 × 224.

# %%
def square_resize(x):
    h, w = x.shape
    s = max(h, w)
    x = cv2.copyMakeBorder(x, (s - h) // 2, s - h - (s - h) // 2, (s - w) // 2, s - w - (s - w) // 2,
                           cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(x, (IMG, IMG), interpolation=cv2.INTER_AREA if s > IMG else cv2.INTER_LINEAR)

def load_filter(name):
    """All lesion crops of one filter as an (N, IMG, IMG) uint8 array, in the row order of `meta`."""
    cp = os.path.join(CACHE_DIR, f"{name}_{IMG}.npy")
    if os.path.exists(cp):
        arr = np.load(cp)
        if len(arr) == len(meta):
            return arr
    npz = os.path.join(FILT_DIR, f"{name}.npz")                 # written by the week-1b notebook
    keys = [f"{d}__{i}" for d, i in zip(meta.dataset, meta.image_id)]
    if os.path.exists(npz):
        with np.load(npz) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise KeyError(f"{len(missing)} images missing in {npz} (e.g. {missing[0]})")
            arr = np.stack([square_resize(z[k]) for k in keys])
    else:
        paths = [os.path.join(FILT_DIR, name, d, f"{i}.png") for d, i in zip(meta.dataset, meta.image_id)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Filter '{name}' not found: run the week-1b notebook first ({npz}).")
        arr = np.stack([square_resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in paths])
    np.save(cp, arr)
    return arr

# %% [markdown]
# ## 4. Model and training

# %%
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
AUG = T.Compose([T.RandomHorizontalFlip(),
                 T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))])

class CropSet(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y.astype(np.float32), train
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float().div(255).unsqueeze(0)
        if self.train:
            x = AUG(x)
        return (x.repeat(3, 1, 1) - MEAN) / STD, torch.tensor(self.y[i])

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_model():
    if ARCH == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if PRETRAINED else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(DEVICE)

@torch.no_grad()
def predict(model, X):
    model.eval()
    dl = DataLoader(CropSet(X, np.zeros(len(X)), False), batch_size=128, num_workers=NUM_WORKERS)
    out = []
    for xb, _ in dl:
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            out.append(torch.sigmoid(model(xb.to(DEVICE)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan

def train_model(Xtr, ytr, Xva, yva, seed):
    seed_everything(seed)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(CropSet(Xtr, ytr, True), batch_size=BATCH, shuffle=True, drop_last=len(Xtr) > BATCH,
                    num_workers=NUM_WORKERS, generator=g)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(xb.to(DEVICE)).squeeze(1), yb.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        auc = safe_auc(yva, predict(model, Xva))
        if np.nan_to_num(auc, nan=0.0) > best:
            best, best_state, best_ep = np.nan_to_num(auc, nan=0.0), copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    return model, best, best_ep

def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    ts = np.unique(p)
    j = [((p >= t) & (y == 1)).sum() / (y == 1).sum() + ((p < t) & (y == 0)).sum() / (y == 0).sum() for t in ts]
    return float(ts[int(np.argmax(j))])

# %% [markdown]
# ## 5. Run all experiments (resumable)
# Each model writes its predictions to its own file, so finished models are skipped on a re-run.

# %%
def filter_ready(f):
    return (os.path.exists(os.path.join(FILT_DIR, f"{f}.npz")) or os.path.exists(os.path.join(FILT_DIR, f))
            or os.path.exists(os.path.join(CACHE_DIR, f"{f}_{IMG}.npy")))

READY = [f for f in FILTERS if filter_ready(f)]
if len(READY) < len(FILTERS):
    print("Not ready yet, skipped for now:", [f for f in FILTERS if f not in READY],
          "-> when week-1b finishes, run this notebook again to add them (finished models are kept).")
key_to_row = pd.Series(np.arange(len(meta)), index=meta.key)
jobs = [(f, exp, src, k) for f in READY for exp, e in EXPERIMENTS.items() for src in e["sources"]
        for k in range(N_FOLDS)]
todo = [j for j in jobs if not os.path.exists(os.path.join(PRED_DIR, "__".join(map(str, j)) + ".csv"))]
print(f"{len(jobs)} models in total, {len(todo)} still to train")

t_start, done = time.time(), 0
for f in READY:
    if not any(j[0] == f for j in todo):
        continue
    X = load_filter(f)
    for (ff, exp, src, k) in [j for j in todo if j[0] == f]:
        e = EXPERIMENTS[exp]
        s = splits[(splits.experiment == exp) & (splits.source == src) & (splits.fold == k)]
        idx = {r: key_to_row[s.key[s.role == r]].values for r in ["train", "val", "test"]}
        t0 = time.time()
        model, val_auc, best_ep = train_model(X[idx["train"]], meta.y.values[idx["train"]],
                                              X[idx["val"]], meta.y.values[idx["val"]], seed=SEED + k)
        thr = youden_threshold(meta.y.values[idx["val"]], predict(model, X[idx["val"]]))
        src_patients = set(meta.patient_id.values[np.concatenate([idx["train"], idx["val"]])])
        evals = {"internal": idx["test"]}
        for t in e["targets_all"]:
            if t == src:
                continue
            m = meta.loc[e["pool"]]
            m = m[(m[e["column"]] == t) & (~m.patient_id.isin(src_patients))]
            evals[t] = key_to_row[m.key].values
        rows = []
        for t, ii in evals.items():
            p = predict(model, X[ii])
            rows.append(pd.DataFrame(dict(filter=f, experiment=exp, source=src, fold=k, target=t,
                                          key=meta.key.values[ii], y=meta.y.values[ii], prob=p, thr=thr,
                                          val_auc=val_auc, best_epoch=best_ep)))
        pd.concat(rows).to_csv(os.path.join(PRED_DIR, f"{f}__{exp}__{src}__{k}.csv"), index=False)
        done += 1
        el = time.time() - t_start
        print(f"[{done}/{len(todo)}] {f:6s} {exp:13s} {src:18s} fold {k} | val AUROC {val_auc:.3f} "
              f"(epoch {best_ep}) | {time.time() - t0:.0f}s | ETA {el / done * (len(todo) - done) / 60:.0f} min")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Quick results: AUROC per filter and change relative to raw images
# Paired DeLong test with Holm correction. The paper reports the patient-level bootstrap computed in notebook 05.

# %%
def compute_midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1)
        i = a
    out = np.empty(n); out[j] = t + 1
    return out

def delong_paired(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs on the same cases (Sun & Xu, 2014)."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([compute_midrank(r) for r in pos]); ty = np.array([compute_midrank(r) for r in neg])
    tz = np.array([compute_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    from scipy.stats import norm
    return aucs[0] - aucs[1], float(2 * norm.sf(abs(z)))

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

preds = pd.concat([pd.read_csv(os.path.join(PRED_DIR, f)) for f in sorted(os.listdir(PRED_DIR)) if f.endswith(".csv")])

def fold_metrics(g):
    y, p, t = g.y.values, g.prob.values, g.thr.values[0]
    yhat = p >= t
    sens = (yhat & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~yhat & (y == 0)).sum() / max((y == 0).sum(), 1)
    return pd.Series(dict(auc=safe_auc(y, p), bal_acc=(sens + spec) / 2, sens=sens, spec=spec, n=len(y)))

GROUP = ["experiment", "source", "target", "filter", "fold"]
per_fold = pd.DataFrame([dict(zip(GROUP, k), **fold_metrics(g)) for k, g in preds.groupby(GROUP)])
summary = per_fold.groupby(["experiment", "source", "target", "filter"]).agg(
    auc_mean=("auc", "mean"), auc_sd=("auc", "std"), bal_acc=("bal_acc", "mean"),
    sens=("sens", "mean"), spec=("spec", "mean"), n=("n", "mean")).reset_index()

ens = preds.groupby(["experiment", "source", "target", "filter", "key"]).agg(y=("y", "first"), prob=("prob", "mean")).reset_index()
comp = []
for (exp, src, tgt), g in ens.groupby(["experiment", "source", "target"]):
    base = g[g["filter"] == "raw"].set_index("key")
    for f in READY:
        cur = g[g["filter"] == f].set_index("key")
        common = base.index.intersection(cur.index)
        if f == "raw" or len(common) == 0:
            comp.append(dict(experiment=exp, source=src, target=tgt, filter=f,
                             auc_ens=safe_auc(base.y[common], base.prob[common]) if f == "raw" else np.nan,
                             delta=0.0 if f == "raw" else np.nan, p=np.nan))
            continue
        y = base.y[common].values
        if len(np.unique(y)) < 2:
            continue
        d, p = delong_paired(y, cur.prob[common].values, base.prob[common].values)
        comp.append(dict(experiment=exp, source=src, target=tgt, filter=f,
                         auc_ens=safe_auc(y, cur.prob[common].values), delta=d, p=p))
comp = pd.DataFrame(comp)
for exp in comp.experiment.unique():
    m = (comp.experiment == exp) & comp.p.notna()
    comp.loc[m, "p_holm"] = holm(comp.loc[m, "p"].values)
table = summary.merge(comp, on=["experiment", "source", "target", "filter"], how="left")
table["filter"] = pd.Categorical(table["filter"], FILTERS, ordered=True)
table = table.sort_values(["experiment", "source", "target", "filter"])
table.to_csv(os.path.join(RES_DIR, "results_table.csv"), index=False)
pd.set_option("display.width", 200)
print(table[["experiment", "source", "target", "filter", "auc_mean", "auc_sd", "auc_ens", "delta", "p_holm",
             "bal_acc"]].round(3).to_string(index=False))

# %% [markdown]
# ### Heat map of ΔAUROC on external targets (* Holm-adjusted p < 0.05)

# %%
ext = table[(table.target != "internal") & (table["filter"] != "raw")]
for exp, g in ext.groupby("experiment"):
    piv = g.pivot_table(index=["source", "target"], columns="filter", values="delta", observed=True)
    pv = g.pivot_table(index=["source", "target"], columns="filter", values="p_holm", observed=True)
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 2.5, 0.55 * len(piv) + 1.5))
    lim = max(0.05, float(np.nanmax(np.abs(piv.values))))
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{s} → {t}" for s, t in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, pp = piv.values[i, j], pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}{'*' if pp < 0.05 else ''}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="ΔAUROC vs raw (ensemble)")
    ax.set_title(f"{exp}: external targets (* Holm-adjusted p < 0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "figures", f"delta_auc_{exp}.png"), dpi=300)
    plt.show()

# %% [markdown]
# ## 7. Scanner separability
# Frozen ImageNet ResNet-18 features of each crop; a linear classifier separates each pair of domains under patient-grouped cross-validation. A lower AUROC means the domains are harder to tell apart.

# %%
if ARCH != "resnet18":
    print("Domain-gap analysis is done once, in the main (ResNet-18) notebook.")
    print("All results are in:", RES_DIR)
else:
    @torch.no_grad()
    def frozen_features(X):
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        m = torchvision.models.resnet18(weights=weights); m.fc = nn.Identity(); m = m.to(DEVICE).eval()
        dl = DataLoader(CropSet(X, np.zeros(len(X)), False), batch_size=128, num_workers=NUM_WORKERS)
        return np.concatenate([m(xb.to(DEVICE)).float().cpu().numpy() for xb, _ in dl])

    def domain_auc(F, dom, groups):
        """Patient-grouped 5-fold AUROC of a linear classifier that separates two domains."""
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        scores = np.zeros(len(dom))
        for tr, te in cv.split(F, dom, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            clf.fit(F[tr], dom[tr]); scores[te] = clf.predict_proba(F[te])[:, 1]
        return roc_auc_score(dom, scores)

    gap_path = os.path.join(RES_DIR, "domain_gap.csv")
    gap_rows = []
    for f in READY:
        feat_path = os.path.join(CACHE_DIR, f"features_{f}_{IMG}.npy")
        F = np.load(feat_path) if os.path.exists(feat_path) else frozen_features(load_filter(f))
        np.save(feat_path, F)
        for exp, e in EXPERIMENTS.items():
            doms = e["targets_all"]
            for a in range(len(doms)):
                for b in range(a + 1, len(doms)):
                    sel = meta.index[meta.index.isin(e["pool"]) & meta[e["column"]].isin([doms[a], doms[b]])]
                    y = (meta.loc[sel, e["column"]] == doms[b]).astype(int).values
                    gap_rows.append(dict(filter=f, experiment=exp, pair=f"{doms[a]} vs {doms[b]}",
                                         domain_auc=domain_auc(F[sel], y, meta.loc[sel, "patient_id"].values)))
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(gap_path, index=False)
    gp = gap.pivot_table(index=["experiment", "pair"], columns="filter", values="domain_auc")[READY]
    print(gp.round(3))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(gp) + 1.5))
    for j, f in enumerate(READY):
        ax.scatter(gp[f].values, np.arange(len(gp)), label=f, s=30)
    ax.set_yticks(range(len(gp))); ax.set_yticklabels([f"{e}: {p}" for e, p in gp.index], fontsize=8)
    ax.set_xlabel("domain-classifier AUROC (lower = scanners less distinguishable)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "figures", "domain_gap.png"), dpi=300)
    plt.show()
    print("All results are in:", RES_DIR)
