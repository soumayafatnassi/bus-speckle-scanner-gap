# %% [markdown]
# # 05 · Statistics and figures for the paper
#
# Reads the predictions of notebooks 03 and 04 (no training) and computes patient-level bootstrap intervals for AUROC and ΔAUROC with Holm correction (Tables S1 and S2), a summary per filter, the forest plot (Figure 2) and the separability-versus-transfer analysis (Figure 3).
#
# **Colab CPU runtime; a few minutes.**

# %%
# ---------------- 0. Setup ----------------
import os, sys, json, zlib
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

PROJECT_DIR = os.environ.get(
    "BUS_PROJECT_DIR", "/content/drive/MyDrive/bus_crossscanner" if IN_COLAB else "./bus_crossscanner")
RES2 = os.path.join(PROJECT_DIR, "results_week2")
NETWORKS = {"ResNet-18": RES2, "EfficientNet-B0": os.path.join(PROJECT_DIR, "results_week2_efficientnet_b0")}
PRIMARY = "ResNet-18"
OUT = os.path.join(PROJECT_DIR, "results_week3")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

FILTERS = ["raw", "median", "lee", "srad", "bm3d"]
B = int(os.environ.get("BUS_BOOT", 10000))   # resamples; enough for stable Holm-adjusted p-values
CHUNK_B = 2000                                # resamples processed at once (limits memory)
SEED = 2026
ALPHA = 0.05

NICE = {"BUSI": "BUSI", "BUS-BRA": "BUS-BRA", "BUS-UCLM": "BUS-UCLM",
        "BRA-GELogiq5": "Logiq 5", "BRA-GELogiq7": "Logiq 7",
        "BRA-ToshibaAplio300": "Aplio 300", "BRA-USystems": "U-Systems", "internal": "internal"}
nice = lambda s: NICE.get(s, s)

meta = pd.read_csv(os.path.join(PROJECT_DIR, "tables", "metadata_all.csv"))
meta["key"] = meta["dataset"] + "/" + meta["image_id"].astype(str)
patient_of = meta.set_index("key")["patient_id"]

parts = []
for net, d in NETWORKS.items():
    pdir = os.path.join(d, "predictions")
    files = sorted(f for f in os.listdir(pdir) if f.endswith(".csv")) if os.path.isdir(pdir) else []
    if not files:
        print(f"{net}: no predictions yet ({pdir})")
        continue
    parts.append(pd.concat([pd.read_csv(os.path.join(pdir, f)) for f in files], ignore_index=True).assign(network=net))
    print(f"{net}: {len(files)} prediction files")
preds = pd.concat(parts, ignore_index=True)
nets = [n for n in NETWORKS if n in set(preds.network)]
assert PRIMARY in nets, "ResNet-18 predictions are missing: run the week-2 notebook first"
present = [f for f in FILTERS if f in set(preds["filter"])]
assert "raw" in present, "raw predictions are missing"
print(f"networks: {nets} | filters: {present} | bootstrap resamples: {B}")

# One score per image: the mean of the five fold models for external targets.
# For the internal target every image is predicted once (out-of-fold), so the mean is that single prediction.
ens = (preds.groupby(["network", "experiment", "source", "target", "filter", "key"])
            .agg(y=("y", "first"), prob=("prob", "mean")).reset_index())
missing = set(ens.key) - set(patient_of.index)
assert not missing, f"{len(missing)} images in predictions are not in metadata_all.csv"
ens["patient"] = patient_of.loc[ens.key].values

# %% [markdown]
# ## 1. Patient-level bootstrap
# Patients are resampled with replacement (10 000 resamples). The same resamples are used for every filter and both networks, and the seed is fixed for each target, so the intervals are exactly reproducible.

# %%
def weighted_auc(score, y, W):
    """AUROC for each row of the weight matrix W (B x n); ties count one half."""
    order = np.argsort(score, kind="mergesort")
    s, yy, Wo = score[order], y[order].astype(float), W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    P = np.add.reduceat(Wo * yy, starts, axis=1)
    N = np.add.reduceat(Wo * (1 - yy), starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(axis=1)
    den = P.sum(axis=1) * N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

rows = []
for (net, exp, src, tgt), g in ens.groupby(["network", "experiment", "source", "target"], sort=False):
    base = g[g["filter"] == "raw"].set_index("key")
    keys = base.index
    for f in present:
        keys = keys.intersection(g[g["filter"] == f].key)
    base = base.loc[keys]
    y = base.y.values.astype(int)
    if len(np.unique(y)) < 2:
        continue
    pats, pidx = np.unique(base.patient.values, return_inverse=True)
    # same patient resamples for every filter and every network evaluated on this target
    rng = np.random.default_rng([SEED, zlib.crc32(f"{exp}|{src}|{tgt}".encode())])
    ones = np.ones((1, len(y)))
    score = {f: g[g["filter"] == f].set_index("key").prob.loc[keys].values for f in present}
    boot = {f: [] for f in present}
    for start in range(0, B, CHUNK_B):
        nb = min(CHUNK_B, B - start)
        W = rng.multinomial(len(pats), np.full(len(pats), 1 / len(pats)), size=nb)[:, pidx].astype(float)
        for f in present:
            boot[f].append(weighted_auc(score[f], y, W))
    boot = {f: np.concatenate(v) for f, v in boot.items()}
    point = {f: float(weighted_auc(score[f], y, ones)[0]) for f in present}
    for f in present:
        r = dict(network=net, experiment=exp, source=src, target=tgt, filter=f, n_images=len(y), n_patients=len(pats),
                 n_malignant=int(y.sum()), auc=point[f],
                 auc_lo=np.nanpercentile(boot[f], 2.5), auc_hi=np.nanpercentile(boot[f], 97.5))
        if f != "raw":
            d = boot[f] - boot["raw"]
            d = d[np.isfinite(d)]
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            r.update(delta=point[f] - point["raw"], delta_lo=np.percentile(d, 2.5),
                     delta_hi=np.percentile(d, 97.5), p_boot=min(1.0, max(p, 1 / len(d))))
        rows.append(r)

res = pd.DataFrame(rows)
for (net, exp), _ in res.groupby(["network", "experiment"]):
    m = (res.network == net) & (res.experiment == exp) & res.p_boot.notna()
    res.loc[m, "p_holm"] = holm(res.loc[m, "p_boot"].values)
res["significant"] = res.p_holm < ALPHA
res["filter"] = pd.Categorical(res["filter"], present, ordered=True)
res["network"] = pd.Categorical(res["network"], nets, ordered=True)
res = res.sort_values(["network", "experiment", "source", "target", "filter"]).reset_index(drop=True)
res.to_csv(os.path.join(OUT, "auc_delta_bootstrap.csv"), index=False)

def fmt_ci(v, lo, hi, signed=False):
    if not np.isfinite(v):
        return ""
    s = "+.3f" if signed else ".3f"
    return f"{v:{s}} [{lo:{s}}, {hi:{s}}]"

show = res.copy()
show["transfer"] = show.source.map(nice) + " → " + show.target.map(nice)
show["AUROC [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(show.auc, show.auc_lo, show.auc_hi)]
show["ΔAUROC [95% CI]"] = [fmt_ci(a, b, c, True) for a, b, c in zip(show.delta, show.delta_lo, show.delta_hi)]
show["p (Holm)"] = show.p_holm.map(lambda p: "" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}"))
show["n (img/pat)"] = show.n_images.astype(str) + "/" + show.n_patients.astype(str)
paper_table = show[["network", "experiment", "transfer", "n (img/pat)", "filter", "AUROC [95% CI]", "ΔAUROC [95% CI]", "p (Holm)"]]
paper_table.to_csv(os.path.join(OUT, "table_auc_ci.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(paper_table.to_string(index=False))

# %% [markdown]
# ## 2. Summary per filter (external targets only)

# %%
ext = res[(res.target != "internal") & (res["filter"] != "raw")].copy()
ext["better"] = ext.significant & (ext.delta > 0)
ext["worse"] = ext.significant & (ext.delta < 0)
summ = (ext.groupby(["network", "experiment", "filter"], observed=True)
           .agg(transfers=("delta", "size"), mean_delta=("delta", "mean"), min_delta=("delta", "min"),
                max_delta=("delta", "max"), sig_better=("better", "sum"), sig_worse=("worse", "sum"))
           .reset_index())
overall = (ext.groupby(["network", "filter"], observed=True)
              .agg(transfers=("delta", "size"), mean_delta=("delta", "mean"), min_delta=("delta", "min"),
                   max_delta=("delta", "max"), sig_better=("better", "sum"), sig_worse=("worse", "sum"))
              .reset_index().assign(experiment="all"))
summ = pd.concat([summ, overall], ignore_index=True)
summ.to_csv(os.path.join(OUT, "summary_by_filter.csv"), index=False)
print(summ.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Figure 2: ΔAUROC relative to raw images with 95% intervals
# Filled points: Holm-adjusted p < 0.05.

# %%
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})

order_rows = []
for exp in ["cross_site", "cross_scanner"]:
    sub = ext[ext.experiment == exp][["source", "target"]].drop_duplicates()
    order_rows += [(exp, s, t) for s, t in sub.sort_values(["source", "target"]).itertuples(index=False)]
ypos = {r: i for i, r in enumerate(order_rows)}
flt = [f for f in present if f != "raw"]

fig, axes = plt.subplots(1, len(flt), figsize=(2.1 * len(flt) + 2.2, 0.32 * len(order_rows) + 1.3),
                         sharey=True, sharex=True)
axes = np.atleast_1d(axes)
lim = float(np.nanmax(np.abs(ext[["delta_lo", "delta_hi"]].values))) * 1.08
NET_COLOR = dict(zip(nets, [BLUE, ORANGE]))
offset = {n: (0.0 if len(nets) == 1 else (-0.18 if i == 0 else 0.18)) for i, n in enumerate(nets)}
for ax, f in zip(axes, flt):
    g = ext[ext["filter"] == f]
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for r in g.itertuples():
        c = NET_COLOR[r.network]
        yy = ypos[(r.experiment, r.source, r.target)] + offset[r.network]
        ax.plot([r.delta_lo, r.delta_hi], [yy, yy], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r.delta], [yy], s=34, zorder=3, edgecolor=c, linewidth=1.6,
                   facecolor=c if r.significant else "white")
    n_site = sum(1 for r in order_rows if r[0] == "cross_site")
    if 0 < n_site < len(order_rows):
        ax.axhline(n_site - 0.5, color=GRID, lw=1)
    ax.set_title(f, color=INK, fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(order_rows)))
axes[0].set_yticklabels([f"{nice(s)} → {nice(t)}" for _, s, t in order_rows])
axes[0].invert_yaxis()
fig.supxlabel("ΔAUROC vs. raw images (95% patient-level bootstrap CI)", fontsize=9, color=INK)
handles = [Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=MUTED, markeredgecolor=MUTED,
                  label="Holm-adjusted p < 0.05"),
           Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor="white", markeredgecolor=MUTED,
                  label="not significant")]
if len(nets) > 1:
    handles = [Line2D([], [], color=NET_COLOR[n], lw=2, label=n) for n in nets] + handles
fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.6, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext_ in ["png", "pdf"]:
    fig.savefig(os.path.join(FIG, f"forest_delta_auc.{ext_}"), dpi=300)
plt.show()

# %% [markdown]
# ## 4. Figure 3: change in scanner separability against change in transfer AUROC

# %%
gap = pd.read_csv(os.path.join(RES2, "domain_gap.csv"))
gp = gap.pivot_table(index=["experiment", "pair"], columns="filter", values="domain_auc")
dis = []
for net, (exp, pair) in [(n, ix) for n in nets for ix in gp.index]:
    row = gp.loc[(exp, pair)]
    a, b = pair.split(" vs ")
    for f in flt:
        if f not in row or not np.isfinite(row[f]):
            continue
        d = ext[(ext.network == net) & (ext.experiment == exp) & (ext["filter"] == f) &
                (((ext.source == a) & (ext.target == b)) | ((ext.source == b) & (ext.target == a)))]
        if d.empty:
            continue
        dis.append(dict(network=net, experiment=exp, pair=pair, filter=f, domain_auc_raw=row["raw"], domain_auc=row[f],
                        d_domain=row[f] - row["raw"], d_auroc=d.delta.mean(), directions=len(d)))
dis = pd.DataFrame(dis)
dis.to_csv(os.path.join(OUT, "dissociation.csv"), index=False)
rhos = {}
for net, gn in dis.groupby("network", sort=False):
    r_, p_ = spearmanr(gn.d_domain, gn.d_auroc)
    rhos[net] = (round(float(r_), 3), round(float(p_), 4))
    print(f"{net}: Spearman rho = {r_:.2f} (p = {p_:.3f}, n = {len(gn)} pair-filter points; points are not independent)")
    for exp, g in gn.groupby("experiment"):
        r2, p2 = spearmanr(g.d_domain, g.d_auroc)
        print(f"  {exp:13s}: rho = {r2:.2f} (p = {p2:.3f}, n = {len(g)})")
dis = dis[dis.network == PRIMARY]
rho, p_rho = rhos[PRIMARY]
print("\nMean domain-classifier AUROC by filter:")
print(gp[present].mean().round(3).to_string())

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.axhline(0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.grid(color=GRID, lw=0.8, zorder=0)
marks = dict(zip(flt, ["o", "s", "^", "D", "v", "P"]))
colors = {"cross_site": BLUE, "cross_scanner": ORANGE}
for r in dis.itertuples():
    ax.scatter(r.d_domain, r.d_auroc, marker=marks[r.filter], s=46, color=colors[r.experiment],
               edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xlabel("Δ domain-classifier AUROC vs. raw\n(negative = scanners harder to tell apart)")
ax.set_ylabel("Δ transfer AUROC vs. raw")
h1 = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=k.replace("_", "-"))
      for k, c in colors.items() if k in set(dis.experiment)]
h2 = [Line2D([], [], marker=marks[f], ls="", color=MUTED, markersize=7, label=f) for f in flt]
ax.legend(handles=h1 + h2, fontsize=8, frameon=False, loc="best")
ax.set_title(f"{PRIMARY}: Spearman ρ = {rho:.2f} (n = {len(dis)})", fontsize=10, color=INK)
fig.tight_layout()
for ext_ in ["png", "pdf"]:
    fig.savefig(os.path.join(FIG, f"dissociation.{ext_}"), dpi=300)
plt.show()

# %% [markdown]
# ## 5. Key numbers

# %%
key = dict(bootstrap_resamples=B, mean_domain_auc=gp[present].mean().round(3).to_dict())
for net, g in ext.groupby("network", observed=True):
    key[net] = dict(external_comparisons=int(len(g)), significant_better=int(g.better.sum()),
                    significant_worse=int(g.worse.sum()),
                    mean_delta_by_filter=g.groupby("filter", observed=True).delta.mean().round(4).to_dict(),
                    spearman_rho_p=rhos.get(net))
json.dump(key, open(os.path.join(OUT, "key_numbers.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(key, indent=2, ensure_ascii=False))
print("\nSignificant external changes:")
print(show[(show.target != "internal") & (show.p_holm < ALPHA)][
    ["network", "experiment", "transfer", "filter", "ΔAUROC [95% CI]", "p (Holm)"]].to_string(index=False))
print("\nAll results are in:", OUT)
