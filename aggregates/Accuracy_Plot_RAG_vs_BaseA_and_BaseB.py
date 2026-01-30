# -*- coding: utf-8 -*-
"""
Figure 3: LLT/PT/SOC accuracy (exact only) comparing Baseline A (fuzz),
Baseline B (rapidfuzz), and RAG Model across MOSAIC/DELTA/DAUNO.

- Single figure only (no Figure 4)
- 3 panels: LLT accuracy / PT accuracy / SOC accuracy
- Error bars: std
- Numeric labels: mean only (2 decimals)
- Colorblind-safe palette (Okabe–Ito)
- Legend at bottom (no overlap)
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===================== CONFIG =====================
AGG_JSON_PATH = "/home/naghmedashti/MedDRA-LLM/aggregates/aggregated_by_variant_20260127_134219.json"
OUT_DIR       = "/home/naghmedashti/MedDRA-LLM/aggregates"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS_DISPLAY = ["MOSAIC", "DELTA", "DAUNO"]
DS_KEY_MAP = {"MOSAIC":"Mosaic", "DELTA":"Delta", "DAUNO":"Dauno"}

# variants in aggregate -> labels shown
VARIANT_TO_LABEL = {
    "newfuzz": "Baseline A",
    "newrapidfuzz": "Baseline B",
    "newrag": "RAG Model",
}

MODELS_DISPLAY = ["Baseline A", "Baseline B", "RAG Model"]

# Okabe–Ito (colorblind-safe)
COLORS = {
    "Baseline A": "#009E73",  # green
    "Baseline B": "#D55E00",  # vermillion
    "RAG Model":  "#0072B2",  # blue
}

# output
OUT_FIG = os.path.join(OUT_DIR, "rag_vs_baselines_LLT_PT_SOC2.png")

# ===================== STYLE =====================
plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

def robust_save(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Saved:", path)

# ===================== LOAD =====================
with open(AGG_JSON_PATH, "r", encoding="utf-8") as f:
    agg = json.load(f)

REC_MAP = {}
for it in agg.get("items", []):
    ds = it.get("dataset")
    lab = VARIANT_TO_LABEL.get((it.get("variant") or "").lower())
    if lab:
        REC_MAP[(ds, lab)] = it

def collect_metric(rec_map, models, mean_key, std_key):
    """Return dict[model] -> (means(list), stds(list)) aligned with DATASETS_DISPLAY."""
    data = {m: ([], []) for m in models}
    for d in DATASETS_DISPLAY:
        ds = DS_KEY_MAP[d]
        for m in models:
            r = rec_map.get((ds, m), {})
            mu = r.get(mean_key)
            sd = r.get(std_key)
            data[m][0].append(np.nan if mu is None else float(mu))
            data[m][1].append(np.nan if sd is None else float(sd))
    return data

# ===================== METRICS (EXACT ONLY) =====================
llt_exact = collect_metric(REC_MAP, MODELS_DISPLAY, "LLT_term_acc_exact__mean", "LLT_term_acc_exact__std")
pt_acc    = collect_metric(REC_MAP, MODELS_DISPLAY, "PT_code_acc__mean",       "PT_code_acc__std")
soc_acc   = collect_metric(REC_MAP, MODELS_DISPLAY, "SOC_acc_option_b__mean",  "SOC_acc_option_b__std")

# ===================== PLOT =====================
x = np.arange(len(DATASETS_DISPLAY))
W = 0.26
cap = 3

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), constrained_layout=False)

def add_mean_labels(ax, xs, means):
    for xi, mu in zip(xs, means):
        if np.isnan(mu):
            continue
        ax.text(xi, mu + 0.02, f"{mu:.2f}", ha="center", va="bottom", fontsize=11)

def panel(ax, title, metric_dict):
    for i, m in enumerate(MODELS_DISPLAY):
        means, stds = metric_dict[m]
        xpos = x + (i - 1) * W

        ax.bar(
            xpos, means, width=W,
            color=COLORS[m], edgecolor="none", alpha=0.95
        )
        ax.errorbar(
            xpos, means, yerr=stds,
            fmt="none", ecolor="black", elinewidth=0.9, capsize=cap
        )
        add_mean_labels(ax, xpos, means)

    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS_DISPLAY, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

panel(axes[0], "LLT accuracy", llt_exact)
panel(axes[1], "PT accuracy",  pt_acc)
panel(axes[2], "SOC accuracy", soc_acc)

# shared legend at bottom
legend_handles = [Patch(facecolor=COLORS[m], label=m) for m in MODELS_DISPLAY]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.03),
    fontsize=12
)

# title above plots (and no overlap with legend)
fig.suptitle("Baseline comparison across MedDRA hierarchy levels", y=1.02, fontsize=16)

# leave room for bottom legend + top suptitle
fig.tight_layout(rect=[0, 0.06, 1, 0.95])

robust_save(fig, OUT_FIG)
plt.close(fig)
