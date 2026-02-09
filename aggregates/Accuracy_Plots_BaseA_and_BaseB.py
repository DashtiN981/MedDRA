# -*- coding: utf-8 -*-
"""
Single plot (like Excel stacked): Baseline A vs Baseline B across MOSAIC/DELTA/DAUNO.
Each bar shows LLT/PT/SOC accuracies as cumulative levels:
  LLT at bottom, PT above (PT-LLT), SOC above (SOC-PT)
=> bar top equals SOC, and LLT/PT boundaries are meaningful.

RAG removed. Reads means from the same aggregate JSON.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ===================== CONFIG =====================
AGG_JSON_PATH = "/home/naghmedashti/MedDRA-LLM/aggregates/aggregated_by_variant_20260209_093745.json"
OUT_DIR       = "/home/naghmedashti/MedDRA-LLM/aggregates"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS_DISPLAY = ["MOSAIC", "DELTA", "DAUNO"]
DS_KEY_MAP = {"MOSAIC":"Mosaic", "DELTA":"Delta", "DAUNO":"Dauno"}

VARIANT_TO_LABEL = {
    "newfuzz": "Baseline A",
    "newrapidfuzz": "Baseline B",
}
MODELS_DISPLAY = ["Baseline A", "Baseline B"]

# === Colors (paper-friendly, NOT gray) ===
# Two base colors (one per baseline) with lightness gradient for LLT/PT/SOC.
BASELINE_BASE_COLOR = {
    "Baseline A": "#2F6CB3",  # blue
    "Baseline B": "#C95A2A",  # orange
}

def shade_towards_white(hex_color, t):
    """Blend a color toward white by factor t in [0,1]."""
    c = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex((1 - t) * c + t * np.ones(3))

def baseline_level_colors(baseline):
    base = BASELINE_BASE_COLOR[baseline]
    # Darkest at bottom (LLT), then lighter (PT), lightest (SOC)
    return {
        "LLT": shade_towards_white(base, 0.05),
        "PT":  shade_towards_white(base, 0.35),
        "SOC": shade_towards_white(base, 0.65),
    }

# Baseline colors for legend markers / outlines (subtle)
BASELINE_EDGE = {
    "Baseline A": shade_towards_white(BASELINE_BASE_COLOR["Baseline A"], 0.0),
    "Baseline B": shade_towards_white(BASELINE_BASE_COLOR["Baseline B"], 0.0),
}

OUT_FIG = os.path.join(OUT_DIR, "baselines_excelstyle_stacked_levels.png")

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

def f(x):
    return np.nan if x is None else float(x)

def get_metrics(ds_key, model):
    r = REC_MAP.get((ds_key, model), {})
    soc = f(r.get("SOC_acc_option_b__mean"))
    pt  = f(r.get("PT_code_acc__mean"))
    llt = f(r.get("LLT_term_acc_exact__mean"))
    return soc, pt, llt

# ===================== PLOT =====================
x = np.arange(len(DATASETS_DISPLAY)) * 0.85  # tighter spacing between dataset groups
W = 0.24  # thinner bars like Excel
gap = 0.06
offsets = {
    "Baseline A": -(W/2 + gap/2),
    "Baseline B": +(W/2 + gap/2),
}

fig, ax = plt.subplots(figsize=(13.0, 5.6))

for m in MODELS_DISPLAY:
    xpos = x + offsets[m]
    soc_list, pt_list, llt_list = [], [], []

    for d in DATASETS_DISPLAY:
        soc, pt, llt = get_metrics(DS_KEY_MAP[d], m)
        soc_list.append(soc)
        pt_list.append(pt)
        llt_list.append(llt)

    soc = np.nan_to_num(np.array(soc_list), nan=0.0)
    pt  = np.nan_to_num(np.array(pt_list),  nan=0.0)
    llt = np.nan_to_num(np.array(llt_list), nan=0.0)

    # Ensure monotonic levels (just in case of weirdness)
    pt  = np.maximum(pt, llt)
    soc = np.maximum(soc, pt)

    # Segment heights that preserve boundaries
    h_llt = llt
    h_pt  = pt - llt
    h_soc = soc - pt

    # Stacked (Excel-style levels)
    colors = baseline_level_colors(m)
    ax.bar(xpos, h_llt, width=W, color=colors["LLT"], edgecolor="none")
    ax.bar(xpos, h_pt,  width=W, bottom=llt, color=colors["PT"],  edgecolor="none")
    ax.bar(xpos, h_soc, width=W, bottom=pt,  color=colors["SOC"], edgecolor="none")

    # Optional: thin baseline-colored outline around the whole bar to distinguish A vs B (NOT hatch)
    ax.bar(xpos, soc, width=W, fill=False, edgecolor=BASELINE_EDGE[m], linewidth=1.2)

    # Label the actual levels (SOC/PT/LLT) clearly (small and neat)
    for xi, s, p, l in zip(xpos, soc, pt, llt):
        ax.text(xi, l + 0.01, f"L:{l:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi, p + 0.01, f"P:{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi, s + 0.01, f"S:{s:.2f}", ha="center", va="bottom", fontsize=9)

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(DATASETS_DISPLAY, fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Baseline comparison across MedDRA hierarchy levels", fontsize=16, pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Legends:
# 1) levels legend (colors)
# 1) level legend (using neutral grayscale swatches to indicate stack order)
level_handles = [
    Patch(facecolor="#3A3A3A", label="LLT accuracy"),
    Patch(facecolor="#8A8A8A", label="PT accuracy"),
    Patch(facecolor="#CFCFCF", label="SOC accuracy"),
]
# 2) baseline legend (solid markers)
baseline_handles = [
    Patch(facecolor=BASELINE_BASE_COLOR["Baseline A"], edgecolor="none", label="Baseline A"),
    Patch(facecolor=BASELINE_BASE_COLOR["Baseline B"], edgecolor="none", label="Baseline B"),
]

'''leg_baseline = fig.legend(
    handles=baseline_handles,
    loc="center left",
    bbox_to_anchor=(0.73, 0.55),
    bbox_transform=fig.transFigure,
    fontsize=11,
)
fig.add_artist(leg_baseline)'''

leg_methods = fig.legend(
    handles=baseline_handles,
    loc="center left",
    bbox_to_anchor=(0.73, 0.55),
    bbox_transform=fig.transFigure,
    fontsize=11,
    title="Methods",
    title_fontsize=11,
    frameon=False,
)
fig.add_artist(leg_methods)


'''fig.legend(
    handles=level_handles,
    loc="center left",
    bbox_to_anchor=(0.73, 0.35),
    bbox_transform=fig.transFigure,
    fontsize=11,
)'''

fig.legend(
    handles=level_handles,
    loc="center left",
    bbox_to_anchor=(0.73, 0.35),
    bbox_transform=fig.transFigure,
    fontsize=11,
    title="Levels",
    title_fontsize=11,
    frameon=False,
)

fig.subplots_adjust(left=0.07, right=0.70, bottom=0.12, top=0.90)
robust_save(fig, OUT_FIG)
plt.close(fig)
