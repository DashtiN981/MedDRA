# -*- coding: utf-8 -*-
"""
Horizontal plot: Baseline A vs Baseline B across MOSAIC/DELTA/DAUNO.
Each bar shows LLT/PT/SOC accuracies as cumulative boundaries (not additive):
  segments are [0->min], [min->mid], [mid->max] after sorting levels.
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
    "Baseline A": "#AA9CC2",  # purple
    "Baseline B": "#7E54C7",  # light purple 
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

OUT_FIG = os.path.join(OUT_DIR, "Accuracy_Plots_BaseA_and_BaseB_horizontal.png")

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
    soc_sd = f(r.get("SOC_acc_option_b__std"))
    pt_sd  = f(r.get("PT_code_acc__std"))
    llt_sd = f(r.get("LLT_term_acc_exact__std"))
    return soc, pt, llt, soc_sd, pt_sd, llt_sd

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(13.2, 5.6))

y = np.arange(len(DATASETS_DISPLAY)) * 0.85
H = 0.24
gap = 0.06
offsets = {
    "Baseline A": -(H/2 + gap/2),
    "Baseline B": +(H/2 + gap/2),
}

for m in MODELS_DISPLAY:
    ypos = y + offsets[m]
    soc_list, pt_list, llt_list = [], [], []
    soc_sd_list, pt_sd_list, llt_sd_list = [], [], []

    for d in DATASETS_DISPLAY:
        soc, pt, llt, soc_sd, pt_sd, llt_sd = get_metrics(DS_KEY_MAP[d], m)
        soc_list.append(soc)
        pt_list.append(pt)
        llt_list.append(llt)
        soc_sd_list.append(soc_sd)
        pt_sd_list.append(pt_sd)
        llt_sd_list.append(llt_sd)

    soc = np.nan_to_num(np.array(soc_list), nan=0.0)
    pt  = np.nan_to_num(np.array(pt_list),  nan=0.0)
    llt = np.nan_to_num(np.array(llt_list), nan=0.0)
    soc_sd = np.nan_to_num(np.array(soc_sd_list), nan=0.0)
    pt_sd  = np.nan_to_num(np.array(pt_sd_list),  nan=0.0)
    llt_sd = np.nan_to_num(np.array(llt_sd_list), nan=0.0)

    colors = baseline_level_colors(m)
    level_specs = [
        ("LLT", llt, llt_sd, colors["LLT"]),
        ("PT",  pt,  pt_sd,  colors["PT"]),
        ("SOC", soc, soc_sd, colors["SOC"]),
    ]

    for i in range(len(DATASETS_DISPLAY)):
        levels = [(name, vals[i], color) for name, vals, _, color in level_specs]
        levels_sorted = sorted(levels, key=lambda t: t[1])
        label_pos = []

        prev = 0.0
        for name, val, color in levels_sorted:
            seg = max(val - prev, 0.0)
            if seg > 0:
                ax.barh(ypos[i], seg, height=H, left=prev, color=color, edgecolor="none")
                label_pos.append((name, prev, val, prev + seg / 2.0))
            prev = val

        for _, vals, sds, color in level_specs:
            if sds[i] > 0:
                ax.errorbar(
                    vals[i],
                    ypos[i],
                    xerr=sds[i],
                    fmt="none",
                    ecolor="#111111",
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.2,
                    zorder=3,
                )

        # Put each level label at the center of its own drawn segment.
        soc_min_width_for_inside_label = 0.09
        soc_outside_pad = 0.008
        for name, x_start, x_end, x_mid in label_pos:
            tag = {"LLT": "LLT", "PT": "PT", "SOC": "SOC"}[name]
            x_text = x_mid
            ha = "center"

            # If SOC segment is too narrow, place label just outside to avoid overlap.
            if name == "SOC" and (x_end - x_start) < soc_min_width_for_inside_label:
                x_text = min(x_end + soc_outside_pad, 1.045)
                ha = "left"

            ax.text(
                x_text,
                ypos[i],
                f"{tag}:{x_end:.2f}",
                ha=ha,
                va="center",
                fontsize=9,
                color="#111111",
                clip_on=False,
                zorder=4,
            )

# Axis formatting
ax.set_yticks(y)
ax.set_yticklabels(DATASETS_DISPLAY, fontsize=12)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_title("Baseline comparison across MedDRA hierarchy levels", fontsize=13, pad=10)
ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.5)

# Legends
level_handles = [
    Patch(facecolor="#3A3A3A", label="LLT accuracy"),
    Patch(facecolor="#8A8A8A", label="PT accuracy"),
    Patch(facecolor="#CFCFCF", label="SOC accuracy"),
]

baseline_handles = [
    Patch(facecolor=BASELINE_BASE_COLOR["Baseline A"], edgecolor="none", label="Baseline A"),
    Patch(facecolor=BASELINE_BASE_COLOR["Baseline B"], edgecolor="none", label="Baseline B"),
]

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

fig.subplots_adjust(left=0.12, right=0.72, bottom=0.12, top=0.90)
robust_save(fig, OUT_FIG)
plt.close(fig)
