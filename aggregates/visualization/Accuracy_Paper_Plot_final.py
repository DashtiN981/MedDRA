# -*- coding: utf-8 -*-
"""
- Figure 3: (a) LLT accuracy (exact vs fuzzy), (b) PT & SOC (Option B)
  * Panel labels: 'a' and 'b' (lowercase)
  * No global suptitle
- Figure 4: Manual (right) vs RAG (center) vs CCR (left)
  * Numeric labels on bars
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===================== CONFIG =====================
AGG_JSON_PATH = "/home/naghmedashti/MedDRA-LLM/aggregates/aggregated_by_variant_20250929_141602.json"
OUT_DIR       = "/home/naghmedashti/MedDRA-LLM/aggregates/visualization"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS_DISPLAY = ["MOSAIC", "DELTA", "DAUNO"]
DS_KEY_MAP = {"MOSAIC":"Mosaic","DELTA":"Delta","DAUNO":"Dauno"}

VARIANT_TO_LABEL = {"fuzz":"Baseline A","rapidfuzz":"Baseline B","rag":"RAG Model"}
MODELS_DISPLAY   = ["Baseline A","Baseline B","RAG Model"]

# constant pallet
COLORS = {"Baseline A":"#1b9e77", "Baseline B":"#d95f02", "RAG Model":"#7570b3"}
MANUAL_COLOR = "#1b9e77"  # green
RAG_COLOR    = "#7570b3"  # purple
CCR_COLOR    = "#d95f02"  # orange

# for Figure 4 aggregate
MANUAL_ACCURACY = {"MOSAIC":0.98,"DELTA":0.95,"DAUNO":0.95}
CCR_VALUES      = {"MOSAIC":0.33,"DELTA":0.32,"DAUNO":0.30}

# for legend position
LEGEND_X = 1.3   
LEGEND_Y = 0.06

# ===================== STYLE =====================
plt.rcParams.update({
    "font.size":10,
    "font.family":"DejaVu Sans",
    "axes.spines.top":False,
    "axes.spines.right":False,
    "axes.linewidth":0.8,
    "legend.frameon":False,
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
    data = {m:([],[]) for m in models}
    for d in DATASETS_DISPLAY:
        ds = DS_KEY_MAP[d]
        for m in models:
            r = rec_map.get((ds, m), {})
            mu = r.get(mean_key)
            sd = r.get(std_key)
            data[m][0].append(np.nan if mu is None else float(mu))
            data[m][1].append(np.nan if sd is None else float(sd))
    return data

# ===================== METRICS =====================
llt_exact = collect_metric(REC_MAP, MODELS_DISPLAY, "LLT_term_acc_exact__mean", "LLT_term_acc_exact__std")
llt_fuzzy = collect_metric(REC_MAP, MODELS_DISPLAY, "LLT_term_acc_fuzzy__mean", "LLT_term_acc_fuzzy__std")
pt_acc    = collect_metric(REC_MAP, MODELS_DISPLAY, "PT_code_acc__mean",       "PT_code_acc__std")
socB_acc  = collect_metric(REC_MAP, MODELS_DISPLAY, "SOC_acc_option_b__mean",  "SOC_acc_option_b__std")

# for Figure 4: mean LLT exact for RAG
RAG_LLTexact = {
    d: float(REC_MAP.get((DS_KEY_MAP[d], "RAG Model"), {}).get("LLT_term_acc_exact__mean", np.nan))
    for d in DATASETS_DISPLAY
}

# ===================== Figure 3 (includes 2 panels) =====================
x = np.arange(len(DATASETS_DISPLAY))
W = 0.28

fig1, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.6))
plt.subplots_adjust(wspace=0.45, bottom=0.18, top=0.90)

# --- a) LLT (exact solid, fuzzy alpha=0.45) ---
for i, m in enumerate(MODELS_DISPLAY):
    mu_e, sd_e = llt_exact[m]; mu_f, sd_f = llt_fuzzy[m]
    # exact
    axA.bar(x + (i-1)*W - 0.07, mu_e, W*0.70, color=COLORS[m], alpha=1.00, edgecolor="none")
    axA.errorbar(x + (i-1)*W - 0.07, mu_e, yerr=sd_e, fmt='none', ecolor='black', elinewidth=0.9, capsize=3)
    # fuzzy
    axA.bar(x + (i-1)*W + 0.07, mu_f, W*0.70, color=COLORS[m], alpha=0.45, edgecolor="none")
    axA.errorbar(x + (i-1)*W + 0.07, mu_f, yerr=sd_f, fmt='none', ecolor='black', elinewidth=0.9, capsize=3)

axA.set_title("LLT accuracy (exact vs fuzzy)", loc="center")
axA.text(-0.12, 1.03, "a", transform=axA.transAxes, fontsize=12, fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(DATASETS_DISPLAY)
axA.set_ylim(0, 1.02); axA.set_ylabel("Accuracy")
axA.grid(axis='y', linestyle=':', alpha=0.4)

# ----- legend (6 items)
handles_a = []
for m in MODELS_DISPLAY:
    handles_a.append(Patch(facecolor=COLORS[m], alpha=1.00, label=f"{m} (exact)"))
    handles_a.append(Patch(facecolor=COLORS[m], alpha=0.45, label=f"{m} (fuzzy)"))

axA.legend(handles=handles_a,
           loc="lower right", bbox_to_anchor=(LEGEND_X, LEGEND_Y),
           ncol=1, borderaxespad=0.4, frameon=False,
           handlelength=1.2, handletextpad=0.6, labelspacing=0.3, fontsize=8)

# --- b) PT/SOC (PT solid, SOC alpha=0.45) ---
for i, m in enumerate(MODELS_DISPLAY):
    mu_pt, sd_pt = pt_acc[m]; mu_sc, sd_sc = socB_acc[m]
    # PT
    axB.bar(x + (i-1)*W - 0.07, mu_pt, W*0.70, color=COLORS[m], alpha=1.00, edgecolor="none")
    axB.errorbar(x + (i-1)*W - 0.07, mu_pt, yerr=sd_pt, fmt='none', ecolor='black', elinewidth=0.9, capsize=3)
    # SOC
    axB.bar(x + (i-1)*W + 0.07, mu_sc, W*0.70, color=COLORS[m], alpha=0.45, edgecolor="none")
    axB.errorbar(x + (i-1)*W + 0.07, mu_sc, yerr=sd_sc, fmt='none', ecolor='black', elinewidth=0.9, capsize=3)

axB.set_title("Mapped PT & SOC accuracy", loc="center")
axB.text(-0.12, 1.03, "b", transform=axB.transAxes, fontsize=12, fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(DATASETS_DISPLAY)
axB.set_ylim(0, 1.02)
axB.grid(axis='y', linestyle=':', alpha=0.4)

# ----- legend (6 items)
handles_b = []
for m in MODELS_DISPLAY:
    handles_b.append(Patch(facecolor=COLORS[m], alpha=1.00, label=f"{m} (PT)"))
    handles_b.append(Patch(facecolor=COLORS[m], alpha=0.45, label=f"{m} (SOC)"))

axB.legend(handles=handles_b,
           loc="lower right", bbox_to_anchor=(LEGEND_X, LEGEND_Y),
           ncol=1, borderaxespad=0.4, frameon=False,
           handlelength=1.2, handletextpad=0.6, labelspacing=0.3, fontsize=8)

robust_save(fig1, os.path.join(OUT_DIR, "Figure3_llt_pt_soc_with_title.png"))
plt.close(fig1)

# ===================== Figure 4 =====================
fig5, ax5 = plt.subplots(figsize=(9.6, 5.6))
plt.subplots_adjust(bottom=0.18, top=0.90)

idx = np.arange(len(DATASETS_DISPLAY))
w = 0.23

manual = [float(MANUAL_ACCURACY[d]) for d in DATASETS_DISPLAY]
rag    = [float(RAG_LLTexact[d])    for d in DATASETS_DISPLAY]
ccr    = [float(CCR_VALUES[d])      for d in DATASETS_DISPLAY]

# راست → چپ: Manual | RAG | CCR
bars_manual = ax5.bar(idx + w, manual, w, color=MANUAL_COLOR, edgecolor="none", alpha=0.98, label="Manual LLT accuracy")
bars_rag    = ax5.bar(idx,     rag,    w, color=RAG_COLOR,    edgecolor="none", alpha=0.98, label="RAG-based LLT accuracy")
bars_ccr    = ax5.bar(idx - w, ccr,    w, color=CCR_COLOR,    edgecolor="none", alpha=0.98, label="CCR")

ax5.set_title("Manual Check LLT Accuracy vs Exact LLT Accuracy and CCR", loc="center")
ax5.set_xticks(idx); ax5.set_xticklabels(DATASETS_DISPLAY)
ax5.set_ylim(0, 1.05); ax5.set_ylabel("Proportion")
ax5.grid(axis='y', linestyle=':', alpha=0.4)

# bar lables
def add_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

add_labels(ax5, bars_manual)
add_labels(ax5, bars_rag)
add_labels(ax5, bars_ccr)

# for change position change LEGEND_X, LEGEND_Y
ax5.legend(loc="lower right", bbox_to_anchor=(LEGEND_X, LEGEND_Y),
           ncol=1, borderaxespad=0.4, frameon=False,
           handlelength=1.2, handletextpad=0.6, labelspacing=0.3, fontsize=9)

robust_save(fig5, os.path.join(OUT_DIR, "Figure4_manual_rag_ccr_wiht_title.png"))
plt.close(fig5)
