"""
Make Figures 3, 4, 5 from aggregated metrics (seed-aggregated) + value labels on bars.
- Figure 3: LLT accuracy (Exact & Fuzzy) — mean ± std  (+labels)
- Figure 4: mapped PT and SOC accuracy — SOC Option B — mean ± std (+labels)
- Figure 5: per dataset: RAG v3 mean LLT (Exact) + CCR + Manual — no error bars (+labels)

Color palette is colorblind-friendly.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
# Path to aggregated JSON produced by your aggregation script
AGG_JSON_PATH = "/home/naghmedashti/MedDRA-LLM/aggregates/aggregated_by_variant_20250929_141602.json"

# Where to save figures
OUT_DIR = "/home/naghmedashti/MedDRA-LLM/aggregates/visualization"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset display order and mapping to keys in aggregated JSON
DATASETS_DISPLAY = ["MOSAIC", "DELTA", "DAUNO"]
DS_KEY_MAP = {"MOSAIC": "Mosaic", "DELTA": "Delta", "DAUNO": "Dauno"}

# Map variant → model label used in plots
VARIANT_TO_LABEL = {"fuzz": "Baseline A", "rapidfuzz": "Baseline B", "rag": "RAG v3"}
MODELS_DISPLAY = ["Baseline A", "Baseline B", "RAG v3"]

# Colorblind-friendly colors
COLORS = {
    "Baseline A": "#1b9e77",  # fuzz
    "Baseline B": "#d95f02",  # rapidfuzz
    "RAG v3":     "#7570b3",  # rag
}
CCR_COLOR    = "#e7298a"
MANUAL_COLOR = "#66a61e"

# Manual & CCR values (adjust if you have updated numbers)
MANUAL_ACCURACY = {"MOSAIC": 0.98, "DELTA": 0.95, "DAUNO": 0.95}
CCR_VALUES      = {"MOSAIC": 0.33, "DELTA": 0.32, "DAUNO": 0.30}
# ===================================================


# ---------- Helpers ----------
def annotate_bars(ax, rects, values, fmt="{:.2f}", dy=0.01, fontsize=9):
    """
    Put numeric labels centered on top of bars.
    values should correspond to rects (length-equal). Skips NaNs.
    dy is vertical offset in axis fraction (converted to data via y-lim).
    """
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for rect, v in zip(rects, values):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + dy*span,
                fmt.format(v), ha='center', va='bottom', fontsize=fontsize)

def collect_metric_across(items, rec_map, models_labels, metric_mean_key, metric_std_key):
    """
    Return dict[label] -> (means_by_dataset, stds_by_dataset), ordered by DATASETS_DISPLAY.
    """
    data = {label: ([], []) for label in models_labels}
    for d in DATASETS_DISPLAY:
        dskey = DS_KEY_MAP[d]
        for label in models_labels:
            it = rec_map.get((dskey, label), {})
            m = it.get(metric_mean_key)
            s = it.get(metric_std_key)
            data[label][0].append(np.nan if m is None else float(m))
            data[label][1].append(np.nan if s is None else float(s))
    return data


# ---------- Load aggregated metrics ----------
with open(AGG_JSON_PATH, "r", encoding="utf-8") as f:
    agg = json.load(f)
items = agg.get("items", [])

# Build (dataset, modelLabel) → record
REC_MAP = {}
for it in items:
    ds = it.get("dataset")
    variant = (it.get("variant") or "").lower()
    label = VARIANT_TO_LABEL.get(variant)
    if label:
        REC_MAP[(ds, label)] = it


# ================= Figure 3: LLT Accuracy (Exact & Fuzzy) =================
llt_exact = collect_metric_across(items, REC_MAP, MODELS_DISPLAY,
                                  "LLT_term_acc_exact__mean", "LLT_term_acc_exact__std")
llt_fuzzy = collect_metric_across(items, REC_MAP, MODELS_DISPLAY,
                                  "LLT_term_acc_fuzzy__mean", "LLT_term_acc_fuzzy__std")

x = np.arange(len(DATASETS_DISPLAY))
width = 0.25

fig3, ax3 = plt.subplots(figsize=(10, 6))
bar_groups = []  # store (rects, values) to annotate

for i, model in enumerate(MODELS_DISPLAY):
    means_e, stds_e = llt_exact[model]
    means_f, stds_f = llt_fuzzy[model]

    r1 = ax3.bar(x + i*width - width, means_e, width/1.8,
                 label=f"{model} – Exact", color=COLORS[model], alpha=1.0)
    ax3.errorbar(x + i*width - width, means_e, yerr=stds_e, fmt='none',
                 ecolor='black', elinewidth=1, capsize=3, alpha=0.8)
    bar_groups.append((r1, means_e))

    r2 = ax3.bar(x + i*width - width/1.8, means_f, width/1.8,
                 label=f"{model} – Fuzzy", color=COLORS[model], alpha=0.5)
    ax3.errorbar(x + i*width - width/1.8, means_f, yerr=stds_f, fmt='none',
                 ecolor='black', elinewidth=1, capsize=3, alpha=0.8)
    bar_groups.append((r2, means_f))

ax3.set_title("LLT Accuracy (Exact & Fuzzy) across Models and Datasets")
ax3.set_xticks(x + width/3)
ax3.set_xticklabels(DATASETS_DISPLAY)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Accuracy")
ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax3.grid(axis='y', linestyle=':', alpha=0.5)

# annotate after y-lim is set
#for rects, vals in bar_groups:
 #   annotate_bars(ax3, rects, vals, fmt="{:.2f}", dy=0.01)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "figure3_llt_accuracy_agg.png")
plt.savefig(fig3_path, dpi=300)
plt.close(fig3)


# ================= Figure 4: mapped PT and SOC accuracy (Option B) =================
pt_acc   = collect_metric_across(items, REC_MAP, MODELS_DISPLAY, "PT_code_acc__mean", "PT_code_acc__std")
socB_acc = collect_metric_across(items, REC_MAP, MODELS_DISPLAY, "SOC_acc_option_b__mean", "SOC_acc_option_b__std")

fig4, ax4 = plt.subplots(figsize=(10, 6))
bar_groups = []

for i, model in enumerate(MODELS_DISPLAY):
    means_pt, stds_pt = pt_acc[model]
    means_sc, stds_sc = socB_acc[model]

    r1 = ax4.bar(x + i*width - width, means_pt, width/1.8,
                 label=f"{model} – PT", color=COLORS[model], alpha=1.0)
    ax4.errorbar(x + i*width - width, means_pt, yerr=stds_pt, fmt='none',
                 ecolor='black', elinewidth=1, capsize=3, alpha=0.8)
    bar_groups.append((r1, means_pt))

    r2 = ax4.bar(x + i*width - width/1.8, means_sc, width/1.8,
                 label=f"{model} – SOC ", color=COLORS[model], alpha=0.5)
    ax4.errorbar(x + i*width - width/1.8, means_sc, yerr=stds_sc, fmt='none',
                 ecolor='black', elinewidth=1, capsize=3, alpha=0.8)
    bar_groups.append((r2, means_sc))

ax4.set_title("Mapped PT and SOC accuracy")
ax4.set_xticks(x + width/3)
ax4.set_xticklabels(DATASETS_DISPLAY)
ax4.set_ylim(0, 1.05)
ax4.set_ylabel("Accuracy")
ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax4.grid(axis='y', linestyle=':', alpha=0.5)

#for rects, vals in bar_groups:
 #   annotate_bars(ax4, rects, vals, fmt="{:.2f}", dy=0.01)

plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "figure4_pt_soc_optionB_agg.png")
plt.savefig(fig4_path, dpi=300)
plt.close(fig4)


# ================= Figure 5: RAG mean LLT (Exact) + CCR + Manual =================
bar_w = 0.22
fig5, ax5 = plt.subplots(figsize=(10, 6))

# برای legend: فقط بارِ اولین بارِ هر برچسب رو نگه می‌داریم
legend_added = {"RAG": False, "CCR": False, "Manual": False}

# کمک‌تابع برای نوشتن عدد روی ستون
def _annot_single(rect, val, fmt="{:.2f}", dy=0.01):
    ymin, ymax = ax5.get_ylim()
    span = ymax - ymin
    ax5.text(rect.get_x() + rect.get_width()/2., rect.get_height() + dy*span,
             fmt.format(val), ha='center', va='bottom', fontsize=9)

# جمع‌آوری داده‌ها به ترتیب نمایش
rag_means  = []
ccr_vals   = []
manual_vals= []
for d in DATASETS_DISPLAY:
    dskey = DS_KEY_MAP[d]
    it = REC_MAP.get((dskey, "RAG v3"), {})
    m = it.get("LLT_term_acc_exact__mean")
    rag_means.append(np.nan if m is None else float(m))
    ccr_vals.append(float(CCR_VALUES.get(d, np.nan)))
    manual_vals.append(float(MANUAL_ACCURACY.get(d, np.nan)))

# رسم به‌صورت داینامیک و مرتب‌شده برای هر دیتاست
for i, d in enumerate(DATASETS_DISPLAY):
    vals = [
        ("RAG",    rag_means[i],  COLORS["RAG v3"]),
        ("CCR",    ccr_vals[i],   CCR_COLOR),
        ("Manual", manual_vals[i], MANUAL_COLOR),
    ]
    # sort by value ascending
    vals.sort(key=lambda t: (np.inf if (t[1] is None or np.isnan(t[1])) else t[1]))

    # سه جایگاه: چپ، وسط، راست
    x_positions = [x[i] - bar_w, x[i], x[i] + bar_w]

    for (label, value, color), xpos in zip(vals, x_positions):
        height = 0.0 if (value is None or np.isnan(value)) else value
        rects = ax5.bar(xpos, height, bar_w, color=color, alpha=0.9,
                        label=(None if legend_added[label] else
                               ("RAG v3 – Mean LLT Accuracy (Exact)" if label=="RAG"
                                else "CCR (clinically correct portion)" if label=="CCR"
                                else "Manual Accuracy")))
        # legend flag
        if not legend_added[label]:
            legend_added[label] = True
        # label روی ستون
        if value is not None and not np.isnan(value):
            _annot_single(rects[0], value, fmt="{:.2f}", dy=0.01)

ax5.set_title("Manual Check LLT Accuracy vs Exact LLT Accuracy and CCR")
ax5.set_xticks(x)
ax5.set_xticklabels(DATASETS_DISPLAY)
ax5.set_ylim(0, 1.05)
ax5.set_ylabel("Proportion")
ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax5.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
fig5_path = os.path.join(OUT_DIR, "figure5_rag_llt_vs_ccr_manual_sorted.png")
plt.savefig(fig5_path, dpi=300)
plt.close(fig5)

print("Saved:", fig3_path)
print("Saved:", fig4_path)
print("Saved:", fig5_path)


