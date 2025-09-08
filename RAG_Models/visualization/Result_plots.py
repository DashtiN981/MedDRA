# Version with transparency instead of hatch patterns

import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ["MOSAIC", "DELTA", "DAUNO"]
models = ["Baseline A", "Baseline B", "RAG v3"]

colors = {
    "Baseline A": "#1f77b4",  # blue
    "Baseline B": "#ff7f0e",  # orange
    "RAG v3": "#2ca02c",      # green
}

llt_exact = {
    "Baseline A": [0.38, 0.24, 0.27],
    "Baseline B": [0.72, 0.69, 0.74],
    "RAG v3":     [0.68, 0.63, 0.66],
}
llt_fuzzy = {
    "Baseline A": [0.44, 0.26, 0.30],
    "Baseline B": [0.80, 0.71, 0.78],
    "RAG v3":     [0.74, 0.66, 0.70],
}

pt_acc = {
    "Baseline A": [0.56, 0.45, 0.44],
    "Baseline B": [0.90, 0.86, 0.88],
    "RAG v3":     [0.87, 0.82, 0.85],
}
soc_acc = {
    "Baseline A": [0.75, 0.73, 0.78],
    "Baseline B": [0.95, 0.93, 0.96],
    "RAG v3":     [0.90, 0.91, 0.96],
}

ccr = [0.33, 0.32, 0.30]
manual_acc = [0.98, 0.95, 0.95]

# ---------------- Figure 3: LLT Accuracy ----------------
x = np.arange(len(datasets))
width = 0.25

fig3, ax3 = plt.subplots(figsize=(10,6))

for i, model in enumerate(models):
    ax3.bar(x + i*width - width, llt_exact[model], 
            width/1.8, label=f"{model} – Exact", color=colors[model], alpha=1.0)
    ax3.bar(x + i*width - width/1.8, llt_fuzzy[model], 
            width/1.8, label=f"{model} – Fuzzy", color=colors[model], alpha=0.5)

ax3.set_title("LLT Accuracy (Exact & Fuzzy) across Models and Datasets")
ax3.set_xticks(x + width/3)
ax3.set_xticklabels(datasets)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Accuracy")
ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax3.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure3_llt_accuracy_transparent.png", dpi=300)
plt.show()

# ---------------- Figure 4: PT and SOC Accuracy ----------------
fig4, ax4 = plt.subplots(figsize=(10,6))

for i, model in enumerate(models):
    ax4.bar(x + i*width - width, pt_acc[model], 
            width/1.8, label=f"{model} – PT", color=colors[model], alpha=1.0)
    ax4.bar(x + i*width - width/1.8, soc_acc[model], 
            width/1.8, label=f"{model} – SOC", color=colors[model], alpha=0.5)

ax4.set_title("PT and SOC Accuracy across Models and Datasets")
ax4.set_xticks(x + width/3)
ax4.set_xticklabels(datasets)
ax4.set_ylim(0, 1.05)
ax4.set_ylabel("Accuracy")
ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax4.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure4_pt_soc_accuracy_transparent.png", dpi=300)
plt.show()

# ---------------- Figure 5: CCR & Manual Review ----------------
fig5, ax5 = plt.subplots(figsize=(9,6))

bar_width = 0.35
ax5.bar(x - bar_width/2, ccr, bar_width, label="CCR (clinically correct portion)", color="#17becf", alpha=0.7)
ax5.bar(x + bar_width/2, manual_acc, bar_width, label="Final Manual Review Accuracy", color="#9467bd", alpha=0.7)

ax5.set_title("Clinical Correctness and Manual Review (RAG v3)")
ax5.set_xticks(x)
ax5.set_xticklabels(datasets)
ax5.set_ylim(0, 1.05)
ax5.set_ylabel("Proportion")
ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax5.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure5_ccr_manual_transparent.png", dpi=300)
plt.show()

"/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure3_llt_accuracy_transparent.png", "/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure4_pt_soc_accuracy_transparent.png", "/home/naghmedashti/MedDRA-LLM/RAG_Models/visualization/figure5_ccr_manual_transparent.png"
