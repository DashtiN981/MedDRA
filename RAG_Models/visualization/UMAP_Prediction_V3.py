import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.metrics.pairwise import cosine_similarity

# === Load files ===
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

with open("./Visualization/other/output_checked_naghmeV3.json", "r", encoding="latin1") as f:
    pred_data = json.load(f)

# === Extract AE prediction records ===
records = []
for row in pred_data:
    true_term = row["true_term"]
    pred_term = row["predicted"]
    if true_term in llt_embeds_raw and pred_term in llt_embeds_raw:
        records.append({
            "true_term": true_term,
            "predicted": pred_term,
            "correct": true_term == pred_term
        })

# === Build embedding set ===
true_terms = [r["true_term"] for r in records]
pred_terms = [r["predicted"] for r in records]
all_terms = set(true_terms + pred_terms)

background_terms = list(set(llt_embeds_raw.keys()) - all_terms)
sampled_bg = np.random.choice(background_terms, size=3000, replace=False)

# Combine all for UMAP
terms_umap = true_terms + pred_terms + list(sampled_bg)
labels = ["Correct" if r["correct"] else "True" for r in records] + \
         ["Correct" if r["correct"] else "Pred" for r in records] + \
         ["Other"] * len(sampled_bg)
embeds = [llt_embeds_raw[t] for t in terms_umap]

# === UMAP projection ===
reducer = umap.UMAP(n_neighbors=100, min_dist=0.9, random_state=42)
umap_result = reducer.fit_transform(embeds)

df = pd.DataFrame(umap_result, columns=["x", "y"])
df["LLT_Term"] = terms_umap
df["Label"] = labels

# === Plotting ===
plt.figure(figsize=(16, 12))
colors = {"Correct": "green", "True": "blue", "Pred": "red", "Other": "lightgray"}

for label in colors:
    subset = df[df["Label"] == label]
    plt.scatter(subset["x"], subset["y"], label=label, s=15, alpha=0.6, color=colors[label])

# === Draw arrows between predicted ↔ true terms for incorrect predictions ===
term_coords = dict(zip(df["LLT_Term"], zip(df["x"], df["y"])))
for r in records:
    if not r["correct"]:
        t1 = term_coords.get(r["true_term"])
        t2 = term_coords.get(r["predicted"])
        if t1 and t2:
            plt.plot([t1[0], t2[0]], [t1[1], t2[1]], color="orange", linewidth=0.5, alpha=0.5)

plt.title("UMAP of LLT Embeddings – Correct vs. Incorrect Predictions", fontsize=16)
plt.axis("off")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("./Visualization/other/output_checked_result_V3.png", dpi=300)
plt.show()
