import json
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

# === Paths ===
LLT_EMB_PATH = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
RESULT_PATH = "/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_reasoning_v3.json"
OUT_IMG = "/home/naghmedashti/MedDRA-LLM/RAG_Models/plots/llt_embedding_neighborhood_umap.png"

NEIGHBOR_K = 100  # Number of nearest LLTs to visualize around GT and Predicted

# === Load Embeddings and Results ===
with open(LLT_EMB_PATH, "r", encoding="latin1") as f:
    llt_emb_dict = json.load(f)
with open(RESULT_PATH, "r", encoding="latin1") as f:
    results = json.load(f)

# Convert all LLT embeddings to numpy
all_terms = list(llt_emb_dict.keys())
all_embs = np.array([np.array(llt_emb_dict[t]) for t in all_terms])

# === Build base set: all LLTs in GT and Prediction ===
llt_in_gt = set(r["true_term"] for r in results)
llt_in_pred = set(r["predicted"] for r in results)
base_terms = llt_in_gt.union(llt_in_pred)

# === Find neighbors for each base term ===
selected_terms = set()
for term in base_terms:
    if term not in llt_emb_dict:
        continue
    term_emb = np.array(llt_emb_dict[term])
    similarities = np.dot(all_embs, term_emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(term_emb) + 1e-6)
    top_indices = np.argsort(similarities)[-NEIGHBOR_K:]
    neighbors = [all_terms[i] for i in top_indices]
    selected_terms.update(neighbors)
selected_terms.update(base_terms)

# === Prepare 2D UMAP projection ===
final_terms = [t for t in selected_terms if t in llt_emb_dict]
final_embs = np.array([llt_emb_dict[t] for t in final_terms])

print(f"Running UMAP on {len(final_terms)} terms...")

reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(final_embs)

# === Determine color coding ===
colors = []
for term in final_terms:
    in_gt = term in llt_in_gt
    in_pred = term in llt_in_pred
    if in_gt and in_pred:
        colors.append("purple")    # Both
    elif in_gt:
        colors.append("green")     # Ground truth only
    elif in_pred:
        colors.append("blue")      # Predicted only
    else:
        colors.append("lightgray") # Neighbor only

# === Plot ===
plt.figure(figsize=(10, 8))
for color in ["green", "blue", "purple", "lightgray"]:
    indices = [i for i, c in enumerate(colors) if c == color]
    label = {
        "green": "Ground Truth Only",
        "blue": "Predicted Only",
        "purple": "Both GT & Pred",
        "lightgray": f"Nearby LLTs (Top-{NEIGHBOR_K})"
    }[color]
    plt.scatter(
        emb_2d[indices, 0],
        emb_2d[indices, 1],
        label=label,
        c=color,
        s=40,
        alpha=0.7,
        edgecolors="none"
    )

plt.title("UMAP Projection of LLT Embeddings Around GT/Pred", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(loc="best", fontsize=10)
plt.tight_layout()

# === Save image ===
os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
plt.savefig(OUT_IMG, dpi=300)
plt.show()
