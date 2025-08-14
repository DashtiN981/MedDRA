import json
import numpy as np
import pandas as pd
import umap
import plotly.express as px

# === Paths ===
LLT_EMB_PATH = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
RESULT_PATH = "/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_reasoning_v3.json"

NEIGHBOR_K = 10  # Only 10 closest LLTs per GT/Prediction

# === Load embeddings and prediction results ===
with open(LLT_EMB_PATH, "r", encoding="latin1") as f:
    llt_emb_dict = json.load(f)
with open(RESULT_PATH, "r", encoding="latin1") as f:
    results = json.load(f)

# Prepare base LLT sets
llt_in_gt = set(r["true_term"] for r in results)
llt_in_pred = set(r["predicted"] for r in results)
base_terms = llt_in_gt.union(llt_in_pred)

# Full list of LLTs and their embeddings
all_terms = list(llt_emb_dict.keys())
all_embs = np.array([llt_emb_dict[t] for t in all_terms])

# Build selected LLTs (GT, Pred, and their neighbors)
selected_terms = set()
for term in base_terms:
    if term not in llt_emb_dict:
        continue
    emb = np.array(llt_emb_dict[term])
    similarities = np.dot(all_embs, emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(emb) + 1e-6)
    top_indices = np.argsort(similarities)[-NEIGHBOR_K:]
    neighbors = [all_terms[i] for i in top_indices]
    selected_terms.update(neighbors)
selected_terms.update(base_terms)

# Final embedding matrix
final_terms = [t for t in selected_terms if t in llt_emb_dict]
final_embs = np.array([llt_emb_dict[t] for t in final_terms])

# Run UMAP projection
reducer = umap.UMAP(n_components=2, random_state=42)
proj = reducer.fit_transform(final_embs)

# Color and label assignment
colors = []
labels = []
for term in final_terms:
    in_gt = term in llt_in_gt
    in_pred = term in llt_in_pred
    if in_gt and in_pred:
        colors.append("Both GT & Pred")
    elif in_gt:
        colors.append("Ground Truth Only")
    elif in_pred:
        colors.append("Predicted Only")
    else:
        colors.append("Nearby LLT")
    labels.append(term)

# Create interactive plot with Plotly
fig = px.scatter(
    x=proj[:, 0], y=proj[:, 1],
    color=colors,
    hover_name=labels,
    title="Interactive UMAP of LLT Embeddings (GT / Pred / Neighbors)",
    labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2"}
)

fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))
fig.update_layout(legend_title_text='LLT Category', width=900, height=700)

# Save HTML interactive file
fig.write_html("/home/naghmedashti/MedDRA-LLM/RAG_Models/plots/interactive_llts_umap_v3.html")
#fig.write_image("/home/naghmedashti/MedDRA-LLM/plots/interactive_llts_umap.png")
fig.show()
