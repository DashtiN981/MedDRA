import json
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go

# === Load files ===
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

with open("./Rag_Models/rag_prompting_reasoning_v3_final.json", "r", encoding="latin1") as f:
    pred_data = json.load(f)

# === Extract prediction records ===
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

# === Prepare embedding data ===
true_terms = [r["true_term"] for r in records]
pred_terms = [r["predicted"] for r in records]
all_terms = set(true_terms + pred_terms)

background_terms = list(set(llt_embeds_raw.keys()) - all_terms)
sampled_bg = np.random.choice(background_terms, size=3000, replace=False)  # Match matplotlib size

terms_umap = true_terms + pred_terms + list(sampled_bg)

# Initial labels
labels = (
    ["True"] * len(true_terms) +
    ["Pred"] * len(pred_terms) +
    ["Other"] * len(sampled_bg)
)

# Adjust to "Correct" if matching
for i, r in enumerate(records):
    if r["correct"]:
        labels[i] = "Correct"
        labels[i + len(records)] = "Correct"

embeds = [llt_embeds_raw[t] for t in terms_umap]

# === UMAP projection ===
reducer = umap.UMAP(n_neighbors=100, min_dist=0.9, random_state=42)
umap_result = reducer.fit_transform(embeds)

df = pd.DataFrame(umap_result, columns=["x", "y"])
df["LLT_Term"] = terms_umap
df["Label"] = labels

term_coords = dict(zip(df["LLT_Term"], zip(df["x"], df["y"])))

# === Plotly figure ===
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="Label",
    hover_data=["LLT_Term"],
    color_discrete_map={
        "Correct": "green",
        "True": "blue",
        "Pred": "red",
        "Other": "lightgray"
    },
    opacity=0.6
)

# Uniform marker size to match matplotlib
fig.update_traces(marker=dict(size=5))

# === Add lines between incorrect predictions
for r in records:
    if not r["correct"]:
        true_pos = term_coords.get(r["true_term"])
        pred_pos = term_coords.get(r["predicted"])
        if true_pos and pred_pos:
            fig.add_trace(go.Scatter(
                x=[true_pos[0], pred_pos[0]],
                y=[true_pos[1], pred_pos[1]],
                mode="lines",
                line=dict(color="orange", width=0.5),
                hoverinfo="skip",
                showlegend=False
            ))

# === Final layout
fig.update_layout(
    title="UMAP of LLT Embeddings – Correct vs. Incorrect Predictions (Plotly Version)",
    legend_title="Label",
    width=1200,
    height=900,
    paper_bgcolor="white"
)

# === Save interactive HTML
fig.write_html("./Visualization/other/Interactive_UMAP_Predictions_V3_Harmonized.html")

# Optional: Save static PNG (if kaleido installed)
# fig.write_image("./Visualization/other/UMAP_Prediction_V3_Harmonized.png", scale=2)
