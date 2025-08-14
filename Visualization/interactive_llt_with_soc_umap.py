import pandas as pd
import json
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load LLT → PT mapping
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')

# === Build LLT_Term → (SOC_Code, SOC_Term) mapping
llt_to_soccode = {}
llt_to_socterm = {}

for _, row in llt_df.iterrows():
    llt_term = row["LLT_Term"]
    pt_code = row["PT_Code"]

    match = pt_df[pt_df["PT_Code"] == pt_code]
    if not match.empty:
        soc_code = match["SOC_Code"].values[0]
        soc_term = match["SOC_Term"].values[0]
        llt_to_soccode[llt_term] = soc_code
        llt_to_socterm[llt_term] = soc_term

# === Load LLT embeddings
with open("./embedding/llt2_embeddings.json", "r", encoding="utf-8") as f:
    llt_emb_dict = json.load(f)

# === Filter: Only terms with both embedding and SOC_Code
filtered_terms = [term for term in llt_emb_dict if term in llt_to_soccode]
filtered_embs = np.array([llt_emb_dict[term] for term in filtered_terms])
filtered_soc_codes = [llt_to_soccode[term] for term in filtered_terms]
filtered_soc_terms = [llt_to_socterm[term] for term in filtered_terms]

print(f"Total LLTs with embedding + SOC_Code: {len(filtered_terms)}")

# === Encode SOC_Code for supervised UMAP training
le = LabelEncoder()
numeric_soc_labels = le.fit_transform(filtered_soc_codes)

# === Supervised UMAP projection based on SOC_Code
reducer = umap.UMAP(n_components=2, random_state=42)
proj = reducer.fit_transform(filtered_embs, y=numeric_soc_labels)  # Supervised by SOC_Code

# === Generate a color for each SOC_Term using matplotlib colormap
unique_socs = sorted(set(filtered_soc_terms))
cmap = cm.get_cmap('tab20b', len(unique_socs))  # or 'tab20', 'nipy_spectral', etc.

# Map each SOC_Term to a distinct hex color
soc_to_color = {
    soc: mcolors.to_hex(cmap(i)) for i, soc in enumerate(unique_socs)
}
colors_assigned = [soc_to_color[soc] for soc in filtered_soc_terms]

# === Plot using Plotly (manual colors)
fig = go.Figure()

for soc in unique_socs:
    indices = [i for i, s in enumerate(filtered_soc_terms) if s == soc]
    fig.add_trace(go.Scatter(
        x=proj[indices, 0],
        y=proj[indices, 1],
        mode='markers',
        marker=dict(size=6, color=soc_to_color[soc]),
        name=soc,
        text=[filtered_terms[i] for i in indices],
        hovertemplate='%{text}<br>' + soc
    ))

fig.update_layout(
    title="Supervised UMAP of LLT Embeddings (Colored by SOC_Term)",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2",
    width=1000,
    height=750,
    legend=dict(
        itemsizing='constant',
        font=dict(size=10)
    )
)

fig.write_html("./Visualization/interactive_llt_with_soc_umap.html")
fig.show()
