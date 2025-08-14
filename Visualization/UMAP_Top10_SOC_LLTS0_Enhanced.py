import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
import plotly.express as px

# File paths
mosaic_path = "./data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
llt_path = "./data/LLT2_Code_English_25_0.csv"
pt_path = "./data/PT2_SOC_25_0.csv"
embedding_path = "./embedding/llt2_embeddings.json"

# Load data
mosaic_df = pd.read_csv(mosaic_path, sep=";", encoding='latin1')
llt_df = pd.read_csv(llt_path, sep=";", encoding='latin1')
pt_df = pd.read_csv(pt_path, sep=";", encoding='latin1')

# Load embeddings
with open(embedding_path, "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)

llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# Merge and filter
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# Map SOC_Code to SOC_Term
soc_code_to_term = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
llt_merged["SOC_Term"] = llt_merged["SOC_Code"].map(soc_code_to_term)

# Top 10 SOCs by frequency in Mosaic AE data
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
filtered_llt = llt_merged[llt_merged["SOC_Code"].isin(top_soc_codes)].copy()

# Select 20 LLTs closest to centroid per SOC
selected_terms, selected_labels, selected_soc_terms = [], [], []

for soc in top_soc_codes:
    group_df = filtered_llt[filtered_llt["SOC_Code"] == soc].copy()
    group_df = group_df.dropna(subset=["embedding"])
    if len(group_df) < 20:
        continue

    vecs = np.vstack(group_df["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    group_df["similarity"] = group_df["embedding"].apply(lambda x: cosine_similarity([x], [centroid])[0][0])
    top_20 = group_df.sort_values("similarity", ascending=False).head(20)

    selected_terms.extend(top_20["LLT_Term"].tolist())
    selected_labels.extend([soc] * len(top_20))
    selected_soc_terms.extend(top_20["SOC_Term"].tolist())

# Background LLTs (gray points)
background_terms = list(set(llt_merged["LLT_Term"].tolist()) - set(selected_terms))
background_embeddings = [llt_embeds_raw[t] for t in background_terms]

# Embeddings for selected terms
selected_embeddings = [llt_embeds_raw[t] for t in selected_terms]

# UMAP projection
all_embeddings = selected_embeddings + background_embeddings
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(all_embeddings)
selected_2d = embedding_2d[:len(selected_embeddings)]
background_2d = embedding_2d[len(selected_embeddings):]

# Static plot with convex hulls
plt.figure(figsize=(16, 12))
colors = plt.cm.tab10(np.linspace(0, 1, len(top_soc_codes)))

# Plot background points in gray
plt.scatter(background_2d[:, 0], background_2d[:, 1], color='lightgray', alpha=0.3, s=10, label="Other LLTs")

# Plot clusters
for idx, soc in enumerate(top_soc_codes):
    indices = [i for i, label in enumerate(selected_labels) if label == soc]
    coords = selected_2d[indices]
    soc_name = soc_code_to_term.get(soc, str(soc))
    plt.scatter(coords[:, 0], coords[:, 1], color=colors[idx], s=25, label=soc_name)

    # Convex hull
    if len(coords) >= 3:
        hull = ConvexHull(coords)
        for simplex in hull.simplices:
            plt.plot(coords[simplex, 0], coords[simplex, 1], color=colors[idx], linewidth=1)

    # Label
    center = coords.mean(axis=0)
    plt.text(center[0], center[1], soc_name, fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

plt.title("UMAP of LLT Terms Grouped by Top-10 SOCs", fontsize=16)
plt.axis("off")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./Visualization/UMAP_Top10_SOC_LLTS0_Enhanced.png", dpi=300)

# Interactive plot
df_plot = pd.DataFrame(selected_2d, columns=["x", "y"])
df_plot["LLT_Term"] = selected_terms
df_plot["SOC_Term"] = selected_soc_terms
fig = px.scatter(df_plot, x="x", y="y", color="SOC_Term", hover_data=["LLT_Term"])
fig.write_html("./Visualization/UMAP_Top10_SOC_LLTS0_Interactive.html")
fig.show()

