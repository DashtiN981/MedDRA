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
llt_df = pd.read_csv(llt_path,  sep=";", encoding='latin1')
pt_df = pd.read_csv(pt_path,  sep=";", encoding='latin1')

# Load embeddings
with open(embedding_path, "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# Merge and clean LLT → PT → SOC
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)
soc_code_to_term = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
llt_merged["SOC_Term"] = llt_merged["SOC_Code"].map(soc_code_to_term)

# Get top 10 SOCs based on Mosaic AE frequency
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
top_soc_terms = [soc_code_to_term.get(code, str(code)) for code in top_soc_codes]

# For each SOC: select 20 central + up to 1000 gray neighbors
selected_entries = []
background_entries = []

for soc in top_soc_codes:
    group_df = llt_merged[llt_merged["SOC_Code"] == soc].copy()
    group_df = group_df.dropna(subset=["embedding"])
    if len(group_df) < 30:
        continue

    vecs = np.vstack(group_df["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    group_df["similarity"] = group_df["embedding"].apply(lambda x: cosine_similarity([x], [centroid])[0][0])
    
    # Top 20 colored
    top_20 = group_df.sort_values("similarity", ascending=False).head(20).copy()
    top_20["is_selected"] = True
    top_20["Display_Color"] = soc_code_to_term.get(soc, str(soc))
    selected_entries.append(top_20)

    # Up to 1000 gray neighbors (excluding top 20)
    rest = group_df[~group_df["LLT_Term"].isin(top_20["LLT_Term"])]
    rest_top_1k = rest.sort_values("similarity", ascending=False).head(1000).copy()
    rest_top_1k["is_selected"] = False
    rest_top_1k["Display_Color"] = "Other LLTs"
    background_entries.append(rest_top_1k)

# Combine all entries for UMAP
umap_df = pd.concat(selected_entries + background_entries, ignore_index=True)
umap_embeddings = np.vstack(umap_df["embedding"].values)
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(umap_embeddings)
umap_df["x"] = embedding_2d[:, 0]
umap_df["y"] = embedding_2d[:, 1]

# Static plot
plt.figure(figsize=(16, 12))
colors = plt.cm.tab10(np.linspace(0, 1, len(top_soc_codes)))

# Gray background points
bg_df = umap_df[umap_df["is_selected"] == False]
plt.scatter(bg_df["x"], bg_df["y"], color='lightgray', alpha=0.3, s=10, label="Other LLTs")

# Colored clusters with convex hulls
for idx, soc in enumerate(top_soc_codes):
    label = soc_code_to_term.get(soc, str(soc))
    soc_df = umap_df[(umap_df["Display_Color"] == label)]
    coords = soc_df[["x", "y"]].values
    color = colors[idx]
    plt.scatter(coords[:, 0], coords[:, 1], color=color, s=25, label=label)

    # Convex Hull over colored LLTs only
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], color=color, linewidth=1)
        except:
            pass

    # Cluster label in center
    center = coords.mean(axis=0)
    plt.text(center[0], center[1], label, fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

plt.title("UMAP of LLT Terms by Top-10 SOCs (Selected + Local Neighbors)", fontsize=16)
plt.axis("off")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./Visualization/UMAP_Top10_SOC_LLTS_Selected_and_Neighbors.png", dpi=300)

# Interactive plot
fig = px.scatter(
    umap_df,
    x="x",
    y="y",
    color="Display_Color",
    hover_data=["LLT_Term", "SOC_Term"],
    title="Interactive UMAP of LLTs: Top-10 SOCs with Local Context",
)
fig.write_html("./Visualization/UMAP_Top10_SOC_LLTS_Interactive_Selected_and_Neighbors.html")
fig.show()
