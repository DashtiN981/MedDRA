import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
import plotly.express as px

# Load input data
mosaic_df = pd.read_csv("./data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=";", encoding='latin1')
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=";", encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=";", encoding='latin1')

with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# Merge data and map SOC
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)
soc_code_to_term = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
llt_merged["SOC_Term"] = llt_merged["SOC_Code"].map(soc_code_to_term)

# Get top-10 SOCs
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
top_soc_terms = [soc_code_to_term.get(code, str(code)) for code in top_soc_codes]

# Collect LLTs per SOC
selected_for_umap = []
for soc in top_soc_codes:
    group_df = llt_merged[llt_merged["SOC_Code"] == soc].copy()
    group_df = group_df.dropna(subset=["embedding"])
    if len(group_df) < 30:
        continue
    # Compute similarity to centroid
    vecs = np.vstack(group_df["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    group_df["similarity"] = group_df["embedding"].apply(lambda x: cosine_similarity([x], [centroid])[0][0])
    # Mark 20 closest as 'selected'
    group_df["is_selected"] = False
    group_df.loc[group_df.sort_values("similarity", ascending=False).head(20).index, "is_selected"] = True
    group_df["Display_Color"] = np.where(group_df["is_selected"], group_df["SOC_Term"], "Other LLTs")
    selected_for_umap.append(group_df)

umap_df = pd.concat(selected_for_umap, ignore_index=True)
umap_embeddings = np.vstack(umap_df["embedding"].values)
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(umap_embeddings)
umap_df["x"] = embedding_2d[:, 0]
umap_df["y"] = embedding_2d[:, 1]

# Static plot
plt.figure(figsize=(16, 12))
soc_palette = plt.cm.tab10(np.linspace(0, 1, len(top_soc_codes)))
soc_color_map = {term: soc_palette[i] for i, term in enumerate(top_soc_terms)}

# Plot gray points first
gray_points = umap_df[umap_df["Display_Color"] == "Other LLTs"]
plt.scatter(gray_points["x"], gray_points["y"], color="lightgray", alpha=0.4, s=10, label="Other LLTs")

# Plot colored points and convex hulls
for term in top_soc_terms:
    df = umap_df[umap_df["SOC_Term"] == term]
    color = soc_color_map[term]
    coords = df[["x", "y"]].values
    plt.scatter(df[df["is_selected"]]["x"], df[df["is_selected"]]["y"], color=color, s=25, label=term)
    # Hull on all points (colored + gray) per SOC
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], color=color, linewidth=1)
        except:
            pass
    # Label in center of hull
    center = coords.mean(axis=0)
    plt.text(center[0], center[1], term, fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

plt.title("UMAP of 20-Top LLT Terms(with Local Neighbors) for Top-10 SOCs", fontsize=16)
plt.axis("off")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./Visualization/UMAP_Top10_SOC_LLTS_Corrected.png", dpi=300)

# Interactive plot
fig = px.scatter(
    umap_df,
    x="x",
    y="y",
    color="Display_Color",
    hover_data=["LLT_Term", "SOC_Term"],
    title="Interactive UMAP of 20-Top LLT Terms"
)
fig.write_html("./Visualization/UMAP_Top10_SOC_LLTS_Corrected_Interactive.html")
fig.show()
