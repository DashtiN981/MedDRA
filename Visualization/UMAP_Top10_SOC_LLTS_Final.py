import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
import umap

# -----------------------------
# Step 1: Load Data
# -----------------------------
mosaic_df = pd.read_csv("./data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=";", encoding='latin1')
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=";", encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=";", encoding='latin1')

with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# -----------------------------
# Step 2: Merge LLT-PT-SOC Data
# -----------------------------
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# Map SOC_Code to SOC_Term
soc_code_to_term = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
llt_merged["SOC_Term"] = llt_merged["SOC_Code"].map(soc_code_to_term)

# -----------------------------
# Step 3: Get Top-10 SOCs
# -----------------------------
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
filtered_llt = llt_merged[llt_merged["SOC_Code"].isin(top_soc_codes)].copy()

# -----------------------------
# Step 4: Select 20 Central LLTs Per SOC
# -----------------------------
selected_df = []
for soc in top_soc_codes:
    group_df = filtered_llt[filtered_llt["SOC_Code"] == soc].copy()
    if len(group_df) < 20:
        continue

    # Compute centroid of SOC embeddings
    vecs = np.vstack(group_df["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    group_df["similarity"] = group_df["embedding"].apply(
        lambda x: cosine_similarity([x], [centroid])[0][0]
    )

    # Select 20 closest LLTs to the centroid
    top_20 = group_df.sort_values("similarity", ascending=False).head(20).copy()
    top_20["Selected"] = True
    selected_df.append(top_20)

selected_df = pd.concat(selected_df, ignore_index=True)
selected_llts = selected_df["LLT_Term"].tolist()

# -----------------------------
# Step 5: Add 500 Background LLTs
# -----------------------------
all_other_llts = list(set(llt_merged["LLT_Term"].tolist()) - set(selected_llts))
background_llts = np.random.choice(all_other_llts, size=500, replace=False).tolist()
background_df = llt_merged[llt_merged["LLT_Term"].isin(background_llts)].copy()
background_df["Selected"] = False

# Merge selected + background LLTs
umap_df = pd.concat([selected_df, background_df], ignore_index=True)
umap_embeddings = np.vstack(umap_df["embedding"].values)

# -----------------------------
# Step 6: UMAP Projection
# -----------------------------
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(umap_embeddings)
umap_df["x"] = embedding_2d[:, 0]
umap_df["y"] = embedding_2d[:, 1]

umap_df["Color"] = np.where(umap_df["Selected"], umap_df["SOC_Term"], "Other LLTs")

# -----------------------------
# Step 7: Plot (Static)
# -----------------------------
plt.figure(figsize=(16, 12))
colors = plt.cm.tab10(np.linspace(0, 1, len(top_soc_codes)))

# Plot background LLTs in gray
plt.scatter(umap_df[umap_df["Color"] == "Other LLTs"]["x"],
            umap_df[umap_df["Color"] == "Other LLTs"]["y"],
            color="lightgray", alpha=0.3, s=10, label="Other LLTs")

# Plot clusters for each SOC
for idx, soc in enumerate(top_soc_codes):
    soc_name = soc_code_to_term.get(soc, str(soc))
    cluster_df = umap_df[umap_df["SOC_Term"] == soc_name]
    coords = cluster_df[["x", "y"]].values
    plt.scatter(cluster_df[cluster_df["Selected"]]["x"],
                cluster_df[cluster_df["Selected"]]["y"],
                color=colors[idx], s=25, label=soc_name)

    # Draw convex hull over all LLTs of the SOC (selected + neighbors)
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], color=colors[idx], linewidth=1)
        except:
            pass

    # Label cluster center
    center = coords.mean(axis=0)
    plt.text(center[0], center[1], soc_name, fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

plt.title("UMAP of LLTs (Top-10 SOCs with 500 neighbors)", fontsize=16)
plt.axis("off")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./Visualization/UMAP_Top10_SOC_LLTS_Final.png", dpi=300)
plt.show()
