import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import random
import colorsys
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull

# -----------------------------
# 1. Load data files
# -----------------------------
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)

llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')

# -----------------------------
# 2. Merge LLT → PT → SOC
# -----------------------------
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# -----------------------------
# 3. Sample 5000 points for UMAP
# -----------------------------
if len(llt_merged) > 5000:
    llt_merged = llt_merged.sample(n=5000, random_state=42)

embeddings = np.array(llt_merged["embedding"].tolist())

# -----------------------------
# 4. UMAP projection
# -----------------------------
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, metric='cosine', random_state=42)
umap_result = reducer.fit_transform(embeddings)
llt_merged["umap_x"] = umap_result[:, 0]
llt_merged["umap_y"] = umap_result[:, 1]

# -----------------------------
# 5. Select Top-k LLT per SOC
# -----------------------------
Top_k = 50
highlighted_points = pd.DataFrame()
for soc, group in llt_merged.groupby("SOC_Term"):
    emb_group = np.array(group["embedding"].tolist())
    sim_matrix = cosine_similarity(emb_group)
    avg_sim = sim_matrix.mean(axis=1)
    top_idx = avg_sim.argsort()[::-1][:Top_k]
    top_llts = group.iloc[top_idx].copy()
    top_llts["highlight"] = True
    highlighted_points = pd.concat([highlighted_points, top_llts], ignore_index=True)

llt_merged["highlight"] = llt_merged["LLT_Term"].isin(highlighted_points["LLT_Term"])

# -----------------------------
# 6. Generate distinct RGB colors for SOCs
# -----------------------------
def generate_distinct_colors(n, lightness=0.5, saturation=0.95):
    hues = [i / n for i in range(n)]
    random.shuffle(hues)
    return [colorsys.hls_to_rgb(h, lightness, saturation) for h in hues]

unique_socs = sorted(highlighted_points["SOC_Term"].unique())
distinct_colors = generate_distinct_colors(len(unique_socs))
soc_to_color = {soc: distinct_colors[i] for i, soc in enumerate(unique_socs)}

# -----------------------------
# 7. Plotting with convex hulls and labels
# -----------------------------
plt.figure(figsize=(22, 12))

# Plot other (non-highlighted) points in gray
others = llt_merged[~llt_merged["highlight"]]
plt.scatter(others["umap_x"], others["umap_y"], 
            s=10, c='lightgray', alpha=0.3, label='Other LLTs')

# Plot top-50 per SOC
for soc in unique_socs:
    sub = highlighted_points[highlighted_points["SOC_Term"] == soc]
    color = soc_to_color[soc]
    plt.scatter(sub["umap_x"], sub["umap_y"],
                s=20, color=color, label=soc, alpha=0.9)

    # Draw convex hull if there are enough points
    if len(sub) >= 3:
        points = sub[["umap_x", "umap_y"]].values
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        plt.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.2, color=color, edgecolor='black', linewidth=0.5)

        # Label cluster at its centroid
        center_x = np.mean(hull_pts[:, 0])
        center_y = np.mean(hull_pts[:, 1])
        plt.text(center_x, center_y, soc, fontsize=7, ha='center', va='center', weight='bold')

# -----------------------------
# 8. Final plot style
# -----------------------------
plt.title("UMAP of LLT Embeddings  Top-50 LLTs per SOC with Cluster Boundaries")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="SOC Terms", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("./Visualization/umap_top50_per_soc_with_clusters.png", dpi=300)
print(" Saved as umap_top50_per_soc_with_clusters.png")
