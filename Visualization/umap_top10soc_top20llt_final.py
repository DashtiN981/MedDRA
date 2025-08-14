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
# 1. Load data
# -----------------------------
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)

llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')

# -----------------------------
# 2. Merge data
# -----------------------------
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# -----------------------------
# 3. UMAP
# -----------------------------
embeddings = np.array(llt_merged["embedding"].tolist())
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, metric='cosine', random_state=42)
umap_result = reducer.fit_transform(embeddings)
llt_merged["umap_x"] = umap_result[:, 0]
llt_merged["umap_y"] = umap_result[:, 1]

# -----------------------------
# 4. Select top 10 SOCs
# -----------------------------
top_socs = llt_merged["SOC_Term"].value_counts().nlargest(10).index.tolist()
llt_top10 = llt_merged[llt_merged["SOC_Term"].isin(top_socs)].copy()

# -----------------------------
# 5. Pick 20 most similar LLTs per SOC
# -----------------------------
highlighted = []
for soc in top_socs:
    group = llt_top10[llt_top10["SOC_Term"] == soc].copy()
    embs = np.array(group["embedding"].tolist())
    sim_matrix = cosine_similarity(embs)
    avg_sim = sim_matrix.mean(axis=1)
    top_indices = avg_sim.argsort()[::-1][:20]
    top_llts = group.iloc[top_indices].copy()
    top_llts["highlight"] = True
    rest_llts = group.drop(index=top_llts.index).copy()
    rest_llts["highlight"] = False
    highlighted.append(pd.concat([top_llts, rest_llts]))

final_df = pd.concat(highlighted, ignore_index=True)

# -----------------------------
# 6. Generate distinct colors
# -----------------------------
def generate_colors(n):
    hues = [i / n for i in range(n)]
    random.shuffle(hues)
    return [colorsys.hls_to_rgb(h, 0.5, 0.8) for h in hues]

soc_colors = generate_colors(len(top_socs))
soc_to_color = {soc: soc_colors[i] for i, soc in enumerate(top_socs)}

# -----------------------------
# 7. Plot
# -----------------------------
plt.figure(figsize=(20, 12))

for soc in top_socs:
    group = final_df[final_df["SOC_Term"] == soc]
    color = soc_to_color[soc]

    # رسم نقاط خاکستری برای LLTهای غیر مشابه
    others = group[~group["highlight"]]
    plt.scatter(others["umap_x"], others["umap_y"], s=10, c='lightgray', alpha=0.3)

    # رسم 20 LLT نزدیک‌تر
    highlights = group[group["highlight"]]
    plt.scatter(highlights["umap_x"], highlights["umap_y"], s=25, color=color, label=soc)

    # رسم مرز محدوده SOC با Convex Hull
    if len(group) >= 3:
        try:
            points = group[["umap_x", "umap_y"]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=2, alpha=0.4)
        except:
            pass

    # برچسب در مرکز ثقل
    center_x = group["umap_x"].mean()
    center_y = group["umap_y"].mean()
    plt.text(center_x, center_y, soc, fontsize=10, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

plt.title("Top-10 SOCs – Top-20 Most Similar LLTs per Cluster with Cluster Borders", fontsize=16)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="SOC Terms", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("./Visualization/umap_top10soc_top20llt_final.png", dpi=300)
print(" Saved a umap_top10soc_top20llt_final.png")
