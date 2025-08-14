import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# -----------------------
# 1. Load LLT Embeddings
# -----------------------
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)

# Convert to DataFrame for easy manipulation
llt_terms = list(llt_embeds_raw.keys())
embeddings = np.array(list(llt_embeds_raw.values()))

'''
embeddings = pd.DataFrame([
    {"LLT_Term": k, **{f"emb_{i}": v[i] for i in range(len(v))}}
    for k, v in llt_embeds_raw.items()
])
'''

# -----------------------
# 2. Load mapping files
# -----------------------
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')

# Merge LLT with PT to get SOC per LLT
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")

# Ensure LLT_Term is unique in merged data
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])

# Add embedding vectors to the merged DataFrame
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_terms)].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# Filter embeddings to match merged data
filtered_embeddings = np.array(llt_merged["embedding"].tolist())

# -----------------------
# 3. Run UMAP
# -----------------------
reducer = umap.UMAP(random_state=42)
umap_result = reducer.fit_transform(filtered_embeddings)
llt_merged["umap_x"] = umap_result[:, 0]
llt_merged["umap_y"] = umap_result[:, 1]

# -----------------------
# 4. Top-k LLT per SOC
# -----------------------
Top_k = 50
highlighted_points = pd.DataFrame()
soc_groups = llt_merged.groupby("SOC_Term")

for soc, group in soc_groups:
    group_embeddings = np.array(group["embedding"].tolist())
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(group_embeddings)
    # Average similarity of each LLT to all others in same SOC
    avg_sim = sim_matrix.mean(axis=1)
    top_indices = avg_sim.argsort()[::-1][:Top_k]  # Top-k LLT
    top_llts = group.iloc[top_indices].copy()
    top_llts["highlight"] = True
    highlighted_points = pd.concat([highlighted_points, top_llts], ignore_index=True)

# Add highlight flag to full dataframe
llt_merged["highlight"] = llt_merged["LLT_Term"].isin(highlighted_points["LLT_Term"])

# -----------------------
# 5. Plotting
# -----------------------
plt.figure(figsize=(20, 10))
sns.set_style("white")

# Plot all points in light gray
plt.scatter(llt_merged["umap_x"], llt_merged["umap_y"], 
            s=10, color="lightgray", label="Other LLTs", alpha=0.4)

# Define distinct colors for each SOC
unique_socs = sorted(highlighted_points["SOC_Term"].unique())
palette = sns.color_palette("hsv", len(unique_socs))
soc_to_color = {soc: palette[i] for i, soc in enumerate(unique_socs)}

# Plot Top-50 per SOC in color
for soc in unique_socs:
    subset = highlighted_points[highlighted_points["SOC_Term"] == soc]
    plt.scatter(subset["umap_x"], subset["umap_y"], 
                s=20, label=soc, color=soc_to_color[soc], alpha=0.9)

# Legend
plt.title("UMAP of LLT Embeddings  Top-50 Similar LLTs Highlighted per SOC", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="SOC Terms", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()

# Save
plt.savefig("umap_top50_per_soc.png", dpi=300)
print(" Plot saved as umap_top50_per_soc.png")
