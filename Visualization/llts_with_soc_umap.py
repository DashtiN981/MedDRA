# visualize_llts_with_soc_umap.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import json

# ----------------------------------------
# Step 1: Load and merge LLT, PT, SOC data
# ----------------------------------------

# Load LLT dictionary
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[['LLT_Code', 'LLT_Term', 'PT_Code']].dropna()

# Load PT → SOC and drop duplicates to avoid duplication
pt_soc_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')
pt_soc_df = pt_soc_df[['PT_Code', 'SOC_Code', 'SOC_Term']].dropna()
pt_soc_df = pt_soc_df.drop_duplicates(subset=["PT_Code"])  #  fix overmerge

# Merge LLT → PT → SOC
merged_df = llt_df.merge(pt_soc_df, on='PT_Code', how='inner')
print(f"Total LLT entries with PT and SOC (deduplicated): {len(merged_df)}")

# ---------------------------------------------------------
# Step 2: Load embeddings and filter LLTs with embeddings
# ---------------------------------------------------------

# Load JSON embeddings and convert to DataFrame (assumed to contain columns: LLT_Term, emb_0, emb_1, ..., emb_n)
with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    embed_data = json.load(f)

embed_df = pd.DataFrame([
    {"LLT_Term": k, **{f"emb_{i}": v[i] for i in range(len(v))}}
    for k, v in embed_data.items()
])


# Merge with embeddings
full_df = merged_df.merge(embed_df, on='LLT_Term', how='inner')
print(f" Total LLTs with valid embedding and SOC: {len(full_df)}")

# Extract embedding matrix
embedding_cols = [col for col in full_df.columns if col.startswith("emb_")]
embeddings = full_df[embedding_cols].values

# Extract labels
soc_labels = full_df['SOC_Term'].values

# -----------------------------------------------------
# Step 3: Dimensionality reduction using UMAP (2D plot)
# -----------------------------------------------------

umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# ----------------------------------------------------
# Step 4: Plot using matplotlib + seaborn for colors
# ----------------------------------------------------

# Prepare color palette with as many unique SOCs as needed
unique_socs = sorted(full_df['SOC_Term'].unique())
palette = sns.color_palette("hsv", len(unique_socs))
soc_to_color = {soc: palette[i] for i, soc in enumerate(unique_socs)}
colors = [soc_to_color[soc] for soc in soc_labels]

plt.figure(figsize=(14, 10))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
            c=colors, s=10, alpha=0.7, linewidths=0)

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=soc, 
                      markerfacecolor=soc_to_color[soc], markersize=6) for soc in unique_socs]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='SOC Terms')
plt.title("UMAP of LLT Embeddings Colored by SOC_Term", fontsize=16)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()
plt.grid(False)
plt.savefig("./Visualization/llts_with_soc_umap.png", dpi=300)
plt.show()
