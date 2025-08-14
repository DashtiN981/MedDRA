import pandas as pd
import numpy as np
import umap
import json
import plotly.express as px

# ----------------------------------------
# Step 1: Load and merge LLT, PT, SOC data
# ----------------------------------------

llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[['LLT_Code', 'LLT_Term', 'PT_Code']].dropna()

pt_soc_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=';', encoding='latin1')
pt_soc_df = pt_soc_df[['PT_Code', 'SOC_Code', 'SOC_Term']].dropna()
pt_soc_df = pt_soc_df.drop_duplicates(subset=["PT_Code"])

merged_df = llt_df.merge(pt_soc_df, on='PT_Code', how='inner')
print(f" Total LLT entries with PT and SOC: {len(merged_df)}")

# ---------------------------------------------------------
# Step 2: Load embeddings
# ---------------------------------------------------------

with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    embed_data = json.load(f)

embed_df = pd.DataFrame([
    {"LLT_Term": k, **{f"emb_{i}": v[i] for i in range(len(v))}}
    for k, v in embed_data.items()
])

# Merge all data
full_df = merged_df.merge(embed_df, on="LLT_Term", how="inner")
print(f" Total LLTs with embeddings: {len(full_df)}")

# Optional: Limit to 5000 for visualization
if len(full_df) > 5000:
    full_df = full_df.sample(n=5000, random_state=42)

# ---------------------------------------------------------
# Step 3: UMAP
# ---------------------------------------------------------

embedding_cols = [col for col in full_df.columns if col.startswith("emb_")]
X = full_df[embedding_cols].values

reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, metric='cosine', random_state=42)
umap_result = reducer.fit_transform(X)

full_df["UMAP_1"] = umap_result[:, 0]
full_df["UMAP_2"] = umap_result[:, 1]

# ---------------------------------------------------------
# Step 4: Plotly interactive plot
# ---------------------------------------------------------

fig = px.scatter(
    full_df,
    x="UMAP_1",
    y="UMAP_2",
    color="SOC_Term",
    hover_data=["LLT_Term", "SOC_Term"],
    title="Interactive UMAP of LLT Embeddings Colored by SOC_Term",
    width=1200,
    height=900
)

fig.update_traces(marker=dict(size=6, opacity=0.7), selector=dict(mode='markers'))
fig.write_html("./Visualization/interactive_umap_5000llts_by_soc.html")
fig.show()
