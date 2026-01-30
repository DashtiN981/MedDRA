import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
# import umap
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
import plotly.express as px

# Load files
mosaic_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=";", encoding='latin1')
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv", sep=";", encoding='latin1')
pt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv", sep=";", encoding='latin1')

with open("/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)

llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# Merge LLT → PT → SOC
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)
soc_code_to_term = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
llt_merged["SOC_Term"] = llt_merged["SOC_Code"].map(soc_code_to_term)

# Top 10 SOCs based on Mosaic AE data
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
top_soc_terms = [soc_code_to_term.get(code, str(code)) for code in top_soc_codes]

selected_llts = []
hull_llts = []

for soc in top_soc_codes:
    group_df = llt_merged[llt_merged["SOC_Code"] == soc].copy()
    group_df = group_df.dropna(subset=["embedding"])
    if len(group_df) < 30:
        continue

    # Compute centroid and similarity
    vecs = np.vstack(group_df["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    group_df["similarity"] = group_df["embedding"].apply(lambda x: cosine_similarity([x], [centroid])[0][0])

    # Select top 20 as colored
    top_20 = group_df.sort_values("similarity", ascending=False).head(20).copy()
    top_20["is_selected"] = True
    top_20["Color"] = soc_code_to_term.get(soc, str(soc))

    # Select up to 1000 more as gray but part of same SOC
    gray_part = group_df[~group_df["LLT_Term"].isin(top_20["LLT_Term"])].sort_values("similarity", ascending=False).head(1000).copy()
    gray_part["is_selected"] = False
    gray_part["Color"] = "gray"

    # For hull, combine both
    hull_llts.append(pd.concat([top_20, gray_part], ignore_index=True))
    selected_llts.append(pd.concat([top_20, gray_part], ignore_index=True))

# Concatenate for UMAP
umap_df = pd.concat(selected_llts, ignore_index=True)
umap_embeddings = np.vstack(umap_df["embedding"].values)
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(umap_embeddings)
umap_df["x"] = embedding_2d[:, 0]
umap_df["y"] = embedding_2d[:, 1]

# Static Plot
plt.figure(figsize=(16, 12))
soc_palette = plt.cm.tab10(np.linspace(0, 1, len(top_soc_codes)))
soc_color_map = {term: soc_palette[i] for i, term in enumerate(top_soc_terms)}

# Gray points
gray_points = umap_df[umap_df["Color"] == "gray"]
plt.scatter(gray_points["x"], gray_points["y"], color="lightgray", alpha=0.3, s=10, label="Other LLTs")

# Draw colored points and hulls
for term in top_soc_terms:
    df = umap_df[(umap_df["Color"] == term)]
    coords = df[["x", "y"]].values
    color = soc_color_map[term]
    plt.scatter(coords[:, 0], coords[:, 1], color=color, s=25, label=term)

    # Hull on all points (colored + gray) of same SOC
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], color=color, linewidth=1)
        except:
            pass

    center = coords.mean(axis=0)
    plt.text(center[0], center[1], term, fontsize=11, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

plt.title("UMAP of LLT Terms by Top-10 SOCs (with Local Neighbors)", fontsize=16)
plt.axis("off")
plt.legend(loc='lower right',
           bbox_to_anchor=(0.3, 0.02),   
           frameon=True, fontsize=10, borderaxespad=0.4)
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/aggregates/visualization/UMAP_Top10_SOC_LLTS_HullOnAll.png", dpi=300)

#  interactive visualization
''' umap_df["Display_Color"] = np.where(
    umap_df["is_selected"], umap_df["Color"], "Other LLTs"
)

fig = px.scatter(
    umap_df,
    x="x",
    y="y",
    color="Display_Color",
    hover_data=["LLT_Term", "SOC_Term"],
    title="Interactive UMAP of LLTs with Selected Terms Highlighted"
)

#fig.update_traces(marker=dict(size=6, opacity=0.85))
fig.update_traces(marker=dict(line=dict(width=0.5, color="black")))

custom_color_map = {
    "Other LLTs": "lightgray",
    **{term: px.colors.qualitative.Plotly[i % 10] for i, term in enumerate(top_soc_terms)}
}
fig.update_traces(marker=dict(line=dict(width=0)))  
fig.update_layout(coloraxis=dict(colorscale=None))
fig.for_each_trace(lambda t: t.update(marker_color=custom_color_map.get(t.name, t.marker.color)))

# output
fig.write_html("/home/naghmedashti/MedDRA-LLM/aggregates/visualization/UMAP_Top10_SOC_LLTS_HullOnAll_Interactive.html")
fig.show() '''

# ---------------- zoomed plot (filter by SOC name) ----------------
target_soc_term = "Erkrankungen des Blutes und des Lymphsystems"
target_df = umap_df[(umap_df["SOC_Term"] == target_soc_term) & (umap_df["is_selected"])].copy()

# Plot with names
plt.figure(figsize=(10, 8))
plt.scatter(target_df["x"], target_df["y"], c="red", s=30, label=target_soc_term)

# Add LLT Term labels
for _, row in target_df.iterrows():
    plt.text(row["x"] + 0.2, row["y"], row["LLT_Term"], fontsize=8, alpha=0.7)

# plt.title(f"Zoomed UMAP for {target_soc_term}", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/aggregates/visualization/Zoom_LLTs_BlutesLymph.png", dpi=300)
plt.show()
