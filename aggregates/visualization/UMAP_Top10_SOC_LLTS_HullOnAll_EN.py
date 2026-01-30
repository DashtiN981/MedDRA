# -*- coding: utf-8 -*-
import os, pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull

# ---------------- translation helpers ----------------
# 1) read from translation file "./data/soc_translation_de_en.csv"  (Columns:de,en)
def load_translation_map(path="./data/soc_translation_de_en.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path,sep=";")
        if {"de","en"}.issubset(set(df.columns)):
            # strip و lowercase for secure accordance
            return {str(r["de"]).strip(): str(r["en"]).strip() for _, r in df.iterrows()}
    return {}

# 2) local dictionary for commom SOC   (Germany → Englosh)
FALLBACK_DE_EN = {
    "Erkrankungen des Gastrointestinaltrakts": "Gastrointestinal disorders",
    "Gefaesserkrankungen": "Vascular disorders",
    "Gefässerkrankungen": "Vascular disorders",
    "Gefäßerkrankungen": "Vascular disorders",
    "Untersuchungen": "Investigations",
    "Erkrankungen des Blutes und des Lymphsystems": "Blood and lymphatic system disorders",
    "Infektionen und parasitäre Erkrankungen": "Infections and infestations",
    "Infektionen und parasitaere Erkrankungen": "Infections and infestations",
    "Allgemeine Erkrankungen und Beschwerden am Verabreichungsort": "General disorders and administration site conditions",
    "Stoffwechsel- und Ernährungsstörungen": "Metabolism and nutrition disorders",
    "Stoffwechsel- und Ernaehrungsstoerungen": "Metabolism and nutrition disorders",
    "Erkrankungen der Nieren und Harnwege": "Renal and urinary disorders",
    "Erkrankungen der Haut und des Unterhautgewebes": "Skin and subcutaneous tissue disorders",
    "Psychiatrische Erkrankungen": "Psychiatric disorders",
}

TRANS_MAP = load_translation_map()

def translate_soc(term: str) -> str:
    if term is None or (isinstance(term, float) and np.isnan(term)):
        return ""
    t = str(term).strip()
    #  first fallback from user file 
    return TRANS_MAP.get(t, FALLBACK_DE_EN.get(t, t))

# ---------------- load data ----------------
mosaic_df = pd.read_csv("./data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=";", encoding='latin1')
llt_df = pd.read_csv("./data/LLT2_Code_English_25_0.csv", sep=";", encoding='latin1')
pt_df = pd.read_csv("./data/PT2_SOC_25_0.csv", sep=";", encoding='latin1')

with open("./embedding/llt2_embeddings.json", "r", encoding="latin1") as f:
    llt_embeds_raw = json.load(f)
llt_embeds_raw = {k: np.array(v) for k, v in llt_embeds_raw.items()}

# ---------------- merge LLT → PT → SOC ----------------
llt_merged = llt_df.merge(pt_df, on="PT_Code", how="left")
llt_merged = llt_merged.drop_duplicates(subset=["LLT_Term"])
llt_merged = llt_merged[llt_merged["LLT_Term"].isin(llt_embeds_raw.keys())].reset_index(drop=True)
llt_merged["embedding"] = llt_merged["LLT_Term"].map(llt_embeds_raw)

# آلمانی → انگلیسی
soc_code_to_term_de = dict(pt_df[["SOC_Code", "SOC_Term"]].drop_duplicates().values)
soc_code_to_term_en = {code: translate_soc(de) for code, de in soc_code_to_term_de.items()}
llt_merged["SOC_Term_EN"] = llt_merged["SOC_Code"].map(soc_code_to_term_en)

# ---------------- pick top-10 SOC codes from Mosaic ----------------
top_soc_codes = mosaic_df["ZB_SOC_Code"].value_counts().nlargest(10).index.tolist()
top_soc_terms_en = [soc_code_to_term_en.get(code, str(code)) for code in top_soc_codes]

# ---------------- select points per SOC ----------------
selected_llts = []
for soc in top_soc_codes:
    g = llt_merged[llt_merged["SOC_Code"] == soc].copy()
    g = g.dropna(subset=["embedding"])
    if len(g) < 30:
        continue

    vecs = np.vstack(g["embedding"].values)
    centroid = np.mean(vecs, axis=0)
    g["similarity"] = g["embedding"].apply(lambda x: cosine_similarity([x], [centroid])[0][0])

    top_20 = g.sort_values("similarity", ascending=False).head(20).copy()
    top_20["is_selected"] = True
    top_20["Color"] = soc_code_to_term_en.get(soc, str(soc))  # English

    gray = g[~g["LLT_Term"].isin(top_20["LLT_Term"])].sort_values("similarity", ascending=False).head(1000).copy()
    gray["is_selected"] = False
    gray["Color"] = "Other LLTs"

    selected_llts.append(pd.concat([top_20, gray], ignore_index=True))

# ---------------- UMAP ----------------
umap_df = pd.concat(selected_llts, ignore_index=True)
umap_embeddings = np.vstack(umap_df["embedding"].values)
reducer = umap.UMAP(n_neighbors=200, min_dist=0.99, random_state=42)
embedding_2d = reducer.fit_transform(umap_embeddings)
umap_df["x"], umap_df["y"] = embedding_2d[:, 0], embedding_2d[:, 1]
# ---------------- global font config ----------------
import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 18
# ---------------- static plot (legend bottom-right) ----------------
plt.figure(figsize=(14, 12))
soc_palette = plt.cm.tab10(np.linspace(0, 1, len(top_soc_terms_en)))
soc_color_map = {term: soc_palette[i] for i, term in enumerate(top_soc_terms_en)}

# gray background
gray_points = umap_df[umap_df["Color"] == "Other LLTs"]
plt.scatter(gray_points["x"], gray_points["y"], color="lightgray", alpha=0.28, s=20, label="Other LLTs")

# clusters + hull + English labels
for term_en in top_soc_terms_en:
    df = umap_df[umap_df["Color"] == term_en]
    if df.empty: 
        continue
    coords = df[["x", "y"]].values
    color = soc_color_map[term_en]
    plt.scatter(coords[:, 0], coords[:, 1], color=color, s=40, label=term_en)

    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            for smp in hull.simplices:
                plt.plot(coords[smp, 0], coords[smp, 1], color=color, linewidth=1)
        except Exception:
            pass

    center = coords.mean(axis=0)
    '''plt.text(center[0], center[1], term_en, 
             bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))'''

# بدون عنوان
plt.axis("off")

# legend 
fig = plt.gcf(); ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()

''' plt.subplots_adjust(bottom=0.16)  
fig.legend(handles, labels,
           loc='lower left',
           bbox_to_anchor=(0.03, 0.02),
           bbox_transform=fig.transFigure,
           frameon=True)

plt.tight_layout()
plt.savefig("./aggregates/visualization/UMAP_Top10_SOC_LLTS_HullOnAll_EN.png", dpi=300)
'''

leg = fig.legend(handles, labels,
                 loc='lower right',              # position
                 bbox_to_anchor=(1.4, 0.03),     # exact position
                 bbox_transform=fig.transFigure,
                 frameon=True)

fig.savefig("./aggregates/visualization/UMAP_Top10_SOC_LLTS_HullOnAll_EN.png",
            dpi=300, bbox_inches="tight", pad_inches=0.5)


# ---------------- zoomed plot (filter by SOC name) ----------------
target_soc_term = "Blood and lymphatic system disorders"
target_df = umap_df[(umap_df["Color"] == target_soc_term) & (umap_df["is_selected"])].copy()

plt.figure(figsize=(10, 6))
plt.scatter(target_df["x"], target_df["y"], c="red", s=30)

for _, row in target_df.iterrows():
    plt.text(row["x"] + 0.2, row["y"], row["LLT_Term"], alpha=0.7, color="black")

plt.axis("off")
plt.tight_layout()
plt.savefig("./aggregates/visualization/Zoom_LLTs_BloodLymph.png", dpi=300)
plt.show()

