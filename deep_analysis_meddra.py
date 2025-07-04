import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA2_LLT_Code_25_0.csv", sep=';', encoding='latin1')
pt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA_PT_SOC_25_0.csv", sep=';', encoding='latin1')

# Merge on PT_Code
merged_df = pd.merge(llt_df, pt_df, on='PT_Code', how='left')

print("Columns in merged dataset:")
print(merged_df.columns.tolist())

# Count LLTs per PT
llt_per_pt = merged_df.groupby('PT_Code')['LLT_Term'].count().sort_values(ascending=False)
print("\n--- LLT count per PT ---")
print("Average LLTs per PT:", round(llt_per_pt.mean(), 2))
print("Max LLTs under a single PT:", llt_per_pt.max())
print("PTs with only one LLT:", (llt_per_pt == 1).sum())

# PTs per SOC
pts_per_soc = merged_df.groupby('SOC_Term')['PT_Code'].nunique().sort_values(ascending=False)
print("\n--- PT count per SOC ---")
print(pts_per_soc.head(10))

# LLTs per SOC
llts_per_soc = merged_df.groupby('SOC_Term')['LLT_Term'].nunique().sort_values(ascending=False)
print("\n--- LLT count per SOC ---")
print(llts_per_soc.head(10))

# LLT/PT ratio
ratio_llt_pt = llts_per_soc / pts_per_soc
print("\n--- LLT/PT ratio per SOC ---")
print(ratio_llt_pt.sort_values(ascending=False).head(10))

# Multi-axial PTs
pt_soc_counts = merged_df.groupby('PT_Code')['SOC_Term'].nunique()
multi_soc_pts = pt_soc_counts[pt_soc_counts > 1]
print("\n--- PTs linked to multiple SOCs ---")
print("Number of PTs with multi-axial SOC assignments:", len(multi_soc_pts))
print("Examples:\n", multi_soc_pts.head())

# Unique counts
unique_llts = merged_df["LLT_Term"].nunique()
unique_pts = merged_df["PT_Term"].nunique()
print(f"\nUnique LLT Terms: {unique_llts}")
print(f"Unique PT Terms: {unique_pts}")

# Sample LLTs per PT
llt_samples_per_pt = merged_df.groupby("PT_Term")["LLT_Term"].apply(lambda x: list(x.unique())[:3]).head(10)
print("\n--- Sample LLTs per PT ---")
print(llt_samples_per_pt)

# LLT term length statistics on unique LLTs
unique_llts = merged_df["LLT_Term"].dropna().unique()
llt_lengths = pd.Series([len(str(term).split()) for term in unique_llts])
print("\n--- LLT Term Length Statistics ---")
print(llt_lengths.describe())

# PTs with only one LLT
llt_count_per_pt = merged_df.groupby("PT_Code")["LLT_Term"].nunique()
pt_with_one_llt = (llt_count_per_pt == 1).sum()
pt_total = llt_count_per_pt.shape[0]
pt_with_one_llt_percentage = (pt_with_one_llt / pt_total) * 100
print("\nPercentage of PTs with only one LLT: {:.2f}%".format(pt_with_one_llt_percentage))

# ------------------ VISUALIZATION ------------------

# Reuse already-known dictionaries if needed here (or just plot from variables)
top_pt = pts_per_soc.head(10)
top_llt = llts_per_soc.head(10)
top_ratio = ratio_llt_pt.sort_values(ascending=False).head(10)

# Translate
soc_translation = {
    "Untersuchungen": "Investigations",
    "Verletzung, Vergiftung und durch Eingriffe bedingte Komplikationen": "Injury, poisoning and procedural complications",
    "Chirurgische und medizinische Eingriffe": "Surgical and medical procedures",
    "Gutartige, boesartige und nicht spezifizierte Neubildungen (einschl. Zysten und Polypen)": "Neoplasms (benign, malignant, unspecified)",
    "Infektionen und parasitaere Erkrankungen": "Infections and infestations",
    "Erkrankungen des Nervensystems": "Nervous system disorders",
    "Erkrankungen des Gastrointestinaltrakts": "Gastrointestinal disorders",
    "Gefaesserkrankungen": "Vascular disorders",
    "Kongenitale, familiaere und genetische Erkrankungen": "Congenital and genetic disorders",
    "Erkrankungen der Haut und des Unterhautgewebes": "Skin and subcutaneous tissue disorders",
    "Skelettmuskulatur-, Bindegewebs- und Knochenerkrankungen": "Musculoskeletal and connective tissue disorders",
    "Schwangerschaft, Wochenbett und perinatale Erkrankungen": "Pregnancy and perinatal conditions",
    "Psychiatrische Erkrankungen": "Psychiatric disorders",
    "Erkrankungen des Blutes und des Lymphsystems": "Blood and lymphatic system disorders",
    "Erkrankungen des Ohrs und des Labyrinths": "Ear and labyrinth disorders",
    "Produktprobleme": "Product issues"
}

top_pt.index = top_pt.index.map(lambda x: soc_translation.get(x, x))
top_llt.index = top_llt.index.map(lambda x: soc_translation.get(x, x))
top_ratio.index = top_ratio.index.map(lambda x: soc_translation.get(x, x))

# Plot 1: PTs by SOC
plt.figure(figsize=(10, 6))
bars1 = plt.barh(top_pt.index, top_pt.values, color='skyblue')
plt.xlabel("Number of PTs")
plt.title("Top 10 SOCs by PT Count")
plt.gca().invert_yaxis()
for bar in bars1:
    plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, str(bar.get_width()), va='center')
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/data/Statistics_Images/pt_count_by_soc_en_labeled.png")

# Plot 2: LLTs by SOC
plt.figure(figsize=(10, 6))
bars2 = plt.barh(top_llt.index, top_llt.values, color='salmon')
plt.xlabel("Number of LLTs")
plt.title("Top 10 SOCs by LLT Count")
plt.gca().invert_yaxis()
for bar in bars2:
    plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, str(bar.get_width()), va='center')
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/data/Statistics_Images/llt_count_by_soc_en_labeled.png")

# Plot 3: LLT/PT Ratio
plt.figure(figsize=(10, 6))
bars3 = plt.barh(top_ratio.index, top_ratio.values, color='seagreen')
plt.xlabel("LLT/PT Ratio")
plt.title("Top 10 SOCs by LLT/PT Ratio")
plt.gca().invert_yaxis()
for bar in bars3:
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
plt.tight_layout()
plt.savefig("/home/naghmedashti/MedDRA-LLM/data/Statistics_Images/llt_pt_ratio_by_soc_en_labeled.png")
