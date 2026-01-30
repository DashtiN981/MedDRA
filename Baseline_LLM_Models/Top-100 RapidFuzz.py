import pandas as pd
from rapidfuzz import fuzz, process

LLT_CSV_PATH = "./data/LLT2_Code_English_25_0.csv"
AE_CSV_PATH  = "./data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv"

def canon(x):
    if pd.isna(x):
        return None
    return str(x).strip().split(".")[0]

# load LLTs
llt_df = pd.read_csv(LLT_CSV_PATH, sep=';', encoding='latin1')
llt_df["LLT_Code"] = llt_df["LLT_Code"].apply(canon)
llt_df = llt_df.dropna(subset=["LLT_Code", "LLT_Term"]).reset_index(drop=True)

llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_terms = llt_df["LLT_Term"].tolist()

# load AE file
ae_df = pd.read_csv(AE_CSV_PATH, sep=';', encoding='latin1')
ae_df = ae_df.dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"])
ae_df["ZB_LLT_Code"] = ae_df["ZB_LLT_Code"].apply(canon)

included = 0
not_included = 0
missing_in_dictionary = 0

for idx, row in ae_df.iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = row["ZB_LLT_Code"]

    # 1) true LLT code must exist in dictionary
    if true_code not in llt_code_to_term:
        missing_in_dictionary += 1
        continue

    true_term = llt_code_to_term[true_code]

    # 2) RapidFuzz top-100
    top = process.extract(
        ae_text,
        llt_terms,
        scorer=fuzz.token_set_ratio,
        limit=100
    )
    top_terms = [t[0] for t in top]

    if true_term in top_terms:
        included += 1
    else:
        not_included += 1

print("Total AE:", len(ae_df))
print("Missing in MedDRA dictionary:", missing_in_dictionary)
print("Included:", included)
print("Not included:", not_included)
print("Percentage included:", included / (included + not_included) * 100 if included + not_included > 0 else 0)
print("Percentage missing:", not_included / (included + not_included) * 100 if included + not_included > 0 else 0)
