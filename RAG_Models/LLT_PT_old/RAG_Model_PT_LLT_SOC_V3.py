"""
File Name: RAG_Model_PT_LLT_SOC_V3.py    === Author: Naghme Dashti / August 2025

RRAG-based Prompting with Explicit Reasoning + Final Answer Line
---------------------------------------------------------------
This script maps predicted LLT -> PT and PT -> SOC and reports LLT/PT  and SOC accuracy
(AE SOC vs predicted primary SOC), with robust handling of Ist_Primary_SOC.
"""

import json
import re
import time
import random
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import OpenAI
from rapidfuzz import fuzz

# =========================
# Local OpenAI-compatible LLM API
# =========================
client = OpenAI(
    api_key="sk-aKGeEFMZB0gXEcE51FTc0A",
    base_url="http://pluto/v1/"
)

# =========================
# Parameters
# =========================
TOP_K = 10
MAX_ROWS = None       # e.g., put number (for example 100) to limit rows; None = all
EMB_DIM = 384

# Datasets
DATASET_NAME     = "KI_Projekt_Mosaic_AE_Codierung_2024_07_03"
DATASET_EMB_NAME = "ae_embeddings_Mosaic"

# Dictionaries and output
LLT_DICTIONARY_NAME      = "LLT2_Code_English_25_0"   # Include LLT_Code, LLT_Term, PT_Code
LLT_DICTIONARY_EMB_NAME  = "llt2_embeddings"
PT_DICTIONARY_NAME       = "PT2_SOC_25_0"
OUTPUT_FILE_NAME         = "Mosaic_output_v3_SOC"       # output file name

# Paths
AE_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{DATASET_NAME}.csv"
AE_EMB_FILE  = f"/home/naghmedashti/MedDRA-LLM/embedding/{DATASET_EMB_NAME}.json"
LLT_CSV_FILE = f"/home/naghmedashti/MedDRA-LLM/data/{LLT_DICTIONARY_NAME}.csv"
LLT_EMB_FILE = f"/home/naghmedashti/MedDRA-LLM/embedding/{LLT_DICTIONARY_EMB_NAME}.json"
PT_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{PT_DICTIONARY_NAME}.csv"

LLM_API_NAME = "Llama-3.3-70B-Instruct"  # or: llama-3.3-70b-instruct-awq
LLM_TEMP = 0.0
LLM_TOKEN = 250

# =========================
# Helpers
# =========================
def canon_code(x) -> str | None:
    """Normalize numeric codes like '10000081.0' -> '10000081', or pick first long digit run."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    m = re.match(r"^(\d+)(?:\.0+)?$", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{3,})", s)
    if m2:
        return m2.group(1)
    return s or None

def norm_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().casefold().split())

# =========================
# Load AE / LLT / PT (+ SOC primary mapping)
# =========================
ae_full = pd.read_csv(AE_CSV_FILE, sep=";", encoding="latin1")
ae_keep = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_full.columns]
if not ae_keep:
    raise ValueError("AE CSV must include at least 'Original_Term_aufbereitet' and 'ZB_LLT_Code'.")
ae_df = ae_full[ae_keep].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]]
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

# Read PT file once (detect columns), then load
pt_header = pd.read_csv(PT_CSV_FILE, sep=";", encoding="latin1", nrows=0).columns
pt_cols = [c for c in ["PT_Code","PT_Term","SOC_Code","SOC_Term","Ist_Primary_SOC"] if c in pt_header]
pt_df = pd.read_csv(PT_CSV_FILE, sep=";", encoding="latin1")[pt_cols]
pt_df["PT_Code"]  = pt_df["PT_Code"].map(canon_code)
pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)

# Lookups
llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_to_pt        = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))
soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"])) if "SOC_Term" in pt_df.columns else {}
pt_meta = pt_df.set_index("PT_Code")[["PT_Term"]].to_dict(orient="index")

# Primary SOC per PT (prefer Ist_Primary_SOC=='Y'; else if PT only has one SOC)
pt_code_to_primary_soc: dict[str, str | None] = {}
if "Ist_Primary_SOC" in pt_df.columns:
    prim = pt_df[pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper().eq("Y")]
    prim = prim.drop_duplicates(subset=["PT_Code"])
    pt_code_to_primary_soc.update(dict(zip(prim["PT_Code"], prim["SOC_Code"])))

pt_code_to_soc_all = (
    pt_df.groupby("PT_Code")["SOC_Code"]
    .apply(lambda s: sorted(set(x for x in s if pd.notna(x))))
    .to_dict()
)
for ptc, soc_list in pt_code_to_soc_all.items():
    if ptc not in pt_code_to_primary_soc:
        pt_code_to_primary_soc[ptc] = soc_list[0] if len(soc_list) == 1 else None

# Term -> LLT_Code
term_norm_to_llt = {}
for _, r in llt_df.iterrows():
    term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

def term_to_llt_code(pred_term: str, allow_fuzzy=True) -> str | None:
    """Exact normalized, then piece-wise, then optional fuzzy (conservative)."""
    t = norm_text(pred_term)
    if not t:
        return None
    if t in term_norm_to_llt:
        return term_norm_to_llt[t]
    for piece in re.split(r"[;,/]+", t):
        p = piece.strip()
        if p and p in term_norm_to_llt:
            return term_norm_to_llt[p]
    if allow_fuzzy:
        hits = get_close_matches(t, list(term_norm_to_llt.keys()), n=1, cutoff=0.94)
        return term_norm_to_llt[hits[0]] if hits else None
    return None

# =========================
# Load Embeddings
# =========================
with open(AE_EMB_FILE,  "r", encoding="latin1") as f:
    ae_emb_raw = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_emb_raw = json.load(f)

ae_emb_dict = {}
for k, v in ae_emb_raw.items():
    ae_emb_dict[k] = np.array(v)
    ae_emb_dict[norm_text(k)] = np.array(v)

llt_emb_dict = {k: np.array(v) for k, v in llt_emb_raw.items()}
llt_terms_all = list(llt_df["LLT_Term"])

if isinstance(MAX_ROWS, int) and MAX_ROWS > 0:
    ae_df = ae_df.iloc[:MAX_ROWS].reset_index(drop=True)
else:
    ae_df = ae_df.reset_index(drop=True)

# =========================
# RAG + prompting
# =========================
results = []
random.seed(42)

for idx, row in ae_df.iterrows():
    ae_text = str(row["Original_Term_aufbereitet"])
    true_LLT_Code = canon_code(row["ZB_LLT_Code"])
    true_LLT_term = llt_code_to_term.get(true_LLT_Code)
    true_PT_Code = llt_to_pt.get(true_LLT_Code)
    true_PT_term = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None

    # AE SOC ground-truth (raw, NO fallback)
    true_SOC_Code_AE = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_SOC_Code_AE = canon_code(v)

    # Build candidates from embeddings (fallback: difflib + random)
    ae_emb = ae_emb_dict.get(ae_text, ae_emb_dict.get(norm_text(ae_text)))
    if ae_emb is not None:
        ae_norm = np.linalg.norm(ae_emb)
        sims = []
        for llt_term, llt_emb in llt_emb_dict.items():
            denom = ae_norm * np.linalg.norm(llt_emb)
            score = float(np.dot(ae_emb, llt_emb) / denom) if denom else 0.0
            sims.append((llt_term, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        candidate_terms = [t for t, _ in sims[:TOP_K]]
    else:
        cand = get_close_matches(ae_text, llt_terms_all, n=TOP_K, cutoff=0.0)
        if len(cand) < TOP_K:
            extra = random.sample(llt_terms_all, k=min(TOP_K - len(cand), len(llt_terms_all)))
            cand += extra
        candidate_terms = cand[:TOP_K]

    if true_LLT_term and true_LLT_term not in candidate_terms:
        candidate_terms.append(true_LLT_term)
    random.shuffle(candidate_terms)

    prompt = (
        "You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term."
        f"\nHere is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        "Here is a list of candidate LLT terms:\n" + "\n".join(f"- {term}" for term in candidate_terms) +
        "\n\nPlease analyze the AE and list, and first provide a short reasoning."
        "\nThen, on a separate line, write the best matching LLT in this format:"
        "\nFinal answer: <LLT_TERM>"
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_API_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMP,
            max_tokens=LLM_TOKEN
        )
        answer = resp.choices[0].message.content.strip()
        answer_line = answer.split("Final answer:")[-1].strip() if "Final answer:" in answer else answer.strip().split("\n")[-1].strip()

        # Map predicted term -> LLT/PT/SOC
        pred_LLT_Code = term_to_llt_code(answer_line, allow_fuzzy=True)
        pred_PT_Code  = llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None
        pred_PT_term  = pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None
        pred_SOC_Code = pt_code_to_primary_soc.get(pred_PT_Code) if pred_PT_Code else None

        # Term-level metrics
        exact_LLT_match = (true_LLT_term is not None and answer_line == true_LLT_term)
        LLT_fuzzy_score = fuzz.ratio(answer_line.lower(), (true_LLT_term or "").lower())
        LLT_fuzzy_match = (true_LLT_term is not None and LLT_fuzzy_score >= 90)

        exact_PT_match = (true_PT_term is not None) and (pred_PT_term is not None) and (pred_PT_term == true_PT_term)
        if (true_PT_term is not None) and (pred_PT_term is not None):
            PT_fuzzy_score = fuzz.ratio(pred_PT_term.lower(), true_PT_term.lower())
            PT_fuzzy_match = PT_fuzzy_score >= 90
        else:
            PT_fuzzy_score = 0.0
            PT_fuzzy_match = False

        pred_LLT_term_std = llt_code_to_term.get(pred_LLT_Code, answer_line)

        results.append({
            "AE_text": ae_text,

            # LLT/PT terms + codes 
            "true_LLT_term": true_LLT_term,
            "pred_LLT_term": pred_LLT_term_std,
            "true_LLT_Code": true_LLT_Code,
            "pred_LLT_Code": pred_LLT_Code,
            "true_PT_term": true_PT_term,
            "pred_PT_term": pred_PT_term,
            "true_PT_Code": true_PT_Code,
            "pred_PT_Code": pred_PT_Code,

            # SOC (AE raw vs predicted primary)
            "true_SOC_Code": true_SOC_Code_AE,
            "pred_SOC_Code": pred_SOC_Code,
            "true_SOC_Term": soc_code_to_term.get(true_SOC_Code_AE) if true_SOC_Code_AE else None,
            "pred_SOC_Term": soc_code_to_term.get(pred_SOC_Code) if pred_SOC_Code else None,

            # optional metrics kept (no SOC extras)
            "exact_LLT_match": exact_LLT_match,
            "LLT_fuzzy_score": LLT_fuzzy_score,
            "LLT_fuzzy_match": LLT_fuzzy_match,
            "exact_PT_match": exact_PT_match,
            "PT_fuzzy_score": PT_fuzzy_score,
            "PT_fuzzy_match": PT_fuzzy_match,

            "model_output": answer
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True LLT/PT: {true_LLT_term}  |  {true_LLT_Code} / {true_PT_Code} ({true_PT_term})")
        print(f"→ Pred LLT/PT: {answer_line} |  {pred_LLT_Code} / {pred_PT_Code} ({pred_PT_term})")
        if true_SOC_Code_AE or pred_SOC_Code:
            print(f"→ SOC (AE vs pred-primary): {true_SOC_Code_AE} vs {pred_SOC_Code}")
        print(f"→ Exact LLT: {exact_LLT_match}, Fuzzy LLT: {LLT_fuzzy_score:.1f} | Exact PT: {exact_PT_match}, Fuzzy PT: {PT_fuzzy_score:.1f}\n")
        time.sleep(0.1)

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        results.append({
            "AE_text": ae_text,
            "true_LLT_term": true_LLT_term,
            "pred_LLT_term": None,
            "true_LLT_Code": true_LLT_Code,
            "pred_LLT_Code": None,
            "true_PT_term": true_PT_term,
            "pred_PT_term": None,
            "true_PT_Code": true_PT_Code,
            "pred_PT_Code": None,
            "true_SOC_Code": true_SOC_Code_AE,
            "pred_SOC_Code": None,
            "true_SOC_Term": soc_code_to_term.get(true_SOC_Code_AE) if true_SOC_Code_AE else None,
            "pred_SOC_Term": None,
            "exact_LLT_match": False,
            "LLT_fuzzy_score": 0.0,
            "LLT_fuzzy_match": False,
            "exact_PT_match": False,
            "PT_fuzzy_score": 0.0,
            "PT_fuzzy_match": False,
            "model_output": None
        })

# =========================
# Save Results
# =========================
out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

out_csv = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_enriched.csv"
pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")

# =========================
# Evaluation (LLT/PT term-based + code-based)
# =========================
y_true = [r["true_LLT_term"] or "" for r in results]
y_pred = [r["pred_LLT_term"] or "" for r in results]
y_pred_fuzzy = [(r["true_LLT_term"] or r["pred_LLT_term"] or "") if r["LLT_fuzzy_match"] else (r["pred_LLT_term"] or "") for r in results]

z_true = [r["true_PT_term"] or "" for r in results]
z_pred = [r["pred_PT_term"] or "" for r in results]
z_pred_fuzzy = [(r["true_PT_term"] or r["pred_PT_term"] or "") if r["PT_fuzzy_match"] else (r["pred_PT_term"] or "") for r in results]

print("Evaluation Report (Exact LLT Match):")
print(classification_report(y_true, y_pred, zero_division=0))
print("\nEvaluation Report (LLT Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

print("Evaluation Report (Exact PT Match):")
print(classification_report(z_true, z_pred, zero_division=0))
print("\nEvaluation Report (PT Fuzzy Match):")
print(classification_report(z_true, z_pred_fuzzy, zero_division=0))

llt_acc = accuracy_score(y_true, y_pred) if any(y_true) else 0.0
llt_f1 = f1_score(y_true, y_pred, average="macro") if any(y_true) else 0.0
llt_precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if any(y_true) else 0.0
llt_recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if any(y_true) else 0.0
llt_fuzzy_acc = (sum(r["LLT_fuzzy_match"] for r in results) / max(1, len(results))) if results else 0.0

pt_acc = accuracy_score(z_true, z_pred) if any(z_true) else 0.0
pt_f1 = f1_score(z_true, z_pred, average="macro") if any(z_true) else 0.0
pt_precision = precision_score(z_true, z_pred, average="macro", zero_division=0) if any(z_true) else 0.0
pt_recall = recall_score(z_true, z_pred, average="macro", zero_division=0) if any(z_true) else 0.0
pt_fuzzy_acc = (sum(r["PT_fuzzy_match"] for r in results) / max(1, len(results))) if results else 0.0

mask_llt = [(r["true_LLT_Code"] is not None) and (r["pred_LLT_Code"] is not None) for r in results]
mask_pt  = [(r["true_PT_Code"]  is not None) and (r["pred_PT_Code"]  is not None) for r in results]

LLT_acc = (sum(1 for i, r in enumerate(results) if mask_llt[i] and r["true_LLT_Code"] == r["pred_LLT_Code"]) / max(1, sum(mask_llt))) if results else 0.0
PT_acc  = (sum(1 for i, r in enumerate(results) if mask_pt[i]  and r["true_PT_Code"]  == r["pred_PT_Code"])  / max(1, sum(mask_pt)))  if results else 0.0

print(f"\n LLT Term Accuracy (exact): {llt_acc:.2f}")
print(f" PT Term Accuracy (exact):  {pt_acc:.2f}")
print(f"LLT F1 (macro): {llt_f1:.2f} | LLT Precision: {llt_precision:.2f} | LLT Recall: {llt_recall:.2f}")
print(f"PT F1 (macro):  {pt_f1:.2f} | PT Precision:  {pt_precision:.2f} | PT Recall:  {pt_recall:.2f}")
print(f"Fuzzy LLT Term Accuracy: {llt_fuzzy_acc:.2f}")
print(f"Fuzzy PT  Term Accuracy: {pt_fuzzy_acc:.2f}")
print(f"LLT Accuracy (code): {LLT_acc:.2f}  [on {sum(mask_llt)} rows]")
print(f"PT  Accuracy (code): {PT_acc:.2f}   [on {sum(mask_pt)} rows]")

# =========================
# Single SOC Accuracy (AE SOC vs predicted primary SOC)
# =========================
pairs = [(r["true_SOC_Code"], r["pred_SOC_Code"]) for r in results if r.get("true_SOC_Code") is not None]
total = len(pairs)
correct = sum(1 for a, b in pairs if (b is not None and str(a) == str(b)))
soc_acc = (correct / total) if total else 0.0
print(f"SOC Accuracy: {soc_acc:.4f} (over {total} rows)")

print("\nSaved:")
print("-", out_json)
print("-", out_csv)
