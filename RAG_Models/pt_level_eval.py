"""
File Name: rag_prompting_reasoning_v3.py    === Author: Naghme Dashti / July 2025

RAG-based Prompting with Explicit Reasoning + Final Answer Line
---------------------------------------------------------------
This script now ALSO maps predicted LLT -> PT and reports BOTH LLT-level and PT-level accuracy,
without skipping rows. If embeddings are missing, it uses a safe fallback candidate list.
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
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",  
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
DATASET_EMB_NAME = "ae_embeddings_Delta"

# Dictionaries and output
LLT_DICTIONARY_NAME      = "LLT2_Code_English_25_0"   # Include LLT_Code, LLT_Term, PT_Code
LLT_DICTIONARY_EMB_NAME  = "llt2_embeddings"
PT_DICTIONARY_NAME       = "PT2_SOC_25_0"
OUTPUT_FILE_NAME         = "Mosaic_output_v3_PT"       # output file name

# Paths
AE_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{DATASET_NAME}.csv"
AE_EMB_FILE  = f"/home/naghmedashti/MedDRA-LLM/embedding/{DATASET_EMB_NAME}.json"
LLT_CSV_FILE = f"/home/naghmedashti/MedDRA-LLM/data/{LLT_DICTIONARY_NAME}.csv"
LLT_EMB_FILE = f"/home/naghmedashti/MedDRA-LLM/embedding/{LLT_DICTIONARY_EMB_NAME}.json"
PT_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{PT_DICTIONARY_NAME}.csv"

LLM_API_NAME = "llama-3.3-70b-instruct-awq" #  Llama-3.3-70B-Instruct
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
# Load AE / LLT / PT
# =========================
ae_df  = pd.read_csv(AE_CSV_FILE, sep=";", encoding="latin1")[["Original_Term_aufbereitet", "ZB_LLT_Code"]]
llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]]
pt_df  = pd.read_csv(PT_CSV_FILE,  sep=";", encoding="latin1")[["PT_Code", "PT_Term", "SOC_Code", "SOC_Term"]]

# Canonicalize & lookups
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

pt_df["PT_Code"] = pt_df["PT_Code"].map(canon_code)

llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_to_pt        = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))
pt_meta = pt_df.set_index("PT_Code")[["PT_Term","SOC_Code","SOC_Term"]].to_dict(orient="index")

# Term -> LLT_Code 
term_norm_to_llt = {}
for _, r in llt_df.iterrows():
    term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"]) #LLT_norm is the normal version of LLT_Term

# Mapping Pred_LLT_Term -> LLT_Code 
def term_to_llt_code(pred_term: str, allow_fuzzy=True) -> str | None:
    """Exact normalized, then piece-wise, then optional fuzzy (conservative)."""
    t = norm_text(pred_term) # text normalization
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
 
# ===== Convert embedding lists to numpy arrays =====

# AE embs keyed by AE text;
# In addition to the primary key, the normalized version of the AE and LLT text is also added to the dictionary
ae_emb_dict = {}
for k, v in ae_emb_raw.items():
    ae_emb_dict[k] = np.array(v)
    ae_emb_dict[norm_text(k)] = np.array(v)   # Normalized key

# LLT embs keyed by LLT term (string)
llt_emb_dict = {k: np.array(v) for k, v in llt_emb_raw.items()}
llt_terms_all = list(llt_df["LLT_Term"])

# Optional row limit
if isinstance(MAX_ROWS, int) and MAX_ROWS > 0:
    ae_df = ae_df.dropna(subset=["Original_Term_aufbereitet"]).iloc[:MAX_ROWS].reset_index(drop=True)
else:
    ae_df = ae_df.dropna(subset=["Original_Term_aufbereitet"]).reset_index(drop=True)

# =========================
# RAG + prompting
# =========================
results = []
random.seed(42)

for idx, row in ae_df.iterrows():
    ae_text = str(row["Original_Term_aufbereitet"])
    true_LLT_Code = canon_code(row["ZB_LLT_Code"])
    true_term = llt_code_to_term.get(true_LLT_Code)           # It may be None but the row will not be deleted
    true_PT_Code = llt_to_pt.get(true_LLT_Code)
    true_PT_Term = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None

    # ---  Try to get AE embedding; if missing, build a fallback candidate list (no continue) ---
    ae_emb = ae_emb_dict.get(ae_text, ae_emb_dict.get(norm_text(ae_text)))
    if ae_emb is not None:
        # manual cosine similarity 
        ae_norm = np.linalg.norm(ae_emb)
        sims = []
        for llt_term, llt_emb in llt_emb_dict.items():
            denom = ae_norm * np.linalg.norm(llt_emb)
            score = float(np.dot(ae_emb, llt_emb) / denom) if denom else 0.0
            sims.append((llt_term, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        candidate_terms = [t for t, _ in sims[:TOP_K]]
    else:
        # Fallback: closest with difflib based on LLT name; if insufficient, fill with random
        cand = get_close_matches(ae_text, llt_terms_all, n=TOP_K, cutoff=0.0)
        if len(cand) < TOP_K:
            extra = random.sample(llt_terms_all, k=min(TOP_K - len(cand), len(llt_terms_all)))
            cand += extra
        candidate_terms = cand[:TOP_K]
    
    # include true_term so the model has a chance of making the correct choice
    if true_term and true_term not in candidate_terms:
        candidate_terms.append(true_term)
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
        response = client.chat.completions.create(
            model=LLM_API_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            ttemperature=LLM_TEMP,
            max_tokens=LLM_TOKEN
        )

        answer = response.choices[0].message.content.strip()
        if "Final answer:" in answer:
            answer_line = answer.split("Final answer:")[-1].strip()
        else:
            answer_line = answer.strip().split("\n")[-1].strip()

        # Map predicted term -> LLT_Code -> PT_Code
        pred_LLT_Code = term_to_llt_code(answer_line, allow_fuzzy=True)
        pred_PT_Code  = llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None
        pred_PT_Term  = pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None

        # Old term metrics 
        exact_match = (true_term is not None and answer_line == true_term)
        fuzzy_score = fuzz.ratio(answer_line.lower(), (true_term or "").lower())
        fuzzy_match = (true_term is not None and fuzzy_score >= 90)

        # New code-based metrics 
        llt_correct = (pred_LLT_Code is not None and true_LLT_Code is not None and pred_LLT_Code == true_LLT_Code)
        pt_correct  = (pred_PT_Code  is not None and true_PT_Code  is not None and pred_PT_Code  == true_PT_Code)

        results.append({
            "AE_text": ae_text,
            "true_LLT_term": true_term,
            "pred_LLT_Term": answer_line,
            # NEW: codes + PT mapping
            "true_PT_Term": true_PT_Term,
            "pred_PT_Term": pred_PT_Term,
            "LLT_correct": llt_correct,
            "PT_correct": pt_correct,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match
            #"true_LLT_Code": true_LLT_Code,
            #"true_PT_Code": true_PT_Code,
            #"pred_LLT_Code": pred_LLT_Code,
            #"pred_PT_Code": pred_PT_Code
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True LLT/PT: {true_term}  |  {true_LLT_Code} / {true_PT_Code} ({true_PT_Term})")
        print(f"→ Pred LLT/PT: {answer_line} |  {pred_LLT_Code} / {pred_PT_Code} ({pred_PT_Term})")
        print(f"→ LLT_correct: {llt_correct} | PT_correct: {pt_correct} | Exact: {exact_match}, Fuzzy: {fuzzy_score:.1f}\n")
        time.sleep(0.1)

    except Exception as e:
        # We don't lose the row even in case of an error; we just report it
        print(f"Error at index {idx}: {e}")
        results.append({
            "AE_text": ae_text,
            "true_LLT_term": true_term,
            "pred_LLT_Term": None,
            "true_PT_Term": true_PT_Term,
            "pred_PT_Term": None,
            "LLT_correct": False,
            "PT_correct": False,
            "exact_match": False,
            "fuzzy_score": 0.0,
            "fuzzy_match": False
            #"true_LLT_Code": true_LLT_Code,
            #"true_PT_Code": true_PT_Code,
            #"pred_LLT_Code": None,
            #"pred_PT_Code": None
        })

# =========================
# Save Results
# =========================
out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}.json"
with open(out_json, "w", encoding="latin1") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

out_csv = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_enriched.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)

# =========================
# Evaluation
# =========================
# Term-based (legacy)
y_true = [r["true_LLT_term"] or "" for r in results]
y_pred = [r["pred_LLT_Term"] or "" for r in results]
y_pred_fuzzy = [ (r["true_LLT_term"] or r["pred_LLT_Term"] or "") if r["fuzzy_match"] else (r["pred_LLT_Term"] or "") for r in results ]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))
print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

acc = accuracy_score(y_true, y_pred) if any(y_true) else 0.0
f1 = f1_score(y_true, y_pred, average="macro") if any(y_true) else 0.0
precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if any(y_true) else 0.0
recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if any(y_true) else 0.0
fuzzy_acc = (sum(r["fuzzy_match"] for r in results) / max(1, len(results))) if results else 0.0

# Code-based: Only on rows that have both required codes
mask_llt = [ (r["true_LLT_Code"] is not None) and (r["pred_LLT_Code"] is not None) for r in results ]
mask_pt  = [ (r["true_PT_Code"]  is not None) and (r["pred_PT_Code"]  is not None) for r in results ]

LLT_acc = (sum(1 for i,r in enumerate(results) if mask_llt[i] and r["true_LLT_Code"]==r["pred_LLT_Code"]) / max(1, sum(mask_llt))) if results else 0.0
PT_acc  = (sum(1 for i,r in enumerate(results) if mask_pt[i]  and r["true_PT_Code"] ==r["pred_PT_Code"])  / max(1, sum(mask_pt)))  if results else 0.0

print(f"\nTerm Accuracy (exact): {acc:.2f}")
print(f"F1 (macro): {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")
print(f"Fuzzy Term Accuracy: {fuzzy_acc:.2f}")
print(f"LLT Accuracy (code): {LLT_acc:.2f}  [on {sum(mask_llt)} rows]")
print(f"PT  Accuracy (code): {PT_acc:.2f}   [on {sum(mask_pt)} rows]")

print("\nSaved:")
print("-", out_json)
print("-", out_csv)
