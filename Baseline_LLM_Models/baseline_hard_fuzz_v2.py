"""
=== File Name: baseline_hard_fuzz.py     === Author: Naghme Dashti / August 2025
(PT & SOC accuracy added + PT/SOC terms added to JSON)
"""

import pandas as pd
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from fuzzywuzzy import fuzz

# ---------------------------
# Config for PT/SOC mapping
# ---------------------------
PT_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"     # must have PT_Code,SOC_Code; PT_Term,SOC_Term optional
LLT_CSV_PATH = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
AE_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
OUT_JSON     = "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/baseline_hard_fuzz_v2.json"

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# Load AE data (keep ZB_SOC_Code if available)
ae_df = pd.read_csv(AE_CSV_PATH, sep=';', encoding='latin1')
ae_cols = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_df.columns]
ae_df = ae_df[ae_cols].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

# Load LLT dictionary (try to keep PT_Code if present)
llt_df = pd.read_csv(LLT_CSV_PATH, sep=';', encoding='latin1')
keep_llt_cols = [c for c in ["LLT_Code", "LLT_Term", "PT_Code", "PT_Term"] if c in llt_df.columns]
llt_df = llt_df[keep_llt_cols].dropna(subset=["LLT_Code", "LLT_Term"]).reset_index(drop=True)

# Build dictionaries
llt_df["LLT_Code"] = llt_df["LLT_Code"].astype(int)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))  # (string keys) for your existing logic

# term -> code (lowercased) for mapping predicted term back to code
llt_term_to_code = {}
for _, r in llt_df.iterrows():
    key = r["LLT_Term"].strip().lower()
    if key not in llt_term_to_code:  # keep first if duplicates
        llt_term_to_code[key] = int(r["LLT_Code"])

# Optional: LLT -> PT mapping if available
has_pt_in_llt = "PT_Code" in llt_df.columns
llt_code_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"])) if has_pt_in_llt else {}

# Optional: PT_Term from LLT file (fallback)
pt_code_to_term_from_llt = {}
if "PT_Term" in llt_df.columns and "PT_Code" in llt_df.columns:
    tmp = llt_df.dropna(subset=["PT_Code", "PT_Term"]).drop_duplicates(subset=["PT_Code"])
    if not tmp.empty:
        tmp["PT_Code"] = tmp["PT_Code"].astype(int)
        pt_code_to_term_from_llt = dict(zip(tmp["PT_Code"], tmp["PT_Term"]))

# Load PT->SOC mapping (+ optional PT_Term/SOC_Term)
pt_code_to_soc = {}
pt_code_to_term = {}
soc_code_to_term = {}
try:
    pt_df = pd.read_csv(PT_CSV_PATH, sep=';', encoding='latin1')
    if not {"PT_Code", "SOC_Code"}.issubset(set(pt_df.columns)):
        raise ValueError("PT/SOC file must contain PT_Code and SOC_Code columns.")
    keep_pt_cols = [c for c in ["PT_Code", "SOC_Code", "PT_Term", "SOC_Term"] if c in pt_df.columns]
    pt_df = pt_df[keep_pt_cols].dropna(subset=["PT_Code", "SOC_Code"]).reset_index(drop=True)
    pt_df["PT_Code"] = pt_df["PT_Code"].astype(int)
    pt_df["SOC_Code"] = pt_df["SOC_Code"].astype(int)
    pt_code_to_soc = dict(zip(pt_df["PT_Code"], pt_df["SOC_Code"]))
    if "PT_Term" in pt_df.columns:
        pt_code_to_term = dict(zip(pt_df["PT_Code"], pt_df["PT_Term"]))
    if "SOC_Term" in pt_df.columns:
        soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"]))
except Exception as e:
    print(f"WARNING: Could not load PT/SOC mapping ({e}). PT/SOC term fields may be None.")

# If PT_Term not present in PT file, fall back to LLT-provided PT_Term (if any)
if not pt_code_to_term and pt_code_to_term_from_llt:
    pt_code_to_term = pt_code_to_term_from_llt

# Parameters
N_CANDIDATES = 100  # realistic set size (more difficult task)
MAX_ROWS = 20       # Number of AE samples for demo
results = []


def map_pred_term_to_llt_code(pred_term, sampled_terms):
    """
    Map model's predicted term to an LLT code.
    1) exact match by lowercased term via global LLT map
    2) fuzzy to sampled terms (backup), then map that term globally
    3) if still not found, return None
    """
    if not isinstance(pred_term, str) or not pred_term.strip():
        return None
    t = pred_term.strip().strip('"').strip("'")
    key = t.lower()

    # exact via global
    if key in llt_term_to_code:
        return llt_term_to_code[key]

    # fallback: fuzzy to sampled terms (the offered candidate list)
    if sampled_terms:
        best = None
        best_score = -1
        for st in sampled_terms:
            s = fuzz.ratio(key, st.lower())
            if s > best_score:
                best, best_score = st, s
        if best is not None and best_score >= 70:
            return llt_term_to_code.get(best.lower(), None)

    return None


# Loop over a small subset of AEs (adjust as needed)
for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    true_llt_code_int = int(row["ZB_LLT_Code"])

    # robust read of SOC gt (if present)
    true_soc_code = None
    if "ZB_SOC_Code" in ae_df.columns:
        val = row.get("ZB_SOC_Code")
        if pd.notna(val):
            true_soc_code = int(val)

    if true_code not in llt_code_to_term:
        continue

    true_term = llt_code_to_term[true_code]

    # Select LLTs excluding the correct one (UNCHANGED by your baseline)
    candidate_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    sampled_df = candidate_pool.sample(min(N_CANDIDATES, len(candidate_pool)), random_state=idx)[["LLT_Term", "LLT_Code"]]
    sampled_terms = sampled_df["LLT_Term"].tolist()
    random.shuffle(sampled_terms)

    # Build prompt (UNCHANGED)
    prompt = (
        f"You are a medical coding assistant. Given the following adverse event description:\n"
        f"\"{ae_text}\"\n"
        f"Choose the most appropriate MedDRA LLT term from the list below:\n\n" +
        "\n".join(f"- {term}" for term in sampled_terms) +
        "\n\nRespond only with the exact chosen term."
    )

    try:
        response = client.chat.completions.create(
            #model="Llama-3.3-70B-Instruct",
            model="llama-3.3-70b-instruct-awq", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()

        # Evaluation (term-level, UNCHANGED)
        exact_match = answer == true_term
        fuzzy_score = fuzz.ratio(answer.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90

        # map predicted term -> LLT_Code (best effort)
        pred_llt_code = map_pred_term_to_llt_code(answer, sampled_terms)

        # derive PT/SOC codes if mappings are available
        true_pt_code = llt_code_to_pt.get(true_llt_code_int) if has_pt_in_llt else None
        pred_pt_code = llt_code_to_pt.get(pred_llt_code) if (has_pt_in_llt and pred_llt_code is not None) else None

        pred_soc_code = pt_code_to_soc.get(pred_pt_code) if (pred_pt_code is not None) else None
        if true_soc_code is None and (true_pt_code is not None):
            true_soc_code = pt_code_to_soc.get(true_pt_code, None)

        # --- NEW: resolve PT/SOC terms (if available)
        true_pt_term = pt_code_to_term.get(true_pt_code) if true_pt_code is not None else None
        pred_pt_term = pt_code_to_term.get(pred_pt_code) if pred_pt_code is not None else None
        true_soc_term = soc_code_to_term.get(true_soc_code) if true_soc_code is not None else None
        pred_soc_term = soc_code_to_term.get(pred_soc_code) if pred_soc_code is not None else None

        # Save row (keep old keys + add new PT/SOC term fields)
        results.append({
            "AE_text": ae_text,
            "true_term": true_term,
            "predicted": answer,

            # also explicit LLT terms (your newer naming)
            "true_llt_term": true_term,
            "pred_llt_term": answer,

            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match,

            # codes (for accuracies)
            "true_llt_code": true_llt_code_int,
            "pred_llt_code": pred_llt_code,
            "true_pt_code": true_pt_code,
            "pred_pt_code": pred_pt_code,
            "true_soc_code": true_soc_code,
            "pred_soc_code": pred_soc_code,

            # ---- NEW: requested term fields ----
            "true_pt_term": true_pt_term,
            "pred_pt_term": pred_pt_term,
            "true_soc_term": true_soc_term,
            "pred_soc_term": pred_soc_term,
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True: {true_term}")
        print(f"→ Predicted: {answer}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")

        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Save predictions (JSON with the new PT/SOC term fields included)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ----- Metrics (unchanged) -----
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

acc = accuracy_score(y_true, y_pred) if y_true else 0.0
f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0

print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")

# Calculate Fuzzy Match Accuracy (custom metric)
fuzzy_accuracy = (sum(r["fuzzy_match"] for r in results) / len(results)) if results else 0.0
print(f"Fuzzy Match Accuracy: {fuzzy_accuracy:.2f}")

# =========================
# PT & SOC Accuracies (code-based) — unchanged behavior
# =========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]

if has_pt_in_llt:
    pt_pairs = _both_present([(r["true_pt_code"], r["pred_pt_code"]) for r in results])
    if pt_pairs:
        pt_acc = sum(int(a == b) for a, b in pt_pairs) / len(pt_pairs)
        print(f"\nPT Accuracy (code): {pt_acc:.4f}  (over {len(pt_pairs)} comparable rows)")
    else:
        print("\nPT Accuracy (code): N/A (no comparable rows)")
else:
    print("\nPT Accuracy (code): N/A (LLT file has no PT_Code column)")

soc_pairs = _both_present([(r["true_soc_code"], r["pred_soc_code"]) for r in results])
if soc_pairs:
    soc_acc = sum(int(a == b) for a, b in soc_pairs) / len(soc_pairs)
    print(f"SOC Accuracy (code): {soc_acc:.4f} (over {len(soc_pairs)} comparable rows)")
else:
    print("SOC Accuracy (code): N/A (no comparable rows)")
