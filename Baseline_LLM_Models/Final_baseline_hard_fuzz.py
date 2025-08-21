"""
=== File Name: Final_baseline_hard_fuzz.py     === Author: Naghme Dashti / August 2025
(PT & SOC accuracies + PT/SOC term fields; handles Ist_Primary_SOC = Y/N with robust fallback + QC prints)
"""

import pandas as pd
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from fuzzywuzzy import fuzz

# ---------------------------
# Paths (adjust if needed)
# ---------------------------
PT_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"     # expects PT_Code,SOC_Code; PT_Term,SOC_Term optional; Ist_Primary_SOC=Y/N optional
LLT_CSV_PATH = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
AE_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
OUT_JSON     = "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Mosaic_output_fuzz.json"

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# ---------------------------
# Load AE (keep SOC ground truth if exists)
# ---------------------------
ae_df = pd.read_csv(AE_CSV_PATH, sep=';', encoding='latin1')
ae_cols = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_df.columns]
ae_df = ae_df[ae_cols].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

# ---------------------------
# Load LLT (keep PT_Code / PT_Term if present — algorithm unchanged)
# ---------------------------
llt_df = pd.read_csv(LLT_CSV_PATH, sep=';', encoding='latin1')
keep_llt_cols = [c for c in ["LLT_Code", "LLT_Term", "PT_Code", "PT_Term"] if c in llt_df.columns]
llt_df = llt_df[keep_llt_cols].dropna(subset=["LLT_Code", "LLT_Term"]).reset_index(drop=True)
llt_df["LLT_Code"] = llt_df["LLT_Code"].astype(int)

# LLT maps
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))  # (string keys)
llt_term_to_code = {}
for _, r in llt_df.iterrows():
    key = r["LLT_Term"].strip().lower()
    if key not in llt_term_to_code:  # keep first if duplicates
        llt_term_to_code[key] = int(r["LLT_Code"])

has_pt_in_llt = "PT_Code" in llt_df.columns
llt_code_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"])) if has_pt_in_llt else {}

# Optional: PT_Term from LLT file (fallback)
pt_code_to_term_from_llt = {}
if "PT_Term" in llt_df.columns and "PT_Code" in llt_df.columns:
    tmp = llt_df.dropna(subset=["PT_Code", "PT_Term"]).drop_duplicates(subset=["PT_Code"]).copy()
    if not tmp.empty:
        tmp["PT_Code"] = tmp["PT_Code"].astype(int)
        pt_code_to_term_from_llt = dict(zip(tmp["PT_Code"], tmp["PT_Term"]))

# ---------------------------
# Load PT->SOC mapping (+ optional PT_Term/SOC_Term + Ist_Primary_SOC handling)
# ---------------------------
pt_code_to_term = {}
soc_code_to_term = {}
pt_code_to_primary_soc = {}  # primary SOC per PT (or None if undefined)
pt_code_to_soc_all = {}      # all SOCs per PT

try:
    pt_df = pd.read_csv(PT_CSV_PATH, sep=';', encoding='latin1')
    if not {"PT_Code", "SOC_Code"}.issubset(set(pt_df.columns)):
        raise ValueError("PT/SOC file must contain PT_Code and SOC_Code columns.")

    keep_pt_cols = [c for c in ["PT_Code", "SOC_Code", "PT_Term", "SOC_Term", "Ist_Primary_SOC"] if c in pt_df.columns]
    pt_df = pt_df[keep_pt_cols].dropna(subset=["PT_Code", "SOC_Code"]).reset_index(drop=True)
    pt_df["PT_Code"] = pt_df["PT_Code"].astype(int)
    pt_df["SOC_Code"] = pt_df["SOC_Code"].astype(int)

    # Terms if present
    if "PT_Term" in pt_df.columns:
        pt_code_to_term = dict(zip(pt_df["PT_Code"], pt_df["PT_Term"]))
    if "SOC_Term" in pt_df.columns:
        soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"]))

    # All SOCs per PT
    pt_code_to_soc_all = (
        pt_df.groupby("PT_Code")["SOC_Code"]
        .apply(lambda s: sorted(set(s.astype(int))))
        .to_dict()
    )

    # Primary SOC per PT using Ist_Primary_SOC == 'Y', robust fallback:
    # - If PT has exactly one SOC -> that one is primary.
    # - If PT has multiple SOCs but no 'Y' -> primary is undefined (None).
    pt_code_to_primary_soc = {}
    if "Ist_Primary_SOC" in pt_df.columns:
        prim_mask = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper().eq("Y")
        prim_rows = pt_df[prim_mask].drop_duplicates(subset=["PT_Code"])
        pt_code_to_primary_soc.update(
            dict(zip(prim_rows["PT_Code"].astype(int), prim_rows["SOC_Code"].astype(int)))
        )

    for ptc, soc_list in pt_code_to_soc_all.items():
        if ptc not in pt_code_to_primary_soc:
            if len(soc_list) == 1:
                pt_code_to_primary_soc[ptc] = soc_list[0]
            else:
                pt_code_to_primary_soc[ptc] = None

    # QC summary
    num_pts = len(pt_code_to_soc_all)
    num_single = sum(1 for _, lst in pt_code_to_soc_all.items() if len(lst) == 1)
    num_undefined = sum(1 for ptc, lst in pt_code_to_soc_all.items() if len(lst) > 1 and (pt_code_to_primary_soc.get(ptc) is None))
    num_with_y = 0
    if "Ist_Primary_SOC" in pt_df.columns:
        y_rows = pt_df[pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper().eq("Y")]
        num_with_y = y_rows["PT_Code"].nunique()
    print(f"[PT/SOC QC] PTs:{num_pts} | with 'Y':{num_with_y} | single-SOC:{num_single} | multi-SOC-noY(primary undefined):{num_undefined}")

except Exception as e:
    print(f"WARNING: Could not load PT/SOC mapping ({e}). PT/SOC metrics/terms may be limited.")

# If PT_Term not present in PT file, fall back to LLT-provided PT_Term (if any)
if not pt_code_to_term and pt_code_to_term_from_llt:
    pt_code_to_term = pt_code_to_term_from_llt

# ---------------------------
# Params
# ---------------------------
N_CANDIDATES = 100  # realistic set size (more difficult task)
MAX_ROWS = 20       # Number of AE samples for demo
SLEEP_SEC = 1.5
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

# ---------------------------
# Main loop (algorithm unchanged)
# ---------------------------
for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_llt_code_int = int(row["ZB_LLT_Code"])
    true_llt_term = llt_code_to_term.get(str(true_llt_code_int))
    if true_llt_term is None:
        continue

    # SOC GT from AE (if present)
    true_soc_code_ae = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_soc_code_ae = int(v)

    # Select LLTs excluding the correct one (UNCHANGED baseline behavior)
    candidate_pool = llt_df[llt_df["LLT_Code"] != true_llt_code_int]
    sampled_df = candidate_pool.sample(min(N_CANDIDATES, len(candidate_pool)), random_state=idx)[["LLT_Term", "LLT_Code"]]
    sampled_terms = sampled_df["LLT_Term"].tolist()
    random.shuffle(sampled_terms)

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
        full_answer = response.choices[0].message.content.strip()
        pred_llt_term = full_answer.strip().split("\n")[-1].strip()

        # Term-level eval
        exact_match = (pred_llt_term == true_llt_term)
        fuzzy_score = fuzz.ratio(pred_llt_term.lower(), true_llt_term.lower())
        fuzzy_match = (fuzzy_score >= 90)

        # map predicted term -> LLT_Code (best effort)
        pred_llt_code = map_pred_term_to_llt_code(pred_llt_term, sampled_terms)

        # derive PT/SOC codes if mappings are available
        true_pt_code = llt_code_to_pt.get(true_llt_code_int) if has_pt_in_llt else None
        pred_pt_code = llt_code_to_pt.get(pred_llt_code) if (has_pt_in_llt and pred_llt_code is not None) else None

        # Primary SOC via PT map:
        true_soc_code_primary = pt_code_to_primary_soc.get(true_pt_code) if true_pt_code is not None else None
        pred_soc_code = pt_code_to_primary_soc.get(pred_pt_code) if pred_pt_code is not None else None

        # If AE SOC not provided, fill from mapping primary (optional to align)
        if true_soc_code_ae is None:
            true_soc_code_ae = true_soc_code_primary

        # All SOCs for reporting/any-of metric
        true_soc_codes_all = pt_code_to_soc_all.get(true_pt_code, []) if true_pt_code is not None else []
        pred_soc_codes_all = pt_code_to_soc_all.get(pred_pt_code, []) if pred_pt_code is not None else []

        # Resolve PT/SOC terms if available
        true_pt_term = pt_code_to_term.get(true_pt_code) if true_pt_code is not None else None
        pred_pt_term = pt_code_to_term.get(pred_pt_code) if pred_pt_code is not None else None
        true_soc_term = soc_code_to_term.get(true_soc_code_ae) if true_soc_code_ae is not None else None
        pred_soc_term = soc_code_to_term.get(pred_soc_code) if pred_soc_code is not None else None
        true_soc_term_primary = soc_code_to_term.get(true_soc_code_primary) if true_soc_code_primary is not None else None

        # Diagnostics flags for missing primary
        pred_primary_soc_missing = (pred_pt_code is not None and pt_code_to_primary_soc.get(pred_pt_code) is None)
        true_primary_soc_missing = (true_pt_code is not None and pt_code_to_primary_soc.get(true_pt_code) is None) if true_pt_code is not None else None

        # Save row (no duplicate: only *_llt_term)
        results.append({
            "AE_text": ae_text,

            "true_llt_term": true_llt_term,
            "pred_llt_term": pred_llt_term,

            "exact_match": bool(exact_match),
            "fuzzy_score": float(fuzzy_score),
            "fuzzy_match": bool(fuzzy_match),

            # codes (for accuracies)
            "true_llt_code": int(true_llt_code_int),
            "pred_llt_code": pred_llt_code,
            "true_pt_code":  int(true_pt_code) if true_pt_code is not None else None,
            "pred_pt_code":  int(pred_pt_code) if pred_pt_code is not None else None,

            # SOCs (AE vs mapping primary)
            "true_soc_code": true_soc_code_ae,               # AE ground truth if present, else mapped primary
            "pred_soc_code": pred_soc_code,                  # predicted primary from mapping
            "true_soc_code_primary": true_soc_code_primary,  # mapping primary for the true PT

            # PT/SOC terms
            "true_pt_term":  true_pt_term,
            "pred_pt_term":  pred_pt_term,
            "true_soc_term": true_soc_term,                  # AE SOC term (or mapped primary if AE missing)
            "pred_soc_term": pred_soc_term,
            "true_soc_term_primary": true_soc_term_primary,

            # all SOCs (for any-of metric)
            "true_soc_codes_all": true_soc_codes_all,
            "pred_soc_codes_all": pred_soc_codes_all,

            # diagnostics
            "pred_primary_soc_missing": pred_primary_soc_missing,
            "true_primary_soc_missing": true_primary_soc_missing,

            # raw model output (optional)
            "model_output": full_answer
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True LLT: {true_llt_term}")
        print(f"→ Pred LLT: {pred_llt_term}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score:.1f} ({'✓' if fuzzy_match else '✗'})\n")

        time.sleep(SLEEP_SEC)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# ---------------------------
# Save predictions (JSON includes PT/SOC codes & terms + all SOCs)
# ---------------------------
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ---------------------------
# Term-level metrics (using *_llt_term only)
# ---------------------------
y_true = [r["true_llt_term"] for r in results]
y_pred = [r["pred_llt_term"] for r in results]
y_pred_fuzzy = [r["true_llt_term"] if r["fuzzy_match"] else r["pred_llt_term"] for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

acc = accuracy_score(y_true, y_pred) if y_true else 0.0
f1  = f1_score(y_true, y_pred, average="macro") if y_true else 0.0

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro):    {recall:.4f}")

# Fuzzy match accuracy
fuzzy_accuracy = (sum(r["fuzzy_match"] for r in results) / len(results)) if results else 0.0
print(f"Fuzzy Match Accuracy: {fuzzy_accuracy:.4f}")

# =========================
# PT & SOC Accuracies (code-based)
# =========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]

# PT accuracy (PT(true LLT) vs PT(pred LLT))
if has_pt_in_llt:
    pt_pairs = _both_present([(r["true_pt_code"], r["pred_pt_code"]) for r in results])
    if pt_pairs:
        pt_acc = sum(int(a == b) for a, b in pt_pairs) / len(pt_pairs)
        print(f"\nPT Accuracy (code): {pt_acc:.4f}  (over {len(pt_pairs)} rows)")
    else:
        print("\nPT Accuracy (code): N/A (no comparable rows)")
else:
    print("\nPT Accuracy (code): N/A (LLT file has no PT_Code column)")

# SOC accuracy (primary vs AE ground-truth)
soc_pairs_ae = _both_present([(r.get("true_soc_code"), r.get("pred_soc_code")) for r in results])
if soc_pairs_ae:
    soc_acc_vs_ae = sum(int(a == b) for a, b in soc_pairs_ae) / len(soc_pairs_ae)
    print(f"SOC Accuracy (primary vs AE): {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
else:
    print("SOC Accuracy (primary vs AE): N/A")

# SOC accuracy (primary vs true primary mapping)
soc_pairs_primary = _both_present([(r.get("true_soc_code_primary"), r.get("pred_soc_code")) for r in results])
if soc_pairs_primary:
    soc_acc_vs_true_primary = sum(int(a == b) for a, b in soc_pairs_primary) / len(soc_pairs_primary)
    print(f"SOC Accuracy (primary vs true primary): {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
else:
    print("SOC Accuracy (primary vs true primary): N/A")

# SOC accuracy (any-of vs AE): true AE SOC is within set of all SOCs of predicted PT
soc_any = []
for r in results:
    ae_soc = r.get("true_soc_code")
    pred_all = r.get("pred_soc_codes_all") or []
    if ae_soc is not None and pred_all:
        soc_any.append(int(ae_soc in pred_all))
if soc_any:
    soc_acc_any = sum(soc_any) / len(soc_any)
    print(f"SOC Accuracy (any-of vs AE):  {soc_acc_any:.4f} (over {len(soc_any)} rows)")
else:
    print("SOC Accuracy (any-of vs AE): N/A")
