# === File Name: Newrapidfuzz.py     === Author: Naghme Dashti / 26 january 2026
# (PT & SOC accuracies + PT/SOC term fields; handles Ist_Primary_SOC = Y/N + Primary_SOC_Code with robust fallback + QC prints)
# Updated: seed-friendly + no-skip (RAG-like), canonical code handling, per-seed outputs, deterministic sampling

import pandas as pd
import numpy as np
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from rapidfuzz import fuzz, process
import re

# ===========================
# Config (adjust paths if needed)
# ===========================
AE_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv"
LLT_CSV_PATH = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
PT_CSV_PATH  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"  # expects PT_Code, SOC_Code; PT_Term, SOC_Term, Ist_Primary_SOC, Primary_SOC_Code optional
OUT_JSON     = "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Dauno_Newoutput_rapidfuzz.json"

# ===========================
# Seed (set this per run: 42 / 43 / 44)
# ===========================
RUN_SEED = 44
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

# ===========================
# Canonicalize numeric-like codes (RAG-style)
# ===========================
def canon_code(x):
    if x is None:
        return None
    s = str(x).strip()
    m = re.match(r"^(\d+)(?:\.0+)?$", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{3,})", s)
    if m2:
        return m2.group(1)
    return s or None

# OpenAI-compatible API client
client = OpenAI(
    api_key="sk-aKGeEFMZB0gXEcE51FTc0A",
    base_url="http://pluto/v1/"
)

# ===========================
# Load AE (keep SOC ground truth if exists)
# ===========================
ae_df = pd.read_csv(AE_CSV_PATH, sep=';', encoding='latin1')
ae_keep = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_df.columns]
ae_df = ae_df[ae_keep].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

# ===========================
# Load LLT (keep PT_Code / PT_Term if present) — canonical codes
# ===========================
llt_df = pd.read_csv(LLT_CSV_PATH, sep=';', encoding='latin1')
llt_keep = [c for c in ["LLT_Code", "LLT_Term", "PT_Code", "PT_Term"] if c in llt_df.columns]
llt_df = llt_df[llt_keep].dropna(subset=["LLT_Code", "LLT_Term"]).reset_index(drop=True)

# canonicalize
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
if "PT_Code" in llt_df.columns:
    llt_df["PT_Code"] = llt_df["PT_Code"].map(canon_code)

# LLT maps (canonical keys)
llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_term_to_code = {}
for _, r in llt_df.iterrows():
    k = r["LLT_Term"].strip().lower()
    if k not in llt_term_to_code:
        llt_term_to_code[k] = r["LLT_Code"]

has_pt_in_llt = "PT_Code" in llt_df.columns
llt_code_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"])) if has_pt_in_llt else {}

# Optional PT_Term from LLT (fallback)
pt_code_to_term_from_llt = {}
if "PT_Term" in llt_df.columns and "PT_Code" in llt_df.columns:
    tmp = llt_df.dropna(subset=["PT_Code", "PT_Term"]).drop_duplicates(subset=["PT_Code"]).copy()
    if not tmp.empty:
        pt_code_to_term_from_llt = dict(zip(tmp["PT_Code"], tmp["PT_Term"]))

# ===========================
# Load PT/SOC mapping (robust primary resolution) — canonical codes
# ===========================
pt_code_to_term = {}
soc_code_to_term = {}
pt_code_to_primary_soc = {}  # primary SOC per PT (or None if undefined)
pt_code_to_soc_all = {}      # all SOCs per PT

try:
    pt_df = pd.read_csv(PT_CSV_PATH, sep=';', encoding='latin1')
    if not {"PT_Code", "SOC_Code"}.issubset(pt_df.columns):
        raise ValueError("PT/SOC file must contain PT_Code and SOC_Code")

    keep_pt = [c for c in ["PT_Code", "SOC_Code", "PT_Term", "SOC_Term", "Ist_Primary_SOC", "Primary_SOC_Code"] if c in pt_df.columns]
    pt_df = pt_df[keep_pt].dropna(subset=["PT_Code", "SOC_Code"]).reset_index(drop=True)

    # canonicalize codes
    pt_df["PT_Code"]  = pt_df["PT_Code"].map(canon_code)
    pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)
    if "Primary_SOC_Code" in pt_df.columns:
        pt_df["Primary_SOC_Code"] = pt_df["Primary_SOC_Code"].map(canon_code)

    # normalize Y/N flag if present
    if "Ist_Primary_SOC" in pt_df.columns:
        pt_df["Ist_Primary_SOC_norm"] = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper()
    else:
        pt_df["Ist_Primary_SOC_norm"] = ""

    # Terms if present
    if "PT_Term" in pt_df.columns:
        pt_code_to_term = dict(zip(pt_df["PT_Code"], pt_df["PT_Term"]))
    if "SOC_Term" in pt_df.columns:
        soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"]))

    # All SOCs per PT (canonical strings)
    pt_code_to_soc_all = (
        pt_df.groupby("PT_Code")["SOC_Code"]
             .apply(lambda s: sorted(set([canon_code(v) for v in s if pd.notna(v)])))
             .to_dict()
    )

    # Primary resolution priority:
    # 1) any row with Ist_Primary_SOC == 'Y' -> that SOC_Code
    # 2) else, Primary_SOC_Code exists and uniquely defined for that PT -> use it
    # 3) else, if PT has exactly one SOC_Code -> use it
    # 4) else -> None
    has_primary_soc_code_col = "Primary_SOC_Code" in pt_df.columns
    pt_code_to_primary_soc = {}
    for ptc, grp in pt_df.groupby("PT_Code"):
        primary = None

        y_rows = grp[grp["Ist_Primary_SOC_norm"] == "Y"]
        if not y_rows.empty and pd.notna(y_rows.iloc[0].get("SOC_Code")):
            primary = canon_code(y_rows.iloc[0]["SOC_Code"])
        else:
            if has_primary_soc_code_col:
                prim_vals = [canon_code(v) for v in grp["Primary_SOC_Code"].dropna().tolist()]
                uniq = sorted(set([v for v in prim_vals if v is not None]))
                if len(uniq) == 1:
                    primary = uniq[0]
            if primary is None:
                all_socs = pt_code_to_soc_all.get(ptc, [])
                if len(all_socs) == 1:
                    primary = all_socs[0]

        pt_code_to_primary_soc[ptc] = primary

    # QC summary
    num_pts = len(pt_code_to_soc_all)
    num_single = sum(1 for _, lst in pt_code_to_soc_all.items() if len(lst) == 1)
    num_with_y = pt_df[pt_df["Ist_Primary_SOC_norm"] == "Y"]["PT_Code"].nunique()
    num_with_primary_code = 0
    if has_primary_soc_code_col:
        tmp = pt_df.dropna(subset=["Primary_SOC_Code"])
        num_with_primary_code = tmp["PT_Code"].nunique()
    num_undefined = sum(1 for _, v in pt_code_to_primary_soc.items() if v is None)

    print(f"[PT/SOC QC] PTs:{num_pts} | with 'Y':{num_with_y} | with Primary_SOC_Code:{num_with_primary_code} | single-SOC:{num_single} | undefined-primary:{num_undefined}")

except Exception as e:
    print(f"WARNING: Could not load PT/SOC mapping ({e}). PT/SOC metrics/terms may be limited.")

# Fallback PT_Term if PT file lacks it
if not pt_code_to_term and pt_code_to_term_from_llt:
    pt_code_to_term = pt_code_to_term_from_llt

# ===========================
# Params
# ===========================
N_CANDIDATES = 100  # Number of Top-K LLTs shown to the model
MAX_ROWS = None     # Number of AE samples
SLEEP_SEC = 1.0
results = []

def map_pred_term_to_llt_code(pred_term, candidate_terms):
    """Return LLT code from predicted term. Exact via global dict; else fuzzy to candidate_terms then global map."""
    if not isinstance(pred_term, str) or not pred_term.strip():
        return None
    t = pred_llt = pred_term.strip().strip('"').strip("'")
    key = t.lower()

    # exact (global)
    if key in llt_term_to_code:
        return llt_term_to_code[key]

    # fuzzy to the shown candidate list
    if candidate_terms:
        best, best_score = None, -1
        for ct in candidate_terms:
            sc = fuzz.ratio(key, ct.lower())
            if sc > best_score:
                best, best_score = ct, sc
        if best is not None and best_score >= 70:
            return llt_term_to_code.get(best.lower(), None)

    return None

def clean_model_term(s: str) -> str:
    """
    Normalize model output:
    - Remove 'Final answer:' if present
    - Remove bullets/numbering prefixes ('-', '•', '1)', '1.' ...)
    - Strip quotes
    - Collapse spaces
    """
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = re.sub(r"^\s*final\s*answer\s*:\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^\s*(?:[-–—•·*]+|\(?\d+\)?[.)]|[A-Za-z]\)|\d+\s*-\s*)\s*", "", t)
    t = t.strip().strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ===========================
# Main loop — no-skip + canonical codes + deterministic candidate shuffle
# ===========================
for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_llt_code = canon_code(row["ZB_LLT_Code"])
    true_llt_term = llt_code_to_term.get(true_llt_code)  # may be None; DO NOT skip

    # SOC GT from AE (if present) canonical
    true_soc_code_ae = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_soc_code_ae = canon_code(v)

    # Build RapidFuzz candidate list (over all LLT terms)
    candidate_terms_all = llt_df["LLT_Term"].tolist()
    # Note: process.extract is deterministic for same inputs; we still seed shuffle below
    closest = process.extract(ae_text, candidate_terms_all, limit=N_CANDIDATES + 10)
    closest_terms = [term for term, score, _ in closest]
    # remove the exact true term if present (when known)
    #if true_llt_term is not None:
    #    closest_terms = [t for t in closest_terms if t != true_llt_term]
    # trim to N, then append true term (if known) to ensure presence like baseline
    closest_terms = closest_terms[:N_CANDIDATES]
    #if true_llt_term is not None and true_llt_term not in closest_terms:
    #    closest_terms.append(true_llt_term)
    # deterministic shuffle for presentation
    random.shuffle(closest_terms)

    prompt = (
        f"You are a medical coding assistant helping to find the best matching MedDRA LLT term.\n"
        f"Here is an adverse event description:\n\"{ae_text}\"\n"
        f"Below is a list of possible MedDRA LLT terms. Select exactly one term that best fits the description.\n"
        f"Respond only with the exact chosen term, without any extra text.\n\n" +
        "\n".join(f"- {term}" for term in closest_terms) + "\n"
    )

    try:
        response = client.chat.completions.create(
            model="nvidia-llama-3.3-70b-instruct-fp8",
            #model="llama-3.3-70b-instruct-awq",
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        full_answer = response.choices[0].message.content.strip()
        raw_last_line = full_answer.split("\n")[-1].strip()
        pred_llt_term = clean_model_term(raw_last_line)

        # Term-level eval (null-safe)
        exact_match = (pred_llt_term == (true_llt_term or ""))
        fuzzy_score = fuzz.ratio(pred_llt_term.lower(), (true_llt_term or "").lower())
        fuzzy_match = (true_llt_term is not None and fuzzy_score >= 90)

        # Map predicted term -> codes (canonical)
        pred_llt_code = map_pred_term_to_llt_code(pred_llt_term, closest_terms)

        true_pt_code = llt_code_to_pt.get(true_llt_code) if has_pt_in_llt else None
        pred_pt_code = llt_code_to_pt.get(pred_llt_code) if (has_pt_in_llt and pred_llt_code is not None) else None

        # Primary SOCs via PT map:
        true_soc_code_primary = pt_code_to_primary_soc.get(true_pt_code) if true_pt_code is not None else None
        pred_soc_code = pt_code_to_primary_soc.get(pred_pt_code) if pred_pt_code is not None else None

        # If AE SOC not provided, fill from true primary
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

        # Flags: primary undefined?
        pred_primary_soc_missing = (pred_pt_code is not None and pt_code_to_primary_soc.get(pred_pt_code) is None)
        true_primary_soc_missing = (true_pt_code is not None and pt_code_to_primary_soc.get(true_pt_code) is None) if true_pt_code is not None else None

        # Save row (canonical codes)
        results.append({
            "AE_text": ae_text,
            "true_llt_term": true_llt_term,
            "pred_llt_term": pred_llt_term,
            "exact_match": bool(exact_match),
            "fuzzy_score": float(fuzzy_score),
            "fuzzy_match": bool(fuzzy_match),

            # codes (canonical strings)
            "true_llt_code": true_llt_code,
            "pred_llt_code": pred_llt_code,
            "true_pt_code":  true_pt_code,
            "pred_pt_code":  pred_pt_code,

            # SOCs (AE vs mapping primary)
            "true_soc_code": true_soc_code_ae,               # AE if present, else true primary
            "pred_soc_code": pred_soc_code,                  # predicted primary
            "true_soc_code_primary": true_soc_code_primary,  # mapping primary for the true PT

            # terms
            "true_pt_term":  true_pt_term,
            "pred_pt_term":  pred_pt_term,
            "true_soc_term": true_soc_term,
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
        # record anyway (no-skip)
        true_pt_code = llt_code_to_pt.get(true_llt_code) if has_pt_in_llt else None
        true_soc_code_primary = pt_code_to_primary_soc.get(true_pt_code) if true_pt_code is not None else None
        if true_soc_code_ae is None:
            true_soc_code_ae = true_soc_code_primary

        results.append({
            "AE_text": ae_text,
            "true_llt_term": true_llt_term,
            "pred_llt_term": None,
            "exact_match": False,
            "fuzzy_score": 0.0,
            "fuzzy_match": False,

            "true_llt_code": true_llt_code,
            "pred_llt_code": None,
            "true_pt_code":  true_pt_code,
            "pred_pt_code":  None,

            "true_soc_code": true_soc_code_ae,
            "pred_soc_code": None,
            "true_soc_code_primary": true_soc_code_primary,

            "true_pt_term":  pt_code_to_term.get(true_pt_code) if true_pt_code is not None else None,
            "pred_pt_term":  None,
            "true_soc_term": soc_code_to_term.get(true_soc_code_ae) if true_soc_code_ae is not None else None,
            "pred_soc_term": None,
            "true_soc_term_primary": soc_code_to_term.get(true_soc_code_primary) if true_soc_code_primary is not None else None,

            "true_soc_codes_all": pt_code_to_soc_all.get(true_pt_code, []) if true_pt_code is not None else [],
            "pred_soc_codes_all": [],

            "pred_primary_soc_missing": None,
            "true_primary_soc_missing": None,

            "model_output": None
        })

# ===========================
# Save predictions — per-seed file
# ===========================
out_json = OUT_JSON.replace(".json", f"_seed{RUN_SEED}.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved:", out_json)

# ===========================
# Term-level metrics (using *_llt_term only) — null-safe
# ===========================
y_true = [r.get("true_llt_term") or "" for r in results]
y_pred = [r.get("pred_llt_term") or "" for r in results]
y_pred_fuzzy = [(r.get("true_llt_term") or r.get("pred_llt_term") or "") if r.get("fuzzy_match") else (r.get("pred_llt_term") or "") for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

acc = accuracy_score(y_true, y_pred) if y_true else 0.0
f1  = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro):    {recall:.4f}")

# Fuzzy match accuracy
fuzzy_accuracy = (sum(1 for r in results if r.get("fuzzy_match")) / len(results)) if results else 0.0
print(f"Fuzzy Match Accuracy: {fuzzy_accuracy:.4f}")

# ===========================
# PT & SOC Accuracies (code-based)
# ===========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]

# PT accuracy (PT(true LLT) vs PT(pred LLT))
if has_pt_in_llt:
    pt_pairs = _both_present([(r.get("true_pt_code"), r.get("pred_pt_code")) for r in results])
    if pt_pairs:
        pt_acc = sum(int(a == b) for a, b in pt_pairs) / len(pt_pairs)
        print(f"\nPT Accuracy (code): {pt_acc:.4f}  (over {len(pt_pairs)} rows)")
    else:
        print("\nPT Accuracy (code): N/A (no comparable rows)")
else:
    print("\nPT Accuracy (code): N/A (LLT file has no PT_Code column)")

# SOC accuracy (primary vs AE ground-truth)  --> Option A
soc_pairs_ae = _both_present([(r.get("true_soc_code"), r.get("pred_soc_code")) for r in results])
if soc_pairs_ae:
    soc_acc_vs_ae = sum(int(a == b) for a, b in soc_pairs_ae) / len(soc_pairs_ae)
    print(f"SOC Accuracy (primary vs AE): {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
    print(f"SOC Accuracy (Option A):      {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
else:
    print("SOC Accuracy (primary vs AE): N/A")
    print("SOC Accuracy (Option A):      N/A")

# SOC accuracy (primary vs true primary mapping)  --> Option B
soc_pairs_primary = _both_present([(r.get("true_soc_code_primary"), r.get("pred_soc_code")) for r in results])
if soc_pairs_primary:
    soc_acc_vs_true_primary = sum(int(a == b) for a, b in soc_pairs_primary) / len(soc_pairs_primary)
    print(f"SOC Accuracy (primary vs true primary): {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
    print(f"SOC Accuracy (Option B):                {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
else:
    print("SOC Accuracy (primary vs true primary): N/A")
    print("SOC Accuracy (Option B):                N/A")

# SOC accuracy (any-of vs AE)
soc_any_flags = []
for r in results:
    ae_soc = r.get("true_soc_code")
    pred_all = r.get("pred_soc_codes_all") or []
    if ae_soc is not None and pred_all:
        soc_any_flags.append(int(ae_soc in pred_all))
if soc_any_flags:
    soc_acc_any = sum(soc_any_flags) / len(soc_any_flags)
    print(f"SOC Accuracy (any-of vs AE):  {soc_acc_any:.4f} (over {len(soc_any_flags)} rows)")
else:
    print("SOC Accuracy (any-of vs AE): N/A")

# ===========================
# Save RUN-LEVEL METRICS (per-seed) as JSON
# ===========================
metrics_payload = {
    "meta": {
        "dataset": AE_CSV_PATH.split("/")[-1].replace(".csv",""),
        "model": "llama-3.3-70b-instruct",
        "n_candidates": N_CANDIDATES,
        "max_rows": MAX_ROWS,
        "seed": RUN_SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "per-AE results saved in OUT_JSON with seed suffix; aggregated metrics saved here",
    },
    "counts": {
        "n_samples": len(results),
        "n_pt_pairs": len(pt_pairs) if has_pt_in_llt else 0,
        "n_soc_optA": len(soc_pairs_ae),
        "n_soc_optB": len(soc_pairs_primary),
        "n_soc_any": len(soc_any_flags),
    },
    "metrics": {
        "LLT_term_acc_exact": float(acc),
        "LLT_term_acc_fuzzy": float(fuzzy_accuracy),
        "LLT_precision_macro": float(precision),
        "LLT_recall_macro": float(recall),
        "LLT_f1_macro": float(f1),
        "PT_code_acc": (float(pt_acc) if has_pt_in_llt and 'pt_acc' in locals() else None),
        "SOC_acc_option_a": (float(soc_acc_vs_ae) if soc_pairs_ae else None),
        "SOC_acc_option_b": (float(soc_acc_vs_true_primary) if soc_pairs_primary else None),
        "SOC_acc_any_of_vs_AE": (float(soc_acc_any) if soc_any_flags else None),
    },
}

metrics_path = OUT_JSON.replace(".json", f"_metrics_seed{RUN_SEED}.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
print("Saved metrics:", metrics_path)
