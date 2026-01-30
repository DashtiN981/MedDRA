"""
File Name: rag_prompting_reasoning_v4_parallel_PT_SOC.py
Author: Naghme Dashti (July 2025) / patched for parallel + PT/SOC mapping

RAG-based Prompting for MedDRA Coding (No Ground Truth Injection)
-----------------------------------------------------------------
- Retrieves Top-K semantically similar LLTs based on MiniLM embeddings
- Does NOT insert the correct LLT into the candidate list
- Prompts LLM to provide reasoning + structured final answer

PATCH:
- Async parallel calls (AsyncOpenAI + Semaphore)
- LLT -> PT and PT -> SOC mapping (primary + any-of)
- Run-level metrics JSON saved (LLT/PT/SOC)
- Optional logging: true_term ∈ Top-K (retrieval success rate, NOT injection)
"""

import json
import re
import time
import random
import asyncio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import AsyncOpenAI
from rapidfuzz import fuzz, process

# =========================
# Local OpenAI-compatible LLM API
# =========================
API_KEY = "sk-aKGeEFMZB0gXEcE51FTc0A"     # 
BASE_URL = "http://pluto/v1/"
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# =========================
# Parameters (keep same behavior as your v4)
# =========================
TOP_K = 100
MAX_ROWS = None
EMB_DIM = 384

# Parallelism
CONCURRENCY = 50
MAX_RETRIES = 4
sem = asyncio.Semaphore(CONCURRENCY)

# Dataset files (v4)
AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings_Dauno.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
AE_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv"
LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"

# NEW: PT dictionary (like parallel version)
PT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"

# Output
OUTPUT_FILE_NAME = "Dauno_100k__v4_parallel_PT_SOC"

# LLM
LLM_API_NAME = "nvidia-llama-3.3-70b-instruct-fp8"
LLM_TEMP = 0.0
LLM_TOKEN = 250

# Optional: log whether GT is in topK (retrieval success, not injection)
LOG_TRUE_IN_TOPK = True


# =========================
# Helpers (copied from parallel version for safe code-mapping)
# =========================
def canon_code(x) -> str | None:
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

def clean_model_term_v4_style(answer: str) -> str:
    """
    Keep v4 behavior:
    - If 'Final answer:' exists: take tail
    - else: last line
    Then light cleanup (quotes/whitespace only) to reduce trivial mismatches.
    """
    if not isinstance(answer, str):
        return ""
    txt = answer.strip()
    if "Final answer:" in txt:
        out = txt.split("Final answer:")[-1].strip()
    else:
        out = txt.splitlines()[-1].strip() if txt.splitlines() else txt.strip()

    out = out.strip().strip('"\''"“”‘’")
    out = re.sub(r"\s+", " ", out).strip()
    return out


async def llm_call_with_retry(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful medical coding assistant."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=LLM_API_NAME,
                    messages=messages,
                    temperature=LLM_TEMP,
                    max_tokens=LLM_TOKEN,
                )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"__ERROR__: {e}"
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.2)


# =========================
# Load AE / LLT / PT (+ SOC primary mapping)
# =========================
ae_full = pd.read_csv(AE_CSV_FILE, sep=";", encoding="latin1")
ae_keep = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_full.columns]
if not ae_keep:
    raise ValueError("AE CSV must include at least 'Original_Term_aufbereitet' and 'ZB_LLT_Code'.")
ae_df = ae_full[ae_keep].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]].dropna().reset_index(drop=True)
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_to_pt        = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))

# PT/SOC mapping (same logic as your parallel script)
pt_header = pd.read_csv(PT_CSV_FILE, sep=";", encoding="latin1", nrows=0).columns
pt_cols = [c for c in ["PT_Code","PT_Term","SOC_Code","SOC_Term","Ist_Primary_SOC","Primary_SOC_Code"] if c in pt_header]
pt_df = pd.read_csv(PT_CSV_FILE, sep=";", encoding="latin1")[pt_cols].copy()

pt_df["PT_Code"]  = pt_df["PT_Code"].map(canon_code)
pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)
if "Primary_SOC_Code" in pt_df.columns:
    pt_df["Primary_SOC_Code"] = pt_df["Primary_SOC_Code"].map(canon_code)

if "Ist_Primary_SOC" in pt_df.columns:
    pt_df["Ist_Primary_SOC_norm"] = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper()
else:
    pt_df["Ist_Primary_SOC_norm"] = ""

soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"])) if "SOC_Term" in pt_df.columns else {}
pt_meta = pt_df.dropna(subset=["PT_Code"]).drop_duplicates(subset=["PT_Code"])
pt_meta = pt_meta.set_index("PT_Code")[["PT_Term"]].to_dict(orient="index")

pt_code_to_soc_all: dict[str, list[str]] = (
    pt_df.dropna(subset=["PT_Code","SOC_Code"])
         .groupby("PT_Code")["SOC_Code"]
         .apply(lambda s: sorted(set(x for x in s if x is not None)))
         .to_dict()
)

pt_code_to_primary_soc: dict[str, str | None] = {}
has_primary_soc_code_col = "Primary_SOC_Code" in pt_df.columns

for ptc, grp in pt_df.dropna(subset=["PT_Code"]).groupby("PT_Code"):
    primary = None
    y_rows = grp[grp["Ist_Primary_SOC_norm"] == "Y"]
    if not y_rows.empty and pd.notna(y_rows.iloc[0].get("SOC_Code")):
        primary = canon_code(y_rows.iloc[0]["SOC_Code"])
    else:
        if has_primary_soc_code_col:
            prim_vals = [canon_code(v) for v in grp["Primary_SOC_Code"].dropna().tolist()]
            uniq = sorted(set(v for v in prim_vals if v is not None))
            if len(uniq) == 1:
                primary = uniq[0]
        if primary is None:
            all_socs = pt_code_to_soc_all.get(ptc, [])
            if len(all_socs) == 1:
                primary = all_socs[0]
    pt_code_to_primary_soc[ptc] = primary

num_pts = len(pt_code_to_soc_all)
num_single = sum(1 for _, lst in pt_code_to_soc_all.items() if len(lst) == 1)
num_with_y = pt_df[pt_df["Ist_Primary_SOC_norm"] == "Y"]["PT_Code"].nunique()
num_with_primary_code = pt_df.dropna(subset=["Primary_SOC_Code"])["PT_Code"].nunique() if has_primary_soc_code_col else 0
num_undefined = sum(1 for _, v in pt_code_to_primary_soc.items() if v is None)
print(f"[PT/SOC QC] PTs:{num_pts} | with 'Y':{num_with_y} | with Primary_SOC_Code:{num_with_primary_code} | single-SOC:{num_single} | undefined-primary:{num_undefined}")

# Term->Code mapping (for robust eval like parallel)
term_norm_to_llt = {}
for _, r in llt_df.iterrows():
    term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

def term_to_llt_code(pred_term: str, allow_fuzzy=True) -> str | None:
    t = norm_text(pred_term)
    if not t:
        return None
    if t in term_norm_to_llt:
        return term_norm_to_llt[t]
    if allow_fuzzy:
        best = process.extractOne(t, list(term_norm_to_llt.keys()), scorer=fuzz.ratio, score_cutoff=94)
        return term_norm_to_llt[best[0]] if best else None
    return None


# =========================
# Load Embeddings (same as v4)
# =========================
with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_emb_raw = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_emb_raw = json.load(f)

ae_emb_dict = {k: np.array(v) for k, v in ae_emb_raw.items()}
llt_emb_dict = {k: np.array(v) for k, v in llt_emb_raw.items()}

# Precompute LLT norms (safe optimization: does NOT change similarity ordering)
llt_terms = list(llt_emb_dict.keys())
llt_mat_norms = {t: float(np.linalg.norm(llt_emb_dict[t])) for t in llt_terms}

# Row limiting
if isinstance(MAX_ROWS, int) and MAX_ROWS > 0:
    ae_df = ae_df.iloc[:MAX_ROWS].reset_index(drop=True)
else:
    ae_df = ae_df.reset_index(drop=True)


# =========================
# One row (async) — keeps v4 logic: NO GT injection
# =========================
async def process_one_row(idx: int, row) -> dict:
    ae_text = str(row["Original_Term_aufbereitet"])
    true_LLT_Code = canon_code(row["ZB_LLT_Code"])
    true_LLT_term = llt_code_to_term.get(true_LLT_Code)

    true_PT_Code = llt_to_pt.get(true_LLT_Code)
    true_PT_term = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None

    # AE SOC (if present)
    true_SOC_Code_AE_raw = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_SOC_Code_AE_raw = canon_code(v)

    # AE embedding lookup (keep v4 behavior: skip if missing)
    if ae_text not in ae_emb_dict:
        return {
            "idx": idx,
            "AE_text": ae_text,
            "true_LLT_term": true_LLT_term,
            "pred_LLT_term": None,
            "true_LLT_Code": true_LLT_Code,
            "pred_LLT_Code": None,
            "true_PT_term": true_PT_term,
            "pred_PT_term": None,
            "true_PT_Code": true_PT_Code,
            "pred_PT_Code": None,
            "true_SOC_Code_AE_raw": true_SOC_Code_AE_raw,
            "true_SOC_Code": None,
            "pred_SOC_Code": None,
            "true_SOC_Code_primary": pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None,
            "true_SOC_codes_all": pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else [],
            "pred_SOC_codes_all": [],
            "exact_LLT_match": False,
            "LLT_fuzzy_score": 0.0,
            "LLT_fuzzy_match": False,
            "model_output": "__SKIP__: missing AE embedding",
            "skip_reason": "missing_ae_embedding",
        }

    ae_emb = ae_emb_dict[ae_text]
    ae_norm = float(np.linalg.norm(ae_emb))

    # Compute similarities (same math as your v4)
    similarities = []
    for term, llt_emb in llt_emb_dict.items():
        denom = (ae_norm * llt_mat_norms[term]) or 1e-12
        sim = float(np.dot(ae_emb, llt_emb) / denom)
        similarities.append((term, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    candidate_terms = [term for term, _ in similarities[:TOP_K]]  # NO injection, NO shuffle

    true_in_topk = None
    true_rank_in_topk = None
    if LOG_TRUE_IN_TOPK and true_LLT_term:
        true_in_topk = (true_LLT_term in candidate_terms)
        true_rank_in_topk = (candidate_terms.index(true_LLT_term) if true_in_topk else None)

    # Prompt (same structure as your v4)
    prompt = (
        f"You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term.\n"
        f"Here is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        f"Here is a list of candidate LLT terms:\n"
        + "\n".join(f"- {term}" for term in candidate_terms) +
        "\n\nPlease analyze the AE and list, and first provide a short reasoning.\n"
        f"Then, on a separate line, write the best matching LLT in this format:\nFinal answer: <LLT_TERM>"
    )

    answer = await llm_call_with_retry(prompt)

    if answer.startswith("__ERROR__:"):
        true_SOC_primary = pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None
        true_SOC_AE_for_eval = true_SOC_Code_AE_raw if true_SOC_Code_AE_raw is not None else true_SOC_primary
        return {
            "idx": idx,
            "AE_text": ae_text,
            "true_LLT_term": true_LLT_term,
            "pred_LLT_term": None,
            "true_LLT_Code": true_LLT_Code,
            "pred_LLT_Code": None,
            "true_PT_term": true_PT_term,
            "pred_PT_term": None,
            "true_PT_Code": true_PT_Code,
            "pred_PT_Code": None,
            "true_SOC_Code_AE_raw": true_SOC_Code_AE_raw,
            "true_SOC_Code": true_SOC_AE_for_eval,
            "pred_SOC_Code": None,
            "true_SOC_Code_primary": true_SOC_primary,
            "true_SOC_codes_all": pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else [],
            "pred_SOC_codes_all": [],
            "exact_LLT_match": False,
            "LLT_fuzzy_score": 0.0,
            "LLT_fuzzy_match": False,
            "model_output": answer,
            "true_in_topk": true_in_topk,
            "true_rank_in_topk": true_rank_in_topk,
        }

    # Keep v4-style parsing
    answer_line = clean_model_term_v4_style(answer)

    # Evaluate LLT (term-based like v4) + map to code/PT/SOC like parallel
    exact_LLT_match = (true_LLT_term is not None and answer_line == true_LLT_term)
    LLT_fuzzy_score = fuzz.ratio(answer_line.lower(), (true_LLT_term or "").lower())
    LLT_fuzzy_match = (true_LLT_term is not None and LLT_fuzzy_score >= 90)

    pred_LLT_Code = term_to_llt_code(answer_line, allow_fuzzy=True)
    pred_LLT_term_std = llt_code_to_term.get(pred_LLT_Code, answer_line)

    pred_PT_Code = llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None
    pred_PT_term = pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None

    # PT fuzzy/exact (same thresholds as parallel)
    exact_PT_match = (true_PT_term is not None) and (pred_PT_term is not None) and (pred_PT_term == true_PT_term)
    if (true_PT_term is not None) and (pred_PT_term is not None):
        PT_fuzzy_score = fuzz.ratio(pred_PT_term.lower(), true_PT_term.lower())
        PT_fuzzy_match = PT_fuzzy_score >= 90
    else:
        PT_fuzzy_score = 0.0
        PT_fuzzy_match = False

    # SOC metrics (same definition as parallel)
    true_SOC_primary = pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None
    pred_SOC_primary = pt_code_to_primary_soc.get(pred_PT_Code) if pred_PT_Code else None

    true_SOC_AE_for_eval = true_SOC_Code_AE_raw if true_SOC_Code_AE_raw is not None else true_SOC_primary

    true_SOC_codes_all = pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else []
    pred_SOC_codes_all = pt_code_to_soc_all.get(pred_PT_Code, []) if pred_PT_Code else []

    return {
        "idx": idx,
        "AE_text": ae_text,
        "true_LLT_term": true_LLT_term,
        "pred_LLT_term": pred_LLT_term_std,
        "true_LLT_Code": true_LLT_Code,
        "pred_LLT_Code": pred_LLT_Code,

        "true_PT_term": true_PT_term,
        "pred_PT_term": pred_PT_term,
        "true_PT_Code": true_PT_Code,
        "pred_PT_Code": pred_PT_Code,

        "true_SOC_Code_AE_raw": true_SOC_Code_AE_raw,
        "true_SOC_Code": true_SOC_AE_for_eval,         # Option A comparator uses this
        "pred_SOC_Code": pred_SOC_primary,

        "true_SOC_Code_primary": true_SOC_primary,      # Option B comparator uses this
        "true_SOC_codes_all": true_SOC_codes_all,
        "pred_SOC_codes_all": pred_SOC_codes_all,

        "exact_LLT_match": exact_LLT_match,
        "LLT_fuzzy_score": LLT_fuzzy_score,
        "LLT_fuzzy_match": LLT_fuzzy_match,

        "exact_PT_match": exact_PT_match,
        "PT_fuzzy_score": PT_fuzzy_score,
        "PT_fuzzy_match": PT_fuzzy_match,

        "model_output": answer,

        # optional logs
        "true_in_topk": true_in_topk,
        "true_rank_in_topk": true_rank_in_topk,
    }


async def run_parallel():
    tasks = [asyncio.create_task(process_one_row(i, r)) for i, r in ae_df.iterrows()]
    out = await asyncio.gather(*tasks)
    out.sort(key=lambda d: d["idx"])
    return out


# =========================
# Run
# =========================
RUN_SEED = 42
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

t0 = time.time()
results = asyncio.run(run_parallel())
print(f"[DONE] Parallel run finished in {(time.time()-t0):.1f}s | n={len(results)} | concurrency={CONCURRENCY}")

# Save per-AE JSON
for r in results:
    r.pop("idx", None)

out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_seed{RUN_SEED}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved per-AE JSON:")
print("-", out_json)


# =========================
# Evaluation (LLT/PT + SOC)
# =========================
# Keep only non-skipped rows for metrics
usable = [r for r in results if not (isinstance(r.get("model_output"), str) and r["model_output"].startswith("__SKIP__"))]

y_true = [r["true_LLT_term"] or "" for r in usable]
y_pred = [r["pred_LLT_term"] or "" for r in usable]
y_pred_fuzzy = [(r["true_LLT_term"] or r["pred_LLT_term"] or "") if r["LLT_fuzzy_match"] else (r["pred_LLT_term"] or "") for r in usable]

z_true = [r["true_PT_term"] or "" for r in usable]
z_pred = [r["pred_PT_term"] or "" for r in usable]
z_pred_fuzzy = [(r["true_PT_term"] or r["pred_PT_term"] or "") if r["PT_fuzzy_match"] else (r["pred_PT_term"] or "") for r in usable]

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
llt_fuzzy_acc = (sum(1 for r in usable if r["LLT_fuzzy_match"]) / max(1, len(usable))) if usable else 0.0

pt_acc = accuracy_score(z_true, z_pred) if any(z_true) else 0.0
pt_f1 = f1_score(z_true, z_pred, average="macro") if any(z_true) else 0.0
pt_precision = precision_score(z_true, z_pred, average="macro", zero_division=0) if any(z_true) else 0.0
pt_recall = recall_score(z_true, z_pred, average="macro", zero_division=0) if any(z_true) else 0.0
pt_fuzzy_acc = (sum(1 for r in usable if r["PT_fuzzy_match"]) / max(1, len(usable))) if usable else 0.0

# Code-based acc like parallel
mask_llt = [(r.get("true_LLT_Code") is not None) and (r.get("pred_LLT_Code") is not None) for r in usable]
mask_pt  = [(r.get("true_PT_Code")  is not None) and (r.get("pred_PT_Code")  is not None) for r in usable]

LLT_code_acc = (sum(1 for i, r in enumerate(usable) if mask_llt[i] and r["true_LLT_Code"] == r["pred_LLT_Code"]) / max(1, sum(mask_llt))) if usable else 0.0
PT_code_acc  = (sum(1 for i, r in enumerate(usable) if mask_pt[i]  and r["true_PT_Code"]  == r["pred_PT_Code"])  / max(1, sum(mask_pt)))  if usable else 0.0

print(f"\nLLT Term Accuracy (exact): {llt_acc:.4f}")
print(f"PT  Term Accuracy (exact): {pt_acc:.4f}")
print(f"LLT F1 (macro): {llt_f1:.4f} | LLT Precision: {llt_precision:.4f} | LLT Recall: {llt_recall:.4f}")
print(f"PT  F1 (macro): {pt_f1:.4f} | PT  Precision: {pt_precision:.4f} | PT  Recall: {pt_recall:.4f}")
print(f"Fuzzy LLT Term Accuracy: {llt_fuzzy_acc:.4f}")
print(f"Fuzzy PT  Term Accuracy: {pt_fuzzy_acc:.4f}")
print(f"LLT Accuracy (code): {LLT_code_acc:.4f}  [on {sum(mask_llt)} rows]")
print(f"PT  Accuracy (code): {PT_code_acc:.4f}   [on {sum(mask_pt)} rows]")


# =========================
# SOC Accuracies (code-based) — Option A / B / Any-of
# =========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]

soc_acc_vs_ae = None
soc_acc_vs_true_primary = None
soc_acc_any = None

soc_pairs_ae = _both_present([(r.get("true_SOC_Code"), r.get("pred_SOC_Code")) for r in usable])
if soc_pairs_ae:
    soc_acc_vs_ae = sum(int(a == b) for a, b in soc_pairs_ae) / len(soc_pairs_ae)
    print(f"SOC Accuracy (Option A, primary vs AE): {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
else:
    print("SOC Accuracy (Option A): N/A")

soc_pairs_primary = _both_present([(r.get("true_SOC_Code_primary"), r.get("pred_SOC_Code")) for r in usable])
if soc_pairs_primary:
    soc_acc_vs_true_primary = sum(int(a == b) for a, b in soc_pairs_primary) / len(soc_pairs_primary)
    print(f"SOC Accuracy (Option B, primary vs true primary): {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
else:
    print("SOC Accuracy (Option B): N/A")

soc_any_flags = []
for r in usable:
    ae_soc = r.get("true_SOC_Code")
    pred_all = r.get("pred_SOC_codes_all") or []
    if ae_soc is not None and pred_all:
        soc_any_flags.append(int(ae_soc in pred_all))
if soc_any_flags:
    soc_acc_any = sum(soc_any_flags) / len(soc_any_flags)
    print(f"SOC Accuracy (any-of vs AE): {soc_acc_any:.4f} (over {len(soc_any_flags)} rows)")
else:
    print("SOC Accuracy (any-of vs AE): N/A")


# =========================
# Save metrics JSON (run-level)
# =========================
true_in_topk_count = sum(1 for r in usable if r.get("true_in_topk") is True) if LOG_TRUE_IN_TOPK else None
true_in_topk_pct = (true_in_topk_count / max(1, len(usable))) if (LOG_TRUE_IN_TOPK and usable) else None

metrics_payload = {
    "meta": {
        "script": "rag_prompting_reasoning_v4_parallel_PT_SOC.py",
        "model": LLM_API_NAME,
        "top_k": TOP_K,
        "max_rows": MAX_ROWS,
        "embedding_dim": EMB_DIM,
        "llm_temp": LLM_TEMP,
        "llm_token": LLM_TOKEN,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "concurrency": CONCURRENCY,
        "max_retries": MAX_RETRIES,
        "run_seed": RUN_SEED,
        "notes": "v4 behavior preserved (no GT injection, no candidate shuffle). Added PT/SOC mapping + async parallel.",
    },
    "counts": {
        "n_total_rows": len(results),
        "n_usable_rows": len(usable),
        "n_skipped_missing_embedding": sum(1 for r in results if r.get("skip_reason") == "missing_ae_embedding"),
        "n_LLT_code_eval": int(sum(mask_llt)),
        "n_PT_code_eval": int(sum(mask_pt)),
        "n_SOC_optA": len(soc_pairs_ae),
        "n_SOC_optB": len(soc_pairs_primary),
        "n_SOC_any": len(soc_any_flags),
        "n_true_in_topk": true_in_topk_count,
    },
    "rates": {
        "pct_true_in_topk": (None if true_in_topk_pct is None else float(true_in_topk_pct)),
    },
    "metrics": {
        "LLT_term_acc_exact": float(llt_acc),
        "LLT_term_acc_fuzzy": float(llt_fuzzy_acc),
        "LLT_precision_macro": float(llt_precision),
        "LLT_recall_macro": float(llt_recall),
        "LLT_f1_macro": float(llt_f1),

        "PT_term_acc_exact": float(pt_acc),
        "PT_term_acc_fuzzy": float(pt_fuzzy_acc),
        "PT_precision_macro": float(pt_precision),
        "PT_recall_macro": float(pt_recall),
        "PT_f1_macro": float(pt_f1),

        "LLT_code_acc": float(LLT_code_acc),
        "PT_code_acc": float(PT_code_acc),

        "SOC_acc_option_a": (None if soc_acc_vs_ae is None else float(soc_acc_vs_ae)),
        "SOC_acc_option_b": (None if soc_acc_vs_true_primary is None else float(soc_acc_vs_true_primary)),
        "SOC_acc_any_of_vs_AE": (None if soc_acc_any is None else float(soc_acc_any)),
    },
}

metrics_json_path = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_metrics_seed{RUN_SEED}.json"
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

print("[Saved metrics JSON]:", metrics_json_path)
