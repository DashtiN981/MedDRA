"""
File Name: RAGNew_parallel.py   === Author: Naghme Dashti / patched for parallel requests
RAG-based Prompting with Explicit Reasoning + Final Answer Line
----------------------------------------------------------------
Parallel version: sends up to CONCURRENCY requests concurrently using AsyncOpenAI + Semaphore.

SOC metrics reported:
- Option A (AE-centric):      primary(pred PT) == AE_SOC   (falls back to true primary if AE SOC missing)
- Option B (dictionary-only): primary(pred PT) == primary(true PT)
- Any-of vs AE:               AE_SOC ∈ all_SOCs(pred PT)
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
API_KEY = "sk-aKGeEFMZB0gXEcE51FTc0A" 
BASE_URL = "http://pluto/v1/"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# =========================
# Parameters
# =========================
TOP_K = 10
MAX_ROWS = None       # e.g., set an int to limit rows; None = all
EMB_DIM = 384

# Parallelism
CONCURRENCY = 30      # <-- اول 10 تست کن، بعد 20/30/50
MAX_RETRIES = 4
sem = asyncio.Semaphore(CONCURRENCY)

# Datasets
DATASET_NAME     = "KI_Projekt_Mosaic_AE_Codierung_2024_07_03"
DATASET_EMB_NAME = "ae_embeddings_Mosaic"

# Dictionaries and output
LLT_DICTIONARY_NAME      = "LLT2_Code_English_25_0"
LLT_DICTIONARY_EMB_NAME  = "llt2_embeddings"
PT_DICTIONARY_NAME       = "PT2_SOC_25_0"
OUTPUT_FILE_NAME         = "Mosaic_outputParallel"

# Paths
AE_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{DATASET_NAME}.csv"
AE_EMB_FILE  = f"/home/naghmedashti/MedDRA-LLM/embedding/{DATASET_EMB_NAME}.json"
LLT_CSV_FILE = f"/home/naghmedashti/MedDRA-LLM/data/{LLT_DICTIONARY_NAME}.csv"
LLT_EMB_FILE = f"/home/naghmedashti/MedDRA-LLM/embedding/{LLT_DICTIONARY_EMB_NAME}.json"
PT_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{PT_DICTIONARY_NAME}.csv"

LLM_API_NAME = "nvidia-llama-3.3-70b-instruct-fp8" # or "Llama-3.3-70B-Instruct" llama-3.3-70b-instruct-awq  or GPT-OSS-120B
LLM_TEMP = 0.0
LLM_TOKEN = 250


# =========================
# Helpers
# =========================
def canon_code(x) -> str | None:
    """Normalize numeric-like codes, e.g., '10000081.0' -> '10000081' (string)."""
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

def clean_model_term(s: str) -> str:
    """Normalize model's last-line answer into a clean LLT term string."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = re.sub(r"^\s*final\s*answer\s*:\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^\s*(?:[-–—•·*]+|\(?\d+\)?[.)]|[A-Za-z]\)|\d+\s*-\s*)\s*", "", t)
    t = t.strip().strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t).strip()
    return t


async def llm_call_with_retry(prompt: str) -> str:
    """Async LLM call with semaphore + retry/backoff. Returns text or '__ERROR__:' prefixed string."""
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

llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]]
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

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

llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_to_pt        = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))
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

term_norm_to_llt = {}
for _, r in llt_df.iterrows():
    term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

def term_to_llt_code(pred_term: str, allow_fuzzy=True) -> str | None:
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
        best = process.extractOne(t, list(term_norm_to_llt.keys()), scorer=fuzz.ratio, score_cutoff=94)
        return term_norm_to_llt[best[0]] if best else None
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
# RAG row processor (async)
# =========================
async def process_one_row(idx: int, row) -> dict:
    ae_text = str(row["Original_Term_aufbereitet"])
    true_LLT_Code = canon_code(row["ZB_LLT_Code"])
    true_LLT_term = llt_code_to_term.get(true_LLT_Code)
    true_PT_Code = llt_to_pt.get(true_LLT_Code)
    true_PT_term = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None

    true_SOC_Code_AE_raw = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_SOC_Code_AE_raw = canon_code(v)

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
        hits = process.extract(ae_text, llt_terms_all, scorer=fuzz.token_set_ratio, limit=TOP_K, score_cutoff=0)
        cand = [t for (t, score, idx_) in hits]
        if len(cand) < TOP_K:
            extra = random.sample(llt_terms_all, k=min(TOP_K - len(cand), len(llt_terms_all)))
            cand += extra
        candidate_terms = cand[:TOP_K]

    #if true_LLT_term and true_LLT_term not in candidate_terms:
    #    candidate_terms.append(true_LLT_term)
    random.shuffle(candidate_terms)

    prompt = (
        "You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term."
        f"\nHere is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        "Here is a list of candidate LLT terms:\n" + "\n".join(f"- {term}" for term in candidate_terms) +
        "\n\nPlease analyze the AE and list, and first provide a short reasoning."
        "\nThen, on a separate line, write the best matching LLT in this format:"
        "\nFinal answer: <LLT_TERM>"
    )

    answer = await llm_call_with_retry(prompt)

    # Error-safe path
    if answer.startswith("__ERROR__:"):
        true_SOC_Code_primary = pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None
        true_SOC_Code_AE_for_eval = true_SOC_Code_AE_raw if true_SOC_Code_AE_raw is not None else true_SOC_Code_primary

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
            "true_SOC_Term_AE": soc_code_to_term.get(true_SOC_Code_AE_raw) if true_SOC_Code_AE_raw else None,
            "true_SOC_Code": true_SOC_Code_AE_for_eval,
            "true_SOC_Term": soc_code_to_term.get(true_SOC_Code_AE_for_eval) if true_SOC_Code_AE_for_eval else None,
            "pred_SOC_Code": None,
            "pred_SOC_Term": None,
            "true_SOC_Code_primary": true_SOC_Code_primary,
            "true_SOC_Term_primary": soc_code_to_term.get(true_SOC_Code_primary) if true_SOC_Code_primary else None,
            "true_SOC_codes_all": pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else [],
            "pred_SOC_codes_all": [],
            "pred_primary_soc_missing": None,
            "true_primary_soc_missing": None,
            "exact_LLT_match": False,
            "LLT_fuzzy_score": 0.0,
            "LLT_fuzzy_match": False,
            "exact_PT_match": False,
            "PT_fuzzy_score": 0.0,
            "PT_fuzzy_match": False,
            "model_output": answer,
        }

    last_line = answer.split("Final answer:")[-1] if "Final answer:" in answer else answer.split("\n")[-1]
    answer_line = clean_model_term(last_line)

    pred_LLT_Code = term_to_llt_code(answer_line, allow_fuzzy=True)
    pred_PT_Code  = llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None
    pred_PT_term  = pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None

    true_SOC_Code_primary = pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None
    pred_SOC_Code_primary = pt_code_to_primary_soc.get(pred_PT_Code) if pred_PT_Code else None
    true_SOC_Code_AE_for_eval = true_SOC_Code_AE_raw if true_SOC_Code_AE_raw is not None else true_SOC_Code_primary

    true_SOC_codes_all = pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else []
    pred_SOC_codes_all = pt_code_to_soc_all.get(pred_PT_Code, []) if pred_PT_Code else []

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

    pred_primary_soc_missing = (pred_PT_Code is not None and pt_code_to_primary_soc.get(pred_PT_Code) is None)
    true_primary_soc_missing = (true_PT_Code is not None and pt_code_to_primary_soc.get(true_PT_Code) is None) if true_PT_Code else None

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
        "true_SOC_Term_AE": soc_code_to_term.get(true_SOC_Code_AE_raw) if true_SOC_Code_AE_raw else None,
        "true_SOC_Code": true_SOC_Code_AE_for_eval,
        "true_SOC_Term": soc_code_to_term.get(true_SOC_Code_AE_for_eval) if true_SOC_Code_AE_for_eval else None,
        "pred_SOC_Code": pred_SOC_Code_primary,
        "pred_SOC_Term": soc_code_to_term.get(pred_SOC_Code_primary) if pred_SOC_Code_primary else None,
        "true_SOC_Code_primary": true_SOC_Code_primary,
        "true_SOC_Term_primary": soc_code_to_term.get(true_SOC_Code_primary) if true_SOC_Code_primary else None,
        "true_SOC_codes_all": true_SOC_codes_all,
        "pred_SOC_codes_all": pred_SOC_codes_all,
        "pred_primary_soc_missing": pred_primary_soc_missing,
        "true_primary_soc_missing": true_primary_soc_missing,
        "exact_LLT_match": exact_LLT_match,
        "LLT_fuzzy_score": LLT_fuzzy_score,
        "LLT_fuzzy_match": LLT_fuzzy_match,
        "exact_PT_match": exact_PT_match,
        "PT_fuzzy_score": PT_fuzzy_score,
        "PT_fuzzy_match": PT_fuzzy_match,
        "model_output": answer,
    }


async def run_parallel():
    tasks = [asyncio.create_task(process_one_row(i, r)) for i, r in ae_df.iterrows()]
    out = await asyncio.gather(*tasks)
    # restore original order
    out.sort(key=lambda d: d["idx"])
    return out


# =========================
# Run
# =========================
RUN_SEED = 44
random.seed(RUN_SEED)

t0 = time.time()
results = asyncio.run(run_parallel())
print(f"[DONE] Parallel run finished in {(time.time()-t0):.1f}s | n={len(results)} | concurrency={CONCURRENCY}")

# Remove idx before saving (optional)
for r in results:
    r.pop("idx", None)


# =========================
# Save Results (PER-AE JSON ONLY — NO CSV)
# =========================
out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_seed{RUN_SEED}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


# =========================
# Evaluation (LLT/PT term-based + code-based)
# =========================
y_true = [r["true_LLT_term"] or "" for r in results]
y_pred = [r["pred_LLT_term"] or "" for r in results]
y_pred_fuzzy = [
    (r["true_LLT_term"] or r["pred_LLT_term"] or "") if r["LLT_fuzzy_match"] else (r["pred_LLT_term"] or "")
    for r in results
]

z_true = [r["true_PT_term"] or "" for r in results]
z_pred = [r["pred_PT_term"] or "" for r in results]
z_pred_fuzzy = [
    (r["true_PT_term"] or r["pred_PT_term"] or "") if r["PT_fuzzy_match"] else (r["pred_PT_term"] or "")
    for r in results
]

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
llt_recall = recall_score(y_true, y_pred, average="macro") if any(y_true) else 0.0
llt_fuzzy_acc = (sum(r["LLT_fuzzy_match"] for r in results) / max(1, len(results))) if results else 0.0

pt_acc = accuracy_score(z_true, z_pred) if any(z_true) else 0.0
pt_f1 = f1_score(z_true, z_pred, average="macro") if any(z_true) else 0.0
pt_precision = precision_score(z_true, z_pred, average="macro", zero_division=0) if any(z_true) else 0.0
pt_recall = recall_score(z_true, z_pred, average="macro") if any(z_true) else 0.0
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
# SOC Accuracies (code-based) — Option A / B / Any-of
# =========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]

soc_acc_vs_ae = None
soc_acc_vs_true_primary = None
soc_acc_any = None

soc_pairs_ae = _both_present([(r.get("true_SOC_Code"), r.get("pred_SOC_Code")) for r in results])
if soc_pairs_ae:
    soc_acc_vs_ae = sum(int(a == b) for a, b in soc_pairs_ae) / len(soc_pairs_ae)
    print(f"SOC Accuracy (primary vs AE): {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
    print(f"SOC Accuracy (Option A):      {soc_acc_vs_ae:.4f} (over {len(soc_pairs_ae)} rows)")
else:
    print("SOC Accuracy (primary vs AE): N/A")
    print("SOC Accuracy (Option A):      N/A")

soc_pairs_primary = _both_present([(r.get("true_SOC_Code_primary"), r.get("pred_SOC_Code")) for r in results])
if soc_pairs_primary:
    soc_acc_vs_true_primary = sum(int(a == b) for a, b in soc_pairs_primary) / len(soc_pairs_primary)
    print(f"SOC Accuracy (primary vs true primary): {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
    print(f"SOC Accuracy (Option B):                {soc_acc_vs_true_primary:.4f} (over {len(soc_pairs_primary)} rows)")
else:
    print("SOC Accuracy (primary vs true primary): N/A")
    print("SOC Accuracy (Option B):                N/A")

soc_any_flags = []
for r in results:
    ae_soc = r.get("true_SOC_Code")
    pred_all = r.get("pred_SOC_codes_all") or []
    if ae_soc is not None and pred_all:
        soc_any_flags.append(int(ae_soc in pred_all))
if soc_any_flags:
    soc_acc_any = sum(soc_any_flags) / len(soc_any_flags)
    print(f"SOC Accuracy (any-of vs AE):  {soc_acc_any:.4f} (over {len(soc_any_flags)} rows)")
else:
    print("SOC Accuracy (any-of vs AE): N/A")

print("\nSaved per-AE JSON:")
print("-", out_json)

# =========================
# Save RUN-LEVEL METRICS (AGGREGATED) as JSON
# =========================
metrics_payload = {
    "meta": {
        "dataset": DATASET_NAME,
        "model": LLM_API_NAME,
        "top_k": TOP_K,
        "max_rows": MAX_ROWS,
        "embedding_dim": EMB_DIM,
        "llm_temp": LLM_TEMP,
        "llm_token": LLM_TOKEN,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "per-AE results saved in OUTPUT_FILE_NAME.json; aggregated metrics saved here",
        "concurrency": CONCURRENCY,
        "max_retries": MAX_RETRIES,
    },
    "counts": {
        "n_samples": len(results),
        "n_LLT_code_eval": int(sum(mask_llt)),
        "n_PT_code_eval": int(sum(mask_pt)),
        "n_SOC_optA": len(soc_pairs_ae),
        "n_SOC_optB": len(soc_pairs_primary),
        "n_SOC_any": len(soc_any_flags),
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
        "LLT_code_acc": float(LLT_acc),
        "PT_code_acc": float(PT_acc),
        "SOC_acc_option_a": (None if soc_acc_vs_ae is None else float(soc_acc_vs_ae)),
        "SOC_acc_option_b": (None if soc_acc_vs_true_primary is None else float(soc_acc_vs_true_primary)),
        "SOC_acc_any_of_vs_AE": (None if soc_acc_any is None else float(soc_acc_any)),
    },
}

metrics_json_path = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_metrics_seed{RUN_SEED}.json"
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

print("[Saved metrics JSON]:", metrics_json_path)
