"""
File Name: rag_prompting_reasoning_v2_parallel_PT_SOC_multiseed.py
Based on: rag_prompting_reasoning_v2.py  (Naghme Dashti / July 2025)
PATCH: async parallel + PT/SOC mapping + multi-seed runs
IMPORTANT: keeps v2 logic (prompt + extract_final_term + top_terms ranking).
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
# Local OpenAI-compatible LLM API  (same as v2 but async)
# =========================
client = AsyncOpenAI(
    api_key="sk-aKGeEFMZB0gXEcE51FTc0A",
    base_url="http://pluto/v1/"
)

# =========================
# Parameters (keep v2 behavior)
# =========================
TOP_K = 30
MAX_ROWS = None
EMB_DIM = 384

AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings_Dauno.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
AE_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv"
LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"

# NEW: PT dictionary for SOC mapping
PT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"

# LLM params (keep v2)
LLM_API_NAME = "nvidia-llama-3.3-70b-instruct-fp8"   # or your fp8 model  Llama-3.3-70B-Instruct
LLM_TEMP = 0.2
LLM_TOKEN = 300

# Parallelism
CONCURRENCY = 50
MAX_RETRIES = 4
sem = asyncio.Semaphore(CONCURRENCY)

# Multi-seed
RUN_SEEDS = [42, 43, 44]   # <- هرچندتا خواستی
SHUFFLE_CANDIDATES = False  # اگر True کنی، seed معنا پیدا می‌کنه (اختیاری)

# Output base name
OUTPUT_PREFIX = "rag_prompting_reasoning_v2_parallel_PT_SOC"


# =========================
# Helpers (same safe mapping logic as v4)
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

# v2 similarity function name was cosine_similarity (shadowing sklearn). keep that style:
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

# v2 extraction logic (UNCHANGED)
def extract_final_term(answer_text, candidate_terms):
    for line in answer_text.splitlines():
        if "final answer:" in line.lower():
            return line.split(":")[-1].strip()
    # Try matching known terms from bottom up
    for line in reversed(answer_text.splitlines()):
        for term in candidate_terms:
            if term.lower() in line.lower():
                return term
    return answer_text.strip().split("\n")[-1].strip()

async def llm_call_with_retry(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a medical coding assistant."},
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
# Load Data (v2 + PT mapping)
# =========================
ae_full = pd.read_csv(AE_CSV_FILE, sep=";", encoding="latin1")
ae_keep = [c for c in ["Original_Term_aufbereitet", "ZB_LLT_Code", "ZB_SOC_Code"] if c in ae_full.columns]
ae_df = ae_full[ae_keep].dropna(subset=["Original_Term_aufbereitet", "ZB_LLT_Code"]).reset_index(drop=True)

llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]].dropna().reset_index(drop=True)
llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
llt_to_pt        = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))

# PT/SOC mapping (same as v4)
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

# Term->Code mapping (for code-based eval, like v4)
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
# Load Embeddings (v2 style)
# =========================
with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_embeddings = json.load(f)

with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_embeddings = json.load(f)

llt_emb_dict = {term: np.array(embedding) for term, embedding in llt_embeddings.items()}

# Row limiting (v2)
if isinstance(MAX_ROWS, int) and MAX_ROWS > 0:
    ae_df = ae_df.iloc[:MAX_ROWS].reset_index(drop=True)
else:
    ae_df = ae_df.reset_index(drop=True)


# =========================
# One row processor (async) — preserves v2 ranking + prompt + extraction
# =========================
async def process_one_row(idx: int, row) -> dict:
    ae_text = row["Original_Term_aufbereitet"]
    true_code = canon_code(row["ZB_LLT_Code"])
    if true_code not in llt_code_to_term:
        return {"idx": idx, "AE_text": ae_text, "skip_reason": "true_code_not_in_dict"}

    true_term = llt_code_to_term[true_code]

    ae_emb = ae_embeddings.get(ae_text)
    if ae_emb is None:
        return {"idx": idx, "AE_text": ae_text, "skip_reason": "missing_ae_embedding"}

    # v2 similarity loop (UNCHANGED)
    similarities = []
    for term, emb in llt_emb_dict.items():
        sim = cosine_similarity(ae_emb, emb)
        similarities.append((term, sim))

    top_terms = sorted(similarities, key=lambda x: x[1], reverse=True)[:TOP_K]
    top_terms = [term for term, _ in top_terms]

    # Optional shuffle for seed diversity (OFF by default)
    if SHUFFLE_CANDIDATES:
        random.shuffle(top_terms)

    # v2 prompt (UNCHANGED content)
    prompt = (
        f"You are a medical coding assistant using the MedDRA terminology.\n"
        f"Here is an Adverse Event (AE): \"{ae_text}\"\n"
        f"You are given a list of candidate MedDRA LLT terms. Your task is to reason step-by-step and select the most appropriate LLT.\n"
        f"1. Analyze the AE text and extract relevant clinical keywords.\n"
        f"2. Compare those keywords with candidate LLT terms.\n"
        f"3. Eliminate unrelated terms.\n"
        f"4. Select the LLT that best matches the AE context.\n"
        f"5. Final answer: respond with only the final LLT term.\n\n"
        f"Candidate LLTs:\n" +
        "\n".join(f"- {term}" for term in top_terms)
    )

    answer = await llm_call_with_retry(prompt)

    if answer.startswith("__ERROR__:"):
        return {
            "idx": idx,
            "AE_text": ae_text,
            "true_term": true_term,
            "predicted": None,
            "exact_match": False,
            "fuzzy_score": 0.0,
            "fuzzy_match": False,
            "model_output": answer,
        }

    # v2 extraction (UNCHANGED)
    answer_line = extract_final_term(answer, top_terms)

    exact_match = (answer_line == true_term)
    fuzzy_score = fuzz.ratio(answer_line.lower(), true_term.lower())
    fuzzy_match = fuzzy_score >= 90

    # NEW: mapping to code/PT/SOC (added, no effect on v2 extraction)
    pred_LLT_Code = term_to_llt_code(answer_line, allow_fuzzy=True)
    true_LLT_Code = true_code

    true_PT_Code = llt_to_pt.get(true_LLT_Code)
    pred_PT_Code = llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None

    true_PT_term = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None
    pred_PT_term = pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None

    # AE SOC if available
    true_SOC_Code_AE_raw = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v):
            true_SOC_Code_AE_raw = canon_code(v)

    true_SOC_primary = pt_code_to_primary_soc.get(true_PT_Code) if true_PT_Code else None
    pred_SOC_primary = pt_code_to_primary_soc.get(pred_PT_Code) if pred_PT_Code else None
    true_SOC_for_optA = true_SOC_Code_AE_raw if true_SOC_Code_AE_raw is not None else true_SOC_primary

    true_SOC_all = pt_code_to_soc_all.get(true_PT_Code, []) if true_PT_Code else []
    pred_SOC_all = pt_code_to_soc_all.get(pred_PT_Code, []) if pred_PT_Code else []

    # PT match metrics (extra)
    exact_PT_match = (true_PT_term is not None) and (pred_PT_term is not None) and (pred_PT_term == true_PT_term)
    if true_PT_term and pred_PT_term:
        PT_fuzzy_score = fuzz.ratio(pred_PT_term.lower(), true_PT_term.lower())
        PT_fuzzy_match = PT_fuzzy_score >= 90
    else:
        PT_fuzzy_score = 0.0
        PT_fuzzy_match = False

    return {
        "idx": idx,
        "AE_text": ae_text,

        # v2 fields (kept)
        "true_term": true_term,
        "predicted": answer_line,
        "exact_match": exact_match,
        "fuzzy_score": fuzzy_score,
        "fuzzy_match": fuzzy_match,

        # NEW mapping fields
        "true_LLT_Code": true_LLT_Code,
        "pred_LLT_Code": pred_LLT_Code,
        "true_PT_Code": true_PT_Code,
        "pred_PT_Code": pred_PT_Code,
        "true_PT_term": true_PT_term,
        "pred_PT_term": pred_PT_term,

        "true_SOC_Code_AE_raw": true_SOC_Code_AE_raw,
        "true_SOC_Code": true_SOC_for_optA,         # Option A comparator
        "pred_SOC_Code": pred_SOC_primary,
        "true_SOC_Code_primary": true_SOC_primary,  # Option B comparator
        "true_SOC_codes_all": true_SOC_all,
        "pred_SOC_codes_all": pred_SOC_all,

        "exact_PT_match": exact_PT_match,
        "PT_fuzzy_score": PT_fuzzy_score,
        "PT_fuzzy_match": PT_fuzzy_match,

        "model_output": answer,
    }


async def run_parallel():
    tasks = [asyncio.create_task(process_one_row(i, r)) for i, r in ae_df.iterrows()]
    out = await asyncio.gather(*tasks)
    out.sort(key=lambda d: d.get("idx", 10**9))
    return out


# =========================
# SOC accuracy helpers
# =========================
def _both_present(pairs):
    return [(a, b) for (a, b) in pairs if (a is not None and b is not None)]


# =========================
# Multi-seed runner
# =========================
for RUN_SEED in RUN_SEEDS:
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)

    t0 = time.time()
    results = asyncio.run(run_parallel())
    dt = time.time() - t0

    # keep only usable rows (not skipped, not error)
    usable = [r for r in results if r.get("skip_reason") is None and not str(r.get("model_output","")).startswith("__ERROR__")]

    # Save per-AE JSON (keep v2 naming style but add seed)
    out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/v2_Results/{OUTPUT_PREFIX}_Dauno_seed{RUN_SEED}.json"
    # remove idx for saving
    save_payload = []
    for r in results:
        rr = dict(r)
        rr.pop("idx", None)
        save_payload.append(rr)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] seed={RUN_SEED} finished in {dt:.1f}s | total={len(results)} | usable={len(usable)} | concurrency={CONCURRENCY}")
    print("Saved per-AE JSON:", out_json)

    # =========================
    # Evaluation (v2 + PT/SOC extras)
    # =========================
    y_true = [r["true_term"] for r in usable]
    y_pred = [r["predicted"] for r in usable]
    y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in usable]

    print("\nEvaluation Report (Exact Match) — v2:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nEvaluation Report (Fuzzy Match) — v2:")
    print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

    acc = accuracy_score(y_true, y_pred) if usable else 0.0
    f1m = f1_score(y_true, y_pred, average="macro") if usable else 0.0
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if usable else 0.0
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if usable else 0.0
    fuzzy_acc = sum(r["fuzzy_match"] for r in usable) / max(1, len(usable))

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1m:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"Fuzzy Match Accuracy: {fuzzy_acc:.4f}")

    # PT term acc
    z_true = [r.get("true_PT_term") or "" for r in usable]
    z_pred = [r.get("pred_PT_term") or "" for r in usable]
    pt_acc = accuracy_score(z_true, z_pred) if any(z_true) else 0.0
    pt_fuzzy_acc = sum(1 for r in usable if r.get("PT_fuzzy_match")) / max(1, len(usable))

    print(f"\nPT Term Accuracy (exact): {pt_acc:.4f}")
    print(f"PT Term Accuracy (fuzzy): {pt_fuzzy_acc:.4f}")

    # SOC accs
    soc_pairs_ae = _both_present([(r.get("true_SOC_Code"), r.get("pred_SOC_Code")) for r in usable])
    soc_pairs_primary = _both_present([(r.get("true_SOC_Code_primary"), r.get("pred_SOC_Code")) for r in usable])

    soc_acc_optA = (sum(int(a == b) for a, b in soc_pairs_ae) / len(soc_pairs_ae)) if soc_pairs_ae else None
    soc_acc_optB = (sum(int(a == b) for a, b in soc_pairs_primary) / len(soc_pairs_primary)) if soc_pairs_primary else None

    soc_any_flags = []
    for r in usable:
        ae_soc = r.get("true_SOC_Code")
        pred_all = r.get("pred_SOC_codes_all") or []
        if ae_soc is not None and pred_all:
            soc_any_flags.append(int(ae_soc in pred_all))
    soc_acc_any = (sum(soc_any_flags) / len(soc_any_flags)) if soc_any_flags else None

    print("\nSOC Accuracy Option A (primary(pred PT) vs AE_SOC/fallback):", "N/A" if soc_acc_optA is None else f"{soc_acc_optA:.4f}")
    print("SOC Accuracy Option B (primary(pred PT) vs primary(true PT)):", "N/A" if soc_acc_optB is None else f"{soc_acc_optB:.4f}")
    print("SOC Accuracy Any-of vs AE:", "N/A" if soc_acc_any is None else f"{soc_acc_any:.4f}")

    # Save run-level metrics JSON
    metrics_payload = {
        "meta": {
            "script": "rag_prompting_reasoning_v2_parallel_PT_SOC_multiseed.py",
            "seed": RUN_SEED,
            "model": LLM_API_NAME,
            "top_k": TOP_K,
            "max_rows": MAX_ROWS,
            "embedding_dim": EMB_DIM,
            "llm_temp": LLM_TEMP,
            "llm_token": LLM_TOKEN,
            "concurrency": CONCURRENCY,
            "shuffle_candidates": SHUFFLE_CANDIDATES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "notes": "v2 logic preserved; added parallel + PT/SOC mapping + multi-seed outputs.",
        },
        "counts": {
            "n_total": len(results),
            "n_usable": len(usable),
            "n_skipped_missing_embedding": sum(1 for r in results if r.get("skip_reason") == "missing_ae_embedding"),
            "n_skipped_true_code_not_in_dict": sum(1 for r in results if r.get("skip_reason") == "true_code_not_in_dict"),
            "n_soc_optA": len(soc_pairs_ae),
            "n_soc_optB": len(soc_pairs_primary),
            "n_soc_any": len(soc_any_flags),
        },
        "metrics": {
            "LLT_term_acc_exact": float(acc),
            "LLT_term_acc_fuzzy": float(fuzzy_acc),
            "LLT_precision_macro": float(precision),
            "LLT_recall_macro": float(recall),
            "LLT_f1_macro": float(f1m),
            "PT_term_acc_exact": float(pt_acc),
            "PT_term_acc_fuzzy": float(pt_fuzzy_acc),
            "SOC_acc_option_a": (None if soc_acc_optA is None else float(soc_acc_optA)),
            "SOC_acc_option_b": (None if soc_acc_optB is None else float(soc_acc_optB)),
            "SOC_acc_any_of_vs_AE": (None if soc_acc_any is None else float(soc_acc_any)),
        },
    }

    metrics_json_path = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/v2_Results/{OUTPUT_PREFIX}_Dauno_metrics_seed{RUN_SEED}.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("Saved metrics JSON:", metrics_json_path)
