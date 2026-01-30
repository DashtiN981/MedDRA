# File: RAG_Model_PT_LLT_SOC_V3_fix.py
# RAG-based MedDRA coding with robust "Final answer" parsing, retry, and safe fallbacks.
# Author: Naghme Dashti (patched) 20 November 2025

import json, re, time, random
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import OpenAI

# =========================
# LLM client
# =========================
client = OpenAI(api_key="sk-aKGeEFMZB0gXEcE51FTc0A", base_url="http://pluto/v1/")

# =========================
# Params (CHANGED: tokens up, retry, thresholds)
# =========================
TOP_K        = 10
MAX_ROWS     = None
EMB_DIM      = 384
RUN_SEED     = 42            # Set with Seed
MAX_RETRY    = 1             # (NEW) One time retry if extraction is not valid
LLM_API_NAME = "Llama-3.3-70B-Instruct"
LLM_TEMP     = 0.0
LLM_TOKEN    = 600           # (CHANGED) Increase to prevent disconnection before Final answer
RF_IN_TEXT_THRESH = 92       # (NEW) Rapidfuzz threshold for matching LLT names within the output text

# =========================
# Dataset/paths (as before)
# =========================
DATASET_NAME     = "KI_Projekt_Mosaic_AE_Codierung_2024_07_03"
DATASET_EMB_NAME = "ae_embeddings_Mosaic"
LLT_DICTIONARY_NAME      = "LLT2_Code_English_25_0"
LLT_DICTIONARY_EMB_NAME  = "llt2_embeddings"
PT_DICTIONARY_NAME       = "PT2_SOC_25_0"
OUTPUT_FILE_NAME         = "Mosaic_output"

AE_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{DATASET_NAME}.csv"
AE_EMB_FILE  = f"/home/naghmedashti/MedDRA-LLM/embedding/{DATASET_EMB_NAME}.json"
LLT_CSV_FILE = f"/home/naghmedashti/MedDRA-LLM/data/{LLT_DICTIONARY_NAME}.csv"
LLT_EMB_FILE = f"/home/naghmedashti/MedDRA-LLM/embedding/{LLT_DICTIONARY_EMB_NAME}.json"
PT_CSV_FILE  = f"/home/naghmedashti/MedDRA-LLM/data/{PT_DICTIONARY_NAME}.csv"

random.seed(RUN_SEED)

# =========================
# Small utils
# =========================
def canon_code(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    s = str(x).strip()
    m = re.match(r"^(\d+)(?:\.0+)?$", s)
    if m: return m.group(1)
    m2 = re.search(r"(\d{3,})", s)
    return (m2.group(1) if m2 else (s or None))

def norm_text(s):
    if s is None: return ""
    return " ".join(str(s).strip().casefold().split())

def clean_model_term(s):
    if not isinstance(s,str): return ""
    t = s.strip()
    t = re.sub(r"^\s*final\s*answer\s*[:\-]\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^\s*(?:[-–—•·*]+|\(?\d+\)?[.)]|[A-Za-z]\)|\d+\s*-\s*)\s*", "", t)
    t = t.strip().strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# (NEW) robust extractor for Final answer with fallbacks
def extract_final_answer(text, candidate_terms):
    """
    1) Regex-catch the final answer line (case-insensitive).
    2) If missing, choose the best candidate whose name appears in text (RapidFuzz).
    3) Else return None (caller may fallback to Top-1 semantic).
    """
    if not text:
        return None, "no_text"

    # 1) regex over whole text, take the LAST match
    pattern = re.compile(r"final\s*answer\s*[:\-]\s*<?\s*([^\n<>]{1,200}?)\s*>?\s*$",
                         flags=re.I|re.S)
    matches = list(pattern.finditer(text))
    if matches:
        ans = clean_model_term(matches[-1].group(1))
        if ans:
            return ans, "ok_final"

    # 2) in-text candidate match by RapidFuzz (token_set_ratio)
    if candidate_terms:
        cand = process.extractOne(
            text, candidate_terms, scorer=fuzz.token_set_ratio,
            score_cutoff=RF_IN_TEXT_THRESH
        )
        if cand:
            return cand[0], "fallback_intext"

    return None, "no_final"

# =========================
# Load data/dicts/embeddings
# =========================
ae_full = pd.read_csv(AE_CSV_FILE, sep=";", encoding="latin1")
ae_keep = [c for c in ["Original_Term_aufbereitet","ZB_LLT_Code","ZB_SOC_Code"] if c in ae_full.columns]
ae_df = ae_full[ae_keep].dropna(subset=["Original_Term_aufbereitet","ZB_LLT_Code"]).reset_index(drop=True)

llt_df = pd.read_csv(LLT_CSV_FILE, sep=";", encoding="latin1")[["LLT_Code","LLT_Term","PT_Code"]]
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
pt_meta = (pt_df.dropna(subset=["PT_Code"]).drop_duplicates(subset=["PT_Code"])
           .set_index("PT_Code")[["PT_Term"]].to_dict(orient="index"))

pt_code_to_soc_all = (
    pt_df.dropna(subset=["PT_Code","SOC_Code"])
         .groupby("PT_Code")["SOC_Code"]
         .apply(lambda s: sorted(set(x for x in s if x is not None)))
         .to_dict()
)

pt_code_to_primary_soc = {}
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

term_norm_to_llt = {}
for _, r in llt_df.iterrows():
    term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

def term_to_llt_code(pred_term, allow_fuzzy=True):
    t = norm_text(pred_term)
    if not t: return None
    if t in term_norm_to_llt: return term_norm_to_llt[t]
    for piece in re.split(r"[;,/]+", t):
        p = piece.strip()
        if p and p in term_norm_to_llt:
            return term_norm_to_llt[p]
    if allow_fuzzy:
        best = process.extractOne(t, list(term_norm_to_llt.keys()),
                                  scorer=fuzz.ratio, score_cutoff=94)
        return term_norm_to_llt[best[0]] if best else None
    return None

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

# =========================
# Main
# =========================
results = []

def build_candidates(ae_text):
    ae_emb = ae_emb_dict.get(ae_text, ae_emb_dict.get(norm_text(ae_text)))
    if ae_emb is not None:
        ae_norm = np.linalg.norm(ae_emb)
        sims = []
        for llt_term, llt_emb in llt_emb_dict.items():
            denom = ae_norm * np.linalg.norm(llt_emb)
            score = float(np.dot(ae_emb, llt_emb) / denom) if denom else 0.0
            sims.append((llt_term, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        cands = [t for t,_ in sims[:TOP_K]]
        top1 = cands[0] if cands else None
        return cands, top1, "dense"
    else:
        hits = process.extract(ae_text, llt_terms_all, scorer=fuzz.token_set_ratio,
                               limit=TOP_K, score_cutoff=0)
        cands = [t for (t,score,idx_) in hits]
        if len(cands) < TOP_K:
            extra = random.sample(llt_terms_all, k=min(TOP_K-len(cands), len(llt_terms_all)))
            cands += extra
        return cands[:TOP_K], (cands[0] if cands else None), "lexical"

def make_prompt(ae_text, candidate_terms, strict=False):
    base = (
        "You are a medical coding assistant.\n"
        f'Adverse Event (AE): "{ae_text}"\n\n'
        "Candidate MedDRA LLT terms:\n" + "\n".join(f"- {t}" for t in candidate_terms) + "\n\n"
    )
    if strict:
        # (NEW) دستور سختگیر برای retry
        return (base +
                "Respond ONLY with a single line: 'Final answer: <LLT_TERM>'\n"
                "No explanation. No extra text.")
    else:
        return (base +
                "First write at most 2–3 short sentences of reasoning (≤60 words total).\n"
                "Then on a NEW line, output exactly:\n"
                "Final answer: <LLT_TERM>\n")

for idx, row in ae_df.iterrows():
    ae_text = str(row["Original_Term_aufbereitet"])
    true_LLT_Code = canon_code(row["ZB_LLT_Code"])
    true_LLT_term = llt_code_to_term.get(true_LLT_Code)
    true_PT_Code  = llt_to_pt.get(true_LLT_Code)
    true_PT_term  = pt_meta.get(true_PT_Code, {}).get("PT_Term") if true_PT_Code else None

    true_SOC_Code_AE_raw = None
    if "ZB_SOC_Code" in ae_df.columns:
        v = row.get("ZB_SOC_Code")
        if pd.notna(v): true_SOC_Code_AE_raw = canon_code(v)

    candidate_terms, top1_dense, cand_mode = build_candidates(ae_text)
    if true_LLT_term and true_LLT_term not in candidate_terms:
        candidate_terms.append(true_LLT_term)
    random.shuffle(candidate_terms)

    parse_status = None
    answer_line  = None
    full_answer  = None

    # ========== First attempt ==========
    try:
        resp = client.chat.completions.create(
            model=LLM_API_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": make_prompt(ae_text, candidate_terms, strict=False)}
            ],
            temperature=LLM_TEMP,
            max_tokens=LLM_TOKEN
        )
        full_answer = resp.choices[0].message.content.strip()
        ans, tag = extract_final_answer(full_answer, candidate_terms)
        answer_line, parse_status = ans, tag
    except Exception as e:
        parse_status = f"api_error:{e}"

    # ========== Retry if needed ==========
    if (answer_line is None) and (MAX_RETRY > 0):
        try:
            resp2 = client.chat.completions.create(
                model=LLM_API_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful medical coding assistant."},
                    {"role": "user", "content": make_prompt(ae_text, candidate_terms, strict=True)}
                ],
                temperature=LLM_TEMP,
                max_tokens=128
            )
            full2 = resp2.choices[0].message.content.strip()
            ans2, tag2 = extract_final_answer(full2, candidate_terms)
            if ans2:
                answer_line, parse_status = ans2, "retry_ok"
                full_answer = full2  # نگه‌دار آخرین پاسخ
        except Exception as e2:
            parse_status = (parse_status or "") + f"|retry_error:{e2}"

    # ========== Final fallback: Top-1 dense ==========
    fallback_used = None
    if answer_line is None:
        if top1_dense:
            answer_line = top1_dense
            parse_status = (parse_status or "") + "|fallback_top1"
            fallback_used = "top1_dense"
        else:
            answer_line = ""   # هیچ چیزی نشد

    # Map predicted term -> codes
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

    results.append({
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
        "LLT_fuzzy_score": float(LLT_fuzzy_score),
        "LLT_fuzzy_match": bool(LLT_fuzzy_match),
        "exact_PT_match": exact_PT_match,
        "PT_fuzzy_score": float(PT_fuzzy_score),
        "PT_fuzzy_match": bool(PT_fuzzy_match),

        # (NEW) diagnostics
        "parse_status": parse_status,
        "candidate_mode": cand_mode,
        "fallback_used": fallback_used,

        "model_output": full_answer
    })

    print(f"[{idx}] AE: {ae_text}")
    print(f"   parse={parse_status}  |  pred='{pred_LLT_term_std}'\n")

# ----- Save per-AE -----
out_json = f"/home/naghmedashti/MedDRA-LLM/RAG_Models/{OUTPUT_FILE_NAME}_seed{RUN_SEED}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved per-AE:", out_json)

# ====== Eval (همانند قبل) ======
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

LLT_code_acc = (sum(1 for i,r in enumerate(results) if mask_llt[i] and r["true_LLT_Code"]==r["pred_LLT_Code"]) / max(1,sum(mask_llt))) if results else 0.0
PT_code_acc  = (sum(1 for i,r in enumerate(results) if mask_pt[i]  and r["true_PT_Code"]==r["pred_PT_Code"])  / max(1,sum(mask_pt)))  if results else 0.0

def _both_present(pairs):
    return [(a,b) for (a,b) in pairs if (a is not None and b is not None)]

soc_pairs_ae = _both_present([(r.get("true_SOC_Code"), r.get("pred_SOC_Code")) for r in results])
soc_acc_vs_ae = (sum(int(a==b) for a,b in soc_pairs_ae)/len(soc_pairs_ae)) if soc_pairs_ae else None

soc_pairs_primary = _both_present([(r.get("true_SOC_Code_primary"), r.get("pred_SOC_Code")) for r in results])
soc_acc_vs_true_primary = (sum(int(a==b) for a,b in soc_pairs_primary)/len(soc_pairs_primary)) if soc_pairs_primary else None

soc_any_flags = []
for r in results:
    ae_soc = r.get("true_SOC_Code")
    pred_all = r.get("pred_SOC_codes_all") or []
    if ae_soc is not None and pred_all:
        soc_any_flags.append(int(ae_soc in pred_all))
soc_acc_any = (sum(soc_any_flags)/len(soc_any_flags)) if soc_any_flags else None

metrics_payload = {
    "meta": {
        "dataset": DATASET_NAME,
        "model": LLM_API_NAME,
        "top_k": TOP_K,
        "embedding_dim": EMB_DIM,
        "llm_temp": LLM_TEMP,
        "llm_token": LLM_TOKEN,
        "seed": RUN_SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
