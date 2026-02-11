from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from threading import Lock
import asyncio

import numpy as np
import pandas as pd
from openai import OpenAI, AsyncOpenAI
from rapidfuzz import fuzz, process

from config import load_config


cfg = load_config()


@dataclass
class RAGResources:
    client: OpenAI
    llt_df: pd.DataFrame
    pt_df: pd.DataFrame
    llt_emb_dict: dict[str, np.ndarray]
    llt_emb_items: list[tuple[str, np.ndarray]]
    llt_terms_all: list[str]
    ae_emb_dict: dict[str, np.ndarray]
    llt_code_to_term: dict[str, str]
    llt_to_pt: dict[str, str]
    soc_code_to_term: dict[str, str]
    pt_meta: dict[str, dict[str, str]]
    pt_code_to_soc_all: dict[str, list[str]]
    pt_code_to_primary_soc: dict[str, str | None]
    term_norm_to_llt: dict[str, str]
    initialized_at: float


_RESOURCES: RAGResources | None = None
_LOCK = Lock()


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


def get_resources() -> RAGResources:
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES

    with _LOCK:
        if _RESOURCES is not None:
            return _RESOURCES

        if not cfg.api_key:
            raise ValueError(
                "Missing API key. Please set MEDDRA_LLM_API_KEY in the environment "
                "before running this script."
            )

        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        # Load AE / LLT / PT (+ SOC primary mapping)
        llt_df = pd.read_csv(cfg.llt_csv_file, sep=";", encoding="latin1")[["LLT_Code", "LLT_Term", "PT_Code"]]
        llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
        llt_df["PT_Code"] = llt_df["PT_Code"].map(canon_code)
        llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

        pt_header = pd.read_csv(cfg.pt_csv_file, sep=";", encoding="latin1", nrows=0).columns
        pt_cols = [c for c in ["PT_Code", "PT_Term", "SOC_Code", "SOC_Term", "Ist_Primary_SOC", "Primary_SOC_Code"] if c in pt_header]
        pt_df = pd.read_csv(cfg.pt_csv_file, sep=";", encoding="latin1")[pt_cols].copy()

        # normalize codes to strings
        pt_df["PT_Code"] = pt_df["PT_Code"].map(canon_code)
        pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)
        if "Primary_SOC_Code" in pt_df.columns:
            pt_df["Primary_SOC_Code"] = pt_df["Primary_SOC_Code"].map(canon_code)

        # normalize Ist_Primary_SOC
        if "Ist_Primary_SOC" in pt_df.columns:
            pt_df["Ist_Primary_SOC_norm"] = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper()
        else:
            pt_df["Ist_Primary_SOC_norm"] = ""

        # Lookups
        llt_code_to_term = dict(zip(llt_df["LLT_Code"], llt_df["LLT_Term"]))
        llt_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))
        soc_code_to_term = dict(zip(pt_df["SOC_Code"], pt_df["SOC_Term"])) if "SOC_Term" in pt_df.columns else {}
        pt_meta = pt_df.dropna(subset=["PT_Code"]).drop_duplicates(subset=["PT_Code"])
        pt_meta = pt_meta.set_index("PT_Code")[["PT_Term"]].to_dict(orient="index")

        # All SOCs per PT
        pt_code_to_soc_all: dict[str, list[str]] = (
            pt_df.dropna(subset=["PT_Code", "SOC_Code"])
            .groupby("PT_Code")["SOC_Code"]
            .apply(lambda s: sorted(set(x for x in s if x is not None)))
            .to_dict()
        )

        # Primary SOC per PT (priority: Y row > unique Primary_SOC_Code > single SOC > None)
        pt_code_to_primary_soc: dict[str, str | None] = {}
        has_primary_soc_code_col = "Primary_SOC_Code" in pt_df.columns

        for ptc, grp in pt_df.dropna(subset=["PT_Code"]).groupby("PT_Code"):
            primary = None

            # case 1: explicit Y
            y_rows = grp[grp["Ist_Primary_SOC_norm"] == "Y"]
            if not y_rows.empty and pd.notna(y_rows.iloc[0].get("SOC_Code")):
                primary = canon_code(y_rows.iloc[0]["SOC_Code"])
            else:
                # case 2: explicit Primary_SOC_Code unique among rows
                if has_primary_soc_code_col:
                    prim_vals = [canon_code(v) for v in grp["Primary_SOC_Code"].dropna().tolist()]
                    uniq = sorted(set(v for v in prim_vals if v is not None))
                    if len(uniq) == 1:
                        primary = uniq[0]
                # case 3: only one SOC for this PT
                if primary is None:
                    all_socs = pt_code_to_soc_all.get(ptc, [])
                    if len(all_socs) == 1:
                        primary = all_socs[0]

            pt_code_to_primary_soc[ptc] = primary

        # QC summary
        num_pts = len(pt_code_to_soc_all)
        num_single = sum(1 for _, lst in pt_code_to_soc_all.items() if len(lst) == 1)
        num_with_y = pt_df[pt_df["Ist_Primary_SOC_norm"] == "Y"]["PT_Code"].nunique()
        num_with_primary_code = pt_df.dropna(subset=["Primary_SOC_Code"])["PT_Code"].nunique() if has_primary_soc_code_col else 0
        num_undefined = sum(1 for _, v in pt_code_to_primary_soc.items() if v is None)
        print(
            f"[PT/SOC QC] PTs:{num_pts} | with 'Y':{num_with_y} | with Primary_SOC_Code:{num_with_primary_code} | "
            f"single-SOC:{num_single} | undefined-primary:{num_undefined}"
        )

        # Term -> LLT_Code (normalized)
        term_norm_to_llt: dict[str, str] = {}
        for _, r in llt_df.iterrows():
            term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

        # Load embeddings
        with open(cfg.ae_emb_file, "r", encoding="latin1") as f:
            ae_emb_raw = json.load(f)
        with open(cfg.llt_emb_file, "r", encoding="latin1") as f:
            llt_emb_raw = json.load(f)

        # AE embs keyed by AE text and its normalized form
        ae_emb_dict: dict[str, np.ndarray] = {}
        for k, v in ae_emb_raw.items():
            ae_emb_dict[k] = np.array(v)
            ae_emb_dict[norm_text(k)] = np.array(v)

        # LLT embs keyed by LLT term (string)
        llt_emb_dict = {k: np.array(v) for k, v in llt_emb_raw.items()}
        llt_emb_items = list(llt_emb_dict.items())
        llt_terms_all = list(llt_df["LLT_Term"])

        random.seed(cfg.run_seed)

        _RESOURCES = RAGResources(
            client=client,
            llt_df=llt_df,
            pt_df=pt_df,
            llt_emb_dict=llt_emb_dict,
            llt_emb_items=llt_emb_items,
            llt_terms_all=llt_terms_all,
            ae_emb_dict=ae_emb_dict,
            llt_code_to_term=llt_code_to_term,
            llt_to_pt=llt_to_pt,
            soc_code_to_term=soc_code_to_term,
            pt_meta=pt_meta,
            pt_code_to_soc_all=pt_code_to_soc_all,
            pt_code_to_primary_soc=pt_code_to_primary_soc,
            term_norm_to_llt=term_norm_to_llt,
            initialized_at=time.time(),
        )

        return _RESOURCES


def term_to_llt_code(pred_term: str, term_norm_to_llt: dict[str, str], allow_fuzzy: bool = True) -> str | None:
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
        # CHANGED: use RapidFuzz instead of difflib.get_close_matches
        best = process.extractOne(t, list(term_norm_to_llt.keys()), scorer=fuzz.ratio, score_cutoff=94)
        return term_norm_to_llt[best[0]] if best else None
    return None


def retrieve_candidates(ae_text: str, resources: RAGResources, top_k: int) -> tuple[list[str], bool]:
    # Build candidates from embeddings (fallback: rapidfuzz + random)  # CHANGED: comment
    ae_emb = resources.ae_emb_dict.get(ae_text, resources.ae_emb_dict.get(norm_text(ae_text)))
    if ae_emb is not None:
        ae_norm = np.linalg.norm(ae_emb)
        sims = []
        for llt_term, llt_emb in resources.llt_emb_items:
            denom = ae_norm * np.linalg.norm(llt_emb)
            score = float(np.dot(ae_emb, llt_emb) / denom) if denom else 0.0
            sims.append((llt_term, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        candidate_terms = [t for t, _ in sims[:top_k]]
        return candidate_terms, False

    # CHANGED: use RapidFuzz for text fallback (if AE embedding missing)
    hits = process.extract(ae_text, resources.llt_terms_all, scorer=fuzz.token_set_ratio, limit=top_k, score_cutoff=0)
    cand = [t for (t, score, idx_) in hits]
    if len(cand) < top_k:
        extra = random.sample(resources.llt_terms_all, k=min(top_k - len(cand), len(resources.llt_terms_all)))
        cand += extra
    candidate_terms = cand[:top_k]
    return candidate_terms, True


def build_prompt(ae_text: str, candidates: list[str], resources: RAGResources) -> str:
    prompt = (
        "You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term."
        f"\nHere is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        "Here is a list of candidate LLT terms:\n" + "\n".join(f"- {term}" for term in candidates) +
        "\n\nPlease analyze the AE and list, and first provide a short reasoning."
        "\nThen, on a separate line, write the best matching LLT in this format:"
        "\nFinal answer: <LLT_TERM>"
    )
    return prompt


def call_llm(prompt: str, resources: RAGResources) -> str:
    resp = resources.client.chat.completions.create(
        model=cfg.llm_api_name,
        messages=[
            {"role": "system", "content": "You are a helpful medical coding assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg.llm_temp,
        max_tokens=cfg.llm_token,
    )
    return resp.choices[0].message.content.strip()


def postprocess_prediction(model_text: str, resources: RAGResources) -> dict:
    last_line = model_text.split("Final answer:")[-1] if "Final answer:" in model_text else model_text.split("\n")[-1]
    answer_line = clean_model_term(last_line)

    pred_LLT_Code = term_to_llt_code(answer_line, resources.term_norm_to_llt, allow_fuzzy=True)
    pred_PT_Code = resources.llt_to_pt.get(pred_LLT_Code) if pred_LLT_Code else None
    pred_PT_term = resources.pt_meta.get(pred_PT_Code, {}).get("PT_Term") if pred_PT_Code else None
    pred_LLT_term_std = resources.llt_code_to_term.get(pred_LLT_Code, answer_line)

    pred_SOC_Code_primary = resources.pt_code_to_primary_soc.get(pred_PT_Code) if pred_PT_Code else None
    pred_SOC_codes_all = resources.pt_code_to_soc_all.get(pred_PT_Code, []) if pred_PT_Code else []

    pred_primary_soc_missing = (pred_PT_Code is not None and resources.pt_code_to_primary_soc.get(pred_PT_Code) is None)

    return {
        "pred_LLT_term": pred_LLT_term_std,
        "pred_LLT_Code": pred_LLT_Code,
        "pred_PT_term": pred_PT_term,
        "pred_PT_Code": pred_PT_Code,
        "pred_SOC_Code": pred_SOC_Code_primary,
        "pred_SOC_Term": resources.soc_code_to_term.get(pred_SOC_Code_primary) if pred_SOC_Code_primary else None,
        "pred_SOC_codes_all": pred_SOC_codes_all,
        "pred_primary_soc_missing": pred_primary_soc_missing,
        "model_output": model_text,
    }


def predict(ae_text: str, *, top_k: int | None = None) -> dict:
    start = time.time()
    resources = get_resources()
    k = top_k if top_k is not None else cfg.top_k

    candidates_retrieved, used_fallback = retrieve_candidates(ae_text, resources, k)
    prompt_candidates = candidates_retrieved.copy()
    random.shuffle(prompt_candidates)

    prompt = build_prompt(ae_text, prompt_candidates, resources)
    model_text = call_llm(prompt, resources)
    pred = postprocess_prediction(model_text, resources)

    latency_ms = int((time.time() - start) * 1000)

    return {
        "pred": pred,
        "candidates_100": prompt_candidates,
        "candidates_retrieved_100": candidates_retrieved,
        "top5_preview": candidates_retrieved[:5],
        "debug": {
            "latency_ms": latency_ms,
            "used_fallback": used_fallback,
        },
    }


if __name__ == "__main__":
    test_text = "Headache and nausea"
    first = predict(test_text)
    first_res_id = id(get_resources())
    second = predict(test_text)
    second_res_id = id(get_resources())
    print("cache reused:", first_res_id == second_res_id)


def batch_predict(ae_texts: list[str], *, top_k: int | None = None) -> list[dict]:
    """
    Batch wrapper around predict(). Must reuse cached resources (no reloading).
    """
    results = []
    for t in ae_texts:
        results.append(predict(str(t), top_k=top_k))
    return results


async def async_batch_predict(
    ae_texts: list[str],
    *,
    top_k: int | None = None,
    concurrency: int = 5,
) -> list[dict]:
    resources = get_resources()
    k = top_k if top_k is not None else cfg.top_k
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    async def _call_llm_with_retry(prompt: str) -> str:
        backoff = 0.25
        last_exc = None
        for _ in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=cfg.llm_api_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful medical coding assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=cfg.llm_temp,
                        max_tokens=cfg.llm_token,
                    )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2
        raise last_exc  # type: ignore[misc]

    async def _predict_one(ae_text: str) -> dict:
        start = time.time()
        candidates_retrieved, used_fallback = retrieve_candidates(ae_text, resources, k)
        prompt_candidates = candidates_retrieved.copy()
        random.shuffle(prompt_candidates)

        prompt = build_prompt(ae_text, prompt_candidates, resources)
        model_text = await _call_llm_with_retry(prompt)
        pred = postprocess_prediction(model_text, resources)
        latency_ms = int((time.time() - start) * 1000)

        return {
            "pred": pred,
            "candidates_100": prompt_candidates,
            "candidates_retrieved_100": candidates_retrieved,
            "top5_preview": candidates_retrieved[:5],
            "debug": {
                "latency_ms": latency_ms,
                "used_fallback": used_fallback,
            },
        }

    tasks = [_predict_one(str(t)) for t in ae_texts]
    return await asyncio.gather(*tasks)


def batch_predict_parallel(
    ae_texts: list[str],
    *,
    top_k: int | None = None,
    concurrency: int = 5,
) -> list[dict]:
    return asyncio.run(
        async_batch_predict(ae_texts, top_k=top_k, concurrency=concurrency)
    )
