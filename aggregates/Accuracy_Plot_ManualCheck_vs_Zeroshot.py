# -*- coding: utf-8 -*-
"""
RAG-based pipeline vs Zero-shot baseline

RAG (clinical validation):
  - LLT accuracy = CCR = mean(manual_check==True) across rows that have a manual flag
  - PT accuracy  = among manual_check==True rows: PT(pred_LLT) == PT(true_LLT)
  - SOC accuracy = among manual_check==True rows: SOC(pred) == SOC(true)
    where SOC(x) := primarySOC( PT( LLT(x) ) )

Zero-shot baseline (seeds):
  - LLT accuracy = exact match rate: pred_LLT_term == true_LLT_term (normalized)
  - PT accuracy  = PT(pred_LLT) == PT(true_LLT) across ALL rows
  - SOC accuracy = SOC(pred) == SOC(true) across ALL rows

Aggregation:
  - RAG: mean±std across 3 reviewers per dataset
  - Zero-shot: mean±std across 3 seeds per dataset

Plot:
  - 3 panels: LLT / PT / SOC
  - Okabe–Ito palette (colorblind-safe)
  - Error bars = std; ONLY mean printed on bars (2 decimals)
  - Saves only PNG
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process


# =========================
# CONFIG
# =========================

MANUAL_FILES = {
    "Dauno": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__full__Isabella.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__full__reviewer2.json",
    ],
    "Delta": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Delta_output__full__Isabella.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Delta_output__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Delta_output__full__reviewer2.json",
    ],
    "Mosaic": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Mosaic_output__full__Isabella.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Mosaic_output__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Mosaic_output__full__reviewer2.json",
    ],
}

ZEROSHOT_FILES = {
    "Dauno": [
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Dauno_output_zeroshot_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Dauno_output_zeroshot_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Dauno_output_zeroshot_seed44.json",
    ],
    "Delta": [
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Delta_output_zeroshot_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Delta_output_zeroshot_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Delta_output_zeroshot_seed44.json",
    ],
    "Mosaic": [
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Mosaic_output_zeroshot_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Mosaic_output_zeroshot_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/Mosaic_output_zeroshot_seed44.json",
    ],
}

LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
PT_CSV_FILE  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"

CSV_SEP = ";"
CSV_ENCODING = "latin1"
FUZZY_CUTOFF = 94

DATASET_ORDER = ["Mosaic", "Delta", "Dauno"]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_FIG_PNG = SCRIPT_DIR / "rag_vs_zeroshot_LLT_PT_SOC.png"


# =========================
# Helpers
# =========================

def canon_code(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    m = re.match(r"^(\d+)(?:\.0+)?$", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{3,})", s)
    if m2:
        return m2.group(1)
    return s or None

def clean_term(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def norm_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().casefold().split())

def norm_term(s: str) -> str:
    return norm_text(clean_term(s))

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_manual_flag(r: dict) -> Optional[bool]:
    for k in ["manual check", "manual_check", "clinical_correct", "clinically_correct", "manual"]:
        if k in r:
            v = r.get(k)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ["true", "1", "yes", "y"]
    return None

def extract_pred_term(r: dict) -> str:
    return (
        r.get("pred_llt_term")
        or r.get("pred_LLT_term")
        or r.get("predicted")
        or r.get("pred")
        or r.get("pred_term")
        or ""
    )

def extract_true_term(r: dict) -> str:
    return (
        r.get("true_LLT_term")
        or r.get("true_llt_term")
        or r.get("true_term")
        or r.get("true")
        or r.get("true_llt")
        or r.get("gold")
        or r.get("ground_truth")
        or ""
    )

def load_mappings(llt_csv: str, pt_csv: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Optional[str]]]:
    # LLT -> PT
    llt_df = pd.read_csv(llt_csv, sep=CSV_SEP, encoding=CSV_ENCODING)[["LLT_Code", "LLT_Term", "PT_Code"]].copy()
    llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
    llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
    llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

    term_norm_to_llt: Dict[str, str] = {}
    for _, row in llt_df.iterrows():
        if row["LLT_norm"] and row["LLT_Code"]:
            term_norm_to_llt.setdefault(row["LLT_norm"], row["LLT_Code"])

    llt_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))

    # PT -> primary SOC
    header = pd.read_csv(pt_csv, sep=CSV_SEP, encoding=CSV_ENCODING, nrows=0).columns
    pt_cols = [c for c in ["PT_Code", "SOC_Code", "Ist_Primary_SOC", "Primary_SOC_Code"] if c in header]
    pt_df = pd.read_csv(pt_csv, sep=CSV_SEP, encoding=CSV_ENCODING)[pt_cols].copy()

    pt_df["PT_Code"]  = pt_df["PT_Code"].map(canon_code)
    pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)
    if "Primary_SOC_Code" in pt_df.columns:
        pt_df["Primary_SOC_Code"] = pt_df["Primary_SOC_Code"].map(canon_code)

    if "Ist_Primary_SOC" in pt_df.columns:
        pt_df["Ist_Primary_SOC_norm"] = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper()
    else:
        pt_df["Ist_Primary_SOC_norm"] = ""

    pt_code_to_soc_all = (
        pt_df.dropna(subset=["PT_Code", "SOC_Code"])
             .groupby("PT_Code")["SOC_Code"]
             .apply(lambda s: sorted(set([x for x in s if x is not None])))
             .to_dict()
    )

    pt_primary_soc: Dict[str, Optional[str]] = {}
    has_primary_soc_code_col = "Primary_SOC_Code" in pt_df.columns

    for ptc, grp in pt_df.dropna(subset=["PT_Code"]).groupby("PT_Code"):
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
        pt_primary_soc[ptc] = primary

    return term_norm_to_llt, llt_to_pt, pt_primary_soc

def term_to_llt_code(term: str, term_norm_to_llt: Dict[str, str]) -> Optional[str]:
    t = norm_text(clean_term(term))
    if not t:
        return None
    if t in term_norm_to_llt:
        return term_norm_to_llt[t]
    for piece in re.split(r"[;,/]+", t):
        p = piece.strip()
        if p and p in term_norm_to_llt:
            return term_norm_to_llt[p]
    best = process.extractOne(t, list(term_norm_to_llt.keys()), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
    return term_norm_to_llt[best[0]] if best else None

def pt_of_term(term: str, term_norm_to_llt: Dict[str, str], llt_to_pt: Dict[str, str]) -> Optional[str]:
    llt_code = term_to_llt_code(term, term_norm_to_llt)
    return llt_to_pt.get(llt_code) if llt_code else None

def soc_of_term(term: str,
                term_norm_to_llt: Dict[str, str],
                llt_to_pt: Dict[str, str],
                pt_primary_soc: Dict[str, Optional[str]]) -> Optional[str]:
    pt = pt_of_term(term, term_norm_to_llt, llt_to_pt)
    return pt_primary_soc.get(pt) if pt else None


# =========================
# Metrics
# =========================

def compute_rag_metrics(rows: List[dict],
                        term_norm_to_llt: Dict[str, str],
                        llt_to_pt: Dict[str, str],
                        pt_primary_soc: Dict[str, Optional[str]]) -> Dict[str, float]:
    """
    RAG (clinical validation):
      - LLT_acc = CCR over rows that have a manual flag  (mean(manual_check==True))
      - PT_acc  = PT(pred) == PT(true) among manual_true rows  (kept as before)
      - SOC_acc = Option B among manual_true rows, computed like RAG.py:
                 primarySOC(PT(pred)) == primarySOC(PT(true))
                 evaluated ONLY on rows where both primary SOCs exist (_both_present)
    """
    manual_flags = []
    pt_match = []

    # --- Option B: evaluate only when both are present (RAG.py style)
    soc_pairs_primary = []  # list of (true_primary_soc, pred_primary_soc)

    for r in rows:
        mf = get_manual_flag(r)
        if mf is None:
            continue

        manual_flags.append(int(mf))

        if mf is not True:
            continue

        pred_term = extract_pred_term(r)
        true_term = extract_true_term(r)

        pred_pt = pt_of_term(pred_term, term_norm_to_llt, llt_to_pt)
        true_pt = pt_of_term(true_term, term_norm_to_llt, llt_to_pt)

        # PT accuracy (keep your previous rule: missing PT -> counts as 0)
        pt_match.append(int(pred_pt is not None and true_pt is not None and pred_pt == true_pt))

        # SOC Option B (RAG.py): compare primary(pred PT) vs primary(true PT)
        pred_soc_primary = pt_primary_soc.get(pred_pt) if pred_pt else None
        true_soc_primary = pt_primary_soc.get(true_pt) if true_pt else None

        # --- RAG.py logic: _both_present -> exclude missing mappings from denominator
        if pred_soc_primary is not None and true_soc_primary is not None:
            soc_pairs_primary.append((true_soc_primary, pred_soc_primary))

    llt_acc = float(np.mean(manual_flags)) if manual_flags else 0.0
    pt_acc  = float(np.mean(pt_match)) if pt_match else 0.0

    if soc_pairs_primary:
        soc_acc = float(np.mean([int(a == b) for a, b in soc_pairs_primary]))
    else:
        soc_acc = 0.0

    return {"LLT_acc": llt_acc, "PT_acc": pt_acc, "SOC_acc": soc_acc}



def compute_zeroshot_metrics(rows: List[dict],
                            term_norm_to_llt: Dict[str, str],
                            llt_to_pt: Dict[str, str],
                            pt_primary_soc: Dict[str, Optional[str]]) -> Dict[str, float]:
    """
    Zero-shot:
      - LLT_acc = exact match pred vs true (normalized)
      - PT_acc  = PT(pred) == PT(true) across all rows
      - SOC_acc = SOC(pred) == SOC(true) across all rows
    """
    llt_match = []
    pt_match = []
    soc_match = []

    for r in rows:
        pred_term = extract_pred_term(r)
        true_term = extract_true_term(r)

        # LLT exact (string-normalized)
        llt_match.append(int(norm_term(pred_term) != "" and norm_term(true_term) != "" and norm_term(pred_term) == norm_term(true_term)))

        pred_pt = pt_of_term(pred_term, term_norm_to_llt, llt_to_pt)
        true_pt = pt_of_term(true_term, term_norm_to_llt, llt_to_pt)
        pt_match.append(int(pred_pt is not None and true_pt is not None and pred_pt == true_pt))

        pred_soc = pt_primary_soc.get(pred_pt) if pred_pt else None
        true_soc = pt_primary_soc.get(true_pt) if true_pt else None
        soc_match.append(int(pred_soc is not None and true_soc is not None and pred_soc == true_soc))

    return {
        "LLT_acc": float(np.mean(llt_match)) if llt_match else 0.0,
        "PT_acc":  float(np.mean(pt_match)) if pt_match else 0.0,
        "SOC_acc": float(np.mean(soc_match)) if soc_match else 0.0,
    }


def mean_std(records: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    arr = np.array([r[key] for r in records], dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


# =========================
# Plotting
# =========================

def plot(df: pd.DataFrame):
    # Okabe–Ito (colorblind-safe)
    C_RAG = "#0072B2"
    C_ZS  = "#D55E00"

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=False)

    x = np.arange(len(df))
    width = 0.36
    cap = 4

    def add_labels(ax, xs, means):
        for xi, mu in zip(xs, means):
            ax.text(xi, mu + 0.02, f"{mu:.2f}",
                    ha="center", va="bottom", fontsize=11)

    def panel(ax, title, rag_mean, rag_std, zs_mean, zs_std):
        ax.bar(
            x - width/2, rag_mean, width,
            yerr=rag_std, capsize=cap,
            color=C_RAG,
            label="RAG-based pipeline (clinical validation)"
        )
        ax.bar(
            x + width/2, zs_mean, width,
            yerr=zs_std, capsize=cap,
            color=C_ZS,
            label="Zero-shot baseline"
        )

        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["dataset"].tolist(), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

        add_labels(ax, x - width/2, rag_mean)
        add_labels(ax, x + width/2, zs_mean)

    panel(
        axes[0], "LLT accuracy",
        df["LLT_rag_mean"].values, df["LLT_rag_std"].values,
        df["LLT_zs_mean"].values,  df["LLT_zs_std"].values
    )
    panel(
        axes[1], "PT accuracy",
        df["PT_rag_mean"].values, df["PT_rag_std"].values,
        df["PT_zs_mean"].values,  df["PT_zs_std"].values
    )
    panel(
        axes[2], "SOC accuracy",
        df["SOC_rag_mean"].values, df["SOC_rag_std"].values,
        df["SOC_zs_mean"].values,  df["SOC_zs_std"].values
    )

    # Title
    fig.suptitle("RAG-based pipeline vs Zero-shot baseline", fontsize=14, y=0.98)

    # Legend bottom centered
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(top=0.85, bottom=0.22)

    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    print(f"[Saved] {OUT_FIG_PNG}")


# =========================
# MAIN
# =========================

def main():
    term_norm_to_llt, llt_to_pt, pt_primary_soc = load_mappings(LLT_CSV_FILE, PT_CSV_FILE)

    # ---- RAG (manual reviewers): mean±std
    rag_summary = []
    for ds, files in MANUAL_FILES.items():
        per_rev = []
        for fp in files:
            rows = load_json(fp)
            per_rev.append(compute_rag_metrics(rows, term_norm_to_llt, llt_to_pt, pt_primary_soc))

        llt_mu, llt_sd = mean_std(per_rev, "LLT_acc")
        pt_mu,  pt_sd  = mean_std(per_rev, "PT_acc")
        soc_mu, soc_sd = mean_std(per_rev, "SOC_acc")

        rag_summary.append({
            "dataset": ds,
            "LLT_rag_mean": llt_mu, "LLT_rag_std": llt_sd,
            "PT_rag_mean":  pt_mu,  "PT_rag_std":  pt_sd,
            "SOC_rag_mean": soc_mu, "SOC_rag_std": soc_sd,
        })
    rag_df = pd.DataFrame(rag_summary)

    # ---- Zero-shot (seeds): mean±std
    zs_summary = []
    for ds, files in ZEROSHOT_FILES.items():
        per_seed = []
        for fp in files:
            rows = load_json(fp)
            per_seed.append(compute_zeroshot_metrics(rows, term_norm_to_llt, llt_to_pt, pt_primary_soc))

        llt_mu, llt_sd = mean_std(per_seed, "LLT_acc")
        pt_mu,  pt_sd  = mean_std(per_seed, "PT_acc")
        soc_mu, soc_sd = mean_std(per_seed, "SOC_acc")

        zs_summary.append({
            "dataset": ds,
            "LLT_zs_mean": llt_mu, "LLT_zs_std": llt_sd,
            "PT_zs_mean":  pt_mu,  "PT_zs_std":  pt_sd,
            "SOC_zs_mean": soc_mu, "SOC_zs_std": soc_sd,
        })
    zs_df = pd.DataFrame(zs_summary)

    out = pd.merge(rag_df, zs_df, on="dataset", how="inner")
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    out = out.sort_values("dataset").reset_index(drop=True)

    print("\n=== Summary (mean±std) ===")
    print(out.to_string(index=False))

    plot(out)


if __name__ == "__main__":
    main()
