# -*- coding: utf-8 -*-
"""
Manual clinical validation vs NewRAG vs Zero-shot

Manual (clinical validation):
  - LLT accuracy = CCR = mean(manual_check==True) across rows that have a manual flag
  - PT accuracy  = among manual_check==True rows: PT(pred_LLT) == PT(true_LLT)
  - SOC accuracy = among manual_check==True rows: SOC(pred) == SOC(true)
    where SOC(x) := primarySOC( PT( LLT(x) ) )

NewRAG (seeds) & Zero-shot (seeds):
  - LLT accuracy = exact match rate: pred_LLT_term == true_LLT_term (normalized)
  - PT accuracy  = PT(pred_LLT) == PT(true_LLT) across ALL rows
  - SOC accuracy = SOC(pred) == SOC(true) across ALL rows

Aggregation:
  - Manual: mean±std across 3 reviewers per dataset
  - NewRAG: mean±std across seed files per dataset
  - Zero-shot: mean±std across seed files per dataset

Plot:
  - Single stacked-level bar chart (Excel-style)
  - Per dataset: Manual | NewRAG | Zero-shot
  - Each bar: stacked LLT (bottom), PT (middle), SOC (top)
    with cumulative levels (h_llt=LLT, h_pt=PT-LLT, h_soc=SOC-PT)
  - Error bars at LLT, PT, SOC boundaries
  - Save PNG
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from matplotlib.patches import Patch


# =========================
# CONFIG
# =========================

MANUAL_FILES = {
    "Dauno": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output_NewRAG__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output_NewRAG__full__reviewer3.json",
    ],
    "Delta": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Delta_output_NewRAG__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Delta_output_NewRAG__full__reviewer3.json",
    ],
    "Mosaic": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Mosaic_output_NewRAG__full__reviewer1.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Mosaic_output_NewRAG__full__reviewer3.json",
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

# TODO: Fill in the NewRAG paths
NEWRAG_FILES = {
    "Dauno": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Dauno_output_NewRAG_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Dauno_output_NewRAG_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Dauno_output_NewRAG_seed44.json",
    ],
    "Delta": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Delta_output_NewRAG_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Delta_output_NewRAG_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Delta_output_NewRAG_seed44.json",
    ],
    "Mosaic": [
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Mosaic_output_NewRAG_seed42.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Mosaic_output_NewRAG_seed43.json",
        "/home/naghmedashti/MedDRA-LLM/RAG_Models/NewRAG_Results/Top_K_100/Mosaic_output_NewRAG_seed44.json",
    ],
}

LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
PT_CSV_FILE  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"

CSV_SEP = ";"
CSV_ENCODING = "latin1"
FUZZY_CUTOFF = 94

DATASET_ORDER = ["Mosaic", "Delta", "Dauno"]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_FIG_PNG = SCRIPT_DIR / "manual_vs_rag_vs_zeroshot_stacked.png"


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
    Manual (clinical validation):
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
    Zero-shot / NewRAG:
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


def shade_towards_white(hex_color: str, t: float) -> str:
    """Blend a hex color towards white by factor t in [0,1]."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * t)
    g = int(g + (255 - g) * t)
    b = int(b + (255 - b) * t)

    return f"#{r:02X}{g:02X}{b:02X}"


def enforce_monotonicity(llt: float, pt: float, soc: float) -> Tuple[float, float, float]:
    pt = max(pt, llt)
    soc = max(soc, pt)
    return llt, pt, soc


# =========================
# Plotting
# =========================

def plot(df: pd.DataFrame):
    # Base colors
    C_MANUAL = "#0072B2"
    C_NEWRAG = "#009E73"
    C_ZS     = "#D55E00"

    # Shades per level
    shades = {
        "LLT": 0.05,
        "PT":  0.35,
        "SOC": 0.65,
    }

    fig, ax = plt.subplots(1, 1, figsize=(13.0, 5.6), constrained_layout=False)

    x = np.arange(len(df))
    width = 0.22
    # Prepare series in consistent order
    series = [
        ("Manual",  C_MANUAL, "manual"),
        ("NewRAG",  C_NEWRAG, "newrag"),
        ("Zero-shot", C_ZS,  "zs"),
    ]

    offsets = [-width, 0.0, width]

    for (label, base_color, key), dx in zip(series, offsets):
        llt = df[f"LLT_{key}_mean"].values
        pt  = df[f"PT_{key}_mean"].values
        soc = df[f"SOC_{key}_mean"].values

        llt_sd = df[f"LLT_{key}_std"].values
        pt_sd  = df[f"PT_{key}_std"].values
        soc_sd = df[f"SOC_{key}_std"].values

        # Robust NaN handling
        llt = np.nan_to_num(llt, nan=0.0)
        pt  = np.nan_to_num(pt,  nan=0.0)
        soc = np.nan_to_num(soc, nan=0.0)

        llt_sd = np.nan_to_num(llt_sd, nan=0.0)
        pt_sd  = np.nan_to_num(pt_sd,  nan=0.0)
        soc_sd = np.nan_to_num(soc_sd, nan=0.0)

        # If a higher level is zero but a lower one exists, carry it upward
        pt = np.where((pt == 0) & (llt > 0), llt, pt)
        soc = np.where((soc == 0) & (pt > 0), pt, soc)

        # Enforce monotonicity
        llt_m = np.zeros_like(llt)
        pt_m  = np.zeros_like(pt)
        soc_m = np.zeros_like(soc)
        for i in range(len(llt)):
            llt_m[i], pt_m[i], soc_m[i] = enforce_monotonicity(llt[i], pt[i], soc[i])

        h_llt = llt_m
        h_pt = pt_m - llt_m
        h_soc = soc_m - pt_m

        # Stacked bars
        ax.bar(
            x + dx, h_llt, width,
            color=shade_towards_white(base_color, shades["LLT"]),
            linewidth=0
        )
        ax.bar(
            x + dx, h_pt, width,
            bottom=h_llt,
            color=shade_towards_white(base_color, shades["PT"]),
            linewidth=0
        )
        ax.bar(
            x + dx, h_soc, width,
            bottom=h_llt + h_pt,
            color=shade_towards_white(base_color, shades["SOC"]),
            linewidth=0
        )

        # Numeric labels at boundaries
        for i in range(len(llt_m)):
            base_off = 0.01
            pt_off = 0.03 if (pt_m[i] - llt_m[i]) < 0.04 else 0.0
            soc_off = 0.03 if (soc_m[i] - pt_m[i]) < 0.04 else 0.0

            ax.text(x[i] + dx, llt_m[i] + base_off, f"L:{llt_m[i]:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")
            ax.text(x[i] + dx, pt_m[i] + base_off + pt_off, f"P:{pt_m[i]:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")
            ax.text(x[i] + dx, soc_m[i] + base_off + soc_off, f"S:{soc_m[i]:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"].tolist(), fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    # Legends (two blocks)
    method_handles = [
        Patch(facecolor=C_MANUAL, edgecolor="none", label="Manual"),
        Patch(facecolor=C_NEWRAG, edgecolor="none", label="NewRAG"),
        Patch(facecolor=C_ZS, edgecolor="none", label="Zero-shot"),
    ]

    level_handles = [
        Patch(facecolor=shade_towards_white("#777777", shades["LLT"]), edgecolor="none", label="LLT"),
        Patch(facecolor=shade_towards_white("#777777", shades["PT"]), edgecolor="none", label="PT"),
        Patch(facecolor=shade_towards_white("#777777", shades["SOC"]), edgecolor="none", label="SOC"),
    ]

    leg1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=11,
        title="Methods",
        title_fontsize=11,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=level_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        frameon=False,
        fontsize=11,
        title="Levels",
        title_fontsize=11,
    )

    fig.subplots_adjust(left=0.07, right=0.72, bottom=0.12, top=0.95)

    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    # fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    print(f"[Saved] {OUT_FIG_PNG}")


# =========================
# MAIN
# =========================

def main():
    term_norm_to_llt, llt_to_pt, pt_primary_soc = load_mappings(LLT_CSV_FILE, PT_CSV_FILE)

    # ---- Manual (reviewers): mean±std
    manual_summary = []
    for ds, files in MANUAL_FILES.items():
        per_rev = []
        for fp in files:
            rows = load_json(fp)
            per_rev.append(compute_rag_metrics(rows, term_norm_to_llt, llt_to_pt, pt_primary_soc))

        llt_mu, llt_sd = mean_std(per_rev, "LLT_acc")
        pt_mu,  pt_sd  = mean_std(per_rev, "PT_acc")
        soc_mu, soc_sd = mean_std(per_rev, "SOC_acc")

        manual_summary.append({
            "dataset": ds,
            "LLT_manual_mean": llt_mu, "LLT_manual_std": llt_sd,
            "PT_manual_mean":  pt_mu,  "PT_manual_std":  pt_sd,
            "SOC_manual_mean": soc_mu, "SOC_manual_std": soc_sd,
        })
    manual_df = pd.DataFrame(manual_summary)

    # ---- NewRAG (seeds): mean±std
    newrag_summary = []
    for ds, files in NEWRAG_FILES.items():
        per_seed = []
        for fp in files:
            rows = load_json(fp)
            per_seed.append(compute_zeroshot_metrics(rows, term_norm_to_llt, llt_to_pt, pt_primary_soc))

        llt_mu, llt_sd = mean_std(per_seed, "LLT_acc")
        pt_mu,  pt_sd  = mean_std(per_seed, "PT_acc")
        soc_mu, soc_sd = mean_std(per_seed, "SOC_acc")

        newrag_summary.append({
            "dataset": ds,
            "LLT_newrag_mean": llt_mu, "LLT_newrag_std": llt_sd,
            "PT_newrag_mean":  pt_mu,  "PT_newrag_std":  pt_sd,
            "SOC_newrag_mean": soc_mu, "SOC_newrag_std": soc_sd,
        })
    newrag_df = pd.DataFrame(newrag_summary)

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

    out = manual_df.merge(newrag_df, on="dataset", how="inner").merge(zs_df, on="dataset", how="inner")
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    out = out.sort_values("dataset").reset_index(drop=True)

    print("\n=== Summary (mean±std) ===")
    print(out.to_string(index=False))

    plot(out)


if __name__ == "__main__":
    main()
