# -*- coding: utf-8 -*-
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

AGGREGATE_JSON = "/home/naghmedashti/MedDRA-LLM/aggregates/aggregated_by_variant_20260115_110422.json"
ZERO_SHOT_VARIANT_NAME = "zeroshot"

LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"
PT_CSV_FILE  = "/home/naghmedashti/MedDRA-LLM/data/PT2_SOC_25_0.csv"

CSV_SEP = ";"
CSV_ENCODING = "latin1"
FUZZY_CUTOFF = 94

# ✅ write outputs to a known-writable folder
OUT_DIR = "/home/naghmedashti/MedDRA-LLM/aggregates"
OUT_FIG_PNG = f"{OUT_DIR}/manual_vs_zeroshot_accuracy.png"
OUT_FIG_PDF = f"{OUT_DIR}/manual_vs_zeroshot_accuracy.pdf"
OUT_CSV     = f"{OUT_DIR}/manual_vs_zeroshot_summary.csv"


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

def norm_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().casefold().split())

def clean_term(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().strip('"\''"“”‘’")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_mappings(
    llt_csv: str,
    pt_csv: str
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Optional[str]]]:
    # LLT
    llt_df = pd.read_csv(llt_csv, sep=CSV_SEP, encoding=CSV_ENCODING)[["LLT_Code", "LLT_Term", "PT_Code"]].copy()
    llt_df["LLT_Code"] = llt_df["LLT_Code"].map(canon_code)
    llt_df["PT_Code"]  = llt_df["PT_Code"].map(canon_code)
    llt_df["LLT_norm"] = llt_df["LLT_Term"].map(norm_text)

    # PT/SOC
    header = pd.read_csv(pt_csv, sep=CSV_SEP, encoding=CSV_ENCODING, nrows=0).columns
    pt_cols = [c for c in ["PT_Code","SOC_Code","Ist_Primary_SOC","Primary_SOC_Code"] if c in header]
    if "PT_Code" not in pt_cols or "SOC_Code" not in pt_cols:
        raise ValueError("PT CSV must include at least PT_Code and SOC_Code.")
    pt_df = pd.read_csv(pt_csv, sep=CSV_SEP, encoding=CSV_ENCODING)[pt_cols].copy()

    pt_df["PT_Code"]  = pt_df["PT_Code"].map(canon_code)
    pt_df["SOC_Code"] = pt_df["SOC_Code"].map(canon_code)
    if "Primary_SOC_Code" in pt_df.columns:
        pt_df["Primary_SOC_Code"] = pt_df["Primary_SOC_Code"].map(canon_code)

    if "Ist_Primary_SOC" in pt_df.columns:
        pt_df["Ist_Primary_SOC_norm"] = pt_df["Ist_Primary_SOC"].astype(str).str.strip().str.upper()
    else:
        pt_df["Ist_Primary_SOC_norm"] = ""

    # lookups
    llt_to_pt = dict(zip(llt_df["LLT_Code"], llt_df["PT_Code"]))

    # primary SOC per PT
    pt_code_to_primary_soc: Dict[str, Optional[str]] = {}
    has_primary_soc_code_col = "Primary_SOC_Code" in pt_df.columns

    # all SOCs per PT (needed for fallback: single SOC)
    pt_code_to_soc_all: Dict[str, List[str]] = (
        pt_df.dropna(subset=["PT_Code","SOC_Code"])
             .groupby("PT_Code")["SOC_Code"]
             .apply(lambda s: sorted(set([x for x in s if x is not None])))
             .to_dict()
    )

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

        pt_code_to_primary_soc[ptc] = primary

    # term_norm -> LLT_Code
    term_norm_to_llt: Dict[str, str] = {}
    for _, r in llt_df.iterrows():
        if r["LLT_norm"] and r["LLT_Code"]:
            term_norm_to_llt.setdefault(r["LLT_norm"], r["LLT_Code"])

    return llt_to_pt, term_norm_to_llt, pt_code_to_primary_soc

def term_to_llt_code(term: str, term_norm_to_llt: Dict[str, str], allow_fuzzy: bool=True) -> Optional[str]:
    t = norm_text(clean_term(term))
    if not t:
        return None
    if t in term_norm_to_llt:
        return term_norm_to_llt[t]
    for piece in re.split(r"[;,/]+", t):
        p = piece.strip()
        if p and p in term_norm_to_llt:
            return term_norm_to_llt[p]
    if allow_fuzzy:
        best = process.extractOne(
            t,
            list(term_norm_to_llt.keys()),
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_CUTOFF
        )
        return term_norm_to_llt[best[0]] if best else None
    return None

def load_manual_json(path: str) -> List[dict]:
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

def parse_manual_metrics(rows, llt_to_pt, term_norm_to_llt, pt_code_to_primary_soc):
    flags = [get_manual_flag(r) for r in rows]
    flags = [f for f in flags if f is not None]
    if not flags:
        raise ValueError("No manual-check boolean field found in manual reviewer JSON.")

    llt_acc = float(np.mean(flags))
    manual_true_rows = [r for r in rows if get_manual_flag(r) is True]

    if not manual_true_rows:
        return {"LLT_acc_manual": llt_acc, "PT_acc_manual": 0.0, "SOC_acc_manual_optB": 0.0}

    pt_ok = 0
    soc_ok = 0

    for r in manual_true_rows:
        pred_term = r.get("pred_LLT_term") or r.get("predicted") or r.get("pred") or ""
        pred_code = term_to_llt_code(pred_term, term_norm_to_llt, allow_fuzzy=True)
        pt_code = llt_to_pt.get(pred_code) if pred_code else None
        if pt_code:
            pt_ok += 1
            if pt_code_to_primary_soc.get(pt_code):
                soc_ok += 1

    return {
        "LLT_acc_manual": llt_acc,
        "PT_acc_manual": float(pt_ok / len(manual_true_rows)),
        "SOC_acc_manual_optB": float(soc_ok / len(manual_true_rows)),
    }

def load_zeroshot_from_aggregate(agg_json: str, variant_name: str) -> pd.DataFrame:
    with open(agg_json, "r", encoding="utf-8") as f:
        agg = json.load(f)

    rows = []
    for it in agg.get("items", []):
        if str(it.get("variant", "")).strip().lower() != variant_name.strip().lower():
            continue
        rows.append({
            "dataset": it.get("dataset"),
            "LLT_zs_mean": it.get("LLT_term_acc_exact__mean"),
            "LLT_zs_std":  it.get("LLT_term_acc_exact__std"),
            "PT_zs_mean":  it.get("PT_code_acc__mean"),
            "PT_zs_std":   it.get("PT_code_acc__std"),
            "SOC_zs_mean": it.get("SOC_acc_option_b__mean"),
            "SOC_zs_std":  it.get("SOC_acc_option_b__std"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No items found for variant='{variant_name}' in aggregate JSON.")
    return df

def main():
    llt_to_pt, term_norm_to_llt, pt_code_to_primary_soc = load_mappings(LLT_CSV_FILE, PT_CSV_FILE)

    # manual
    manual_records = []
    for dataset, files in MANUAL_FILES.items():
        for fp in files:
            rows = load_manual_json(fp)
            m = parse_manual_metrics(rows, llt_to_pt, term_norm_to_llt, pt_code_to_primary_soc)
            manual_records.append({"dataset": dataset, "reviewer": Path(fp).name, **m})

    manual_df = pd.DataFrame(manual_records)
    manual_agg = (manual_df.groupby("dataset")
                  .agg(
                      LLT_manual_mean=("LLT_acc_manual","mean"),
                      LLT_manual_std =("LLT_acc_manual","std"),
                      PT_manual_mean =("PT_acc_manual","mean"),
                      PT_manual_std  =("PT_acc_manual","std"),
                      SOC_manual_mean=("SOC_acc_manual_optB","mean"),
                      SOC_manual_std =("SOC_acc_manual_optB","std"),
                  ).reset_index())

    # zeroshot
    zs_df = load_zeroshot_from_aggregate(AGGREGATE_JSON, ZERO_SHOT_VARIANT_NAME)

    out = pd.merge(manual_agg, zs_df, on="dataset", how="inner")
    ds_order = ["Mosaic", "Delta", "Dauno"]
    out["dataset"] = pd.Categorical(out["dataset"], categories=ds_order, ordered=True)
    out = out.sort_values("dataset").reset_index(drop=True)

    summary_cols = [
        "dataset",
        "LLT_manual_mean","LLT_manual_std","LLT_zs_mean","LLT_zs_std",
        "PT_manual_mean","PT_manual_std","PT_zs_mean","PT_zs_std",
        "SOC_manual_mean","SOC_manual_std","SOC_zs_mean","SOC_zs_std",
    ]
    out[summary_cols].to_csv(OUT_CSV, index=False)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(out))
    width = 0.36

    def panel(ax, title, m_mean, m_std, z_mean, z_std):
        ax.bar(x - width/2, m_mean, width, yerr=m_std, capsize=4, label="Manual (reviewers)")
        ax.bar(x + width/2, z_mean, width, yerr=z_std, capsize=4, label="Zero-shot (seeds)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(out["dataset"].astype(str))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    panel(axes[0], "LLT accuracy",
          out["LLT_manual_mean"].values, out["LLT_manual_std"].values,
          out["LLT_zs_mean"].values, out["LLT_zs_std"].values)

    panel(axes[1], "PT accuracy",
          out["PT_manual_mean"].values, out["PT_manual_std"].values,
          out["PT_zs_mean"].values, out["PT_zs_std"].values)

    panel(axes[2], "SOC accuracy (Option B)",
          out["SOC_manual_mean"].values, out["SOC_manual_std"].values,
          out["SOC_zs_mean"].values, out["SOC_zs_std"].values)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Manual (clinical) vs Zero-shot accuracy across datasets", y=1.02, fontsize=12)

    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")

    print("\nSaved:")
    print(" -", OUT_FIG_PNG)
    print(" -", OUT_FIG_PDF)
    print(" -", OUT_CSV)
    print("\nSummary:")
    print(out[summary_cols])

if __name__ == "__main__":
    main()
