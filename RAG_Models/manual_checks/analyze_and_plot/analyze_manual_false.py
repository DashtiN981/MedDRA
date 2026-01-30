# analyze_manual_false.py
import json, hashlib
from pathlib import Path
from typing import Any, Optional
import pandas as pd

# ---- files path (checked) ----
INPUTS = [
    "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__checked__Isabella.json",
    "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__checked__reviewer1.json",
    "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/Dauno_output__checked__reviewer2.json",
]

def norm_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true","yes","y","1"}:  return True
        if v in {"false","no","n","0"}:   return False
    return None

def get_first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d: return d[k]
    return default

def manual_flag(rec: dict) -> Optional[bool]:
    pos = ["manual check","manual_check","manualChecked","manual",
           "is_manual_correct","final_manual_check","manual_check_result",
           "checked_manual","is_correct_after_manual"]
    for k in pos:
        if k in rec: return norm_bool(rec[k])
    # معکوس‌های رایج
    neg = ["is_error_after_manual","manual_incorrect"]
    for k in neg:
        if k in rec:
            v = norm_bool(rec[k])
            return (not v) if v is not None else None
    return None

def stable_record_id(ae: str, gold: str, pred: str, file_index: int, row_index: int) -> str:
    base = f"{ae}|||{gold}|||{pred}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{file_index}-{row_index}-{h}"

def try_load_list(fp: Path):
    data = json.load(open(fp, "r", encoding="utf-8"))
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list): return v
    return []

def reviewer_from_name(name: str) -> str:
    if "Isabella" in name: return "Isabella"
    if "reviewer1" in name: return "reviewer1"
    if "reviewer2" in name: return "reviewer2"
    return "unknown"

rows = []
per_file_stats = []

for fi, path in enumerate(INPUTS):
    p = Path(path)
    data = try_load_list(p)
    reviewer = reviewer_from_name(p.name)

    # if find full files then use them for analysis
    p_full = Path(str(p).replace("__checked__", "__full__"))
    full_count = None
    if p_full.exists():
        try:
            full_count = len(try_load_list(p_full))
        except Exception:
            full_count = None

    manual_false_count = 0
    for ri, rec in enumerate(data):
        m = manual_flag(rec)
        ae   = get_first(rec, ["AE_text","ae_text","AE","ae"], "")
        pred = get_first(rec, ["pred_LLT_term","predicted_LLT","predicted_LLT_term","model_pred","prediction"], "")
        gold = get_first(rec, ["true_LLT_term","gold_LLT","gold_LLT_term","ground_truth_LLT","label"], "")

        rec_id = get_first(rec, ["id","record_id","row_id"])
        if rec_id is None:
            rec_id = stable_record_id(ae or "", gold or "", pred or "", fi, ri)

        if m is False:
            manual_false_count += 1
            rows.append({
                "reviewer": reviewer,
                "AE_text": ae,
                "pred_LLT_term": pred,
                "true_LLT_term": gold,
                "record_id": rec_id,
                "source_file": p.name,
            })

    per_file_stats.append({
        "source_file": p.name,
        "reviewer": reviewer,
        "records_in_checked": len(data),
        "records_in_full_if_found": full_count,
        "manual_false_count": manual_false_count,
        "manual_false_rate_checked": (manual_false_count/len(data) if len(data) else None),
        "manual_false_rate_full": (manual_false_count/full_count if full_count else None),
    })

# --- خروجی‌ها ---
df = pd.DataFrame(rows, columns=["reviewer","AE_text","pred_LLT_term","true_LLT_term","record_id","source_file"])
stats = pd.DataFrame(per_file_stats)

df.to_csv("/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/analyze_and_plot/Dauno_manual_false_cases_all_reviewers.csv", index=False)
stats.to_csv("/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/analyze_and_plot/Dauno_manual_false_summary_by_file.csv", index=False)

print("Saved tables:")
print("/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/analyze_and_plot/Dauno_manual_false_cases_all_reviewers.csv")
print("/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks/analyze_and_plot/Dauno_manual_false_summary_by_file.csv")
