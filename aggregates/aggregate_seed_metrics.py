"""
Aggregate per-seed metrics by (dataset, variant) where filenames follow:
  DatasetName_output_{variant}_metrics_seed{SEED}.json
with variant in {RAG, fuzz, rapidfuzz, zeroshot} (case-insensitive).

- Reads all *_metrics_seed*.json recursively under given roots.
- Parses dataset/variant/seed from filename (source of truth).
- Groups by (dataset, variant), dedup seeds, and computes mean/std/n for each numeric metric.
- Saves one combined CSV + JSON with all groups.

Author: you :)
"""
import os, json, math, glob, re
from collections import defaultdict, OrderedDict
import csv
from datetime import datetime

# ---------- CONFIG ----------
ROOTS = [
    "/home/naghmedashti/MedDRA-LLM/RAG_Models",
    "/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models",
]

ROOTS

# Output folder:
OUT_DIR = "/home/naghmedashti/MedDRA-LLM/aggregates"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: restrict to these datasets/models (leave empty to include all)
INCLUDE_DATASETS = set()   # e.g., {"KI_Projekt_Mosaic_AE_Codierung_2024_07_03", "KI_Projekt_Delta_AE_Codierung_2023_02_08", "KI_Projekt_Dauno_AE_Codierung_2022_10_20"}
INCLUDE_VARIANTS = set()  # e.g., {"RAG","fuzz","rapidfuzz","zeroshot"} (case-insensitive) empty equval ALL

# file name pattern
#  e.g. Mosaic_output_rapidfuzz_metrics_seed42.json
FNAME_RE = re.compile(
    r"(?P<dataset>.+?)_output_(?P<variant>NewRAG|Newfuzz|Newrapidfuzz|zeroshot)_metrics_seed(?P<seed>\d+)\.json$",
    re.IGNORECASE
)

def _is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def _nanmean_std(vals):
    """Return (mean, std, n) over numeric vals ignoring None; returns (None, None, 0) if empty."""
    xs = [v for v in vals if v is not None]
    n = len(xs)
    if n == 0:
        return (None, None, 0)
    if n == 1:
        return (xs[0], 0.0, 1)
    m = sum(xs)/n
    var = sum((x - m)**2 for x in xs) / (n - 1)
    return (m, math.sqrt(var), n)

def _collect_files(roots):
    files = []
    for r in roots:
        pattern = os.path.join(r, "**", "*_metrics_seed*.json")
        files.extend(glob.glob(pattern, recursive=True))
    # Unique path
    return sorted(set(files))

def _parse_fname(fp):
    base = os.path.basename(fp)
    m = FNAME_RE.search(base)
    if not m:
        return None
    dataset = m.group("dataset")
    variant = m.group("variant").lower()  # normalize to lower: rag/fuzz/rapidfuzz/zeroshot
    seed = int(m.group("seed"))
    return {"dataset": dataset, "variant": variant, "seed": seed}

def load_rows():
    rows = []
    for fp in _collect_files(ROOTS):
        info = _parse_fname(fp)
        if not info:
            # Files whose names do not match the pattern will be rejected.
            continue
        dataset = info["dataset"]
        variant = info["variant"]
        seed    = info["seed"]

        if INCLUDE_DATASETS and dataset not in INCLUDE_DATASETS:
            continue
        if INCLUDE_VARIANTS and variant.lower() not in {v.lower() for v in INCLUDE_VARIANTS}:
            continue

        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        metrics = obj.get("metrics", {})
        counts  = obj.get("counts", {})
        # Keep only numeric metrics (or None); non-numeric ones are ignored in aggregation
        clean_metrics = {}
        for k, v in metrics.items():
            if v is None:
                clean_metrics[k] = None
            elif _is_number(v):
                clean_metrics[k] = float(v)
            else:
                # non-numeric  → None
                clean_metrics[k] = None

        rows.append({
            "file": fp,
            "dataset": dataset,
            "variant": variant,  # rag/fuzz/rapidfuzz/zeroshot
            "seed": seed,
            "metrics": clean_metrics,
            "counts": counts
        })
    return rows

def aggregate(rows):
    # Collect metric keysم
    metric_keys = set()
    for r in rows:
        metric_keys.update(r["metrics"].keys())

    # grouping based on (dataset, variant)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["variant"])].append(r)

    agg = []
    for (dataset, variant), items in sorted(groups.items()):
        # Remove duplicate seeds within a group
        seen_seeds = set()
        uniq_items = []
        for it in items:
            s = it["seed"]
            if s in seen_seeds:
                # If there are multiple files with the same name/result from a seed, the first one is sufficient.
                continue
            seen_seeds.add(s)
            uniq_items.append(it)

        seeds_sorted = sorted(seen_seeds)
        rec = OrderedDict()
        rec["dataset"] = dataset
        rec["variant"] = variant  # rag/fuzz/rapidfuzz/zeroshot
        rec["seeds"] = seeds_sorted
        rec["n_runs"] = len(uniq_items)
        rec["sum_n_samples"] = sum(int(it["counts"].get("n_samples", 0)) for it in uniq_items)

        # Average/standard deviation for each metric
        for mk in sorted(metric_keys):
            vals = [it["metrics"].get(mk) for it in uniq_items]
            m, s, n = _nanmean_std(vals)
            rec[f"{mk}__mean"] = (None if m is None else round(m, 6))
            rec[f"{mk}__std"]  = (None if s is None else round(s, 6))
            rec[f"{mk}__n"]    = n

        agg.append(rec)
    return agg

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("[Saved]", path)

def save_csv(recs, path):
    if not recs:
        print("[WARN] No records; CSV not written.")
        return
    # Column order: Basic + Metrics
    base_cols = ["dataset", "variant", "seeds", "n_runs", "sum_n_samples"]
    other_cols = []
    for r in recs:
        for k in r.keys():
            if k not in base_cols and k not in other_cols:
                other_cols.append(k)
    fieldnames = base_cols + other_cols
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in recs:
            row = dict(r)
            if isinstance(row.get("seeds"), list):
                row["seeds"] = ",".join(str(x) for x in row["seeds"])
            w.writerow(row)
    print("[Saved]", path)

def main():
    rows = load_rows()
    if not rows:
        print("[INFO] No metrics files found. Check ROOTS or filename pattern.")
        return
    agg = aggregate(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUT_DIR, f"aggregated_by_variant_{ts}.json")
    csv_path  = os.path.join(OUT_DIR, f"aggregated_by_variant_{ts}.csv")

    save_json({"items": agg, "meta": {"created_at": ts, "num_input_files": len(rows)}}, json_path)
    save_csv(agg, csv_path)

if __name__ == "__main__":
    main()