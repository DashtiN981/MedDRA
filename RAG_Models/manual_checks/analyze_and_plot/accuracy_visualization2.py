import json
import os
import matplotlib.pyplot as plt

BASE_DIR = "/home/naghmedashti/MedDRA-LLM/RAG_Models/manual_checks"

DATASETS  = ["Dauno", "Delta", "Mosaic"]
REVIEWERS = ["Isabella", "reviewer1", "reviewer2"]

def load_json_list(path: str):
    with open(path, "r", encoding="latin1") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []

def get_manual_flag(d: dict):
    keys = [
        "manual check",
        "manual_check",
        "final_manual_check",
        "manualChecked",
        "is_manual_correct",
        "manual_check_result",
        "checked_manual",
        "is_correct_after_manual",
    ]
    for k in keys:
        if k in d:
            val = d[k]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                v = val.strip().lower()
                if v in {"true","yes","y","1"}:
                    return True
                if v in {"false","no","n","0"}:
                    return False
    return None

# ساختار: metrics[dataset][metric_name] = لیست به طول 3 (برای هر ریویور)
metrics = {
    ds: {
        "exact":  [],
        "fuzzy":  [],
        "manual": []
    } for ds in DATASETS
}

for ds in DATASETS:
    for reviewer in REVIEWERS:
        fp = os.path.join(
            BASE_DIR,
            f"{ds}_output__full__{reviewer}.json"
        )
        if not os.path.exists(fp):
            print(f"[WARN] File not found: {fp}")
            # اگر فایل نبود، مقدار None می‌گذاریم
            metrics[ds]["exact"].append(0.0)
            metrics[ds]["fuzzy"].append(0.0)
            metrics[ds]["manual"].append(0.0)
            continue

        data = load_json_list(fp)
        total = len(data)

        exact_count = 0
        fuzzy_count = 0
        manual_true = 0

        for d in data:
            if not isinstance(d, dict):
                continue
            if d.get("exact_LLT_match") is True:
                exact_count += 1
            if d.get("LLT_fuzzy_match") is True:
                fuzzy_count += 1
            mflag = get_manual_flag(d)
            if mflag is True:
                manual_true += 1

        if total == 0:
            exact_acc  = 0.0
            fuzzy_acc  = 0.0
            manual_acc = 0.0
        else:
            exact_acc  = 100 * exact_count / total
            fuzzy_acc  = 100 * fuzzy_count / total
            manual_acc = 100 * manual_true / total

        metrics[ds]["exact"].append(exact_acc)
        metrics[ds]["fuzzy"].append(fuzzy_acc)
        metrics[ds]["manual"].append(manual_acc)

# ---- رسم 3 ساب‌پلات: برای هر دیتاست یک نمودار ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

bar_width = 0.25
x = range(len(REVIEWERS))

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]

    exact_accs  = metrics[ds]["exact"]
    fuzzy_accs  = metrics[ds]["fuzzy"]
    manual_accs = metrics[ds]["manual"]

    bars_exact = ax.bar([i - bar_width for i in x], exact_accs,
                        width=bar_width, label="Exact Match")
    bars_fuzzy = ax.bar(x, fuzzy_accs,
                        width=bar_width, label="Fuzzy Match")
    bars_manual = ax.bar([i + bar_width for i in x], manual_accs,
                         width=bar_width, label="Manual Check")

    ax.set_title(ds)
    ax.set_xticks(list(x))
    ax.set_xticklabels(REVIEWERS, rotation=15)
    ax.set_ylim(0, 100)
    if idx == 0:
        ax.set_ylabel("Accuracy (%)")

    # نوشتن درصد روی ستون‌ها
    def annotate(bars, vals):
        for b, v in zip(bars, vals):
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, y + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    annotate(bars_exact, exact_accs)
    annotate(bars_fuzzy, fuzzy_accs)
    annotate(bars_manual, manual_accs)

# فقط یک legend مشترک
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

fig.suptitle("Accuracy per Dataset and Reviewer (FULL files only)", y=1.03)
plt.tight_layout()

out_path = os.path.join(BASE_DIR, "analyze_and_plot", "accuracy_per_dataset_and_reviewer.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
plt.show()

print("Saved plot:", out_path)
