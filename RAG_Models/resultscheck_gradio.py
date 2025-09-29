# resultscheck_gradio.py
import os, json, datetime as dt
import gradio as gr
from datetime import datetime, timezone

# ====== Paths ======
HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "MedDRA-LLM", "RAG_Models")
OUT_DIR  = os.path.join(BASE_DIR, "manual_checks")
os.makedirs(OUT_DIR, exist_ok=True)

# فقط فایل‌های خروجیِ سه دیتاست با الگوی *_seed*.json
def list_server_jsons():
    if not os.path.isdir(BASE_DIR):
        return []
    files = [f for f in os.listdir(BASE_DIR) if f.endswith(".json") and "_seed" in f]
    files.sort()
    return [os.path.join(BASE_DIR, f) for f in files]

def entry_key(e):
    return (e.get("AE_text",""), e.get("true_LLT_term",""), e.get("pred_LLT_term",""))

# ====== Core fns ======
def load_data(reviewer, selected_path, use_auto_accept):
    """
    بر اساس reviewer و فایل انتخاب‌شده، داده را لود می‌کند،
    exact_LLT_match را (درصورت انتخاب) خودکار تایید می‌کند،
    و اولین آیتمِ نیازمند مانوال‌چک را برمی‌گرداند.
    """
    if not reviewer:
        return (gr.update(value=""), "Enter Reviewer ID.",
                None, None, None, None, None, gr.update(visible=False))

    # اعتبارسنجی مسیر
    if not selected_path:
        return (gr.update(value=""), f"No dataset selected. [cwd={os.getcwd()}]",
                None, None, None, None, None, gr.update(visible=False))

    path = os.path.abspath(os.path.expanduser(selected_path))
    if not os.path.exists(path):
        return (gr.update(value=""), f"Server path NOT FOUND: {path} [cwd={os.getcwd()}]",
                None, None, None, None, None, gr.update(visible=False))

    # لود ورودی
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        return (gr.update(value=""), f"JSON load error: {e}",
                None, None, None, None, None, gr.update(visible=False))

    # مسیر خروجی: per-dataset + per-reviewer
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}__checked__{reviewer}.json")

    # لود پیشرفت قبلی
    results = []
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path, "r", encoding="utf-8"))
        except Exception:
            results = []
    done = {entry_key(r) for r in results}

    # auto-accept for exact matches
    if use_auto_accept:
        changed = False
        for e in data:
            k = entry_key(e)
            if k in done:
                continue
            if e.get("exact_LLT_match", False) is True:
                e["manual check"] = True
                e["reviewer"] = reviewer
                e["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                results.append(e)
                done.add(k)
                changed = True
        if changed:
            json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2)

    # پیدا کردن اولین رکوردِ باقی‌مانده برای مانوال‌چک
    idx = None
    for i, e in enumerate(data):
        if entry_key(e) not in done and not e.get("exact_LLT_match", False):
            idx = i
            break

    if idx is None:
        # همه بررسی شده‌اند
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2)
        status = f"✅ All entries reviewed for: {os.path.basename(path)}\nSaved: {out_path}"
        return (gr.update(value=""), status, data, results, out_path, None, reviewer,
                gr.update(value=out_path, visible=True))

    curr = data[idx]
    status = (f"Dataset: {os.path.basename(path)}\n"
              f"Progress: {len(results)+1} / {len(data)}")
    return (json.dumps(curr, indent=2, ensure_ascii=False), status,
            data, results, out_path, idx, reviewer, gr.update(visible=False))

def save_and_next(choice, state_data, state_results, out_path, idx, reviewer):
    if state_data is None or idx is None or out_path is None:
        return (gr.update(value=""), "Load a dataset first.",
                state_data, state_results, out_path, idx, reviewer, gr.update(visible=False))

    data = state_data
    results = state_results or []

    curr = data[idx]
    curr["manual check"] = (choice == "true")
    curr["reviewer"] = reviewer
    curr["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    results.append(curr)

    # ذخیره
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2)

    # بعدی
    done = {entry_key(r) for r in results}
    next_idx = None
    for i, e in enumerate(data):
        if entry_key(e) not in done and not e.get("exact_LLT_match", False):
            next_idx = i
            break

    if next_idx is None:
        status = f"✅ Finished this dataset.\nSaved: {out_path}"
        return (gr.update(value=""), status,
                data, results, out_path, None, reviewer, gr.update(value=out_path, visible=True))

    curr = data[next_idx]
    status = f"Progress: {len(results)} / {len(data)}"
    return (json.dumps(curr, indent=2, ensure_ascii=False), status,
            data, results, out_path, next_idx, reviewer, gr.update(visible=False))

# ====== UI ======
with gr.Blocks() as demo:
    gr.Markdown("## MedDRA – Manual Review (Gradio)")

    with gr.Row():
        reviewer = gr.Textbox(label="Reviewer ID", value="doctor1", scale=2)
        use_auto = gr.Checkbox(label="Auto-accept exact matches", value=True, scale=1)

    # لیست فایل‌های موجود روی سرور (Dropdown)
    server_files = list_server_jsons()
    dataset = gr.Dropdown(choices=server_files, value=(server_files[0] if server_files else None),
                          label="Select dataset (server)", interactive=True)

    load_btn = gr.Button("Load selected dataset", variant="primary")

    item_json = gr.Code(label="Current entry (read-only JSON)", language="json", interactive=False, lines=18)
    status = gr.Textbox(label="Status", interactive=False)

    # states
    state_data = gr.State()
    state_results = gr.State()
    state_out = gr.State()
    state_idx = gr.State()
    state_reviewer = gr.State()

    with gr.Row():
        choice = gr.Radio(choices=["true","false"], value="true", label="Manual check", scale=1, interactive=True)
        save_btn = gr.Button("Save & Next", scale=2)
        download = gr.File(label="Download current results", visible=False, scale=2)

    # رویدادها
    load_btn.click(
        load_data,
        inputs=[reviewer, dataset, use_auto],
        outputs=[item_json, status, state_data, state_results, state_out, state_idx, state_reviewer, download]
    )
    save_btn.click(
        save_and_next,
        inputs=[choice, state_data, state_results, state_out, state_idx, state_reviewer],
        outputs=[item_json, status, state_data, state_results, state_out, state_idx, state_reviewer, download]
    )

# احراز هویت ساده برای تست
if __name__ == "__main__":
    demo.launch(share=True, auth=[("doctor1","pass123"), ("doctor2","pass456")])
