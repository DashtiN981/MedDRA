# resultscheck_by_gradio.py
import os, json
from datetime import datetime, timezone
import gradio as gr

# ====== Paths ======
HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "MedDRA-LLM", "RAG_Models")
OUT_DIR  = os.path.join(BASE_DIR, "manual_checks")
os.makedirs(OUT_DIR, exist_ok=True)

def list_server_jsons():

    """List *_seed*.json files from RAG_Models"""

    if not os.path.isdir(BASE_DIR):
        return []
    files = [f for f in os.listdir(BASE_DIR) if f.endswith(".json") and "_seed" in f]
    files.sort()
    return [os.path.join(BASE_DIR, f) for f in files]

def entry_key(e):
    return (e.get("AE_text",""), e.get("true_LLT_term",""), e.get("pred_LLT_term",""))

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _write_full_snapshot(data, results, out_path_full, reviewer, default_unreviewed=False):
    """
    write a file with entries equval input file.
    - If object checked, manual_check= (True/False).
    - If object not checked: manual check = default_unreviewed (False).
    """
    decided = {
        (r.get("AE_text",""), r.get("true_LLT_term",""), r.get("pred_LLT_term","")): r
        for r in results
    }
    full = []
    for e in data:
        k = (e.get("AE_text",""), e.get("true_LLT_term",""), e.get("pred_LLT_term",""))
        if k in decided:
            full.append(decided[k])
        else:
            e2 = dict(e)
            e2["manual check"] = bool(default_unreviewed)  # False for unchecked items
            e2["reviewer"] = reviewer
            full.append(e2)
    with open(out_path_full, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

# ====== Core fns ======
def load_data(reviewer, selected_path, use_auto_accept):
    """
    - Load selected file
    - auto-accept are done
    - write/update output of checked items and full 
    - return first items which should be check
    """
    if not reviewer:
        return (gr.update(value=""), "Enter Reviewer ID.",
                None, None, None, None, None, None, gr.update(visible=False))

    if not selected_path:
        return (gr.update(value=""), f"No dataset selected. [cwd={os.getcwd()}]",
                None, None, None, None, None, None, gr.update(visible=False))

    path = os.path.abspath(os.path.expanduser(selected_path))
    if not os.path.exists(path):
        return (gr.update(value=""), f"Server path NOT FOUND: {path} [cwd={os.getcwd()}]",
                None, None, None, None, None, None, gr.update(visible=False))

    # load input
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        return (gr.update(value=""), f"JSON load error: {e}",
                None, None, None, None, None, None, gr.update(visible=False))

    # output paths (per dataset + per reviewer)
    base = os.path.splitext(os.path.basename(path))[0]
    out_checked = os.path.join(OUT_DIR, f"{base}__checked__{reviewer}.json")
    out_full    = os.path.join(OUT_DIR, f"{base}__full__{reviewer}.json")

    # load previous progress
    results = []
    if os.path.exists(out_checked):
        try:
            results = json.load(open(out_checked, "r", encoding="utf-8"))
        except Exception:
            results = []
    done = {entry_key(r) for r in results}

    # auto-accept exact matches
    if use_auto_accept:
        changed = False
        for e in data:
            k = entry_key(e)
            if k in done:
                continue
            if e.get("exact_LLT_match", False) is True:
                e = dict(e)
                e["manual check"] = True
                e["reviewer"] = reviewer
                e["reviewed_at"] = utc_iso()
                results.append(e)
                done.add(k)
                changed = True
        if changed:
            with open(out_checked, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # write full (unchecked= False)
    _write_full_snapshot(data, results, out_full, reviewer, default_unreviewed=False)

    # find first remaining manual item
    idx = None
    for i, e in enumerate(data):
        if entry_key(e) not in done and not e.get("exact_LLT_match", False):
            idx = i
            break

    if idx is None:
        status = (f"All entries reviewed for: {os.path.basename(path)}\n"
                  f"Saved checked: {out_checked}\n"
                  f"Saved full:    {out_full} (size={len(data)})")
        return (gr.update(value=""), status,
                data, results, out_checked, out_full, None, reviewer, gr.update(value=out_full, visible=True))

    curr = data[idx]
    status = (f"Dataset: {os.path.basename(path)}\n"
              f"Progress: {len(results)} / {len(data)} reviewed\n"
              f"Remaining: {len(data) - len(results)}")
    return (json.dumps(curr, indent=2, ensure_ascii=False), status,
            data, results, out_checked, out_full, idx, reviewer, gr.update(visible=False))

def save_and_next(choice, state_data, state_results, out_checked, out_full, idx, reviewer):
    if state_data is None or idx is None or out_checked is None or out_full is None:
        return (gr.update(value=""), "Load a dataset first.",
                state_data, state_results, out_checked, out_full, idx, reviewer, gr.update(visible=False))

    data = state_data
    results = state_results or []

    # current item
    curr = dict(data[idx])
    curr["manual check"] = (choice == "true")
    curr["reviewer"] = reviewer
    curr["reviewed_at"] = utc_iso()
    results.append(curr)

    # save checked
    with open(out_checked, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # save full (unchecked = False)
    _write_full_snapshot(data, results, out_full, reviewer, default_unreviewed=False)

    # next item
    done = {entry_key(r) for r in results}
    next_idx = None
    for i, e in enumerate(data):
        if entry_key(e) not in done and not e.get("exact_LLT_match", False):
            next_idx = i
            break

    if next_idx is None:
        status = (f"Finished this dataset.\n"
                  f"Saved checked: {out_checked}\n"
                  f"Saved full:    {out_full} (size={len(data)})")
        return (gr.update(value=""), status,
                data, results, out_checked, out_full, None, reviewer, gr.update(value=out_full, visible=True))

    curr = data[next_idx]
    status = f"Progress: {len(results)} / {len(data)} reviewed"
    return (json.dumps(curr, indent=2, ensure_ascii=False), status,
            data, results, out_checked, out_full, next_idx, reviewer, gr.update(visible=False))

# ====== UI ======
with gr.Blocks() as demo:
    gr.Markdown("## MedDRA – Manual Review (Gradio)")

    with gr.Row():
        reviewer = gr.Textbox(label="Reviewer ID", value="doctor1", scale=2)
        use_auto = gr.Checkbox(label="Auto-accept exact matches", value=True, scale=1)

    server_files = list_server_jsons()
    dataset = gr.Dropdown(choices=server_files, value=(server_files[0] if server_files else None),
                          label="Select dataset (server)", interactive=True)

    load_btn = gr.Button("Load selected dataset", variant="primary")

    item_json = gr.Code(label="Current entry (read-only JSON)", language="json", interactive=False, lines=18)
    status = gr.Textbox(label="Status", interactive=False)

    # states
    state_data = gr.State()
    state_results = gr.State()
    state_out_checked = gr.State()
    state_out_full = gr.State()
    state_idx = gr.State()
    state_reviewer = gr.State()

    with gr.Row():
        choice = gr.Radio(choices=["true","false"], value="true", label="Manual check", scale=1, interactive=True)
        save_btn = gr.Button("Save & Next", scale=2)
        download_full = gr.File(label="Download FULL results (all records)", visible=False, scale=2)

    load_btn.click(
        load_data,
        inputs=[reviewer, dataset, use_auto],
        outputs=[item_json, status, state_data, state_results, state_out_checked, state_out_full, state_idx, state_reviewer, download_full]
    )
    save_btn.click(
        save_and_next,
        inputs=[choice, state_data, state_results, state_out_checked, state_out_full, state_idx, state_reviewer],
        outputs=[item_json, status, state_data, state_results, state_out_checked, state_out_full, state_idx, state_reviewer, download_full]
    )

#user athuntication for test
if __name__ == "__main__":
    demo.launch(share=True, auth=[("reviewer1","987asd"), ("reviewer2","123qwe")])
