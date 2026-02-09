from __future__ import annotations

import os
import socket
import time
from typing import Any

import gradio as gr
import pandas as pd

from rag_core import batch_predict_parallel, get_resources


def find_free_port(start: int = 7860, end: int = 7870) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free ports available in range {start}-{end}.")


def _load_file(file_obj) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    if file_obj is None:
        empty = pd.DataFrame()
        return empty, empty, [], ""

    path = file_obj.name
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, sep=",")
        except Exception:
            df = pd.read_csv(path, sep=";")

    preview = df.head(5)
    columns = list(df.columns)

    if "AE_Term" in columns:
        default_col = "AE_Term"
    elif "AE_Text" in columns:
        default_col = "AE_Text"
    else:
        default_col = columns[0] if columns else ""

    return df, preview, columns, default_col


def _on_file_change(file_obj):
    df, preview, columns, default_col = _load_file(file_obj)
    return (
        preview,
        gr.Dropdown(choices=columns, value=default_col),
        df,
    )


def _top5_item_to_term(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("llt_term") or item.get("term") or item.get("LLT_Term") or "")
    return str(item)


def _run_predictions(df: pd.DataFrame, ae_column: str):
    if df is None or df.empty or not ae_column:
        empty = pd.DataFrame()
        return (
            empty,
            [],
            empty,
            0,
            "Item 0 / 0",
            "",
            "No data loaded.",
            gr.Dropdown(choices=[], value=None),
        )

    top_k = int(os.getenv("TOP_K", "100"))
    concurrency = int(os.getenv("CONCURRENCY", "50"))
    max_rows = int(os.getenv("MAX_ROWS", "0"))

    if max_rows > 0:
        df_used = df.iloc[:max_rows].copy()
    else:
        df_used = df.copy()
    #df_used = df.iloc[:max_rows].copy()
    ae_texts = df_used[ae_column].fillna("").astype(str).tolist()
    predictions = batch_predict_parallel(ae_texts, top_k=top_k, concurrency=concurrency)
    print(f"[Demo3] requested={len(ae_texts)} got_predictions={len(predictions)}")

    rows = []
    n = len(df_used)
    for i in range(n):
        row_id = df_used.index.tolist()[i]
        ae_text = ae_texts[i]
        pred = predictions[i] if i < len(predictions) else {}
        pred_fields = (pred or {}).get("pred", {})
        top5_preview = (pred or {}).get("top5_preview") or []
        decision = "pending" if i < len(predictions) else "prediction_failed"
        rows.append(
            {
                "row_id": row_id,
                "AE_text": ae_text,
                "pred_LLT_term": pred_fields.get("pred_LLT_term") or "",
                "pred_LLT_Code": pred_fields.get("pred_LLT_Code") or "",
                "pred_PT_Code": pred_fields.get("pred_PT_Code") or "",
                "pred_SOC_Code": pred_fields.get("pred_SOC_Code") or "",
                "pred_SOC_Term": pred_fields.get("pred_SOC_Term") or "",
                "top5_terms": " | ".join(_top5_item_to_term(i) for i in top5_preview[:5]),
                "selected_LLT_term": "",
                "selected_LLT_Code": "",
                "selection_source": "",
                "decision": decision,
            }
        )

    review_df = pd.DataFrame(
        rows,
        columns=[
            "row_id",
            "AE_text",
            "pred_LLT_term",
            "pred_LLT_Code",
            "pred_PT_Code",
            "pred_SOC_Code",
            "pred_SOC_Term",
            "top5_terms",
            "selected_LLT_term",
            "selected_LLT_Code",
            "selection_source",
            "decision",
        ],
    )

    view_df = review_df[
        ["row_id", "AE_text", "pred_LLT_term", "selected_LLT_term", "selected_LLT_Code", "selection_source", "decision"]
    ].copy()

    progress_md, ae_text_full, pred_md, top5_dropdown = _render_current(0, review_df, predictions)
    return review_df, predictions, view_df, 0, progress_md, ae_text_full, pred_md, top5_dropdown


def _render_current(cursor: int, review_df: pd.DataFrame, predictions: list[dict]):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        return "Item 0 / 0", "", "No row selected.", gr.Dropdown(choices=[], value=None)

    row = review_df.iloc[cursor]
    pred = predictions[cursor]
    pred_fields = pred.get("pred", {})
    debug = pred.get("debug", {})
    top5_terms = [_top5_item_to_term(i) for i in (pred.get("top5_preview") or [])[:5] if _top5_item_to_term(i)]
    pred_md = "\n".join(
        [
            f"**Predicted LLT:** {pred_fields.get('pred_LLT_term')} (Code: {pred_fields.get('pred_LLT_Code')})",
            f"**Predicted PT Code:** {pred_fields.get('pred_PT_Code')}",
            f"**Predicted SOC:** {pred_fields.get('pred_SOC_Code')} ({pred_fields.get('pred_SOC_Term')})",
            f"**Latency (ms):** {debug.get('latency_ms')} | **Used fallback:** {debug.get('used_fallback')}",
        ]
    )

    default_term = top5_terms[0] if top5_terms else None
    if default_term not in top5_terms:
        default_term = None
    progress_md = f"Item {cursor + 1} / {len(predictions)}"
    return progress_md, str(row.get("AE_text", "")), pred_md, gr.Dropdown(choices=top5_terms, value=default_term)


def _move_cursor(delta: int, cursor: int, review_df: pd.DataFrame, predictions: list[dict]):
    if review_df is None or review_df.empty:
        return cursor, "Item 0 / 0", "", "No row selected.", gr.Dropdown(choices=[], value=None)
    new_cursor = min(max(cursor + delta, 0), len(predictions) - 1)
    progress_md, ae_text_full, pred_md, top5_dropdown = _render_current(new_cursor, review_df, predictions)
    return new_cursor, progress_md, ae_text_full, pred_md, top5_dropdown


def _derive_llt_code(term: str) -> str | None:
    if not term:
        return None
    resources = get_resources()
    t = " ".join(str(term).strip().casefold().split())
    return resources.term_norm_to_llt.get(t)


def _apply_accept(
    cursor: int,
    review_df: pd.DataFrame,
    predictions: list[dict],
):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        view_df = pd.DataFrame()
        return view_df, review_df, cursor, "Item 0 / 0", "", "No row selected.", gr.Dropdown(choices=[], value=None), ""

    pred_fields = predictions[cursor].get("pred", {})
    row_id = review_df.iloc[cursor]["row_id"]
    review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = pred_fields.get("pred_LLT_term") or ""
    review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = pred_fields.get("pred_LLT_Code") or ""
    review_df.loc[review_df["row_id"] == row_id, "selection_source"] = "model"
    review_df.loc[review_df["row_id"] == row_id, "decision"] = "accepted"

    next_cursor = cursor + 1 if cursor + 1 < len(predictions) else cursor
    progress_md, ae_text_full, pred_md, top5_dropdown = _render_current(next_cursor, review_df, predictions)

    view_df = review_df[
        ["row_id", "AE_text", "pred_LLT_term", "selected_LLT_term", "selected_LLT_Code", "selection_source", "decision"]
    ].copy()
    return view_df, review_df, next_cursor, progress_md, ae_text_full, pred_md, top5_dropdown, ""


def _apply_reject(
    cursor: int,
    chosen_term: str,
    manual_term: str,
    review_df: pd.DataFrame,
    predictions: list[dict],
):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        view_df = pd.DataFrame()
        return view_df, review_df, cursor, "Item 0 / 0", "", "No row selected.", gr.Dropdown(choices=[], value=None), ""

    manual_term_clean = (manual_term or "").strip()
    chosen_term_effective = manual_term_clean or (chosen_term or "")

    if manual_term_clean:
        selection_source = "manual"
    elif chosen_term_effective:
        selection_source = "top5"
    else:
        selection_source = ""

    row_id = review_df.iloc[cursor]["row_id"]
    if chosen_term_effective:
        code = _derive_llt_code(chosen_term_effective)
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = chosen_term_effective
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = code or ""
        review_df.loc[review_df["row_id"] == row_id, "selection_source"] = selection_source
        review_df.loc[review_df["row_id"] == row_id, "decision"] = "rejected_with_replacement"
    else:
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = ""
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = ""
        review_df.loc[review_df["row_id"] == row_id, "selection_source"] = ""
        review_df.loc[review_df["row_id"] == row_id, "decision"] = "rejected"

    next_cursor = cursor + 1 if cursor + 1 < len(predictions) else cursor
    progress_md, ae_text_full, pred_md, top5_dropdown = _render_current(next_cursor, review_df, predictions)

    view_df = review_df[
        ["row_id", "AE_text", "pred_LLT_term", "selected_LLT_term", "selected_LLT_Code", "selection_source", "decision"]
    ].copy()
    return view_df, review_df, next_cursor, progress_md, ae_text_full, pred_md, top5_dropdown, ""


def _export_xlsx(review_df: pd.DataFrame):
    if review_df is None or review_df.empty:
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"/tmp/MedDRA_RAG_Demo3_review_{ts}.xlsx"
    export_df = review_df[
        ["row_id", "AE_text", "pred_LLT_term", "selected_LLT_term", "selected_LLT_Code", "decision", "selection_source"]
    ].copy()
    export_df.columns = [
        "row_id",
        "AE_text",
        "pred_LLT_term",
        "final_LLT_term",
        "final_LLT_Code",
        "decision",
        "selection_source",
    ]
    export_df.to_excel(out_path, index=False)
    return out_path


def build_app() -> gr.Blocks:
    css = """
    .gradio-container { max-width: 1100px; margin: 0 auto; }

    /* Restore scrollable container for the dataframe */
    #results_df .table-wrap {
      max-height: 520px !important;
      overflow: auto !important;
    }

    /* Wrap long text inside cells, but keep scrolling */
    #results_df table td, #results_df table th {
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: anywhere !important;
    }
    """
    with gr.Blocks(title="MedDRA RAG Prototype – Demo 3", css=css) as demo:
        gr.Markdown("# MedDRA RAG Prototype – Demo 3")

        file_input = gr.File(label="Upload File (.csv or .xlsx)")
        ae_column = gr.Dropdown(label="AE column", choices=[])
        run_btn = gr.Button("Run Predictions")

        preview_df = gr.Dataframe(label="Preview (First 5 Rows)")

        try:
            results_df = gr.Dataframe(
                label="Results",
                headers=[
                    "row_id",
                    "AE_text",
                    "pred_LLT_term",
                    "selected_LLT_term",
                    "selected_LLT_Code",
                    "selection_source",
                    "decision",
                ],
                elem_id="results_df",
                interactive=False,
                max_height=520,
            )
        except TypeError:
            results_df = gr.Dataframe(
                label="Results",
                headers=[
                    "row_id",
                    "AE_text",
                    "pred_LLT_term",
                    "selected_LLT_term",
                    "selected_LLT_Code",
                    "selection_source",
                    "decision",
                ],
                elem_id="results_df",
                interactive=False,
            )

        gr.Markdown("## Current Item")
        with gr.Group():
            progress_md = gr.Markdown("Item 0 / 0")
            ae_full_box = gr.Textbox(label="AE text (full)", lines=3, interactive=False)
            pred_box = gr.Markdown()
        top5_select = gr.Dropdown(label="Top suggestions", choices=[], allow_custom_value=True)
        gr.Markdown(
            "✅ **Accept**: confirm the model prediction (keep model output).  \n"
            "❌ **Reject**: replace the prediction by choosing from **Top suggestions** (top-5) or typing a **Manual LLT term**."
        )
        manual_term = gr.Textbox(label="Manual LLT term (optional)")

        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            accept_btn = gr.Button("Accept")
            reject_btn = gr.Button("Reject")
            export_btn = gr.Button("Download Reviewed Excel")

        export_file = gr.File(label="Download File")

        df_state = gr.State(pd.DataFrame())
        pred_state = gr.State([])
        review_state = gr.State(pd.DataFrame())
        cursor_state = gr.State(0)

        file_input.change(
            _on_file_change,
            inputs=[file_input],
            outputs=[preview_df, ae_column, df_state],
        )

        run_btn.click(
            _run_predictions,
            inputs=[df_state, ae_column],
            outputs=[
                review_state,
                pred_state,
                results_df,
                cursor_state,
                progress_md,
                ae_full_box,
                pred_box,
                top5_select,
            ],
        )

        prev_btn.click(
            lambda cursor, review_df, predictions: _move_cursor(-1, cursor, review_df, predictions),
            inputs=[cursor_state, review_state, pred_state],
            outputs=[cursor_state, progress_md, ae_full_box, pred_box, top5_select],
        )

        next_btn.click(
            lambda cursor, review_df, predictions: _move_cursor(1, cursor, review_df, predictions),
            inputs=[cursor_state, review_state, pred_state],
            outputs=[cursor_state, progress_md, ae_full_box, pred_box, top5_select],
        )

        accept_btn.click(
            _apply_accept,
            inputs=[cursor_state, review_state, pred_state],
            outputs=[
                results_df,
                review_state,
                cursor_state,
                progress_md,
                ae_full_box,
                pred_box,
                top5_select,
                manual_term,
            ],
        )

        reject_btn.click(
            _apply_reject,
            inputs=[cursor_state, top5_select, manual_term, review_state, pred_state],
            outputs=[
                results_df,
                review_state,
                cursor_state,
                progress_md,
                ae_full_box,
                pred_box,
                top5_select,
                manual_term,
            ],
        )

        export_btn.click(
            _export_xlsx,
            inputs=[review_state],
            outputs=[export_file],
        )

    return demo


def main() -> None:
    get_resources()
    share_env = os.getenv("GRADIO_SHARE", "false").strip().lower()
    share_enabled = share_env in {"1", "true", "yes"}
    desired_port = int(os.getenv("PORT", "7860"))
    free_port = find_free_port(start=desired_port, end=7870)

    print(
        f"[Demo3] SHARE={share_enabled} PORT={free_port} "
        f"TOP_K={os.getenv('TOP_K','100')} CONCURRENCY={os.getenv('CONCURRENCY','50')} "
        f"MAX_ROWS={os.getenv('MAX_ROWS','0')}"
    )
    print(f"[gradio] share enabled: {share_enabled}")

    app = build_app()
    launch_result = app.launch(
        server_name="0.0.0.0",
        server_port=free_port,
        show_api=False,
        share=share_enabled,
    )
    if share_enabled:
        share_url = None
        if isinstance(launch_result, tuple) and launch_result:
            share_url = launch_result[-1]
        if share_url:
            print(f"[gradio] public URL: {share_url}")
        else:
            print("[gradio] public URL: (check Gradio logs)")


if __name__ == "__main__":
    main()
