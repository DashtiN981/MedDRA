from __future__ import annotations

import os
import socket
import time
from typing import Any

import gradio as gr
import pandas as pd
from rapidfuzz import fuzz, process

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

    preview = df.head(10)
    columns = list(df.columns)

    if "AE_Term" in columns:
        default_col = "AE_Term"
    elif "AE_Text" in columns:
        default_col = "AE_Text"
    else:
        default_col = columns[0] if columns else ""

    return df, preview, columns, default_col


def _on_load(file_obj):
    df, preview, columns, default_col = _load_file(file_obj)
    return (
        preview,
        gr.Dropdown(choices=columns, value=default_col),
        df,
    )


def _run_predictions(
    df: pd.DataFrame,
    ae_column: str,
    top_k: int,
    concurrency: int,
    max_rows_to_run: int,
):
    if df is None or df.empty or not ae_column:
        empty = pd.DataFrame()
        return empty, [], empty, 0, "No data loaded.", gr.Dropdown(choices=[], value=None)

    df_used = df.iloc[:max_rows_to_run].copy()
    ae_texts = df_used[ae_column].fillna("").astype(str).tolist()
    predictions = batch_predict_parallel(ae_texts, top_k=top_k, concurrency=concurrency)

    rows = []
    for idx, (row_id, ae_text, pred) in enumerate(
        zip(df_used.index.tolist(), ae_texts, predictions), start=1
    ):
        pred_fields = pred.get("pred", {})
        top5 = pred.get("top5_preview") or []
        top5_terms = " | ".join(_top5_item_to_term(item) for item in top5[:5])

        rows.append(
            {
                "row_id": row_id,
                "AE_text": ae_text,
                "pred_LLT_term": pred_fields.get("pred_LLT_term"),
                "pred_LLT_Code": pred_fields.get("pred_LLT_Code"),
                "pred_PT_Code": pred_fields.get("pred_PT_Code"),
                "pred_SOC_Code": pred_fields.get("pred_SOC_Code"),
                "pred_SOC_Term": pred_fields.get("pred_SOC_Term"),
                "top5_terms": top5_terms,
                "selected_LLT_term": "",
                "selected_LLT_Code": "",
                "selection_source": "",
                "decision": "pending",
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

    current_md, top5_dropdown = _render_current(0, review_df, predictions)

    view_df = review_df[
        [
            "row_id",
            "AE_text",
            "pred_LLT_term",
            "top5_terms",
            "pred_PT_Code",
            "pred_SOC_Code",
            "pred_SOC_Term",
            "selected_LLT_term",
            "selected_LLT_Code",
            "selection_source",
            "decision",
        ]
    ].copy()

    return review_df, predictions, view_df, 0, current_md, top5_dropdown


def _extract_top5_terms(pred: dict) -> list[str]:
    top5 = pred.get("top5_preview") or []
    terms: list[str] = []
    for item in top5[:5]:
        if isinstance(item, dict):
            terms.append(item.get("llt_term") or item.get("term") or item.get("LLT_Term") or "")
        else:
            terms.append(str(item))
    return [t for t in terms if t]


def _top5_item_to_term(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("llt_term") or item.get("term") or item.get("LLT_Term") or "")
    return str(item)


def _render_current(cursor: int, review_df: pd.DataFrame, predictions: list[dict]):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        return "No row selected.", gr.Dropdown(choices=[], value=None)

    row = review_df.iloc[cursor]
    pred = predictions[cursor]
    pred_fields = pred.get("pred", {})
    debug = pred.get("debug", {})

    md = "\n".join(
        [
            f"**AE_text:** {row.get('AE_text', '')}",
            f"**Predicted LLT:** {pred_fields.get('pred_LLT_term')} (Code: {pred_fields.get('pred_LLT_Code')})",
            f"**Predicted PT Code:** {pred_fields.get('pred_PT_Code')}",
            f"**Predicted SOC:** {pred_fields.get('pred_SOC_Code')} ({pred_fields.get('pred_SOC_Term')})",
            f"**used_fallback:** {debug.get('used_fallback')} | **latency_ms:** {debug.get('latency_ms')}",
        ]
    )

    top5_terms = _extract_top5_terms(pred)
    predicted_term = pred_fields.get("pred_LLT_term")
    default_term = top5_terms[0] if top5_terms else None
    if default_term not in top5_terms:
        default_term = None
    return md, gr.Dropdown(choices=top5_terms, value=default_term)


def _derive_llt_code(term: str) -> str | None:
    if not term:
        return None
    resources = get_resources()
    t = " ".join(str(term).strip().casefold().split())
    exact = resources.term_norm_to_llt.get(t)
    if exact:
        return exact

    best = process.extractOne(
        t,
        list(resources.term_norm_to_llt.keys()),
        scorer=fuzz.ratio,
        score_cutoff=94,
    )
    if best:
        return resources.term_norm_to_llt.get(best[0])
    return None


def _accept_for_cursor(
    cursor: int,
    review_df: pd.DataFrame,
    predictions: list[dict],
):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        view_df = pd.DataFrame()
        return view_df, review_df, cursor, "No row selected.", gr.Dropdown(choices=[], value=None), ""

    pred = predictions[cursor]
    pred_fields = pred.get("pred", {})

    row_id = review_df.iloc[cursor]["row_id"]
    review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = pred_fields.get("pred_LLT_term") or ""
    review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = pred_fields.get("pred_LLT_Code") or ""
    review_df.loc[review_df["row_id"] == row_id, "selection_source"] = "model"
    review_df.loc[review_df["row_id"] == row_id, "decision"] = "validated"

    next_cursor = cursor + 1 if cursor + 1 < len(predictions) else cursor
    current_md, top5_dropdown = _render_current(next_cursor, review_df, predictions)
    view_df = review_df[
        [
            "row_id",
            "AE_text",
            "pred_LLT_term",
            "top5_terms",
            "pred_PT_Code",
            "pred_SOC_Code",
            "pred_SOC_Term",
            "selected_LLT_term",
            "selected_LLT_Code",
            "selection_source",
            "decision",
        ]
    ].copy()
    return view_df, review_df, next_cursor, current_md, top5_dropdown, ""


def _reject_for_cursor(
    cursor: int,
    chosen_term: str,
    manual_term: str,
    review_df: pd.DataFrame,
    predictions: list[dict],
):
    if review_df is None or review_df.empty or cursor < 0 or cursor >= len(predictions):
        view_df = pd.DataFrame()
        return view_df, review_df, cursor, "No row selected.", gr.Dropdown(choices=[], value=None), ""

    pred = predictions[cursor]
    top5 = pred.get("top5_preview") or []
    top5_terms = _extract_top5_terms(pred)
    manual_term_clean = (manual_term or "").strip()
    chosen_term_effective = manual_term_clean or chosen_term

    has_replacement = bool(chosen_term_effective)
    chosen_code = None
    if chosen_term_effective:
        for item in top5[:5]:
            if isinstance(item, dict):
                term = item.get("llt_term") or item.get("term") or item.get("LLT_Term")
                if term == chosen_term_effective:
                    chosen_code = item.get("llt_code") or item.get("LLT_Code")
                    break
    if not chosen_code:
        chosen_code = _derive_llt_code(chosen_term_effective)

    if manual_term_clean:
        selection_source = "manual"
    elif chosen_term_effective in top5_terms:
        selection_source = "top5"
    else:
        selection_source = "manual"

    row_id = review_df.iloc[cursor]["row_id"]
    if has_replacement:
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = chosen_term_effective or ""
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = chosen_code or ""
        review_df.loc[review_df["row_id"] == row_id, "selection_source"] = selection_source
        review_df.loc[review_df["row_id"] == row_id, "decision"] = "rejected_with_replacement"
    else:
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_term"] = ""
        review_df.loc[review_df["row_id"] == row_id, "selected_LLT_Code"] = ""
        review_df.loc[review_df["row_id"] == row_id, "selection_source"] = ""
        review_df.loc[review_df["row_id"] == row_id, "decision"] = "rejected"

    next_cursor = cursor + 1 if cursor + 1 < len(predictions) else cursor
    current_md, top5_dropdown = _render_current(next_cursor, review_df, predictions)
    view_df = review_df[
        [
            "row_id",
            "AE_text",
            "pred_LLT_term",
            "top5_terms",
            "pred_PT_Code",
            "pred_SOC_Code",
            "pred_SOC_Term",
            "selected_LLT_term",
            "selected_LLT_Code",
            "selection_source",
            "decision",
        ]
    ].copy()
    return view_df, review_df, next_cursor, current_md, top5_dropdown, ""


def _move_cursor(delta: int, cursor: int, review_df: pd.DataFrame, predictions: list[dict]):
    if review_df is None or review_df.empty:
        return cursor, "No row selected.", gr.Dropdown(choices=[], value=None)
    new_cursor = min(max(cursor + delta, 0), len(predictions) - 1)
    current_md, top5_dropdown = _render_current(new_cursor, review_df, predictions)
    return new_cursor, current_md, top5_dropdown


def _goto_cursor(target: int, review_df: pd.DataFrame, predictions: list[dict]):
    if review_df is None or review_df.empty:
        return 0, "No row selected.", gr.Dropdown(choices=[], value=None)
    new_cursor = min(max(int(target or 0), 0), len(predictions) - 1)
    current_md, top5_dropdown = _render_current(new_cursor, review_df, predictions)
    return new_cursor, current_md, top5_dropdown


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MedDRA RAG Prototype – Demo 2 (Batch Review)") as demo:
        gr.Markdown("# MedDRA RAG Prototype – Demo 2 (Batch Review)")

        file_input = gr.File(label="Upload File (.csv or .xlsx)")
        load_btn = gr.Button("Load File")

        ae_column = gr.Dropdown(label="AE column", choices=[])
        top_k_input = gr.Number(label="top_k", value=int(os.getenv("TOP_K", "30")))
        concurrency_input = gr.Number(label="concurrency", value=int(os.getenv("CONCURRENCY", "5")))
        max_rows_input = gr.Number(label="max_rows_to_run", value=10)
        run_btn = gr.Button("Run Predictions")

        preview_df = gr.Dataframe(label="Preview (First 10 Rows)")
        results_df = gr.Dataframe(
            label="Results",
            headers=[
                "row_id",
                "AE_text",
                "pred_LLT_term",
                "top5_terms",
                "pred_PT_Code",
                "pred_SOC_Code",
                "pred_SOC_Term",
                "selected_LLT_term",
                "selected_LLT_Code",
                "selection_source",
                "decision",
            ],
        )
        export_btn = gr.Button("Export to Excel")
        export_file = gr.File(label="Download Export")

        gr.Markdown("## Validation Panel")
        current_md = gr.Markdown()
        top5_select = gr.Dropdown(
            label="Choose LLT from top-5",
            choices=[],
            allow_custom_value=True,
        )
        gr.Markdown("Accept = keep model output. Reject = provide replacement from Top-5 or Manual field.")
        manual_term = gr.Textbox(label="Manual LLT term (optional)")
        prev_btn = gr.Button("Prev")
        next_btn = gr.Button("Next")
        accept_btn = gr.Button("Accept")
        reject_btn = gr.Button("Reject")
        goto_input = gr.Number(label="Go to row", value=0)

        df_state = gr.State(pd.DataFrame())
        pred_state = gr.State([])
        review_state = gr.State(pd.DataFrame())
        cursor_state = gr.State(0)

        load_btn.click(
            _on_load,
            inputs=[file_input],
            outputs=[preview_df, ae_column, df_state],
        )

        run_btn.click(
            _run_predictions,
            inputs=[df_state, ae_column, top_k_input, concurrency_input, max_rows_input],
            outputs=[results_df, pred_state, review_state, cursor_state, current_md, top5_select],
        )

        prev_btn.click(
            lambda cursor, review_df, predictions: _move_cursor(-1, cursor, review_df, predictions),
            inputs=[cursor_state, review_state, pred_state],
            outputs=[cursor_state, current_md, top5_select],
        )

        next_btn.click(
            lambda cursor, review_df, predictions: _move_cursor(1, cursor, review_df, predictions),
            inputs=[cursor_state, review_state, pred_state],
            outputs=[cursor_state, current_md, top5_select],
        )

        goto_input.change(
            _goto_cursor,
            inputs=[goto_input, review_state, pred_state],
            outputs=[cursor_state, current_md, top5_select],
        )

        accept_btn.click(
            _accept_for_cursor,
            inputs=[cursor_state, review_state, pred_state],
            outputs=[results_df, review_state, cursor_state, current_md, top5_select, manual_term],
        )

        reject_btn.click(
            _reject_for_cursor,
            inputs=[
                cursor_state,
                top5_select,
                manual_term,
                review_state,
                pred_state,
            ],
            outputs=[results_df, review_state, cursor_state, current_md, top5_select, manual_term],
        )

        def _export_review(review_df: pd.DataFrame):
            if review_df is None or review_df.empty:
                return None
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"/tmp/MedDRA_RAG_Demo2_review_{ts}.xlsx"
            review_df.to_excel(out_path, index=False)
            return out_path

        export_btn.click(
            _export_review,
            inputs=[review_state],
            outputs=[export_file],
        )

    return demo


def main() -> None:
    get_resources()
    desired_port = int(os.getenv("PORT", "7860"))
    free_port = find_free_port(start=desired_port, end=7870)
    app = build_app()
    share_env = os.getenv("SHARE", "false").strip().lower()
    share_enabled = share_env in {"1", "true", "yes"}
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
