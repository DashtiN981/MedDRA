from __future__ import annotations

import os
import socket
from typing import Any

import gradio as gr
import pandas as pd

from rag_core import get_resources, predict


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().casefold().split())


def _term_to_llt_code(term: str, term_norm_to_llt: dict[str, str]) -> str | None:
    t = _norm_text(term)
    if not t:
        return None
    if t in term_norm_to_llt:
        return term_norm_to_llt[t]
    return None


def _format_field(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value)


def _build_top5_table(top5_preview: list[Any]) -> pd.DataFrame:
    resources = get_resources()
    rows = []

    for idx, item in enumerate(top5_preview[:5], start=1):
        if isinstance(item, dict):
            llt_term = item.get("llt_term") or item.get("term") or item.get("LLT_Term") or ""
            llt_code = item.get("llt_code") or item.get("LLT_Code")
            score = item.get("score")
        else:
            llt_term = str(item)
            llt_code = None
            score = None

        if not llt_code:
            llt_code = _term_to_llt_code(llt_term, resources.term_norm_to_llt)

        pt_code = resources.llt_to_pt.get(llt_code) if llt_code else None
        soc_code = resources.pt_code_to_primary_soc.get(pt_code) if pt_code else None

        rows.append(
            {
                "rank": idx,
                "llt_term": llt_term,
                "llt_code": llt_code,
                "score": score,
                "pt_code": pt_code,
                "soc_code": soc_code,
            }
        )

    return pd.DataFrame(rows, columns=["rank", "llt_term", "llt_code", "score", "pt_code", "soc_code"])


def _predict_ui(ae_text: str):
    if not ae_text or not ae_text.strip():
        return (
            "Please enter an AE description to get a prediction.",
            pd.DataFrame(columns=["rank", "llt_term", "llt_code", "score", "pt_code", "soc_code"]),
            {},
        )

    result = predict(ae_text.strip())
    pred = result.get("pred", {})

    md = "\n".join(
        [
            f"**Predicted LLT:** {_format_field(pred.get('pred_LLT_term'))} (Code: {_format_field(pred.get('pred_LLT_Code'))})",
            f"**Predicted PT:** {_format_field(pred.get('pred_PT_term'))} (Code: {_format_field(pred.get('pred_PT_Code'))})",
            f"**Predicted SOC:** {_format_field(pred.get('pred_SOC_Term'))} (Code: {_format_field(pred.get('pred_SOC_Code'))})",
        ]
    )

    top5 = result.get("top5_preview") or []
    table = _build_top5_table(top5)

    return md, table, result


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MedDRA RAG Prototype – Demo 1") as demo:
        gr.Markdown("# MedDRA RAG Prototype – Demo 1")

        ae_input = gr.Textbox(
            label="AE Description",
            lines=5,
            placeholder="Enter AE description here...",
        )
        predict_btn = gr.Button("Predict")

        output_md = gr.Markdown()
        output_table = gr.Dataframe(
            headers=["rank", "llt_term", "llt_code", "score", "pt_code", "soc_code"],
            datatype=["number", "str", "str", "number", "str", "str"],
            row_count=5,
            col_count=6,
        )

        with gr.Accordion("Debug", open=False):
            output_json = gr.JSON()

        predict_btn.click(
            _predict_ui,
            inputs=[ae_input],
            outputs=[output_md, output_table, output_json],
        )

    return demo


def find_free_port(start: int = 7860, end: int = 7870) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free ports available in range {start}-{end}.")


def main() -> None:
    get_resources()
    desired_port = int(os.getenv("PORT", "7860"))
    free_port = find_free_port(start=desired_port, end=7870)
    share_env = os.getenv("GRADIO_SHARE", "false").strip().lower()
    share_enabled = share_env in {"1", "true", "yes"}
    print(f"[gradio] share enabled: {share_enabled}")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=free_port,
        show_api=False,
        share=share_enabled,
    )


if __name__ == "__main__":
    main()
