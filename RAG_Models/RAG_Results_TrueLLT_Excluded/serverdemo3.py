from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from rag_core import batch_predict_parallel, get_resources


app = Flask(__name__)


@dataclass
class JobState:
    file_name: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    review_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cursor: int = 0
    run_status: str = "idle"
    run_started_at: float | None = None
    run_finished_at: float | None = None
    run_total_rows: int = 0
    run_processed_rows: int = 0
    run_progress_pct: float = 0.0
    run_error: str = ""


JOBS: dict[str, JobState] = {}


def _job_or_none(job_id: str | None) -> JobState | None:
    if not job_id:
        return None
    return JOBS.get(job_id)


def _top5_item_to_term(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("llt_term") or item.get("term") or item.get("LLT_Term") or "")
    return str(item)


def _derive_llt_code(term: str) -> str | None:
    if not term:
        return None
    resources = get_resources()
    normalized = " ".join(str(term).strip().casefold().split())
    return resources.term_norm_to_llt.get(normalized)


def _read_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.filename or ""
    if file_name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_csv(uploaded_file, sep=",")
    except Exception:
        uploaded_file.stream.seek(0)
        return pd.read_csv(uploaded_file, sep=";")


def _default_ae_column(columns: list[str]) -> str:
    if "AE_Term" in columns:
        return "AE_Term"
    if "AE_Text" in columns:
        return "AE_Text"
    return columns[0] if columns else ""


def _preview_records(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.head(n).fillna("").to_dict(orient="records")


def _build_review_table(df_used: pd.DataFrame, ae_texts: list[str], predictions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i in range(len(df_used)):
        row_id = int(df_used.index.tolist()[i])
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
                "top5_terms": " | ".join(_top5_item_to_term(it) for it in top5_preview[:5]),
                "selected_LLT_term": "",
                "selected_LLT_Code": "",
                "selection_source": "",
                "decision": decision,
            }
        )

    return pd.DataFrame(
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


def _results_view(review_df: pd.DataFrame) -> list[dict[str, Any]]:
    if review_df is None or review_df.empty:
        return []
    view_df = review_df[
        [
            "row_id",
            "AE_text",
            "pred_LLT_term",
            "selected_LLT_term",
            "selected_LLT_Code",
            "selection_source",
            "decision",
        ]
    ].copy()
    return view_df.fillna("").to_dict(orient="records")


def _candidate_rows(prediction: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    resources = get_resources()
    terms = (prediction or {}).get("candidates_retrieved_100") or []
    if not terms:
        return []

    rows = []
    denom = max(limit - 1, 1)
    for idx, term in enumerate(terms[:limit], start=1):
        llt_code = resources.term_norm_to_llt.get(" ".join(str(term).strip().casefold().split()))
        pt_code = resources.llt_to_pt.get(llt_code) if llt_code else None
        soc_code = resources.pt_code_to_primary_soc.get(pt_code) if pt_code else None
        score = round(100 - ((idx - 1) * (70 / denom)), 1)
        rows.append(
            {
                "rank": idx,
                "llt_term": term,
                "llt_code": llt_code or "",
                "pt_code": pt_code or "",
                "soc_code": soc_code or "",
                "score": score,
            }
        )
    return rows


def _current_payload(job: JobState) -> dict[str, Any]:
    if job.review_df is None or job.review_df.empty or not job.predictions:
        return {
            "progress": "Item 0 / 0",
            "cursor": 0,
            "current": None,
            "top5_choices": [],
            "table": _results_view(job.review_df),
            "candidates": [],
        }

    cursor = min(max(job.cursor, 0), len(job.predictions) - 1)
    job.cursor = cursor
    row = job.review_df.iloc[cursor]
    pred = job.predictions[cursor]
    pred_fields = (pred or {}).get("pred", {})
    debug = (pred or {}).get("debug", {})
    top5_terms = [_top5_item_to_term(it) for it in (pred.get("top5_preview") or [])[:5] if _top5_item_to_term(it)]

    current = {
        "row_id": int(row.get("row_id")),
        "ae_text": str(row.get("AE_text", "")),
        "pred_LLT_term": pred_fields.get("pred_LLT_term") or "",
        "pred_LLT_Code": pred_fields.get("pred_LLT_Code") or "",
        "pred_PT_Code": pred_fields.get("pred_PT_Code") or "",
        "pred_SOC_Code": pred_fields.get("pred_SOC_Code") or "",
        "pred_SOC_Term": pred_fields.get("pred_SOC_Term") or "",
        "latency_ms": debug.get("latency_ms"),
        "used_fallback": bool(debug.get("used_fallback")),
        "selected_LLT_term": str(row.get("selected_LLT_term", "")),
        "selected_LLT_Code": str(row.get("selected_LLT_Code", "")),
        "selection_source": str(row.get("selection_source", "")),
        "decision": str(row.get("decision", "")),
    }

    return {
        "progress": f"Item {cursor + 1} / {len(job.predictions)}",
        "cursor": cursor,
        "current": current,
        "top5_choices": top5_terms,
        "table": _results_view(job.review_df),
        "candidates": _candidate_rows(pred, limit=10),
    }


def _run_predictions_background(job_id: str, ae_column: str) -> None:
    job = JOBS.get(job_id)
    if job is None:
        return

    job.run_status = "running"
    job.run_started_at = time.time()
    job.run_finished_at = None
    job.run_total_rows = 0
    job.run_processed_rows = 0
    job.run_progress_pct = 0.0
    job.run_error = ""
    job.predictions = []
    job.review_df = pd.DataFrame()
    job.cursor = 0

    try:
        top_k = int(os.getenv("TOP_K", "100"))
        concurrency = int(os.getenv("CONCURRENCY", "50"))
        max_rows = int(os.getenv("MAX_ROWS", "0"))

        df_used = job.df.iloc[:max_rows].copy() if max_rows > 0 else job.df.copy()
        ae_texts = df_used[ae_column].fillna("").astype(str).tolist()
        total = len(ae_texts)
        job.run_total_rows = total

        if total == 0:
            job.predictions = []
            job.review_df = _build_review_table(df_used, ae_texts, [])
            job.run_status = "completed"
            job.run_progress_pct = 100.0
            job.run_finished_at = time.time()
            return

        predictions: list[dict[str, Any]] = []
        chunk_size = max(1, concurrency)

        for start in range(0, total, chunk_size):
            chunk_texts = ae_texts[start : start + chunk_size]
            chunk_predictions = batch_predict_parallel(
                chunk_texts,
                top_k=top_k,
                concurrency=min(concurrency, len(chunk_texts)),
            )
            predictions.extend(chunk_predictions)
            job.run_processed_rows = len(predictions)
            job.run_progress_pct = min(99.0, (job.run_processed_rows / total) * 100.0)

        review_df = _build_review_table(df_used, ae_texts, predictions)
        job.predictions = predictions
        job.review_df = review_df
        job.cursor = 0
        job.run_status = "completed"
        job.run_progress_pct = 100.0
        job.run_finished_at = time.time()
        print(f"[demo3-flask] requested={total} got_predictions={len(predictions)}")
    except Exception as exc:
        job.run_status = "error"
        job.run_error = str(exc)
        job.run_finished_at = time.time()


def _find_free_port(start: int = 7860, end: int = 7870) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free ports available in range {start}-{end}.")


def _maybe_start_ngrok(port: int) -> str | None:
    enable = os.getenv("ENABLE_NGROK", "false").strip().lower() in {"1", "true", "yes"}
    if not enable:
        return None

    try:
        from pyngrok import ngrok
    except Exception:
        print("[demo3-flask] ENABLE_NGROK is true but pyngrok is not installed.")
        return None

    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if token:
        ngrok.set_auth_token(token)

    tunnel = ngrok.connect(addr=port, bind_tls=True)
    public_url = getattr(tunnel, "public_url", None)
    if public_url:
        print(f"[demo3-flask] public URL: {public_url}")
    return public_url


@app.get("/")
def index() -> str:
    return render_template("index_demo3.html")


@app.post("/api/upload")
def api_upload():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a .csv or .xlsx file."}), 400

    try:
        df = _read_uploaded_dataframe(uploaded_file)
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 400

    columns = [str(c) for c in df.columns.tolist()]
    default_col = _default_ae_column(columns)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobState(file_name=uploaded_file.filename, df=df)

    return jsonify(
        {
            "job_id": job_id,
            "file_name": uploaded_file.filename,
            "columns": columns,
            "default_column": default_col,
            "preview": _preview_records(df),
            "row_count": int(len(df)),
        }
    )


@app.post("/api/run")
def api_run():
    payload = request.get_json(silent=True) or {}
    job_id = str(payload.get("job_id") or "")
    job = _job_or_none(job_id)
    if job is None:
        return jsonify({"error": "Invalid job_id. Upload file again."}), 400

    ae_column = str(payload.get("ae_column") or "").strip()
    if not ae_column or ae_column not in job.df.columns:
        return jsonify({"error": "Please select a valid AE column."}), 400

    if job.run_status == "running":
        return jsonify({"error": "Prediction is already running for this file."}), 400

    thread = threading.Thread(target=_run_predictions_background, args=(job_id, ae_column), daemon=True)
    thread.start()
    return jsonify({"status": "running"})


@app.get("/api/run_status/<job_id>")
def api_run_status(job_id: str):
    job = _job_or_none(job_id)
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400

    now = time.time()
    started = job.run_started_at
    finished = job.run_finished_at

    if started is None:
        elapsed_ms = 0
    elif finished is None:
        elapsed_ms = int((now - started) * 1000)
    else:
        elapsed_ms = int((finished - started) * 1000)

    payload: dict[str, Any] = {
        "status": job.run_status,
        "progress_pct": round(job.run_progress_pct, 1),
        "processed_rows": int(job.run_processed_rows),
        "total_rows": int(job.run_total_rows),
        "elapsed_ms": elapsed_ms,
        "error": job.run_error,
    }
    if job.run_status == "completed":
        payload["result"] = _current_payload(job)
    return jsonify(payload)


@app.get("/api/state/<job_id>")
def api_state(job_id: str):
    job = _job_or_none(job_id)
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400
    return jsonify(_current_payload(job))


@app.post("/api/nav")
def api_nav():
    payload = request.get_json(silent=True) or {}
    job = _job_or_none(payload.get("job_id"))
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400

    delta = int(payload.get("delta") or 0)
    if job.predictions:
        job.cursor = min(max(job.cursor + delta, 0), len(job.predictions) - 1)

    return jsonify(_current_payload(job))


@app.post("/api/accept")
def api_accept():
    payload = request.get_json(silent=True) or {}
    job = _job_or_none(payload.get("job_id"))
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400

    if job.review_df.empty or not job.predictions:
        return jsonify({"error": "No prediction data loaded."}), 400

    cursor = job.cursor
    pred_fields = (job.predictions[cursor] or {}).get("pred", {})
    row_id = job.review_df.iloc[cursor]["row_id"]

    job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_term"] = pred_fields.get("pred_LLT_term") or ""
    job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_Code"] = pred_fields.get("pred_LLT_Code") or ""
    job.review_df.loc[job.review_df["row_id"] == row_id, "selection_source"] = "model"
    job.review_df.loc[job.review_df["row_id"] == row_id, "decision"] = "accepted"

    if cursor + 1 < len(job.predictions):
        job.cursor = cursor + 1

    return jsonify(_current_payload(job))


@app.post("/api/reject")
def api_reject():
    payload = request.get_json(silent=True) or {}
    job = _job_or_none(payload.get("job_id"))
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400

    if job.review_df.empty or not job.predictions:
        return jsonify({"error": "No prediction data loaded."}), 400

    chosen_term = str(payload.get("chosen_term") or "").strip()
    manual_term = str(payload.get("manual_term") or "").strip()
    chosen_effective = manual_term or chosen_term

    if manual_term:
        source = "manual"
    elif chosen_effective:
        source = "top5"
    else:
        source = ""

    row_id = job.review_df.iloc[job.cursor]["row_id"]

    if chosen_effective:
        code = _derive_llt_code(chosen_effective)
        job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_term"] = chosen_effective
        job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_Code"] = code or ""
        job.review_df.loc[job.review_df["row_id"] == row_id, "selection_source"] = source
        job.review_df.loc[job.review_df["row_id"] == row_id, "decision"] = "rejected_with_replacement"
    else:
        job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_term"] = ""
        job.review_df.loc[job.review_df["row_id"] == row_id, "selected_LLT_Code"] = ""
        job.review_df.loc[job.review_df["row_id"] == row_id, "selection_source"] = ""
        job.review_df.loc[job.review_df["row_id"] == row_id, "decision"] = "rejected"

    if job.cursor + 1 < len(job.predictions):
        job.cursor += 1

    return jsonify(_current_payload(job))


@app.get("/api/export/<job_id>")
def api_export(job_id: str):
    job = _job_or_none(job_id)
    if job is None:
        return jsonify({"error": "Invalid job_id."}), 400

    if job.review_df is None or job.review_df.empty:
        return jsonify({"error": "No review data to export."}), 400

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"/tmp/MedDRA_RAG_Demo3_review_{ts}.xlsx"

    export_df = job.review_df[
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

    return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))


if __name__ == "__main__":
    get_resources()

    desired_port = int(os.getenv("PORT", "7860"))
    free_port = _find_free_port(start=desired_port, end=7870)

    print(
        f"[demo3-flask] PORT={free_port} "
        f"TOP_K={os.getenv('TOP_K', '100')} CONCURRENCY={os.getenv('CONCURRENCY', '50')} "
        f"MAX_ROWS={os.getenv('MAX_ROWS', '0')}"
    )

    _maybe_start_ngrok(free_port)
    app.run(host="0.0.0.0", port=free_port, debug=False)
