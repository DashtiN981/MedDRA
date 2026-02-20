from __future__ import annotations

import os
import socket
from typing import Any

from flask import Flask, jsonify, render_template, request

from rag_core import get_resources, norm_text, predict

app = Flask(__name__)


EXAMPLES = [
    "Patient experienced severe headache with nausea and dizziness.",
    "Chest pain radiating to left arm with shortness of breath.",
    "Skin rash with itching and redness on both forearms.",
    "Persistent dry cough and low-grade fever for three days.",
    "Blurred vision with photophobia and eye pain.",
]


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

    try:
        tunnel = ngrok.connect(addr=port, bind_tls=True)
        public_url = getattr(tunnel, "public_url", None)
        if public_url:
            print(f"[demo3-flask] public URL: {public_url}")
        return public_url
    except Exception as exc:
        msg = str(exc)
        if "ERR_NGROK_334" in msg or "already online" in msg:
            print(
                "[demo3-flask] ngrok endpoint already online (ERR_NGROK_334). "
                "Stop the other ngrok session or disable reserved domain there."
            )
            print(
                "[demo3-flask] Continuing without a new ngrok tunnel. "
                "Local app is still running."
            )
        else:
            print(f"[demo3-flask] ngrok startup failed: {exc}")
            print("[demo3-flask] Continuing without ngrok tunnel.")
        return None


def _candidate_rows(ae_text: str, candidates: list[str], limit: int = 10) -> list[dict[str, Any]]:
    resources = get_resources()
    rows: list[dict[str, Any]] = []
    denom = max(limit - 1, 1)

    for idx, term in enumerate(candidates[:limit], start=1):
        llt_code = resources.term_norm_to_llt.get(norm_text(term))
        pt_code = resources.llt_to_pt.get(llt_code) if llt_code else None
        pt_term = resources.pt_meta.get(pt_code, {}).get("PT_Term") if pt_code else None
        soc_code = resources.pt_code_to_primary_soc.get(pt_code) if pt_code else None
        soc_term = resources.soc_code_to_term.get(soc_code) if soc_code else None

        # Visual confidence from retrieval rank only (model confidence is not returned by rag_core).
        score = round(100 - ((idx - 1) * (65 / denom)), 1)

        rows.append(
            {
                "rank": idx,
                "llt_term": term,
                "llt_code": llt_code,
                "pt_term": pt_term,
                "pt_code": pt_code,
                "soc_term": soc_term,
                "soc_code": soc_code,
                "score": score,
            }
        )
    return rows


@app.get("/")
def index():
    return render_template("index.html", examples=EXAMPLES)


@app.get("/api/examples")
def api_examples():
    return jsonify({"examples": EXAMPLES})


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    ae_text = str(payload.get("text", "")).strip()
    top_k = int(payload.get("top_k", 100) or 100)

    if not ae_text:
        return jsonify({"error": "Please provide an adverse event description."}), 400

    try:
        response = predict(ae_text, top_k=top_k)
        pred = response.get("pred", {})
        candidates = response.get("candidates_retrieved_100", [])
        debug = response.get("debug", {})

        result = {
            "input_text": ae_text,
            "prediction": {
                "llt_term": pred.get("pred_LLT_term"),
                "llt_code": pred.get("pred_LLT_Code"),
                "pt_term": pred.get("pred_PT_term"),
                "pt_code": pred.get("pred_PT_Code"),
                "soc_term": pred.get("pred_SOC_Term"),
                "soc_code": pred.get("pred_SOC_Code"),
                "soc_codes_all": pred.get("pred_SOC_codes_all") or [],
                "primary_soc_missing": bool(pred.get("pred_primary_soc_missing")),
            },
            "top5_preview": response.get("top5_preview", []),
            "candidates": _candidate_rows(ae_text, candidates, limit=10),
            "debug": {
                "latency_ms": debug.get("latency_ms"),
                "used_fallback": bool(debug.get("used_fallback")),
            },
        }
        return jsonify(result)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Warm startup to avoid first-request delay.
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
