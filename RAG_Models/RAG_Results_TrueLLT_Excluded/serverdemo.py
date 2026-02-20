from __future__ import annotations

import os

from serverdemo3 import app, _find_free_port, _maybe_start_ngrok, get_resources


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
