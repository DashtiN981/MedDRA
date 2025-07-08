import json
import numpy as np

LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt_embeddings.json"

with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    try:
        llt_embeddings = json.load(f)
        print(f"[OK] Loaded LLT embeddings. Type: {type(llt_embeddings)}")

        # اگر dict بود و کلیدها رشته بودند (مثلاً "0", "1", ...)
        if isinstance(llt_embeddings, dict):
            # بررسی کنیم آیا شبیه {"0": {...}, "1": {...}} هست؟
            first_key = list(llt_embeddings.keys())[0]
            print("Sample key:", first_key)
            if isinstance(llt_embeddings[first_key], str):
                # اگر مقدار هم string باشه، یعنی باید json.loads بشه
                llt_embeddings = [json.loads(v) for v in llt_embeddings.values()]
            else:
                # مقدارها دیکشنری هستند
                llt_embeddings = list(llt_embeddings.values())

    except Exception as e:
        print(f"[ERROR] Problem loading LLT embeddings: {e}")
        raise

# حالا ازش dict می‌سازیم
llt_emb_dict = {item["term"]: np.array(item["embedding"]) for item in llt_embeddings}
print("✅ Sample element:", llt_embeddings[0])