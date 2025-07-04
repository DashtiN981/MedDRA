# embed_llts_robust.py

import pandas as pd
import json
import time
from openai import OpenAI

# Init client
client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

# Load LLT terms
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv", sep=";", encoding="latin1")
terms = llt_df["LLT_Term"].dropna().astype(str).unique().tolist()

def get_embedding_via_chat(text):
    """Generate embeddings via chat prompt from LLM."""
    try:
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are an embedding generator. Return only a JSON array of float numbers representing a sentence embedding."},
                {"role": "user", "content": f"Generate embedding for: {text}"}
            ],
            temperature=0.0,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f" Error for '{text}': {e}")
        return None

embeddings = {}
for idx, term in enumerate(terms[:1000]):
    emb = get_embedding_via_chat(term)
    if emb:
        embeddings[term] = emb
        print(f" [{idx+1}] Embedded: {term}")
    else:
        print(f"  [{idx+1}] Failed: {term}")
    time.sleep(0.5)

# Save
with open("llt_embeddings_llama.json", "w") as f:
    json.dump(embeddings, f, indent=2)
