# embed_llts.py

from openai import OpenAI
import pandas as pd
import json
import time

# Initialize client for local LLM API
client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

# Load LLT list
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/LLT_Code_English_25_0.csv", sep=';', encoding='latin1')
terms = llt_df["LLT_Term"].astype(str).tolist()  

def get_embedding(text):
    """Prompt the model to output an embedding list for a given text."""
    resp = client.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are an embedding generator. Only output a JSON list of floats."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        max_tokens=512
    )
    return json.loads(resp.choices[0].message.content.strip())

# Batch process (with optional limit)
limit = 1000  # for fast embedding
embeddings = {}
for idx, term in enumerate(terms[:limit]):
    try:
        embeddings[term] = get_embedding(term)
        time.sleep(0.5)  # to respect rate limits
        print(f"[{idx+1}/{limit}] embedded")
    except Exception as e:
        print(f"Error at {idx}, {term}: {e}")

# Save to file
with open("embed_llt_sample.json", "w") as f:
    json.dump(embeddings, f)
