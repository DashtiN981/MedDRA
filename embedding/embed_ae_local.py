# embed_ae_local.py

from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from tqdm import tqdm

# Load AE terms from CSV
ae_df = pd.read_csv("./data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_terms = ae_df["Original_Term_aufbereitet"].dropna().unique().tolist()

# Load the same model used for LLT embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dictionary to store AE => embedding
ae_embeddings = {}

# Generate embeddings
for ae in tqdm(ae_terms, desc="Embedding AE terms"):
    try:
        emb = model.encode(ae).tolist()
        ae_embeddings[ae] = emb
    except Exception as e:
        print(f" Error embedding AE '{ae}': {e}")

# Save embeddings to file
with open("ae_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(ae_embeddings, f, ensure_ascii=False, indent=2)

print(" Done! AE embeddings saved to ae_embeddings.json")
