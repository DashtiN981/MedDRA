from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from tqdm import tqdm

# Load LLT terms from CSV
llt_df = pd.read_excel("./data/LLT2_Code_English_25_0.xlsx")
terms = llt_df["LLT_Term"].astype(str).tolist()

# Load sentence transformer model (offline if already cached)
model = SentenceTransformer("all-MiniLM-L6-v2")  # or another local model

# Dictionary to store term => embedding
embeddings = {}

# Loop through each term and generate embedding
for term in tqdm(terms, desc="Embedding LLT terms"):
    try:
        emb = model.encode(term).tolist()  # convert numpy array to list
        embeddings[term] = emb
    except Exception as e:
        print(f"Error embedding term '{term}': {e}")

# Save all embeddings to JSON file
with open("llt2_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

print("Done! All embeddings saved to llt_embeddings.json")
