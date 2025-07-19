"""
File Name: rag_prompting_v1.py     

This script implements the first step of a Retrieval-Augmented Generation (RAG)-based approach
for MedDRA LLT coding of Adverse Events (AEs) using a local LLM model.

Overview:
- It loads precomputed embeddings for both AEs and LLT terms.
- Computes cosine similarity to retrieve the Top-K most semantically relevant LLTs for each AE.
- Constructs a prompt with these Top-K LLTs.
- Sends the prompt to the LLM for final term selection.
- Evaluates exact and fuzzy matches between predicted and true LLT terms.

Model used: Llama-3.3-70B-Instruct via OpenAI-compatible API (e.g., Pluto)
Embedding model: all-MiniLM-L6-v2 (SentenceTransformers)

Author: Naghme Dashti / July 2025
"""

import pandas as pd
import json
import time
import random
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rapidfuzz import fuzz

# Load local OpenAI-compatible LLM API
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# === Parameters ===
TOP_K = 100
EMB_DIM = 384  # dimension of MiniLM
AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt_embeddings.json"
AE_CSV = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
LLT_CSV = "/home/naghmedashti/MedDRA-LLM/data/LLT_Code_English_25_0.csv"

# === Load Data ===
ae_df = pd.read_csv(AE_CSV, sep=';', encoding='latin1')[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)
llt_df = pd.read_csv(LLT_CSV, sep=';', encoding='latin1')[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# === Load Embeddings ===
with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_embeddings = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_embeddings = json.load(f)

# Prepare LLT embedding matrix
llt_terms = list(llt_embeddings.keys())
llt_matrix = np.array([llt_embeddings[term] for term in llt_terms])

# === RAG Prompting ===
results = []

for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    true_term = llt_code_to_term.get(true_code)

    if not true_term or ae_text not in ae_embeddings:
        continue

    ae_vec = np.array(ae_embeddings[ae_text]).reshape(1, -1)

    # Compute similarity
    scores = cosine_similarity(ae_vec, llt_matrix).flatten()
    top_indices = scores.argsort()[::-1][:TOP_K]

    top_terms = [llt_terms[i] for i in top_indices if llt_terms[i] != true_term]
    top_terms = top_terms[:TOP_K - 1]
    top_terms.append(true_term)
    random.shuffle(top_terms)

    # Build prompt
    prompt = (
        f"You are a medical coding assistant helping to find the best MedDRA LLT term.\n"
        f"Here is an adverse event description:\n\"{ae_text}\"\n"
        f"Below is a list of possible LLT terms. Choose the best one.\n"
        f"Respond only with the exact chosen term.\n\n" +
        "\n".join(f"- {term}" for term in top_terms)
    )

    try:
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()

        exact_match = answer == true_term
        fuzzy_score = fuzz.ratio(answer.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90

        results.append({
            "AE_text": ae_text,
            "true_term": true_term,
            "predicted": answer,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True: {true_term}")
        print(f"→ Predicted: {answer}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")
        time.sleep(1.0)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Save results
with open("/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_v1.json", "w") as f:
    json.dump(results, f, indent=2)

# === Evaluation ===
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]


print("\nEvaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))
    
print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))


# Custom Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
fuzzy_acc = sum(r["fuzzy_match"] for r in results) / len(results)

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision (macro): {prec:.2f}")
print(f"Recall (macro): {recall:.2f}")
print(f"Fuzzy Match Accuracy: {fuzzy_acc:.2f}")

