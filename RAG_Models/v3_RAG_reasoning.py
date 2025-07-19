"""
rag_prompting_reasoning_v3.py

RAG-based Prompting with Explicit Reasoning + Final Answer Line
---------------------------------------------------------------
This script implements a refined version of the RAG-based model for MedDRA coding.

Features:
- Uses MiniLM embeddings to retrieve Top-K semantically similar LLTs for each AE.
- The prompt asks the LLM to first provide a justification (reasoning).
- Then explicitly output the final selected LLT term in a separate line with:
    Final answer: <LLT_TERM>
- Improves interpretability and separation of reasoning vs prediction.

"""

import json
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import OpenAI
from rapidfuzz import fuzz
import time

# --- Load Embeddings ---
AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt_embeddings.json"

with open(AE_EMB_FILE, "r", encoding="utf-8") as f:
    ae_emb_dict = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="utf-8") as f:
    llt_emb_dict = json.load(f)

# Convert embedding lists to numpy arrays
llt_emb_dict = {k: np.array(v) for k, v in llt_emb_dict.items()}
ae_emb_dict = {k: np.array(v) for k, v in ae_emb_dict.items()}

# --- Load AE and LLT data ---
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# --- Parameters ---
K = 100
results = []

# --- Initialize API ---
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# --- Main Loop ---
for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    if true_code not in llt_code_to_term:
        continue
    true_term = llt_code_to_term[true_code]

    # Get AE embedding
    if ae_text not in ae_emb_dict:
        continue
    ae_emb = ae_emb_dict[ae_text]

    # Compute cosine similarity with LLTs
    similarities = [(term, float(np.dot(ae_emb, llt_emb) / (np.linalg.norm(ae_emb) * np.linalg.norm(llt_emb))))
                    for term, llt_emb in llt_emb_dict.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)

    candidate_terms = [term for term, _ in similarities[:K]]
    if true_term not in candidate_terms:
        candidate_terms.append(true_term)
    random.shuffle(candidate_terms)

    # --- Prompt ---
    prompt = (
        f"You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term."
        f"Here is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        f"Here is a list of candidate LLT terms:\n"
        + "\n".join(f"- {term}" for term in candidate_terms)
        + "\n\nPlease analyze the AE and list, and first provide a short reasoning."
        f"Then, on a separate line, write the best matching LLT in this format:\nFinal answer: <LLT_TERM>"
    )

    try:
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=250
        )

        answer = response.choices[0].message.content.strip()

        if "Final answer:" in answer:
            answer_line = answer.split("Final answer:")[-1].strip()
        else:
            answer_line = answer.strip().split("\n")[-1].strip()

        exact_match = answer_line == true_term
        fuzzy_score = fuzz.ratio(answer_line.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90

        results.append({
            "AE_text": ae_text,
            "true_term": true_term,
            "predicted": answer_line,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True: {true_term}")
        print(f"→ Predicted: {answer_line}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score:.1f} ({'✓' if fuzzy_match else '✗'})\n")
        time.sleep(1.0)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# --- Save Results ---
with open("/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_reasoning_v3.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Evaluate ---
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

# Custom Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
fuzzy_acc = sum(r["fuzzy_match"] for r in results) / len(results)

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")
print(f"Fuzzy Match Accuracy: {fuzzy_acc:.2f}")
