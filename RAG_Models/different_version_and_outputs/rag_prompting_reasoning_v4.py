"""
File Name: rag_prompting_reasoning_v4.py    === Author: Naghme Dashti / July 2025

RAG-based Prompting for MedDRA Coding (No Ground Truth Injection)
-----------------------------------------------------------------
This version removes the ground truth term from candidate LLTs to simulate a real-world scenario.

Features:
- Retrieves Top-K semantically similar LLTs based on MiniLM embeddings
- Does NOT insert the correct LLT into the candidate list
- Prompts LLM to provide reasoning + structured final answer
"""

import json
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import OpenAI
from rapidfuzz import fuzz
import time

# === Setup LLM API ===
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# === Parameters ===
TOP_K = 30
MAX_ROWS = 100  # Or set to None to evaluate all rows
EMB_DIM = 384

AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings_Mosaic.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
AE_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"

# === Load Data ===
ae_df = pd.read_csv(AE_CSV_FILE, sep=';', encoding='latin1')[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)
llt_df = pd.read_csv(LLT_CSV_FILE, sep=';', encoding='latin1')[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_emb_dict = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_emb_dict = json.load(f)

llt_emb_dict = {k: np.array(v) for k, v in llt_emb_dict.items()}
ae_emb_dict = {k: np.array(v) for k, v in ae_emb_dict.items()}

# === RAG with Reasoning (No true_term injection) ===
results = []

for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    if true_code not in llt_code_to_term:
        continue
    true_term = llt_code_to_term[true_code]

    if ae_text not in ae_emb_dict:
        continue
    ae_emb = ae_emb_dict[ae_text]

    # Compute similarities
    similarities = [(term, float(np.dot(ae_emb, llt_emb) / (np.linalg.norm(ae_emb) * np.linalg.norm(llt_emb))))
                    for term, llt_emb in llt_emb_dict.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)

    candidate_terms = [term for term, _ in similarities[:TOP_K]]

    # === Prompt Construction ===
    prompt = (
        f"You are a medical coding assistant. Your job is to reason through the best MedDRA LLT term.\n"
        f"Here is an Adverse Event (AE):\n\"{ae_text}\"\n\n"
        f"Here is a list of candidate LLT terms:\n"
        + "\n".join(f"- {term}" for term in candidate_terms) +
        "\n\nPlease analyze the AE and list, and first provide a short reasoning.\n"
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

# === Save Results ===
with open("/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_reasoning_v4.json", "w") as f:
    json.dump(results, f, indent=2)

# === Evaluation ===
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

# === Summary Metrics ===
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
