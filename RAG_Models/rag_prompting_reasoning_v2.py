"""
rag_prompting_reasoning_v2.py

This script implements a RAG-based MedDRA LLT term selection system using an LLM
with embedded Chain-of-Thought reasoning. For each Adverse Event (AE) description,
the top-K most semantically similar LLT terms are retrieved based on cosine similarity
between Sentence Transformer embeddings. The LLM is prompted to explain its reasoning
and then select the most appropriate LLT term.

Key Features:
- Embedding-based retrieval of Top-K LLTs from a pre-embedded LLT pool.
- Chain-of-Thought prompting to encourage reasoning by the LLM.
- Robust parsing of LLM outputs to extract the final selected LLT term.
- Evaluation of exact and fuzzy match accuracy.

Author: Naghme Dashti / July 2025
"""

import json
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from rapidfuzz import fuzz
from numpy.linalg import norm
import time

# ==== Configuration ====
K = 100  # Number of similar LLTs to retrieve
TEMPERATURE = 0.0
MODEL_NAME = "Llama-3.3-70B-Instruct"

# ==== File Paths ====
AE_FILE = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv"
LLT_FILE = "/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv"
AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt_embeddings.json"

# ==== Load Data ====
ae_df = pd.read_csv(AE_FILE, sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

llt_df = pd.read_csv(LLT_FILE, sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_embeddings = json.load(f)
with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_embeddings = json.load(f)

# ==== Embedding Utilities ====
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10)

# ==== LLM Client ====
client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

# ==== Prompt Template ====
def build_reasoning_prompt(ae_text, candidates):
    return (
        f"You are a clinical coding assistant.\n"
        f"Given the following Adverse Event (AE):\n\"{ae_text}\"\n\n"
        f"Here is a list of possible MedDRA LLT terms:\n\n" +
        "\n".join(f"- {term}" for term in candidates) +
        "\n\nThink step-by-step to determine the best match.\n"
        f"Then write your final selection on a new line, in this format:\n"
        f"Final answer: <LLT term>"
    )

# ==== Run Evaluation ====
results = []

for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    if true_code not in llt_code_to_term or ae_text not in ae_embeddings:
        continue
    true_term = llt_code_to_term[true_code]

    # Retrieve top-K similar LLTs
    ae_vec = np.array(ae_embeddings[ae_text])
    scores = []
    for term, vec in llt_embeddings.items():
        sim = cosine_similarity(ae_vec, np.array(vec))
        scores.append((term, sim))
    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:K]
    top_terms = [term for term, _ in top_k]

    # Make sure true term is in the list
    if true_term not in top_terms:
        top_terms[-1] = true_term
    np.random.shuffle(top_terms)

    # Build the prompt with reasoning instruction
    prompt = build_reasoning_prompt(ae_text, top_terms)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()

        # Parse answer line
        prediction_line = ""
        for line in answer.splitlines():
            if line.lower().startswith("final answer:"):
                prediction_line = line.split(":", 1)[-1].strip()
                break

        if not prediction_line:
            prediction_line = answer.strip().split("\n")[-1].strip()

        exact_match = prediction_line == true_term
        fuzzy_score = fuzz.ratio(prediction_line.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90

        results.append({
            "ae_text": ae_text,
            "true_term": true_term,
            "predicted_term": prediction_line,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match,
            "raw_answer": answer
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True: {true_term}")
        print(f"→ Predicted: {prediction_line}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")

        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# ==== Save Results ====
with open("/home/naghmedashti/MedDRA-LLM/RAG_Models/rag_prompting_reasoning_v2.json", "w") as f:
    json.dump(results, f, indent=2)

# ==== Evaluation ====
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted_term"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted_term"] for r in results]

print("\nEvaluation Report (Exact Match):")
print(classification_report(y_true, y_pred))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy))

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
