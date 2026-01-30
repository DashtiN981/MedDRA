"""
File Name: rag_prompting_reasoning_v2.py    === Author: Naghme Dashti / July 2025

This script implements an LLM-based MedDRA coding assistant using embedding-based RAG retrieval 
combined with reasoning-augmented prompting. For each AE, it retrieves Top-K similar LLT terms 
(based on cosine similarity), generates a chain-of-thought explanation using the LLM, and then 
extracts the best matching term. Final predictions are evaluated using exact and fuzzy matching metrics.

"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from rapidfuzz import fuzz
import time


# Load local OpenAI-compatible LLM API
client = OpenAI(
    api_key="sk-aKGeEFMZB0gXEcE51FTc0A",
    base_url="http://pluto/v1/"
)

# === Parameters ===
TOP_K = 30
MAX_ROWS = None
EMB_DIM = 384  # dimension of MiniLM

AE_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/ae_embeddings_Dauno.json"
LLT_EMB_FILE = "/home/naghmedashti/MedDRA-LLM/embedding/llt2_embeddings.json"
AE_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Dauno_AE_Codierung_2022_10_20.csv"
LLT_CSV_FILE = "/home/naghmedashti/MedDRA-LLM/data/LLT2_Code_English_25_0.csv"

# === Load Data ===
ae_df = pd.read_csv(AE_CSV_FILE, sep=';', encoding='latin1')[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)
llt_df = pd.read_csv(LLT_CSV_FILE, sep=';', encoding='latin1')[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))


# === Load Embeddings ===
with open(AE_EMB_FILE, "r", encoding="latin1") as f:
    ae_embeddings = json.load(f)

with open(LLT_EMB_FILE, "r", encoding="latin1") as f:
    llt_embeddings = json.load(f)


# === Similarity Function ===
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

# === Extraction Logic ===
def extract_final_term(answer_text, candidate_terms):
    for line in answer_text.splitlines():
        if "final answer:" in line.lower():
            return line.split(":")[-1].strip()
    # Try matching known terms from bottom up
    for line in reversed(answer_text.splitlines()):
        for term in candidate_terms:
            if term.lower() in line.lower():
                return term
    return answer_text.strip().split("\n")[-1].strip()

# Convert LLT embeddings to dictionary
llt_emb_dict = {term: np.array(embedding) for term, embedding in llt_embeddings.items()}



# === RAG Prompting ===
results = []

for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    if true_code not in llt_code_to_term:
        continue
    true_term = llt_code_to_term[true_code]

    # Get AE embedding
    ae_emb = ae_embeddings.get(ae_text)
    if ae_emb is None:
        continue

    similarities = []
    for term, emb in llt_emb_dict.items():
        sim = cosine_similarity(ae_emb, emb)
        similarities.append((term, sim))

    top_terms = sorted(similarities, key=lambda x: x[1], reverse=True)[:TOP_K]
    top_terms = [term for term, _ in top_terms]

    # Build reasoning-augmented prompt
    prompt = (
        f"You are a medical coding assistant using the MedDRA terminology.\n"
        f"Here is an Adverse Event (AE): \"{ae_text}\"\n"
        f"You are given a list of candidate MedDRA LLT terms. Your task is to reason step-by-step and select the most appropriate LLT.\n"
        f"1. Analyze the AE text and extract relevant clinical keywords.\n"
        f"2. Compare those keywords with candidate LLT terms.\n"
        f"3. Eliminate unrelated terms.\n"
        f"4. Select the LLT that best matches the AE context.\n"
        f"5. Final answer: respond with only the final LLT term.\n\n"
        f"Candidate LLTs:\n" +
        "\n".join(f"- {term}" for term in top_terms)
    )

    try:
        response = client.chat.completions.create(
            model="nvidia-llama-3.3-70b-instruct-fp8", 
            # model="llama-3.3-70b-instruct-awq",Llama-3.3-70B-Instruct
            messages=[
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        answer_line = extract_final_term(answer, top_terms)

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
with open("/home/naghmedashti/MedDRA-LLM/RAG_Models/v2_Results/Dauno_output_result_v2.json", "w") as f:
    json.dump(results, f, indent=2)

# === Evaluation ===
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]


#if len(y_true) > 0:
print("\nEvaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))
    
print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))
#else:
    #print("No successful predictions to evaluate.")

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