import pandas as pd
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fuzzywuzzy import fuzz

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# Load AE data
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

# Load LLT dictionary
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# Parameters
N_CANDIDATES = 100  # realistic set size (more difficult task)
results = []

# Loop over a small subset of AEs (adjust as needed)
for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    
    if true_code not in llt_code_to_term:
        continue

    true_term = llt_code_to_term[true_code]

    # Select LLTs excluding the correct one
    candidate_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    sampled_terms = candidate_pool.sample(N_CANDIDATES, random_state=idx)["LLT_Term"].tolist()
    random.shuffle(sampled_terms)

    # Build prompt
    prompt = (
        f"You are a medical coding assistant. Given the following adverse event description:\n"
        f"\"{ae_text}\"\n"
        f"Choose the most appropriate MedDRA LLT term from the list below:\n\n" +
        "\n".join(f"- {term}" for term in sampled_terms) +
        "\n\nRespond only with the exact chosen term."
    )

    try:
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()

        # Evaluation
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

        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Save predictions
with open("/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/baseline_hard_fuzz.json", "w") as f:
    json.dump(results, f, indent=2)

# Prepare metrics
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]

# Output placeholder - evaluation below
from sklearn.metrics import classification_report
print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy, zero_division=0))

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")

# Calculate Fuzzy Match Accuracy (custom metric)
fuzzy_accuracy = sum(r["fuzzy_match"] for r in results) / len(results)
print(f"Fuzzy Match Accuracy: {fuzzy_accuracy:.2f}")