# Re-run the baseline with fixed kernel (after reset)
import pandas as pd
import random
import time
import json
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from fuzzywuzzy import fuzz
from openai import OpenAI

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",  # placeholder
    base_url="http://pluto/v1/"
)

# Load AE data (English)
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

# Load LLT dictionary (English)
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/LLT_Code_English_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))
llt_term_to_code = dict(zip(llt_df["LLT_Term"], llt_df["LLT_Code"]))

# Parameters
N_CANDIDATES = 100  # Number of LLTs shown to the model
MAX_ROWS = 20       # Number of AE samples for demo
results = []

# Sample run
for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    true_term = llt_code_to_term.get(true_code, None)
    if not true_term:
        continue

    # Distractors only
    candidate_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    sampled = candidate_pool.sample(N_CANDIDATES, random_state=idx)
    sampled_terms = sampled["LLT_Term"].tolist()
    sampled_codes = sampled["LLT_Code"].tolist()
    candidates = [f"[{c}] {t}" for c, t in zip(sampled_codes, sampled_terms)]

    # Shuffle candidate list
    random.shuffle(candidates)

    # Prompt (realistic with distractors only)
    prompt = (
        f"You are a medical coding assistant. Given the following adverse event description:\n"
        f"\"{ae_text}\"\n"
        f"Select the most appropriate MedDRA LLT term with its code from the list below:\n\n" +
        "\n".join(f"- {item}" for item in candidates) +
        "\n\nRespond only with the code and the exact term (e.g., [10000001] TermName)."
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

        predicted_term = answer.split(']')[-1].strip()
        predicted_code = answer.split(']')[0].replace('[', '').strip()

        fuzzy_score = fuzz.ratio(predicted_term.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90
        exact_match = predicted_term == true_term

        results.append({
            "AE_text": ae_text,
            "true_term": true_term,
            "true_code": true_code,
            "predicted_term": predicted_term,
            "predicted_code": predicted_code,
            "exact_match": exact_match,
            "fuzzy_score": fuzzy_score,
            "fuzzy_match": fuzzy_match
        })
        
        print(f"[{idx}] AE: {ae_text}")
        print(f"→ True: [{true_code}] {true_term}")
        print(f"→ Predicted: [{predicted_code}] {predicted_term}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")
        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Save predictions
with open("/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/baseline_negativetest.json", "w") as f:
    json.dump(results, f, indent=2)

# Evaluate
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted_term"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted_term"] for r in results]

report_exact = classification_report(y_true, y_pred, zero_division=0)
report_fuzzy = classification_report(y_true, y_pred_fuzzy, zero_division=0)

report_exact, report_fuzzy

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")