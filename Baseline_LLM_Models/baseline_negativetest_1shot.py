import pandas as pd
import random
import json
import time
from fuzzywuzzy import fuzz
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# Load AE data
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

# Load English LLT dictionary
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# Parameters
N_CANDIDATES = 100
results = []

# 1-shot example to include in every prompt
example_ae = "headache"
example_code = "10019211"
example_term = "Headache"
example_str = f"Example:\nAE: \"{example_ae}\"\n→ [{example_code}] {example_term}\n"

# Iterate over a subset of data
for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    true_term = llt_code_to_term.get(true_code, None)
    if not true_term:
        continue

    # Sample distractors and find the closest matching distractors
    distractor_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    sampled = distractor_pool.sample(N_CANDIDATES, random_state=idx)
    sampled_terms = sampled["LLT_Term"].tolist()
    sampled_codes = sampled["LLT_Code"].tolist()

    # Adding a check to handle the scenario when no direct match is available
    candidates = list(zip(sampled_codes, sampled_terms))
    random.shuffle(candidates)
    candidate_lines = [f"- [{code}] {term}" for code, term in candidates]

    # Prompt  (realistic with distractors+example)
    prompt = (
        "You are a medical coding assistant.\n\n"
        f"{example_str}\n"
        f"Now, given the following adverse event:\n\"{ae_text}\"\n\n"
        "Select the most appropriate MedDRA LLT term with its code from the list below:\n\n" +
        "\n".join(candidate_lines) +
        "\n\nRespond ONLY with the code and term in this format: [LLT_Code] LLT_Term"
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

        # Use fuzzy score to find the closest match
        fuzzy_score = fuzz.ratio(predicted_term.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90  # Threshold set to 90 for fuzzy match
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
with open("/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/baseline_negativetest_1shot.json", "w") as f:
    json.dump(results, f, indent=2)

# Evaluation
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted_term"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted_term"] for r in results]

print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred))
print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy))

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")