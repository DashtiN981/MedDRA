import pandas as pd
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, classification_report
from fuzzywuzzy import fuzz
import re

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

# Load AE data
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

# Load LLT dictionary
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# Parameters
N_CANDIDATES = 100
results = []

# One-shot example
example_ae = "Mild headache and dizziness"
example_candidates = ["Headache", "Dizziness", "Nausea"]
example_answer = "Headache"

for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    
    if true_code not in llt_code_to_term:
        continue

    true_term = llt_code_to_term[true_code]

    candidate_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    sampled_terms = candidate_pool.sample(N_CANDIDATES, random_state=idx)["LLT_Term"].tolist()
    sampled_terms.append(true_term)
    random.shuffle(sampled_terms)
    numbered_terms = [f"{i+1}. {term}" for i, term in enumerate(sampled_terms)]

    prompt = (
        "You are a medical coding assistant. Your task is to map adverse event descriptions to the most appropriate MedDRA LLT term from a given list.\n\n"
        "Example:\n"
        f"Description: \"{example_ae}\"\n"
        "Candidates:\n" +
        "\n".join(f"{i+1}. {term}" for i, term in enumerate(example_candidates)) +
        f"\nAnswer: {example_answer}\n\n"
        "Now solve the following:\n"
        f"Description: \"{ae_text}\"\n"
        "Candidates:\n" +
        "\n".join(numbered_terms) +
        "\nAnswer:"
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
        raw_output = response.choices[0].message.content.strip()

        # Extract predicted LLT term
        match = re.search(r'\bAnswer:\s*(?:\d+\.\s*)?(.*)', raw_output)
        if match:
            answer = match.group(1).strip()
        else:
            lines = raw_output.split("\n")
            answer = lines[-1].strip()

        exact_match = answer == true_term
        fuzzy_score = fuzz.ratio(answer.lower(), true_term.lower())
        fuzzy_match = fuzzy_score >= 90

        results.append({
            "AE": ae_text,
            "True": true_term,
            "Predicted": answer,
            "Exact": exact_match,
            "Fuzzy": fuzzy_match
        })

        print(f"[{idx}] {ae_text}")
        print(f"→ True: {true_term}")
        print(f"→ Predicted: {answer}")
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")

        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Evaluation
y_true = [r["True"] for r in results]
y_pred = [r["Predicted"] for r in results]
y_pred_fuzzy = [r["True"] if r["Fuzzy"] else r["Predicted"] for r in results]

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy (Exact Match): {acc:.2f}")
print("\n📊 Classification Report (Exact Match):")
print(classification_report(y_true, y_pred))

print("\n📊 Classification Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy))

# Optional: Save to file
with open("baseline_hard_v2.json", "w") as f:
    json.dump(results, f, indent=2)
