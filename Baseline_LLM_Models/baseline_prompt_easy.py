import pandas as pd
import random
from openai import OpenAI
import json
import time
from sklearn.metrics import accuracy_score, f1_score

# Initialize OpenAI-compatible client for local LLM inference
client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

# Load AE (Adverse Event) descriptions
ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

# Load MedDRA LLT terms
llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/clean_data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)

# Build dictionary from LLT_Code to LLT_Term
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

# Parameters
N_CANDIDATES = 100  # Number of LLTs shown to the model
MAX_ROWS = 20       # Number of AE samples for demo

results = []

for idx, row in ae_df.iloc[:MAX_ROWS].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))  # Convert to string for dictionary matching

    if true_code not in llt_code_to_term:
        continue

    true_term = llt_code_to_term[true_code]

    # Get N-1 random LLTs excluding the correct one
    candidate_pool = llt_df[llt_df["LLT_Code"] != int(true_code)]
    random_candidates = candidate_pool.sample(N_CANDIDATES - 1, random_state=idx)
    llt_terms = [true_term] + list(random_candidates["LLT_Term"])
    random.shuffle(llt_terms)

    # Build prompt for LLM
    prompt = (
        f"You are a clinical coding assistant.\n"
        f"Given the following adverse event (AE) description:\n"
        f"\"{ae_text}\"\n\n"
        f"Choose the most appropriate MedDRA LLT term from the list below:\n\n" +
        "\n".join(f"- {term}" for term in llt_terms) +
        "\n\nRespond only with the most relevant term, and nothing else."
    )

    try:
        # Query the local LLM model
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
        prediction = answer

        results.append({
            "ae_text": ae_text,
            "true_term": true_term,
            "predicted_term": prediction
        })

        print(f"[{idx}] AE: {ae_text}")
        print(f"True: {true_term}")
        print(f"Predicted: {prediction}\n")

        time.sleep(1.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Evaluation
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted_term"] for r in results]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save results
with open("/home/naghmedashti/MedDRA-LLM/Baseline_LLM_Models/baseline_prompt_easy.json", "w") as f:
    json.dump(results, f, indent=2)
