import pandas as pd
import random
from openai import OpenAI
import time
import json
from sklearn.metrics import accuracy_score, f1_score
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
from rapidfuzz import fuzz
from rapidfuzz import process

# from groq import Groq

# client = Groq(api_key="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")

# Initialize OpenAI-compatible API client
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",
    base_url="http://pluto/v1/"
)

ae_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv", sep=';', encoding='latin1')
ae_df = ae_df[["Original_Term_aufbereitet", "ZB_LLT_Code"]].dropna().reset_index(drop=True)

llt_df = pd.read_csv("/home/naghmedashti/MedDRA-LLM/data/MedDRA1_LLT_Code_25_0.csv", sep=';', encoding='latin1')
llt_df = llt_df[["LLT_Code", "LLT_Term"]].dropna().reset_index(drop=True)
llt_code_to_term = dict(zip(llt_df["LLT_Code"].astype(str), llt_df["LLT_Term"]))

N_CANDIDATES = 100
results = []

for idx, row in ae_df.iloc[:20].iterrows():
    ae_text = row["Original_Term_aufbereitet"]
    true_code = str(int(row["ZB_LLT_Code"]))
    if true_code not in llt_code_to_term:
        continue
    true_term = llt_code_to_term[true_code]

    candidate_terms = llt_df["LLT_Term"].tolist()
    #closest_terms = [term for term, score in process.extract(ae_text, candidate_terms, limit=N_CANDIDATES + 10)]
    closest_terms = [term for term, score, _ in process.extract(ae_text, candidate_terms, limit=N_CANDIDATES + 10)]
    closest_terms = [term for term in closest_terms if term != true_term]
    closest_terms = closest_terms[:N_CANDIDATES]
    closest_terms.append(true_term)
    random.shuffle(closest_terms)

    prompt = (
        f"You are a medical coding assistant helping to find the best matching MedDRA LLT term.\n"
        f"Here is an adverse event description:\n\"{ae_text}\"\n"
        f"Below is a list of possible MedDRA LLT terms. Select exactly one term that best fits the description.\n"
        f"Respond only with the exact chosen term, without any extra text.\n\n" +
        "\n".join(f"- {term}" for term in closest_terms) +
        "\n"
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
        print(f"→ Exact: {exact_match}, Fuzzy: {fuzzy_score} ({'✓' if fuzzy_match else '✗'})\n")

        time.sleep(1.0)

    except Exception as e:
        print(f"Error at index {idx}: {e}")

with open("baseline_hard_rapidfuzz.json", "w") as f:
    json.dump(results, f, indent=2)

# Prepare metrics
y_true = [r["true_term"] for r in results]
y_pred = [r["predicted"] for r in results]
y_pred_fuzzy = [r["true_term"] if r["fuzzy_match"] else r["predicted"] for r in results]

# Output placeholder - evaluation below
from sklearn.metrics import classification_report
print("Evaluation Report (Exact Match):")
print(classification_report(y_true, y_pred))

print("\nEvaluation Report (Fuzzy Match):")
print(classification_report(y_true, y_pred_fuzzy))

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nAccuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")