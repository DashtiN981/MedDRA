import json
# 
input_file = "/home/naghmedashti/MedDRA-LLM/RAG_Models/Dauno_output_NewRAG_seed44.json"
output_file = "/home/naghmedashti/MedDRA-LLM/RAG_Models/Dauno_output_NewRAG.json"

# 
fields_to_keep = [
    "AE_text",
    "true_LLT_term",
    "pred_LLT_term",
    "exact_LLT_match",
    "LLT_fuzzy_score",
    "LLT_fuzzy_match",
    "model_output"
]

# read input file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# filter data
filtered = []
for obj in data:
    new_obj = {k: obj.get(k, None) for k in fields_to_keep}
    filtered.append(new_obj)

# save output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"Saved {len(filtered)} objects to {output_file}")
