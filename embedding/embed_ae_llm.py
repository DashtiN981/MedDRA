# embed_ae_llm.py

from openai import OpenAI
import pandas as pd
import json
import time

# Initialize the OpenAI-compatible client pointing to local LLM server
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",  # Use your actual API key
    base_url="http://pluto/v1/"           # Internal server URL
)

# Load AE terms from the CSV file
ae_df = pd.read_csv(
    "/home/naghmedashti/MedDRA-LLM/data/KI_Projekt_Mosaic_AE_Codierung_2024_07_03.csv",
    sep=';',
    encoding='latin1'
)

# Extract the AE descriptions (text inputs to embed)
ae_texts = ae_df["Original_Term_aufbereitet"].dropna().astype(str).tolist()

# Define function to get embedding from the server model
def get_embedding(text):
    """
    Ask the local LLM server to return an embedding vector for a given AE text.
    The model is instructed to return a JSON list of floats only.
    """
    try:
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",  # You can change to other available model
            messages=[
                {
                    "role": "system",
                    "content": "You are an embedding generator. Only output a JSON list of floats."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.0,
            max_tokens=512
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f" Error embedding: {e}")
        return None

# Dictionary to store embeddings for each AE
ae_embeddings = {}

# Loop over all AE descriptions (only 320 entries, so it's manageable)
for idx, text in enumerate(ae_texts):
    emb = get_embedding(text)
    if emb:
        ae_embeddings[text] = emb
        print(f"[{idx+1}/{len(ae_texts)}]  Embedded: {text[:60]}...")
    else:
        print(f"[{idx+1}/{len(ae_texts)}]  Failed to embed: {text[:60]}...")
    
    time.sleep(1.5)  # Optional delay to avoid server overload

# Save the resulting AE embeddings to JSON file
with open("ae_embeddings.json", "w") as f:
    json.dump(ae_embeddings, f, indent=2)

print("All AE embeddings saved to ae_embeddings.json")
