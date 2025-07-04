from openai import OpenAI
import json
import numpy as np

# Initialize OpenAI-compatible client with your local LLM API
client = OpenAI(
    api_key="sk-BEYOnuDXHm5OcYLc5xKX6w",         # Replace with your actual API key
    base_url="http://pluto/v1/"          # Local API base URL
)

# Input sentence to be converted into an embedding
text = "Patient had a severe headache after injection"

# Query the LLM to generate an embedding for the sentence
response = client.chat.completions.create(
    model="Llama-3.3-70B-Instruct",  # Use any available model
    messages=[
        {
            "role": "system",
            "content": (
                "You are an embedding generator. For any input sentence, return only a list of numbers representing its semantic embedding. "
                "Do not return any explanation or description. Only output the raw list."
            )
        },
        {
            "role": "user",
            "content": f"Generate embedding for the following sentence:\n{text}"
        }
    ],
    temperature=0.0,
    max_tokens=512
)

# Raw model output (string representation of the list)
raw_output = response.choices[0].message.content.strip()

# Attempt to parse the raw output as a list of numbers
try:
    embedding = json.loads(raw_output)
    if isinstance(embedding, list):
        embedding = np.array(embedding)
        print("Embedding successfully generated.")
        print("First 10 values:", embedding[:10])
    else:
        print("Output is not a valid list.")
except Exception as e:
    print("Failed to parse embedding:", e)
    print("Raw output:", raw_output)