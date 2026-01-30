from openai import OpenAI

client = OpenAI(api_key="sk-aKGeEFMZB0gXEcE51FTc0A", base_url="http://pluto/v1")

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant!"
        },
        {
            "role": "user",
            "content": "What do you know about the KatherLab?",
        }
    ],
      #model="medgemma-27b-it-q6",
      #model="medgemma-27b-it-fp8-dynamic",
      #model="llama-3.3-70b-instruct",
      model="nvidia-llama-3.3-70b-instruct-fp8",
      #model="GPT-OSS-120B",
      #model="llama-3.3-70b-instruct-awq",
      #model="Llama-4-Maverick-17B-128E-Instruct-FP8",
      #model="GLM-4.7-FP8",
)

print(response.choices[0].message.content)