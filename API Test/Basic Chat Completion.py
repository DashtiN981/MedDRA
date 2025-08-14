from openai import OpenAI

client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

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
      model="Llama-3.3-70B-Instruct",
      #model="llama-3.3-70b-instruct-awq",
    
)

print(response.choices[0].message.content)