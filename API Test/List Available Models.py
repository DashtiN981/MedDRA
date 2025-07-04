from openai import OpenAI

client = OpenAI(api_key="sk-BEYOnuDXHm5OcYLc5xKX6w", base_url="http://pluto/v1/")

print("List of available Models: ")
print(client.models.list())