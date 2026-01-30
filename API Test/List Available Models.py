from openai import OpenAI

client = OpenAI(api_key="sk-aKGeEFMZB0gXEcE51FTc0A", base_url="http://pluto/v1/")

print("List of available Models: ")
print(client.models.list())