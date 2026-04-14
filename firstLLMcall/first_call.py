from openai import OpenAI
import config

API_KEY = config.API_KEY
client = OpenAI(api_key=API_KEY)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "What is AI?"},
        temperature=0
    ]
)

print(response.choices[0].message.content)