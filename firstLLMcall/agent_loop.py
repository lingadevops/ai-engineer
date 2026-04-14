import os
from opeanai import OpenAI
import config

API_KEY = config.API_KEY
API = config.API

client = OpenAI(
    api_key = API_KEY,
    base_url = API
)

message = [
    {"role": "system", "content": "You are a helpful assistant."
    },
    {"role": "user", "content": "What are three things an AI agent can do that a regular chatbot cannot?"}

]

while True:
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = message,
    )
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "stop":
        print(response.choices[0].message.content)
        break
    else:
        break

