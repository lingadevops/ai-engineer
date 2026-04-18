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
questions = [
    "What is an agent?",
    "How is that different from a chatbot?",
    "Give me one example.",
]

for question in questions:
    message.append({"role": "user", "content": question})
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=message,
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "stop":
            reply = response.choices[0].message.content
            print(f"Q: {question}")
            print(f"A: {reply}")
            print()
            message.append({"role": "assistant", "content": reply})
            break
        else:
            break
