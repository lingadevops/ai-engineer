import os
import json
from openai import OpenAI
from tools import check_calendar, search_web, get_user_preferences, TOOLS_SCHEMA

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

MAX_ITERATIONS = 10

system_prompt = """You are a helpful personal assistant.
Use your tools to find information when needed.
Before calling a tool, use 'Thought:' to explain your reasoning.
After a tool result, use 'Observation:' to note what you learned.
Provide clear, concise answers."""

def run_agent(user_message, conversation_history=None):
    
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=messages,
            tools=TOOLS_SCHEMA
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                try:
                    if name == "check_calendar":
                        result = check_calendar(**args)
                    elif name == "search_web":
                        result = search_web(**args)
                    elif name == "get_user_preferences":
                        result = get_user_preferences(**args)
                    elif name == "broken_tool":
                        from tools import broken_tool
                        result = broken_tool(**args)
                    else:
                        result = f"Unknown tool: {name}"
                except Exception as e:
                    result = f"Error: {str(e)}. Please try a different approach."
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            #print(msg.content)
            return msg.content


response = run_agent("What's on my calendar for monday?")
print(response)