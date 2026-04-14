import config
from openai import OpenAI
API_KEY = config.API_KEY
OpenAI.api = config.API

client = OpenAI(
    api_key=API_KEY,
    base_url=config.API
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check the user calendar for a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check the calendar for, in YYYY-MM-DD format"
                    }
                },
                "required": ["date"]
        }
    }
    }

]

def check_calendar(date):
    return "10am: Team standup, 2pm: Dentist appointment"

def execute_tool(name, args):
    if name == "check_calendar":
        return check_calendar(**args)
    return f"Unknown tool: {name}"

system_message = "You are a helpful assistant."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "What's on my calendar today?"}
]

response = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=messages,
    tools=tools,
)

finish_reason = response.choices[0].finish_reason
print(f"Finish reason: {finish_reason}")

if finish_reason == "tool_calls":
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        name = tool_call.function.name
        import json
        args = json.loads(tool_call.function.arguments)
        result = execute_tool(name, args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    final_response = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=messages,
        tools=tools,
    )
    print(final_response.choices[0].message.content)
else:
    print(response.choices[0].message.content)