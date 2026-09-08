import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")
messages = [{"role": "user", "content": "What time is it now ?"}]

print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

client = OpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


def get_weather(city):
    return f"the weather in {city} is sunny"


def get_time():
    return datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005


available_tools = {
    "get_weather": get_weather,
    "get_time": get_time,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city."}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "get the current local time",
            "parameters": {
                "type": "object",
                "propertiens": {},
            },
        },
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    max_tokens=1000,
)

message = response.choices[0].message

if message.tool_calls:
    messages.append(message)

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function = available_tools[function_name]

        arguments = json.loads(tool_call.function.arguments)
        result = function(**arguments)

        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )

        print("TOOL RESULT:")
        print(result)


response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    max_tokens=1000,
)

message = response.choices[0].message

print("MODEL REQUESTED:")
print(message.tool_calls)
