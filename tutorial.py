import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")
messages = [{"role": "user", "content": "What's the weather in Algiers?"}]

print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

client = OpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


def get_weather(city):
    return f"the weather in {city} is sunny"


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
    }
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
        if tool_call.function.name == "get_weather":
            arguments = json.loads(tool_call.function.arguments)
            result = get_weather(arguments["city"])

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
