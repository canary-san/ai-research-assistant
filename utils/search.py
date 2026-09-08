import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")
query = "what are the best phones under 2000 dollars in 2026 ?"

print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

client = OpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


def search_web(query):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))
    search_results = tavily_client.search(query, max_results=5)

    search_text = "\n".join(
        f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:2000]}\n"
        for result in search_results["results"]
    )
    return search_text


available_tools = {"search_web": search_web}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for answers to the query.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to search the web for.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": """
You are an AI research assistant.
Use the provided search results to answer the user's question.
Be accurate, concise, and do not invent information.
""",
    },
    {"role": "user", "content": query},
]

completion = client.beta.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    max_tokens=1000,
)

message = completion.choices[0].message

if message.tool_calls:
    messages.append(message)

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function = available_tools.get(function_name)
        if function is None:
            print(f"unknown tool: {function_name}")
            continue

        arguments = json.loads(tool_call.function.arguments)
        result = function(**arguments)

        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )

        print("TOOL RESULT:")
        print(result)


class SearchResults(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SearchResults]


completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ResearchResponse,
    max_tokens=1000,
)
response = completion.choices[0].message.parsed


print("\nSOURCES:")
for source in message.sources:
    print(f"- {source.title}")
    print(f"  {source.url}")
