import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model = "gemini-3.5-flash"

print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

client = OpenAI(api_key=key, base_url=base_url)

print("Client created successfully!")


tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))

response = tavily_client.search("what is json?")

print(response)

search_text = "\n".join(
    f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n"
    for result in response["results"]
)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


messages = [
    {
        "role": "system",
        "content": """
        You are an AI research assistant.
        Analyze the provided search results and extract the information relevant to the user's question.
        Be accurate, concise, and do not invent information.
        Separate facts from uncertainty.
        """,
    },
    {"role": "user", "content": json.dumps(response)},
]

completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ResearchResponse,
    max_tokens=1000,
)

result = completion.choices[0].message.parsed
print(result)
