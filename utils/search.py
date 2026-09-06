import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model = "gemini-3.5-flash"
query = "what is json ?"


print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

client = OpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))
search_results = tavily_client.search(query, max_results=5)

print(search_results)

search_text = "\n".join(
    f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:2000]}\n"
    for result in search_results["results"]
)
print(search_text)


class SearchResults(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SearchResults]


messages = [
    {
        "role": "system",
        "content": """
You are an AI research assistant.
Use the provided search results to answer the user's question.
Be accurate, concise, and do not invent information.
""",
    },
    {
        "role": "user",
        "content": f"""
Question: {query}

Search results:
{search_text}
""",
    },
]
completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ResearchResponse,
    max_tokens=1000,
)

response = completion.choices[0].message.parsed
print(response.answer)

print("\nSOURCES:")
for source in response.sources:
    print(f"- {source.title}")
    print(f"  {source.url}")
