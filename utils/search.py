import os

from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

client = OpenAI(
    api_key=os.getenv("GEMINI-API-KEY"), base_url=os.getenv("GEMINI-BASE-URL")
)
model = "gemini-3.5-flash"

tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))

response = tavily_client.search("what is json?")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


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
    {"role": "user", "content": response},
]

completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=SearchResult,
    max_tokens=1000,
)
