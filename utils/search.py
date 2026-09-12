# ============================================================
# AI Research Assistant: Web Search + Structured Research Output
# This script sets up an OpenAI-compatible client, performs a web
# search, and asks the model to answer the user question using
# the retrieved results.
# ============================================================

import asyncio
import json
import logging
import os

import nest_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

# ------------------------------------------------------------
# 1) Load environment variables and initialize the async client
# ------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")
question = "what are the best phones under 2000 dollars in 2026 ?"

# Create the OpenAI-compatible client.
client = AsyncOpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


# ------------------------------------------------------------
# 2) Structured output models for research answers
# ------------------------------------------------------------
class SearchResults(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SearchResults]


class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)


# ------------------------------------------------------------
# 3) Build the initial conversation for the model
# ------------------------------------------------------------
messages = (
    [
        {
            "role": "system",
            "content": (
                "You are a search planner. Given a research question, generate "
                "exactly 3 diverse, specific web search queries that together "
                "would fully answer it. Each query must be self-contained."
            ),
        },
        {"role": "user", "content": question},
    ],
)


async def generate_queries(question: str) -> list[str]:
    messages.append(question)
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=QueryPlan,
        max_tokens=3000,
    )
    queries = completion.choices[0].message.parsed.queries

    return queries


# ------------------------------------------------------------
# 4) Function tool: search the web
# ------------------------------------------------------------
async def search_web(query: str) -> str:
    # Tavily is used to retrieve fresh live search results.
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))
    search_results = await tavily_client.async_search(query, max_results=5)

    # Keep only the most useful parts of each result so the model
    # has readable context to work with.
    search_text = "\n".join(
        f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:2000]}\n"
        for result in search_results["results"]
    )
    return search_text


# This dictionary lets us call the tool by name later.
available_tools = {"search_web": search_web}

# ------------------------------------------------------------
# 5) Define the function-calling schema for the model
# ------------------------------------------------------------
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


# Ask whether external information is needed before answering.
async def check_search_needed(queries: list[str]):
    for query in queries:
        messages.append(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an AI research assistant. "
                        "Before answering, decide whether external information is needed. "
                        "If needed, use the web search tool."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(query),
                },
            ]
        )

    completion = await client.beta.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=2000,
    )

    response = completion.choices[0].message
    return response


# ------------------------------------------------------------
# 6) Handle tool calls from the model
# ------------------------------------------------------------
async def tool_call_handler(response):

    if response.tool_calls:
        # Add the model's tool call response to the conversation history.
        messages.append(response)

        # Run each tool call requested by the model.
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            function = available_tools.get(function_name)
            if function is None:
                print(f"unknown tool: {function_name}")
                continue

            arguments = json.loads(tool_call.function.arguments)
            result = await function(**arguments)

            # Add the tool result back into the chat so the model can use it.
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

            print("TOOL RESULT:")
            print(result)

    return messages


# ------------------------------------------------------------
# 7) Request and print the structured final response.
# ------------------------------------------------------------
async def main():
    queries = await generate_queries(question)

    response = await check_search_needed(queries)

    if response.tool_calls:
        await tool_call_handler(response)

        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ResearchResponse,
            max_tokens=2000,
        )
        response = completion.choices[0].message.parsed
    else:
        # model didn't request a search
        final_response = response

    print("\nFINAL ANSWER:")
    print(final_response.answer)

    print("\nSOURCES:")
    for source in final_response.sources:
        print(f"- {source.title}")
        print(f"  {source.url}")
        print(f"  {source.snippet}")


if __name__ == "main":
    asyncio.run(main)
