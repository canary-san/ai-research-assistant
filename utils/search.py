# ============================================================
# AI Research Assistant: Web Search + Structured Research Output
# This script sets up an OpenAI-compatible client, performs a web
# search, and asks the model to answer the user question using
# the retrieved results.
# ============================================================

import asyncio
import json
import os

import nest_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tavily import TavilyClient

# ------------------------------------------------------------
# 1) Load environment variables and initialize the async client
# ------------------------------------------------------------
load_dotenv()
nest_asyncio.apply()

key = os.getenv("GROQ_API_KEY")
base_url = os.getenv("GROQ_BASE_URL")
model = os.getenv("GROQ_MODEL")
question = "what are the best phones under 2000 dollars in 2026 ?"

print("Key exists:", key is not None)
print("Key empty:", key == "")
print("Key length:", len(key) if key else 0)
print("Base URL:", repr(base_url))

# This creates the client that will talk to the reasoning model.
client = AsyncOpenAI(api_key=key, base_url=base_url)
print("Client created successfully!")


# ------------------------------------------------------------
# 2) Function tool: search the web
# ------------------------------------------------------------
async def search_web(question):
    # Tavily is used to retrieve fresh live search results.
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))
    search_results = await tavily_client.async_search(question, max_results=5)

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
# 3) Define the function-calling schema for the model
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
                    "question": {
                        "type": "string",
                        "description": "The search question to search the web for.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }
]

# ------------------------------------------------------------
# 4) Build the initial conversation for the model
# ------------------------------------------------------------
messages = [
    {
        "role": "system",
        "content": (
            "You are a search planner. Given a research question, generate "
            "exactly 3 diverse, specific web search queries that together "
            "would fully answer it. Each query must be self-contained."
        ),
    },
    {"role": "user", "content": question},
]


async def generate_queries(question: str) -> list[str]:
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=QueryPlan,
        max_tokens=3000,
    )
    queries = completion.choices[0].message.parsed.queries

    messages.append(
        {
            "role": "assistant",
            "content": f"generated search queries: {queries}",
        }
    )
    return queries


queries = generate_queries(question)


# Ask the model to respond to the question.and decide whether it needs to
# call the web-search tool before answering.
async def check_search_needed(queries: list[str]):
    messages.append(
        {
            "role": "user",
            "content": (
                "You are an AI research assistant. "
                "before answering. If current or external information is needed, "
                "use the web search tool."
                "\n".join(queries)
            ),
        },
    )
    completion = await client.beta.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=2000,
    )

    message = completion.choices[0].message
    return message


# ------------------------------------------------------------
# 5) Handle tool calls from the model
# ------------------------------------------------------------
async def tool_call_handler(message):

    if message.tool_calls:
        # Add the model's tool call response to the conversation history.
        messages.append(message)

        # Run each tool call requested by the model.
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function = available_tools.get(function_name)
            if function is None:
                print(f"unknown tool: {function_name}")
                continue

            # Parse the JSON arguments passed to the tool.
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
# 6) Structured output models for research answers
# ------------------------------------------------------------
class SearchResults(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SearchResults]


class QueryPlan(BaseModel):
    queries: list[str]


# ------------------------------------------------------------
# 7) Final answer: ask the model to return structured JSON
# ------------------------------------------------------------
completion = client.beta.chat.completions.parse(
    model=model, messages=messages, response_format=ResearchResponse, max_tokens=2000
)
response = completion.choices[0].message.parsed

# Print the final answer and source list.
print("\nFINAL ANSWER:")
print(response.answer)

print("\nSOURCES:")
for source in response.sources:
    print(f"- {source.title}")
    print(f"  {source.url}")
    print(f"  {source.snippet}")
