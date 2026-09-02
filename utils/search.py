import os
from tavily import TavilyClient


tavily_client= TavilyClient(api_key=os.getenv("TAVILY-API-KEY"))

response= tavily_client.search("who is json?")

