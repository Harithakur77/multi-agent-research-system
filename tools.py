
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import tavily
from rich import print
load_dotenv()

tavily_client = TavilyClient( api_key=os.getenv("TAVILY_API_KEY") )

@tool

def web_search(query: str) -> str:
     """Search the web for recent and reliable information on a topic. Returns titles, URLs, and snippets. """ 
     results = tavily_client.search( query=query, max_results=5 ) 
     return str(results)
     
     
print(web_search.invoke("What are the recent news about the war?"))