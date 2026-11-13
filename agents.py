from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

#1st agent

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )

#2nd Agent

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )


#writer_prompt

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert resarch writer.write clear, structured, and well-researched content based on the information provided by the user. Ensure that your writing is coherent, engaging, and informative. Avoid plagiarism and provide proper citations when necessary."),
    ("human", """write a detailed resarch report on the topic below.

Topic: {topic}

Resarch Gathered: {research_gathered}

structure the report as :
- Introduction
-key Findings(minimum 3 well_explained key findings)
-Conclusion 
- Sources (list all URLs found in the resarch)

BE detailed,factual and professional. """),


])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert research critic. your task is to critically evaluate the research report provided by the user. Assess the clarity, coherence, and depth of the content. Identify any gaps in information, logical inconsistencies, or areas that require further elaboration. Provide constructive feedback and suggestions for improvement."),
    ("human", """Critically evaluate the research report below.

report: {report}

respond in this exact format:

score:x/10
- ...
- ...

strengths :
- ...
- ...

Areas for improvement:
- ...
- ...

one line verdict: 
..."""),
])

critic_chain= critic_prompt | llm | StrOutputParser()
