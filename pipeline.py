from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

def run_search_pipeline(topic : str)-> str:
    search_agent = build_search_agent()

    state={}

    #search agent working
    print("\n"+"="*50)
    print("step 1 -search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"find recent and reliable information on the topic: {topic}")],
    })

    state["search_results"] = search_result["messages"][-1]["content"]

    print("\n search result",state ['search_results'])