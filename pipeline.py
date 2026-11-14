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

    state["search_results"] = search_result["messages"][-1].content

    print("\n search result",state ['search_results'])

    #step2 - reader agent

    print("\n"+"="*50)
    print("step 2 -reader agent is working ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages":[("user",
            f"based on the following search results about '{topic}',"
            f"pick the most relevant URLs and scrape their content for deeper reading. Here are the search results:\n\n"
            f"search results:\n{state['search_results'][:800]}"
        )]
    })

    state["scraped_content"] = reader_result['messages'][-1].content

    print("\nscraped content",state['scraped_content'])


    #step 3 - writer agent

    print("\n"+"="*50)
    print("step 3 -writer agent is working ...")
    print("="*50)

    research_gathered = (
        f"search results:\n{state['search_results']}\n\n"
        f"detailed scraped content:\n{state['scraped_content']}"
    )

    state["report"]=writer_chain.invoke({
        "topic": topic,
        "research_gathered": research_gathered
    })
    print("\nfinal report\n", state["report"])


    #critic agent

    print("\n"+"="*50)
    print("step 4 - critic is reviewing the report ...")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
    print("\n critic report\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a topic to research: ")
    run_search_pipeline(topic)