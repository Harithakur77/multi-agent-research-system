"""
Streamlit UI for the Multi-Agent Research System.

Run with:  streamlit run app.py
Place this file in the same folder as agents.py, pipeline.py and tools.py.
"""

import time
import traceback

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import (  # noqa: E402
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .step-title {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def message_text(msg) -> str:
    """Extract text from a LangChain message object or a dict."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", msg)

    # Some models return a list of content blocks
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_pipeline_with_progress(topic: str) -> dict:
    """Same 4 steps as pipeline.py, but with live progress in the UI."""
    state = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1 - Search agent
        st.write("🔎 **Step 1/4** — Search agent is gathering information...")
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {
                "messages": [
                    ("user", f"find recent and reliable information on the topic: {topic}")
                ],
            }
        )
        state["search_results"] = message_text(search_result["messages"][-1])
        st.write(f"✅ Search done in {time.time() - t0:.1f}s")

        # Step 2 - Reader agent
        st.write("📖 **Step 2/4** — Reader agent is scraping top sources...")
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"based on the following search results about '{topic}', "
                        f"pick the most relevant URLs and scrape their content for deeper reading. "
                        f"Here are the search results:\n\n"
                        f"search results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = message_text(reader_result["messages"][-1])
        st.write(f"✅ Reading done in {time.time() - t0:.1f}s")

        # Step 3 - Writer
        st.write("✍️ **Step 3/4** — Writer is drafting the report...")
        t0 = time.time()
        research_gathered = (
            f"search results:\n{state['search_results']}\n\n"
            f"detailed scraped content:\n{state['scraped_content']}"
        )
        report = writer_chain.invoke({"topic": topic, "research": research_gathered})
        state["report"] = message_text(report)
        st.write(f"✅ Report written in {time.time() - t0:.1f}s")

        # Step 4 - Critic
        st.write("🧐 **Step 4/4** — Critic is reviewing the report...")
        t0 = time.time()
        feedback = critic_chain.invoke({"report": state["report"]})
        state["feedback"] = message_text(feedback)
        st.write(f"✅ Review done in {time.time() - t0:.1f}s")

        status.update(label="Pipeline complete ✅", state="complete", expanded=False)

    return state


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"topic":..., "result":...}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Research System")
    st.caption("Search → Reader → Writer → Critic")

    st.markdown("---")
    st.subheader("History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            if st.button(item["topic"][:40], key=f"hist_{idx}", use_container_width=True):
                st.session_state.topic = item["topic"]
                st.session_state.result = item["result"]
                st.rerun()
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("No research yet.")

    st.markdown("---")
    st.caption("Powered by LangChain agents")


# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.title("Multi-Agent Research System")
st.write("Enter a topic and four agents will search, read, write and critique a report for you.")

with st.form("research_form", clear_on_submit=False):
    topic_input = st.text_input(
        "Research topic",
        value=st.session_state.topic,
        placeholder="e.g. Latest advancements in solid-state batteries",
    )
    submitted = st.form_submit_button("🚀 Run research", type="primary")

if submitted:
    topic = topic_input.strip()
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        st.session_state.topic = topic
        st.session_state.result = None
        try:
            result = run_pipeline_with_progress(topic)
            st.session_state.result = result
            st.session_state.history.append({"topic": topic, "result": result})
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
result = st.session_state.result
if result:
    st.markdown("---")
    st.subheader(f"Results: {st.session_state.topic}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔎 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        st.markdown(result["search_results"])

    with tab_scraped:
        st.markdown(result["scraped_content"])