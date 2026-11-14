"""
Multi-Agent Research System - Streamlit UI

Run with:  python -m streamlit run app.py
Keep this file next to agents.py, pipeline.py and tools.py.
Optional: put .streamlit/config.toml next to it for the matching theme.
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
# Page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Atelier",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');

    :root {
        --ink: #1B1F27;
        --muted: #6B7280;
        --navy: #1F3A5F;
        --gold: #B08D57;
        --paper: #FAF8F4;
        --card: #FFFFFF;
        --line: #E5E0D6;
    }

    html, body, [class*="css"], .stMarkdown, p, li, label {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 1100px;}

    /* Hero */
    .eyebrow {
        letter-spacing: 0.22em; text-transform: uppercase; font-size: 0.72rem;
        color: var(--gold); font-weight: 600; margin-bottom: 0.4rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif; font-size: 2.9rem; font-weight: 600;
        line-height: 1.1; color: var(--navy); margin: 0 0 0.6rem 0;
    }
    .hero-sub {color: var(--muted); font-size: 1.02rem; max-width: 640px; margin-bottom: 0.4rem;}
    .rule {border: none; border-top: 1px solid var(--line); margin: 1.4rem 0 1.6rem 0;}

    /* Inputs */
    .stTextInput input {
        background: var(--card); border: 1px solid var(--line); border-radius: 8px;
        padding: 0.8rem 1rem; font-size: 1rem;
    }
    .stTextInput input:focus {border-color: var(--navy); box-shadow: 0 0 0 1px var(--navy);}
    div[data-testid="stForm"] {border: none; padding: 0;}

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        background: var(--navy); color: #fff; border: 1px solid var(--navy);
        border-radius: 8px; padding: 0.55rem 1.4rem; font-weight: 500;
        letter-spacing: 0.02em; transition: all .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
        background: #16294a; border-color: #16294a; color: #fff;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent; color: var(--ink); border: 1px solid var(--line);
        text-align: left; font-weight: 400; padding: 0.45rem 0.8rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--card); border-color: var(--navy); color: var(--navy);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {border-right: 1px solid var(--line);}
    .side-brand {font-family: 'Playfair Display', serif; font-size: 1.35rem; color: var(--navy); font-weight: 600;}
    .side-label {
        letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.68rem;
        color: var(--muted); font-weight: 600; margin: 1.2rem 0 0.5rem 0;
    }

    /* Pipeline tracker */
    .tracker {display: flex; gap: 12px; margin: 0.5rem 0 1.4rem 0;}
    .tstep {
        flex: 1; background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: 0.9rem 1rem; position: relative;
    }
    .tstep .num {font-size: 0.7rem; letter-spacing: 0.16em; color: var(--muted); font-weight: 600;}
    .tstep .name {font-family: 'Playfair Display', serif; font-size: 1.15rem; color: var(--ink); margin-top: 2px;}
    .tstep .desc {font-size: 0.8rem; color: var(--muted); margin-top: 2px;}
    .tstep.active {border-color: var(--navy); box-shadow: 0 2px 10px rgba(31,58,95,.12);}
    .tstep.active .num {color: var(--navy);}
    .tstep.done {border-color: var(--gold);}
    .tstep.done .num {color: var(--gold);}
    .tstep.error {border-color: #B3402F;}
    .tstep.error .num {color: #B3402F;}
    .tstep.active::after {
        content: ""; position: absolute; left: 0; bottom: 0; height: 3px; width: 100%;
        background: linear-gradient(90deg, var(--navy), var(--gold), var(--navy));
        background-size: 200% 100%; animation: slide 1.6s linear infinite;
        border-radius: 0 0 10px 10px;
    }
    @keyframes slide {from {background-position: 200% 0;} to {background-position: 0 0;}}

    /* Metrics */
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.7rem; color: var(--muted);
    }
    div[data-testid="stMetricValue"] {font-family: 'Playfair Display', serif; color: var(--navy);}

    /* Tabs + report */
    .stTabs [data-baseweb="tab-list"] {gap: 1.6rem; border-bottom: 1px solid var(--line);}
    .stTabs [data-baseweb="tab"] {padding: 0.6rem 0; font-weight: 500; color: var(--muted);}
    .stTabs [aria-selected="true"] {color: var(--navy);}
    .stTabs [data-baseweb="tab-highlight"] {background-color: var(--navy);}
    .report-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 1.2rem 2rem; margin-top: 1rem;
    }
    .report-card h1, .report-card h2, .report-card h3 {
        font-family: 'Playfair Display', serif; color: var(--navy);
    }
    .section-title {font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--navy); margin: 0;}
    .section-sub {color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Pipeline definition + helpers
# ----------------------------------------------------------------------------
STEPS = [
    ("Search", "Gathering sources"),
    ("Read", "Extracting content"),
    ("Write", "Drafting the report"),
    ("Review", "Critical feedback"),
]


def render_tracker(placeholder, current: int, failed: bool = False, finished: bool = False):
    """Draw the 4-step progress tracker."""
    cards = []
    for i, (name, desc) in enumerate(STEPS):
        if finished or i < current:
            cls, tag = "done", "COMPLETE"
        elif i == current:
            cls, tag = ("error", "FAILED") if failed else ("active", "IN PROGRESS")
        else:
            cls, tag = "", "PENDING"
        cards.append(
            f'<div class="tstep {cls}"><div class="num">0{i + 1} · {tag}</div>'
            f'<div class="name">{name}</div><div class="desc">{desc}</div></div>'
        )
    placeholder.markdown(f'<div class="tracker">{"".join(cards)}</div>', unsafe_allow_html=True)


def message_text(msg) -> str:
    """Extract text from a LangChain message object or a dict."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", msg)
    if isinstance(content, list):
        parts = []
        for block in content:
            parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_pipeline(topic: str, tracker) -> dict:
    """Same 4 steps as pipeline.py, with a live progress tracker."""
    state = {"timings": []}
    current = 0
    try:
        # 1 - Search
        render_tracker(tracker, 0)
        t0 = time.time()
        search_agent = build_search_agent()
        res = search_agent.invoke(
            {"messages": [("user", f"find recent and reliable information on the topic: {topic}")]}
        )
        state["search_results"] = message_text(res["messages"][-1])
        state["timings"].append(time.time() - t0)

        # 2 - Read
        current = 1
        render_tracker(tracker, 1)
        t0 = time.time()
        reader_agent = build_reader_agent()
        res = reader_agent.invoke(
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
        state["scraped_content"] = message_text(res["messages"][-1])
        state["timings"].append(time.time() - t0)

        # 3 - Write
        current = 2
        render_tracker(tracker, 2)
        t0 = time.time()
        research = (
            f"search results:\n{state['search_results']}\n\n"
            f"detailed scraped content:\n{state['scraped_content']}"
        )
        state["report"] = message_text(writer_chain.invoke({"topic": topic, "research_gathered": research}))
        state["timings"].append(time.time() - t0)

        # 4 - Review
        current = 3
        render_tracker(tracker, 3)
        t0 = time.time()
        state["feedback"] = message_text(critic_chain.invoke({"report": state["report"]}))
        state["timings"].append(time.time() - t0)

        render_tracker(tracker, 4, finished=True)
        return state
    except Exception:
        render_tracker(tracker, current, failed=True)
        raise


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("topic", "")
st.session_state.setdefault("history", [])

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">◆ Research Atelier</div>', unsafe_allow_html=True)
    st.caption("Four agents. One considered report.")

    st.markdown('<div class="side-label">Recent research</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for i in range(len(st.session_state.history) - 1, -1, -1):
            item = st.session_state.history[i]
            label = item["topic"] if len(item["topic"]) <= 34 else item["topic"][:33] + "…"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.topic = item["topic"]
                st.session_state.result = item["result"]
                st.rerun()
        if st.button("Clear history", key="clear_hist", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Nothing yet.")

    st.markdown('<div class="side-label">Pipeline</div>', unsafe_allow_html=True)
    st.caption("Search → Read → Write → Review")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="eyebrow">Multi-Agent Research</div>
    <div class="hero-title">From question to<br>well-sourced report.</div>
    <div class="hero-sub">Enter a topic. Specialised agents will search the web, read the best
    sources, write a structured report and critique it.</div>
    <hr class="rule">
    """,
    unsafe_allow_html=True,
)

with st.form("research_form", clear_on_submit=False):
    topic_input = st.text_input(
        "Research topic",
        value=st.session_state.topic,
        placeholder="e.g. Recent advances in solid-state batteries",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Begin research")

tracker_slot = st.empty()

if submitted:
    topic = topic_input.strip()
    if not topic:
        st.warning("Please enter a topic to begin.")
    else:
        st.session_state.topic = topic
        st.session_state.result = None
        try:
            result = run_pipeline(topic, tracker_slot)
            st.session_state.result = result
            st.session_state.history.append({"topic": topic, "result": result})
        except Exception as e:
            msg = str(e)
            if "429" in msg or "no credits" in msg.lower() or "rate limit" in msg.lower():
                st.error(
                    "The LLM provider rejected the request: no credits left or rate limit reached. "
                    "Add credits, wait a moment, or switch to a free provider (Groq / Ollama) in agents.py."
                )
            else:
                st.error(f"Something went wrong: {msg}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
result = st.session_state.result
if result:
    # Keep tracker visible as 'complete' when reopening from history
    if not submitted:
        render_tracker(tracker_slot, 4, finished=True)

    st.markdown(
        f'<div class="section-title">{st.session_state.topic}</div>'
        f'<div class="section-sub">Research complete</div>',
        unsafe_allow_html=True,
    )

    total = sum(result.get("timings", [])) if result.get("timings") else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Report length", f"{len(result['report'].split()):,} words")
    c2.metric("Total time", f"{total:.0f}s" if total else "—")
    c3.metric("Agents run", "4")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Report", "Critic's review", "Search results", "Source material"]
    )

    with tab_report:
        with st.container(border=False):
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(result["report"])
            st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "Download report (.md)",
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