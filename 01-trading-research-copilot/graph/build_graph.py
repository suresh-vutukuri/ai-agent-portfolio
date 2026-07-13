"""Assembles the LangGraph pipeline: fetch -> analyze -> synthesize.

LangSmith tracing (LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY/LANGCHAIN_PROJECT,
see .env.example) needs no extra setup here: langchain-core's callback
manager and langsmith's client both read those as plain environment
variables on every run, and CompiledStateGraph.invoke() is itself a
LangChain Runnable, so the whole graph (and the ChatOpenAI call inside
synthesize_node) gets traced automatically once the vars are set and loaded
before .invoke() is called (see run_copilot.py's load_dotenv() call).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nodes import (
    compute_htf_bias_node,
    fetch_htf_node,
    fetch_ltf_node,
    find_ltf_candidates_node,
)
from state import CopilotState
from synthesis_node import synthesize_node


def build_graph() -> CompiledStateGraph:
    """Build and compile the trading research copilot's LangGraph pipeline.

    Node sequence: fetch_htf -> compute_htf_bias -> fetch_ltf ->
    find_ltf_candidates -> synthesize. Each node is defensive (see nodes.py
    and synthesis_node.py): failures are recorded in state['errors'] rather
    than raising, so downstream nodes still run - and the synthesis step can
    acknowledge what's missing - instead of the whole run crashing.

    Returns:
        A compiled LangGraph graph ready to `.invoke(initial_state)`.
    """
    graph = StateGraph(CopilotState)

    graph.add_node("fetch_htf", fetch_htf_node)
    graph.add_node("compute_htf_bias", compute_htf_bias_node)
    graph.add_node("fetch_ltf", fetch_ltf_node)
    graph.add_node("find_ltf_candidates", find_ltf_candidates_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "fetch_htf")
    graph.add_edge("fetch_htf", "compute_htf_bias")
    graph.add_edge("compute_htf_bias", "fetch_ltf")
    graph.add_edge("fetch_ltf", "find_ltf_candidates")
    graph.add_edge("find_ltf_candidates", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
