# graph.py

from typing import Any

from langgraph.graph import StateGraph, END

from memory import AgentState
from agents.planner import planner_agent
from agents.retriever import retriever_agent
from agents.writer import writer_agent
from agents.reflector import reflector_agent


def should_retry(state: AgentState) -> str:
    """Decide whether to retry retrieval or end the workflow."""
    return "retriever" if state.get("needs_revision", False) else END


def build_graph() -> Any:
    """
    Build and compile the LangGraph workflow:
    query -> planner -> retriever -> writer -> reflector -> (retriever or END)
    """
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("planner", planner_agent)
    g.add_node("retriever", retriever_agent)
    g.add_node("writer", writer_agent)
    g.add_node("reflector", reflector_agent)

    # Entry point
    g.set_entry_point("planner")

    # Edges
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "writer")
    g.add_edge("writer", "reflector")
    g.add_conditional_edges("reflector", should_retry, {"retriever": "retriever", END: END})

    app = g.compile()
    return app
