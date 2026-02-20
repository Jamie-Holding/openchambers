"""Hansard agent graph definition."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.src.chatbot.nodes.classify import classify_node
from backend.src.chatbot.nodes.generate import generate_node
from backend.src.chatbot.nodes.resolve import resolve_node
from backend.src.chatbot.nodes.retrieve import retrieve_node
from backend.src.chatbot.state import AgentState


def route_after_resolve(state: AgentState) -> str:
    """Route to retrieve or end (when resolve asked a clarifying question)."""
    if state.get("last_turn_was_ai_question"):
        return END
    return "retrieve"


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Build and compile the Hansard agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("resolve", resolve_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "resolve")
    builder.add_conditional_edges("resolve", route_after_resolve, ["retrieve", END])
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)
