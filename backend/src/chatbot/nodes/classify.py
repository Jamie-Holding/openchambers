"""Classify user intent and extract structured context updates."""

from backend.src.chatbot.state import AgentState


def classify_node(state: AgentState) -> dict:
    """Stub - will be replaced with LLM structured output extraction."""
    last_message = state["messages"][-1].content
    return {
        "active_context": {
            **state["active_context"],
            "search_query": last_message,
        },
    }
