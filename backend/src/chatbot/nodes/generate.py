"""Synthesize a response from retrieval results."""

from langchain_core.messages import AIMessage

from backend.src.chatbot.state import AgentState


def generate_node(state: AgentState) -> dict:
    """Stub - will be replaced with LLM synthesis call."""
    return {"messages": [AIMessage(content="[stub] Graph executed successfully.")]}
