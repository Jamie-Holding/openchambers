"""Hansard chatbot agent: graph creation and streaming."""

from collections.abc import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from backend.src.chatbot.graph import build_graph


def create_hansard_agent(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Build and compile the Hansard agent graph."""
    return build_graph(checkpointer=checkpointer)


async def ask_agent(
    graph: CompiledStateGraph, thread_id: str, text: str
) -> AsyncIterator[str]:
    """Stream responses from the Hansard agent.

    Yields tokens from the generate node's LLM call. For non-streamed
    responses (disambiguation questions, no-results), yields the full
    message after the graph completes.
    """
    config = {"configurable": {"thread_id": thread_id}}
    streamed = False

    async for event in graph.astream_events(
        {"messages": [("user", text)]}, config, version="v2"
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "generate"
        ):
            content = event["data"]["chunk"].content
            if content:
                streamed = True
                yield content

    if not streamed:
        state = await graph.aget_state(config)
        messages = state.values.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            yield messages[-1].content
