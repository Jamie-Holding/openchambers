"""Hansard chatbot agent with tool-calling and conversation persistence."""

from collections.abc import AsyncIterator

from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI

from backend.config.settings import AGENT_DEBUG, AGENT_MODEL_NAME, EMBEDDING_MODEL_NAME
from backend.src.chatbot.prompt import AGENT_PROMPT
from backend.src.chatbot.tools import HansardRetrievalTool


llm = ChatOpenAI(
    model=AGENT_MODEL_NAME,
    temperature=0.0,
    top_p=1.0,
)

hansard_tool = HansardRetrievalTool(model_name=EMBEDDING_MODEL_NAME, top_k=10, min_similarity=0.1)


def create_hansard_agent(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    parties_str = ", ".join(sorted(hansard_tool.parties))
    system_prompt = AGENT_PROMPT.format(parties=parties_str)

    graph = create_agent(
        model=llm,
        tools=[
            hansard_tool.fetch,
            hansard_tool.list_people,
            hansard_tool.get_mp_voting_record
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        debug=AGENT_DEBUG
    )
    return graph


async def ask_agent(graph: CompiledStateGraph, thread_id: str, text: str) -> AsyncIterator[str]:
    """Stream responses from the Hansard agent.

    Args:
        graph: Compiled LangGraph agent.
        thread_id: Unique identifier for the conversation thread.
        text: User's question or message.

    Yields:
        Response tokens as they are generated.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # We use 'astream_events' to capture token-level details
    # 'v2' is the current standard for LangGraph events
    async for event in graph.astream_events(
            {"messages": [("user", text)]},
            config,
            version="v2"
    ):
        # We specifically look for chat model stream events
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content  # Yield tokens one by one
