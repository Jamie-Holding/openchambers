"""Classify user intent and extract structured context updates."""

from langchain_core.messages import SystemMessage

from backend.src.chatbot.prompts.classify import CLASSIFY_PROMPT
from backend.src.chatbot.schemas import ActiveContext, Classification
from backend.src.chatbot.state import AgentState
from backend.src.chatbot.utils import hansard_tool, llm

MAX_MESSAGES = 8


async def classify_node(state: AgentState) -> dict:
    """Call the LLM to classify user intent and extract context updates."""
    parties_str = ", ".join(sorted(hansard_tool.list_parties()))
    active_context = state.get("active_context", ActiveContext())
    if hasattr(active_context, "model_dump"):
        active_context_str = str(active_context.model_dump())
    else:
        active_context_str = str(active_context)

    system = SystemMessage(content=CLASSIFY_PROMPT.format(
        parties=parties_str,
        active_context=active_context_str,
        last_turn_was_ai_question=state.get("last_turn_was_ai_question", False),
    ))

    recent_messages = state["messages"][-MAX_MESSAGES:]
    classifier = llm.with_structured_output(Classification)
    result = await classifier.ainvoke([system, *recent_messages])

    return {
        "user_intent": result.user_intent,
        "context_update": result.context_update.model_dump(),
        "need_votes": result.need_votes,
    }
