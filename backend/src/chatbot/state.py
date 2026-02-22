"""Agent state definitions."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from backend.src.chatbot.schemas import ActiveContext, ContextUpdate


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_context: ActiveContext
    last_turn_was_ai_question: bool
    retrieval_result: dict | None
    user_intent: str | None
    context_update: ContextUpdate | None
    need_votes: bool
