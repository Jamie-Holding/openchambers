"""Agent state definitions."""

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class ContextUpdate(BaseModel):
    """LLM-facing schema: fields the classifier extracts from the user message."""

    person_names: list[str] = []
    parties: list[str] = []
    date_text: str | None = None
    search_query: str | None = None


class ActiveContext(BaseModel):
    """Full context: includes classifier fields plus resolved fields."""

    person_ids: list[int] = []
    person_names: list[str] = []
    parties: list[str] = []
    date_text: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    search_query: str | None = None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_context: ActiveContext
    last_turn_was_ai_question: bool
    retrieval_result: dict | None
    user_intent: str | None
    context_update: ContextUpdate | None
    need_votes: bool
