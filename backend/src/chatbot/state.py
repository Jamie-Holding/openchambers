"""Agent state definitions."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ActiveContext(TypedDict):
    person_ids: list[int]
    person_names: list[str]
    parties: list[str]
    date_from: str | None
    date_to: str | None
    search_query: str | None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_context: ActiveContext
    last_turn_was_ai_question: bool
    retrieval_result: dict | None
