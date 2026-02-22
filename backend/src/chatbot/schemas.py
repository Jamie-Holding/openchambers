"""Pydantic schemas for the agent."""

from typing import Literal

from pydantic import BaseModel


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


class Classification(BaseModel):
    """Structured output from the classifier LLM."""

    user_intent: Literal["new_query", "refine_query", "answer_to_question"]
    context_update: ContextUpdate
    need_votes: bool
