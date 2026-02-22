"""Classify user intent and extract structured context updates."""

from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from backend.src.chatbot.state import AgentState, ContextUpdate
from backend.src.chatbot.utils import hansard_tool, llm

MAX_MESSAGES = 8


class Classification(BaseModel):
    """Structured output from the classifier LLM."""

    user_intent: Literal["new_query", "refine_query", "answer_to_question"]
    context_update: ContextUpdate
    need_votes: bool


CLASSIFY_PROMPT = """You are a query classifier for a UK parliamentary search system.

Analyse the user's latest message and return structured JSON.

## Intents

- "new_query": A brand new search topic unrelated to the previous conversation.
- "refine_query": The user is modifying a previous search (e.g. changing date, party, or asking a follow-up on the same topic).
- "answer_to_question": The user is responding to a clarification question from the assistant (e.g. selecting an MP from a list, confirming a choice).

## Fields to extract

- person_names: List of MP/speaker names mentioned. Empty list if none.
- parties: List of political parties mentioned. Empty list if none. Valid parties: {parties}
- date_text: The raw date expression from the user (e.g. "last 6 months", "in 2025", "since January"). null if no date mentioned.
- search_query: The semantic core of the query with names, parties, and dates stripped out. e.g. "What has Labour said about housing in 2025?" -> "housing". null if the user is just answering a question.
- need_votes: true if the user is asking about voting records, voting patterns, or how someone voted. false otherwise.

## Current context

{active_context}

## Last turn was AI question: {last_turn_was_ai_question}

If last_turn_was_ai_question is true, the user is likely responding to a clarification. Consider "answer_to_question" as the intent.

## Rules

- Only extract what the user explicitly mentions. Do not infer or guess.
- For refine_query, only set fields the user is changing. Leave others as defaults.
- For date_text, extract the exact phrasing the user used. Do not convert to dates.
"""


async def classify_node(state: AgentState) -> dict:
    """Call the LLM to classify user intent and extract context updates."""
    parties_str = ", ".join(sorted(hansard_tool.list_parties()))
    active_context = state["active_context"]
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
