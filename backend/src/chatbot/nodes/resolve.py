"""Resolve dates/names/parties and decide whether to retrieve or ask."""

from langchain_core.messages import AIMessage

from src.chatbot.dates import parse_dates
from src.chatbot.messages.resolve import (
    PERSON_DISAMBIGUATION,
    PERSON_NOT_FOUND,
    format_person_options,
)
from src.chatbot.schemas import ActiveContext, ContextUpdate
from src.chatbot.state import AgentState
from src.chatbot.utils import hansard_tool


def _merge_context(
    active_context: ActiveContext, context_update: ContextUpdate, intent: str
) -> ActiveContext:
    """Merge classifier output into active context based on intent."""
    if intent == "new_query":
        return ActiveContext(
            person_names=context_update.person_names,
            parties=context_update.parties,
            date_text=context_update.date_text,
            search_query=context_update.search_query,
        )

    if intent == "refine_query":
        data = active_context.model_dump()
        if context_update.person_names:
            data["person_names"] = context_update.person_names
            data["person_ids"] = []
        if context_update.parties:
            data["parties"] = context_update.parties
        if context_update.date_text is not None:
            data["date_text"] = context_update.date_text
            data["date_from"] = None
            data["date_to"] = None
        if context_update.search_query is not None:
            data["search_query"] = context_update.search_query
        return ActiveContext(**data)

    # answer_to_question - don't merge, keep existing context
    return active_context


def _resolve_people(active_context: ActiveContext) -> tuple[ActiveContext, str | None]:
    """Resolve person names to IDs. Returns (updated_context, ask_message_or_None)."""
    if not active_context.person_names or active_context.person_ids:
        return active_context, None

    person_ids = []
    for name in active_context.person_names:
        matches = hansard_tool.list_people(name)

        if len(matches) == 1:
            # Exact match - add the ID.
            person_ids.append(matches[0]["person_id"])

        elif len(matches) == 0:
            # No matches - ask user to rephrase.
            return active_context, PERSON_NOT_FOUND.format(name=name)

        else:
            # Ambiguous name - ask user to clarify.
            options = format_person_options(matches)
            return active_context, PERSON_DISAMBIGUATION.format(
                name=name, options=options
            )

    updated = active_context.model_copy(update={"person_ids": person_ids})
    return updated, None


def _resolve_person_disambiguation(
    active_context: ActiveContext, user_message: str
) -> tuple[ActiveContext, str | None]:
    """Resolve a person disambiguation answer (number or name) to a person ID."""
    if not active_context.person_names or active_context.person_ids:
        return active_context, None

    for name in active_context.person_names:
        matches = hansard_tool.list_people(name)

        if len(matches) < 2:
            continue

        text = user_message.strip()

        # Try parsing as a number selection
        idx = int(text) - 1 if text.isdigit() else None
        if idx is not None and 0 <= idx < len(matches):
            updated = active_context.model_copy(
                update={
                    "person_ids": [matches[idx]["person_id"]],
                    "person_names": [matches[idx]["display_name"]],
                }
            )
            return updated, None

        # Try matching by name - only accept if exactly one match
        text_lower = text.lower()
        name_matches = [m for m in matches if text_lower in m["display_name"].lower()]
        if len(name_matches) == 1:
            updated = active_context.model_copy(
                update={
                    "person_ids": [name_matches[0]["person_id"]],
                    "person_names": [name_matches[0]["display_name"]],
                }
            )
            return updated, None

        # Ambiguous or invalid answer - ask again
        options = format_person_options(matches)
        return active_context, PERSON_DISAMBIGUATION.format(name=name, options=options)

    return active_context, None


def _ask(active_context: ActiveContext, message: str) -> dict:
    """Return an early-exit dict that asks the user a clarifying question."""
    return {
        "active_context": active_context,
        "last_turn_was_ai_question": True,
        "messages": [AIMessage(content=message)],
    }


async def resolve_node(state: AgentState) -> dict:
    """Resolve context: merge, resolve people, parse dates.

    Any step can return early with a clarifying question (ASK).
    The graph routes ASK responses to END so the user sees the question.
    """
    intent = state.get("user_intent", "new_query")
    context_update_raw = state.get("context_update") or {}
    context_update = ContextUpdate(**context_update_raw)
    active_context = state.get("active_context", ActiveContext())

    if not isinstance(active_context, ActiveContext):
        active_context = ActiveContext(**active_context)

    # Merge classifier output or handle disambiguation answer
    if intent == "answer_to_question":
        user_message = state["messages"][-1].content
        active_context, ask_msg = _resolve_person_disambiguation(
            active_context, user_message
        )
        if ask_msg:
            return _ask(active_context, ask_msg)
    else:
        active_context = _merge_context(active_context, context_update, intent)

    # Resolve person names to IDs
    active_context, ask_msg = _resolve_people(active_context)
    if ask_msg:
        return _ask(active_context, ask_msg)

    # Parse dates
    date_from, date_to, ask_msg = await parse_dates(active_context.date_text)
    if ask_msg:
        return _ask(active_context, ask_msg)

    active_context = active_context.model_copy(
        update={"date_from": date_from, "date_to": date_to}
    )

    return {
        "active_context": active_context,
        "last_turn_was_ai_question": False,
    }
