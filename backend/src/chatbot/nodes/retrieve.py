"""Run search tools based on resolved context."""

import asyncio
from itertools import product

from src.chatbot.schemas import ActiveContext
from src.chatbot.state import AgentState
from src.chatbot.utils import hansard_tool


def _build_filter_combos(active_context: ActiveContext) -> list[dict]:
    """Build a list of filter dicts from person_ids and parties.

    Each combo is a dict with person_id and party keys, suitable for
    passing to hansard_tool.fetch(). When a list is empty, the filter
    is omitted (None).
    """
    person_ids = active_context.person_ids or [None]
    parties = active_context.parties or [None]

    return [
        {"person_id": pid, "party": party}
        for pid, party in product(person_ids, parties)
    ]


async def retrieve_node(state: AgentState) -> dict:
    """Fetch quotes and votes in parallel based on resolved context."""
    active_context = state.get("active_context", ActiveContext())
    if not isinstance(active_context, ActiveContext):
        active_context = ActiveContext(**active_context)

    need_votes = state.get("need_votes", False)

    # Quote retrieval - one task per person/party combination
    quote_tasks = []
    if active_context.search_query:
        for combo in _build_filter_combos(active_context):
            quote_tasks.append(
                asyncio.to_thread(
                    hansard_tool.fetch,
                    query=active_context.search_query,
                    date_from=active_context.date_from,
                    date_to=active_context.date_to,
                    **combo,
                )
            )

    # Vote retrieval - one task per person_id
    vote_tasks = []
    if need_votes and active_context.person_ids and active_context.search_query:
        for person_id in active_context.person_ids:
            vote_tasks.append(
                asyncio.to_thread(
                    hansard_tool.get_mp_voting_record,
                    person_id=person_id,
                    search_term=active_context.search_query,
                )
            )

    quote_batches, vote_batches = await asyncio.gather(
        asyncio.gather(*quote_tasks),
        asyncio.gather(*vote_tasks),
    )

    quotes = [r for batch in quote_batches for r in batch]
    votes = [r for batch in vote_batches for r in batch]

    return {"retrieval_result": {"quotes": quotes, "votes": votes}}
