"""Hansard retrieval tools for semantic search and metadata lookup."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import and_, desc, func, literal, select
from sqlalchemy.orm import Session

from backend.src.data.db import init_db
from backend.src.data.database.models import (
    Embedding,
    Membership,
    Person,
    Utterance,
    UtteranceChunk,
    MPPolicySummary,
)


class HansardRetrievalTool:
    """Fetch relevant Hansard chunks using filtered semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        top_k: int = 20,
        min_similarity: float | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.engine = init_db()
        self.parties = self._fetch_parties()

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        party: str | None = None,
        person_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_similarity: float | None = None,
    ) -> list[Utterance]:
        """Search chunks and return distinct parent utterances.

        Searches against chunk embeddings but returns the full parent
        utterance for exact quoting. Deduplicates results so multiple
        matching chunks from the same utterance only return it once.

        Args:
            query_embedding: The query vector to search against.
            party: Filter by political party.
            person_id: Filter by speaker's person ID.
            date_from: Filter by minimum date (inclusive).
            date_to: Filter by maximum date (inclusive).
            min_similarity: Minimum cosine similarity threshold (0 to 1).
                Results below this threshold are excluded.

        Returns:
            List of Utterance objects ordered by semantic similarity.
        """
        with Session(self.engine) as session:
            conditions = []

            if party:
                conditions.append(func.lower(Utterance.party_at_time) == party.lower())

            if person_id:
                conditions.append(Utterance.person_id == person_id)

            if date_from:
                conditions.append(Utterance.date >= date_from)

            if date_to:
                conditions.append(Utterance.date <= date_to)

            # Subquery: find best (minimum) distance per utterance
            best_chunks = (
                select(
                    UtteranceChunk.utterance_id,
                    func.min(
                        Embedding.embedding.op("<#>")(query_embedding)
                    ).label("best_distance"),
                )
                .join(Embedding, Embedding.chunk_id == UtteranceChunk.id)
                .group_by(UtteranceChunk.utterance_id)
                .subquery()
            )

            # Main query: join utterances to their best distances
            stmt = select(Utterance).join(
                best_chunks, best_chunks.c.utterance_id == Utterance.id
            )

            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Filter by minimum similarity threshold
            # <#> returns negative inner product, so distance = -similarity
            effective_min_sim = min_similarity if min_similarity is not None else self.min_similarity
            if effective_min_sim is not None:
                stmt = stmt.where(best_chunks.c.best_distance <= literal(-effective_min_sim))

            stmt = stmt.order_by(best_chunks.c.best_distance).limit(self.top_k)

            return list(session.execute(stmt).scalars().all())

    def fetch(
        self,
        query: str,
        party: str | None = None,
        person_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_similarity: float | None = None,
    ) -> list[dict[str, object]]:
        """
        Fetch and format Hansard utterance data for the agent.

        Date format: YYYY-MM-DD

        Args:
            query: The search query text.
            party: Filter by political party.
            person_id: Filter by speaker's person ID.
            date_from: Filter by minimum date (inclusive).
            date_to: Filter by maximum date (inclusive).
            min_similarity: Minimum cosine similarity threshold (0 to 1).
                Results below this threshold are excluded.

        Returns:
            List of structured dicts containing matching Hansard utterances,
            or an empty list if no results were found.
        """

        embedding = self.model.encode([query], convert_to_numpy=True)[0]

        results = self._vector_search(
            embedding,
            party=party,
            person_id=person_id,
            date_from=date_from,
            date_to=date_to,
            min_similarity=min_similarity,
        )

        return [self._format_search_result(utt) for utt in results]

    def _format_search_result(self, utt: Utterance) -> dict[str, object]:
        """Format an utterance as a structured dict.

        Args:
            utt: Utterance object to format.

        Returns:
            Structured dict with speaker info, original text, and context.
        """
        return {
            "date": str(utt.date),
            "text": utt.original_utterance,
            "speaker": {
                "name": utt.speakername,
                "office": utt.speakeroffice,
            },
            "party": utt.party_at_time,
            "speech_id": utt.speech_id,
            "context": {
                "topic": utt.minor_heading,
                "department": utt.major_heading,
                "session": utt.oral_heading,
                "main_question": {
                    "speaker": utt.question_speaker,
                    "text": utt.original_question_text,
                } if utt.original_question_text else None,
                "context_question": {
                    "speaker": utt.context_question_speaker,
                    "text": utt.original_context_question_text,
                } if utt.original_context_question_text else None,
            },
        }

    def list_parties(self) -> list[str]:
        """Get a list of distinct political parties from the database."""
        return self.parties

    def _fetch_parties(self) -> list[str]:
        """Return all valid party_at_time values present in the Hansard database."""
        with Session(self.engine) as session:
            stmt = select(Membership.party).distinct()
            parties = session.execute(stmt).scalars().all()
        return [party for party in parties if party]

    def list_people(self, person_name: str) -> list[dict[str, object]]:
        """Get a list of distinct speaker names and person IDs from the database for a user written person name."""
        with Session(self.engine) as session:
            # Subquery to get the latest membership per person
            latest_membership = (
                select(
                    Membership.person_id,
                    Membership.party,
                    func.row_number()
                    .over(
                        partition_by=Membership.person_id,
                        order_by=desc(Membership.start_date),
                    )
                    .label("rn"),
                )
                .subquery()
            )

            stmt = (
                select(Person.id, Person.display_name, latest_membership.c.party)
                .outerjoin(
                    latest_membership,
                    and_(
                        latest_membership.c.person_id == Person.id,
                        latest_membership.c.rn == 1,
                    ),
                )
                .where(Person.display_name.ilike(f"%{person_name}%"))
                .distinct()
            )
            rows = session.execute(stmt).all()

        return [
            {
                "person_id": row.id,
                "display_name": row.display_name,
                "current_party": row.party,
            }
            for row in rows
        ]
    
    def get_mp_voting_record(
        self,
        person_id: int,
        search_term: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """
        Fetch an MP's voting record on a policy area for a given person_id and search_term,
        using the latest available period_id *within that filtered subset*.
        """
        with Session(self.engine) as session:
            # Latest period for THIS person + THIS search term filter
            latest_period_id = (
                select(func.max(MPPolicySummary.period_id))
                .where(
                    MPPolicySummary.person_id == person_id,
                    MPPolicySummary.name.ilike(f"%{search_term}%"),
                    MPPolicySummary.period_id.isnot(None),
                )
                .scalar_subquery()
            )

            stmt = (
                select(
                    MPPolicySummary.person_id,
                    MPPolicySummary.name,
                    MPPolicySummary.policy_description,
                    MPPolicySummary.context_description,
                    MPPolicySummary.mp_stance_label,
                    MPPolicySummary.mp_policy_alignment_score,
                    MPPolicySummary.num_votes_same,
                    MPPolicySummary.num_strong_votes_same,
                    MPPolicySummary.num_votes_different,
                    MPPolicySummary.num_strong_votes_different,
                    MPPolicySummary.num_votes_absent,
                    MPPolicySummary.num_strong_votes_absent,
                    MPPolicySummary.num_votes_abstain,
                    MPPolicySummary.num_strong_votes_abstain,
                )
                .where(
                    MPPolicySummary.person_id == person_id,
                    MPPolicySummary.name.ilike(f"%{search_term}%"),
                    MPPolicySummary.period_id == latest_period_id,  # 👈 latest in-filter period
                )
                .distinct(MPPolicySummary.person_id, MPPolicySummary.name)
                .order_by(
                    MPPolicySummary.person_id,
                    MPPolicySummary.name,
                    desc(MPPolicySummary.mp_policy_alignment_score),
                    desc(MPPolicySummary.num_votes_same + MPPolicySummary.num_votes_different),
                )
                .limit(limit)
            )

            rows = session.execute(stmt).all()

        results = []
        for row in rows:
            num_same = row.num_votes_same or 0
            num_different = row.num_votes_different or 0
            num_absent = row.num_votes_absent or 0
            num_abstain = row.num_votes_abstain or 0

            actual_votes = num_same + num_different
            total_opportunities = actual_votes + num_absent + num_abstain

            results.append({
                "person_id": row.person_id,
                "policy_name": row.name,
                "policy_description": row.policy_description,
                "context_description": row.context_description,
                "mp_stance_label": row.mp_stance_label,
                "mp_policy_alignment_score": row.mp_policy_alignment_score,
                "num_votes_same": row.num_votes_same,
                "num_strong_votes_same": row.num_strong_votes_same,
                "num_votes_different": row.num_votes_different,
                "num_strong_votes_different": row.num_strong_votes_different,
                "num_votes_absent": row.num_votes_absent,
                "num_strong_votes_absent": row.num_strong_votes_absent,
                "num_votes_abstain": row.num_votes_abstain,
                "num_strong_votes_abstain": row.num_strong_votes_abstain,
                "total_votes": actual_votes,
                "total_opportunities": total_opportunities,
                "percent_aligned": round(num_same / actual_votes * 100, 1) if actual_votes else 0,
                "percent_opposed": round(num_different / actual_votes * 100, 1) if actual_votes else 0,
                "percent_absent": round(num_absent / total_opportunities * 100, 1) if total_opportunities else 0,
                "percent_abstain": round(num_abstain / total_opportunities * 100, 1) if total_opportunities else 0,
            })
        return results

