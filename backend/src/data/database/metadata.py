"""Repository for Hansard metadata database operations."""

from typing import Any

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from backend.src.data.database.base import BaseRepository
from backend.src.data.database.models import Division, Membership, MPPolicySummary, Person, Vote


class MetadataRepository(BaseRepository):
    """Repository for Hansard metadata: people, memberships, votes."""

    def _insert_batched(
        self,
        records: list[dict[str, Any]],
        model_class: type,
        desc: str,
        batch_size: int = 1000,
    ) -> None:
        """Insert records into the database in batches with progress bar.

        Args:
            records: List of dicts to insert.
            model_class: SQLAlchemy model class.
            desc: Description for progress bar.
            batch_size: Number of records per batch.
        """
        total = len(records)
        with tqdm(total=total, desc=desc, unit="rows") as pbar:
            for i in range(0, total, batch_size):
                batch = records[i : i + batch_size]
                objs = [model_class(**row) for row in batch]
                with self.Session() as session:
                    session.add_all(objs)
                    session.commit()
                pbar.update(len(batch))

    def truncate_tables(self) -> None:
        """Truncate all metadata tables (person, membership, division, vote, mp_policy_summary)."""
        with self.Session() as session:
            session.execute(text("TRUNCATE TABLE mp_policy_summary CASCADE"))
            session.execute(text("TRUNCATE TABLE vote CASCADE"))
            session.execute(text("TRUNCATE TABLE division CASCADE"))
            session.execute(text("TRUNCATE TABLE membership CASCADE"))
            session.execute(text("TRUNCATE TABLE person CASCADE"))
            session.commit()

    def insert_people(self, people_df: pd.DataFrame, batch_size: int = 1000) -> None:
        """Insert people records into the database.

        Args:
            people_df: DataFrame with columns matching the Person model.
            batch_size: Number of records to insert per batch.
        """
        records = people_df.to_dict(orient="records")
        self._insert_batched(records, Person, "Inserting people", batch_size)

    def insert_memberships(self, memberships_df: pd.DataFrame, batch_size: int = 1000) -> None:
        """Insert membership records into the database.

        Args:
            memberships_df: DataFrame with columns matching the Membership model.
            batch_size: Number of records to insert per batch.
        """
        records = memberships_df.to_dict(orient="records")
        self._insert_batched(records, Membership, "Inserting memberships", batch_size)

    def insert_divisions(self, divisions_df: pd.DataFrame, batch_size: int = 1000) -> dict[str, int]:
        """Insert division records into the database.

        Args:
            divisions_df: DataFrame with columns matching the Division model.
            batch_size: Number of records to insert per batch.

        Returns:
            Mapping from division_key to generated division id.
        """
        records = divisions_df.to_dict(orient="records")
        total = len(records)
        division_key_to_id = {}

        with tqdm(total=total, desc="Inserting divisions", unit="rows") as pbar:
            for i in range(0, total, batch_size):
                batch = records[i : i + batch_size]
                division_objs = [Division(**row) for row in batch]
                with self.Session() as session:
                    session.add_all(division_objs)
                    session.commit()
                    # Build mapping from division_key to id
                    for div in division_objs:
                        division_key_to_id[div.division_key] = div.id
                pbar.update(len(batch))

        return division_key_to_id

    def insert_votes(
        self,
        votes_df: pd.DataFrame,
        division_key_to_id: dict[str, int],
        batch_size: int = 1000,
    ) -> None:
        """Insert vote records into the database in batches.

        Args:
            votes_df: DataFrame with division_key, person_id, membership_id, vote.
            division_key_to_id: Mapping from division_key to division id.
            batch_size: Number of votes to insert per batch.
        """
        votes_df = votes_df.copy()
        votes_df["division_id"] = votes_df["division_key"].map(division_key_to_id)
        votes_df = votes_df.drop(columns=["division_key"])

        records = votes_df.to_dict(orient="records")
        self._insert_batched(records, Vote, "Inserting votes", batch_size)

    def insert_mp_policy_summaries(
        self,
        summaries_df: pd.DataFrame,
        batch_size: int = 1000,
    ) -> None:
        """Insert MP policy summary records into the database in batches.

        Args:
            summaries_df: DataFrame with columns matching MPPolicySummary model.
            batch_size: Number of records to insert per batch.
        """
        records = summaries_df.to_dict(orient="records")
        self._insert_batched(records, MPPolicySummary, "Inserting policy summaries", batch_size)

    def update_party_at_time(self) -> None:
        """Update 'party_at_time' column in utterances based on memberships.

        Joins utterances with memberships by person_id and date range to
        determine which party the speaker belonged to at the time of speaking.

        Note:
            Memberships and utterances must be inserted first.
        """
        with self.Session() as session:
            session.execute(text("""
                UPDATE utterance u
                SET party_at_time = m.party
                FROM membership m
                WHERE u.person_id = m.person_id
                  AND u.date >= m.start_date
                  AND u.date <= m.end_date;
            """))
            session.commit()
