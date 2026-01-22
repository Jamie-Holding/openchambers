"""Loader for TheyWorkForYou metadata (people, memberships, votes)."""

import json
import os

import numpy as np
import pandas as pd

from ..utils import extract_person_id


class Metadata:
    """Loader for TheyWorkForYou metadata (People, Memberships, Votes)."""

    def __init__(self, metadata_dir: str) -> None:
        """Initialise the metadata loader.

        Args:
            metadata_dir: Directory containing TheyWorkForYou metadata files.
        """
        self.metadata_dir = metadata_dir
        self.PEOPLE_PATH = os.path.join(self.metadata_dir, "people.json")
        self.MEMBERSHIPS_PATH = os.path.join(self.metadata_dir, "people.json") # Note: same file as people.
        self.VOTES_PATH = os.path.join(self.metadata_dir, "votes.parquet")
        self.DIVISIONS_PATH = os.path.join(self.metadata_dir, "divisions.parquet")
        self.PEOPLE_REQUIRED_COLUMNS = [
            "id",
            "given_name",
            "family_name",
            "display_name"
        ]
        self.MEMBERSHIP_REQUIRED_COLUMNS = [
            "membership_id",
            "person_id",
            "party",
            "post_id",
            "start_date",
            "end_date",
            "start_reason",
            "end_reason",
            "historichansard_id",
        ]
        self.DIVISIONS_REQUIRED_COLUMNS = [
            "division_key",
            "vote_date",
            "description",
        ]
        self.VOTES_REQUIRED_COLUMNS = [
            "division_key",
            "person_id",
            "membership_id",
            "vote",
        ]

    def load_memberships(self) -> pd.DataFrame:
        """Load and normalise memberships data from TheyWorkForYou.

        Returns:
            DataFrame with columns: membership_id, person_id, party, post_id,
            start_date, end_date, start_reason, end_reason, historichansard_id.
        """
        with open(self.MEMBERSHIPS_PATH, "r") as f:
            data = json.load(f)

        memberships = data["memberships"]
        df = pd.DataFrame(memberships)

        # Rename core identifiers.
        df = df.rename(columns={
            "id": "membership_id",
            "on_behalf_of_id": "party",
        })

        # Parse dates.
        for col in ("start_date", "end_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Extract historic Hansard ID.
        df["historichansard_id"] = df.get("identifiers").apply(
            self._extract_historic_id
        )

        # Drop NULL person_id rows.
        df = df[~df["person_id"].isnull()]
        df["person_id"] = df["person_id"].apply(extract_person_id)

        # Helps with NaT format insertion to DB downstream.
        df.replace({np.nan: None}, inplace=True)

        return df[self.MEMBERSHIP_REQUIRED_COLUMNS]

    def _extract_historic_id(self, identifiers: list[dict] | None) -> str | None:
        """Extract the historic Hansard ID from identifier records.

        Args:
            identifiers: List of identifier dicts with 'scheme' and 'identifier' keys.

        Returns:
            The historic Hansard ID if found, otherwise None.
        """
        if not isinstance(identifiers, list):
            return None
        for item in identifiers:
            if item.get("scheme") == "historichansard_id":
                return item.get("identifier")
        return None

    def load_divisions(self) -> pd.DataFrame:
        """Load division (vote event) metadata.

        Returns:
            DataFrame with columns: division_key, vote_date, description.
        """
        divisions = pd.read_parquet(self.DIVISIONS_PATH)

        column_name_mapping = {
            "key": "division_key",
            "date": "vote_date",
            "division_name": "description",
        }
        divisions = divisions.rename(columns=column_name_mapping)

        return divisions[self.DIVISIONS_REQUIRED_COLUMNS]

    def load_votes(self) -> pd.DataFrame:
        """Load individual MP votes.

        Returns:
            DataFrame with columns: division_key, person_id, membership_id, vote.
        """
        votes = pd.read_parquet(self.VOTES_PATH)

        # Drop original vote column, use effective_vote instead.
        votes.drop(columns=["vote"], inplace=True)
        votes = votes.rename(columns={"effective_vote": "vote"})

        return votes[self.VOTES_REQUIRED_COLUMNS]

    def _reconcile_person_name(
        self, other_names: list[dict] | None
    ) -> tuple[str | None, str | None, str | None]:
        """Reconcile a person's name from name records.

        Prefers records with note='Main' and the latest start_date.

        Args:
            other_names: List of name record dicts.

        Returns:
            Tuple of (given_name, family_name, display_name).
        """
        if not other_names or not isinstance(other_names, list):
            return None, None, None

        # Pick the most relevant record.
        # Prefer 'Main' note, latest start_date if multiple.
        main_records = [r for r in other_names if r.get("note") == "Main"]
        if main_records:
            main_records.sort(key=lambda r: r.get("start_date", ""), reverse=True)
            record = main_records[0]
        else:
            # Fallback: take the first one.
            record = other_names[0]

        # Extract given_name.
        given_name = (
                record.get("given_name")
                or record.get("additional_name")
                or (record.get("name").split()[0] if record.get("name") else None)
        )

        # Extract family_name.
        family_name = (
                record.get("family_name")
                or record.get("surname")
                or (record.get("name").split()[-1] if record.get("name") else None)
        )

        # Build display_name.
        honorific = record.get("honorific_prefix")
        if honorific:
            if family_name:
                display_name = f"{honorific} {family_name}"
            elif given_name:
                display_name = f"{honorific} {given_name}"
            else:
                display_name = honorific
        else:
            parts = [part for part in [given_name, family_name] if part]
            display_name = " ".join(parts) if parts else None

        return given_name, family_name, display_name

    def load_people(self) -> pd.DataFrame:
        """Load and normalise people data from TheyWorkForYou.

        Returns:
            DataFrame with columns: id, given_name, family_name, display_name.
        """
        with open(self.PEOPLE_PATH, "r") as f:
            data = json.load(f)

        people = data["persons"]
        people = pd.DataFrame(people)
        people = people[~people["other_names"].isnull()] # Remove records without a name.

        # Parse the most appropriate up-to-date names from the json structure.
        people[["given_name", "family_name", "display_name"]] = people.apply(
            lambda row: pd.Series(self._reconcile_person_name(row["other_names"])),
            axis=1
        )

        # Extract person ID.
        people["id"] = people["id"].apply(extract_person_id)

        return people[self.PEOPLE_REQUIRED_COLUMNS]



