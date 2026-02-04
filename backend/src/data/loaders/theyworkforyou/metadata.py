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
        self.POLICIES_PATH = os.path.join(self.metadata_dir, "policies.json")
        self.POLICY_CALCS_PATH = os.path.join(self.metadata_dir, "policy_calc_to_load.parquet")
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
        self.MP_POLICY_SUMMARY_REQUIRED_COLUMNS = [
            "person_id",
            "policy_id",
            "period_id",
            "name",
            "policy_description",
            "context_description",
            "distance_score",
            "start_year",
            "end_year",
            "num_votes_same",
            "num_strong_votes_same",
            "num_votes_different",
            "num_strong_votes_different",
            "num_votes_absent",
            "num_strong_votes_absent",
            "num_votes_abstain",
            "num_strong_votes_abstain",
            "division_ids",
            "mp_policy_alignment_score",
            "mp_stance_label",
        ]

    def _load_json(self, path: str) -> dict | list:
        """Load JSON data from a file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON data.
        """
        with open(path, "r") as f:
            return json.load(f)

    def load_memberships(self) -> pd.DataFrame:
        """Load and normalise memberships data from TheyWorkForYou.

        Returns:
            DataFrame with columns: membership_id, person_id, party, post_id,
            start_date, end_date, start_reason, end_reason, historichansard_id.
        """
        data = self._load_json(self.MEMBERSHIPS_PATH)
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
        data = self._load_json(self.PEOPLE_PATH)
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

    def _calculate_alignment_score(
        self,
        num_votes_same: int,
        num_strong_votes_same: int,
        num_votes_different: int,
        num_strong_votes_different: int,
    ) -> float | None:
        """Calculate MP policy alignment score from voting counts.

        Args:
            num_votes_same: Number of votes aligned with policy.
            num_strong_votes_same: Number of strong votes aligned with policy.
            num_votes_different: Number of votes against policy.
            num_strong_votes_different: Number of strong votes against policy.

        Returns:
            Alignment score from 0-100, or None if no votes cast.
        """
        votes_for_policy = num_votes_same + num_strong_votes_same
        votes_against_policy = num_votes_different + num_strong_votes_different
        total_cast_votes = votes_for_policy + votes_against_policy
        if total_cast_votes == 0:
            return None
        return (votes_for_policy / total_cast_votes) * 100

    def _score_to_stance_label(self, score: float | None) -> str:
        """Convert alignment score to stance label.

        Args:
            score: Alignment score from 0-100, or None if no votes.

        Returns:
            Human-readable stance label.
        """
        if pd.isna(score):
            return "No voting evidence"
        if score >= 95:
            return "consistently voted for"
        elif score >= 85:
            return "almost always voted for"
        elif score >= 60:
            return "generally voted for"
        elif score >= 40:
            return "voted a mixture of for and against"
        elif score >= 15:
            return "generally voted against"
        elif score >= 5:
            return "almost always voted against"
        elif score >= -1:
            return "consistently voted against"
        else:
            raise ValueError(f"Unexpected alignment score detected: {score}")

    def load_mp_policy_summaries(self) -> pd.DataFrame:
        """Load MP policy alignment summaries.

        Joins policy_calc_to_load.parquet with policies.json and computes
        derived alignment scores and stance labels.

        Returns:
            DataFrame with columns matching MP_POLICY_SUMMARY_REQUIRED_COLUMNS.
        """
        policies = self._load_json(self.POLICIES_PATH)
        policies_df = pd.DataFrame(policies)[["id", "name", "policy_description", "context_description"]]
        policies_df = policies_df.rename(columns={"id": "policy_id"})

        # Load policy calculations
        calcs_df = pd.read_parquet(self.POLICY_CALCS_PATH)

        # Join on policy_id
        df = calcs_df.merge(policies_df, on="policy_id", how="left")

        # Compute derived fields
        df["mp_policy_alignment_score"] = df.apply(
            lambda row: self._calculate_alignment_score(
                row["num_votes_same"],
                row["num_strong_votes_same"],
                row["num_votes_different"],
                row["num_strong_votes_different"],
            ),
            axis=1,
        )
        df["mp_stance_label"] = df["mp_policy_alignment_score"].apply(self._score_to_stance_label)

        # Convert division_ids from numpy.int64 to native Python int
        df["division_ids"] = df["division_ids"].apply(
            lambda x: [int(i) for i in x] if isinstance(x, (list, np.ndarray)) else x
        )

        # Replace NaN with None for DB insertion
        df.replace({np.nan: None}, inplace=True)

        return df[self.MP_POLICY_SUMMARY_REQUIRED_COLUMNS]
