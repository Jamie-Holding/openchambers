"""Loader for TheyWorkForYou Hansard debate XML files."""

import datetime
import logging
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from ..base import BaseLoader
from ..utils import extract_person_id

logger = logging.getLogger(__name__)


class Debates(BaseLoader):
    """Loader for TheyWorkForYou Hansard XML files."""

    def __init__(self, source_path: str, start_date: str | None = None) -> None:
        """Initialise the debates loader.

        Args:
            source_path: Directory containing the XML debate files.
            start_date: Optional date (YYYY-MM-DD) to filter files from.
        """
        super().__init__(source_path)
        all_files = [f for f in os.listdir(source_path) if f.endswith(".xml")]
        self.files = self._filter_latest_versions(all_files)
        if start_date is not None:
            start_date_parsed = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            self.files = [f for f in self.files if datetime.datetime.strptime(self._extract_date(f), "%Y-%m-%d").date() >= start_date_parsed]
        self.files = sorted(self.files)

    def load_batch(self, batch_number: int, batch_size: int) -> pd.DataFrame:
        """Load a single batch of TheyWorkForYou XML files.

        Args:
            batch_number: Zero-indexed batch number to load.
            batch_size: Number of files to load in this batch.

        Returns:
            DataFrame containing utterances from the batch, or empty DataFrame
            if no files remain.
        """
        start = batch_number * batch_size
        batch_files = self.files[start:start + batch_size]
        if not batch_files:
            return pd.DataFrame()  # no more batches

        # Load and parse each XML file in the batch.
        dfs = []
        for file in batch_files:
            path = os.path.join(self.source_path, file)
            try:
                dfs.append(self._parse_xml(path))
            except Exception as e:
                logger.error(f"Failed to parse {file}: {e}")

        if not dfs:
            return pd.DataFrame()  # Batch was empty.

        return pd.concat(dfs, ignore_index=True)

    def _parse_xml(self, xml_path: str) -> pd.DataFrame:
        """Parse UK parliamentary debate XML into a DataFrame.

        Creates one row per utterance, including speech content and all parent
        metadata such as headings and question context.

        Args:
            xml_path: Path to the XML file.

        Returns:
            DataFrame with columns for speech attributes and parent metadata.
        """
        # Parse the XML.
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Store current context as we traverse.
        context_columns = [
            "oral_heading",
            "major_heading",
            "minor_heading",
            "current_statement",
            "current_statement_id",
            "current_statement_speaker",
            "current_question",
            "current_question_id",
            "current_question_speaker",
            "current_context_question",
            "current_context_question_id",
            "current_context_question_speaker",
            "current_context_question_type"
        ]
        context = {col: None for col in context_columns}

        # Traverse all elements.
        records = []
        for elem in root.iter():
            # Helper to reset all context below headings
            def reset_all_context() -> None:
                context["current_statement"] = None
                context["current_statement_id"] = None
                context["current_statement_speaker"] = None
                context["current_question"] = None
                context["current_question_id"] = None
                context["current_question_speaker"] = None
                context["current_context_question"] = None
                context["current_context_question_id"] = None
                context["current_context_question_speaker"] = None
                context["current_context_question_type"] = None

            # Update context when we encounter headings.
            # Headings reset all context below them.
            if elem.tag == "oral-heading":
                context["oral_heading"] = self._normalize_whitespace(elem.text)
                context["major_heading"] = None
                context["minor_heading"] = None
                reset_all_context()

            elif elem.tag == "major-heading":
                context["major_heading"] = self._normalize_whitespace(elem.text)
                context["minor_heading"] = None
                reset_all_context()

            elif elem.tag == "minor-heading":
                context["minor_heading"] = self._normalize_whitespace(elem.text)
                reset_all_context()

            # Process speech elements (actual utterances).
            elif elem.tag == "speech":
                # Extract all paragraph text first.
                paragraphs = []
                for p in elem.findall("p"):
                    p_text = "".join(p.itertext()).strip()
                    if p_text:
                        paragraphs.append(p_text)

                speech_text = " ".join(paragraphs)
                speech_type = elem.get("type", "")

                # Determine utterance type from speech type attribute
                is_statement = speech_type == "Start Statement"
                is_main_question = speech_type == "Start Question"
                is_supplementary_question = speech_type == "Start SupplementaryQuestion"
                is_intervention = speech_type == "Start Intervention"
                is_question = is_main_question or is_supplementary_question or is_intervention
                # Continuation speech after an intervention is also an answer
                is_continuation_after_intervention = (
                    speech_type == "Continuation Speech"
                    and context["current_context_question_type"] == "intervention"
                )
                is_answer = "Answer" in speech_type or is_continuation_after_intervention

                # Update context based on speech type
                if is_statement:
                    # New statement: update statement and reset all question context
                    context["current_statement"] = speech_text
                    context["current_statement_id"] = elem.get("id")
                    context["current_statement_speaker"] = elem.get("speakername")
                    context["current_question"] = None
                    context["current_question_id"] = None
                    context["current_question_speaker"] = None
                    context["current_context_question"] = None
                    context["current_context_question_id"] = None
                    context["current_context_question_speaker"] = None
                    context["current_context_question_type"] = None
                elif is_main_question:
                    # New main question: update main question and reset context question
                    context["current_question"] = speech_text
                    context["current_question_id"] = elem.get("id")
                    context["current_question_speaker"] = elem.get("speakername")
                    context["current_context_question"] = None
                    context["current_context_question_id"] = None
                    context["current_context_question_speaker"] = None
                    context["current_context_question_type"] = None
                elif is_supplementary_question:
                    # Supplementary question: update context question, keep main question
                    context["current_context_question"] = speech_text
                    context["current_context_question_id"] = elem.get("id")
                    context["current_context_question_speaker"] = elem.get("speakername")
                    context["current_context_question_type"] = "supplementary"
                elif is_intervention:
                    # Intervention: update context question, keep main question
                    context["current_context_question"] = speech_text
                    context["current_context_question_id"] = elem.get("id")
                    context["current_context_question_speaker"] = elem.get("speakername")
                    context["current_context_question_type"] = "intervention"

                record = {
                    # File metadata.
                    "xml_path": xml_path,
                    "date": self._extract_date(xml_path),

                    # Speech attributes.
                    "speech_id": elem.get("id"),
                    "speakername": elem.get("speakername"),
                    "person_id": elem.get("person_id"),
                    "speakeroffice": elem.get("speakeroffice"),
                    "type": speech_type,
                    "colnum": elem.get("colnum"),
                    "time": elem.get("time"),
                    "url": elem.get("url"),
                    "oral_qnum": elem.get("oral-qnum"),
                    "nospeaker": elem.get("nospeaker"),

                    # Statement/Question/Answer tracking.
                    "is_statement": is_statement,
                    "is_question": is_question,
                    "is_main_question": is_main_question,
                    "is_supplementary_question": is_supplementary_question,
                    "is_intervention": is_intervention,
                    "is_answer": is_answer,
                    "statement_text": context["current_statement"],
                    "statement_id": context["current_statement_id"],
                    "statement_speaker": context["current_statement_speaker"],
                    "question_text": context["current_question"],
                    "question_id": context["current_question_id"],
                    "question_speaker": context["current_question_speaker"],
                    "context_question_text": context["current_context_question"],
                    "context_question_id": context["current_context_question_id"],
                    "context_question_speaker": context["current_context_question_speaker"],
                    "context_question_type": context["current_context_question_type"],

                    # Original versions (before any summarization)
                    "original_statement_text": context["current_statement"],
                    "original_question_text": context["current_question"],
                    "original_context_question_text": context["current_context_question"],

                    # Parent context.
                    "oral_heading": context["oral_heading"],
                    "major_heading": context["major_heading"],
                    "minor_heading": context["minor_heading"],

                    # Speech content.
                    "utterance": speech_text,
                    "original_utterance": speech_text,
                    "num_paragraphs": len(paragraphs)
                }

                records.append(record)

        # Convert to dataframe.
        df = pd.DataFrame(records)

        if df.empty:
            return df
        
        if "nospeaker" in df.columns:
            df = df[df["nospeaker"] != "true"].reset_index(drop=True)

        # Extract person ID from person ID string.
        df["person_id"] = df["person_id"].apply(extract_person_id)
        return df

    def _normalize_whitespace(self, text: str | None) -> str | None:
        """Normalize whitespace in text by collapsing newlines and multiple spaces.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text with single spaces, or None if input is None/empty.
        """
        if not text:
            return None
        return " ".join(text.split())

    def _extract_date(self, xml_path: str) -> str:
        """Extract the date from the XML filename.

        Args:
            xml_path: Path to the XML file (e.g. 'debates2025-09-16a.xml').

        Returns:
            Date string in YYYY-MM-DD format.

        Raises:
            ValueError: If no date pattern found in filename.
        """
        date = re.search(r"(\d{4}-\d{2}-\d{2})", xml_path)
        if date:
            date = date.group(1)
        else:
            raise ValueError("Date not found in filename.")
        return date

    def _filter_latest_versions(self, files: list[str]) -> list[str]:
        """Filter to keep only the latest version of each date's debate file.

        Files with letter suffixes (e.g., 'a', 'b', 'c') represent versions.
        For each date, only the file with the highest letter suffix is kept.
        Files without letter suffixes are always kept.

        Args:
            files: List of filenames to filter.

        Returns:
            Filtered list with only the latest version per date.
        """
        date_to_best: dict[str, tuple[str, str]] = {}

        for f in files:
            date = self._extract_date(f)
            match = re.search(r"\d{4}-\d{2}-\d{2}([a-z])?\.xml$", f)
            suffix = match.group(1) if match and match.group(1) else ""

            if date not in date_to_best:
                date_to_best[date] = (suffix, f)
            else:
                current_suffix, _ = date_to_best[date]
                if suffix > current_suffix:
                    date_to_best[date] = (suffix, f)

        return [filename for _, filename in date_to_best.values()]
