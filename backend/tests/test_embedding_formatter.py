"""Unit tests for EmbeddingFormatter."""

import pandas as pd
import pytest

from src.data.transformers.embedding_text_formatter import EmbeddingFormatter

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def formatter():
    """Create an EmbeddingFormatter with default settings."""
    return EmbeddingFormatter(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _make_row(**kwargs):
    """Build a pd.Series matching EmbeddingFormatter's expected input."""
    defaults = {
        "utterance": "Some text",
        "speakername": None,
        "speakeroffice": None,
        "is_answer": False,
        "context_question_text": None,
        "context_question_speaker": None,
        "context_question_type": None,
        "original_context_question_text": None,
        "question_text": None,
        "question_speaker": None,
        "original_question_text": None,
        "statement_text": None,
        "statement_speaker": None,
        "original_statement_text": None,
        "minor_heading": None,
        "major_heading": None,
        "oral_heading": None,
        "date": None,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# _summary_label
# ---------------------------------------------------------------------------


class TestSummaryLabel:
    def test_same_text_returns_empty(self, formatter):
        assert formatter._summary_label("some text", "some text") == ""

    def test_different_text_returns_label(self, formatter):
        assert (
            formatter._summary_label("short summary", "very long original text")
            == " (LLM summary)"
        )

    def test_none_current_returns_empty(self, formatter):
        assert formatter._summary_label(None, "original") == ""

    def test_none_original_returns_empty(self, formatter):
        assert formatter._summary_label("current", None) == ""

    def test_nan_current_returns_empty(self, formatter):
        assert formatter._summary_label(float("nan"), "original") == ""

    def test_nan_original_returns_empty(self, formatter):
        assert formatter._summary_label("current", float("nan")) == ""


# ---------------------------------------------------------------------------
# _format_utterance
# ---------------------------------------------------------------------------


class TestFormatUtterance:
    def test_speaker_with_name(self, formatter):
        row = _make_row(utterance="We need more housing.", speakername="John Smith")
        result = formatter._format_utterance(row)
        assert result.startswith("John Smith: We need more housing.")

    def test_speaker_with_office(self, formatter):
        row = _make_row(
            utterance="Policy update.",
            speakername="John Smith",
            speakeroffice="Minister for Housing",
        )
        result = formatter._format_utterance(row)
        assert "John Smith (Minister for Housing): Policy update." in result

    def test_no_speaker(self, formatter):
        row = _make_row(utterance="Anonymous statement.")
        result = formatter._format_utterance(row)
        first_line = result.split("\n")[0]
        assert first_line == "Anonymous statement."

    def test_answer_with_supplementary_question(self, formatter):
        row = _make_row(
            utterance="The answer is yes.",
            speakername="Minister",
            is_answer=True,
            context_question_text="What about housing?",
            context_question_speaker="Jane Doe",
            context_question_type="supplementary",
            original_context_question_text="What about housing?",
        )
        result = formatter._format_utterance(row)
        assert "Responding to Jane Doe: What about housing?" in result
        assert "intervention" not in result

    def test_answer_with_intervention(self, formatter):
        row = _make_row(
            utterance="I will address that.",
            speakername="Minister",
            is_answer=True,
            context_question_text="Point of order!",
            context_question_speaker="Mr Speaker",
            context_question_type="intervention",
            original_context_question_text="Point of order!",
        )
        result = formatter._format_utterance(row)
        assert "Responding to intervention from Mr Speaker: Point of order!" in result

    def test_answer_with_main_question(self, formatter):
        row = _make_row(
            utterance="The government's position is clear.",
            speakername="Minister",
            is_answer=True,
            question_text="What is the government's housing policy?",
            question_speaker="Jane Doe",
            original_question_text="What is the government's housing policy?",
        )
        result = formatter._format_utterance(row)
        assert (
            "Main parliamentary question from Jane Doe: "
            "What is the government's housing policy?" in result
        )

    def test_answer_with_statement(self):
        formatter = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_statement=True,
        )
        row = _make_row(
            utterance="I agree with the statement.",
            speakername="MP",
            is_answer=True,
            statement_text="Opening statement on housing.",
            statement_speaker="Secretary of State",
            original_statement_text="Opening statement on housing.",
        )
        result = formatter._format_utterance(row)
        assert (
            "Statement from Secretary of State: Opening statement on housing." in result
        )

    def test_context_parts_ordering(self, formatter):
        row = _make_row(
            utterance="Some speech.",
            speakername="MP",
            minor_heading="Housing Supply",
            major_heading="DCLG",
            oral_heading="Oral Questions",
            date="2025-03-01",
        )
        result = formatter._format_utterance(row)
        assert "Topic: Housing Supply" in result
        assert "Department: DCLG" in result
        assert "Session: Oral Questions" in result
        assert "Date: 2025-03-01" in result
        # Verify ordering
        topic_pos = result.index("Topic:")
        dept_pos = result.index("Department:")
        session_pos = result.index("Session:")
        date_pos = result.index("Date:")
        assert topic_pos < dept_pos < session_pos < date_pos

    def test_summarized_context_question_gets_label(self, formatter):
        row = _make_row(
            utterance="The answer.",
            speakername="Minister",
            is_answer=True,
            context_question_text="Short summary of question.",
            context_question_speaker="MP",
            context_question_type="supplementary",
            original_context_question_text="Very long original question text that was summarized by the LLM.",
        )
        result = formatter._format_utterance(row)
        assert "(LLM summary)" in result
        assert "Responding to MP (LLM summary): Short summary of question." in result

    def test_non_answer_skips_question_context(self, formatter):
        row = _make_row(
            utterance="A regular speech.",
            speakername="MP",
            is_answer=False,
            context_question_text="This should not appear.",
            context_question_speaker="Other MP",
            question_text="Nor should this.",
            question_speaker="Another MP",
            statement_text="Or this.",
            statement_speaker="Yet Another MP",
        )
        result = formatter._format_utterance(row)
        assert "This should not appear" not in result
        assert "Nor should this" not in result
        assert "Or this" not in result

    def test_no_optional_fields_no_context_section(self, formatter):
        row = _make_row(utterance="Just the text.")
        result = formatter._format_utterance(row)
        assert "---\nCONTEXT:" not in result
        assert result == "Just the text."


# ---------------------------------------------------------------------------
# TestIncludeFlags
# ---------------------------------------------------------------------------


class TestIncludeFlags:
    """Constructor flags control which context fields appear in formatted output."""

    def _answer_row(self):
        """Build an answer row with all context fields populated."""
        return _make_row(
            utterance="The answer.",
            speakername="Minister",
            is_answer=True,
            context_question_text="Supplementary question.",
            context_question_speaker="MP A",
            context_question_type="supplementary",
            original_context_question_text="Supplementary question.",
            question_text="Main question?",
            question_speaker="MP B",
            original_question_text="Main question?",
            statement_text="Opening statement.",
            statement_speaker="Secretary",
            original_statement_text="Opening statement.",
        )

    def test_include_main_question_true(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_main_question=True,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Main parliamentary question from MP B: Main question?" in result

    def test_include_main_question_false(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_main_question=False,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Main parliamentary question" not in result
        assert "Main question?" not in result

    def test_include_context_question_true(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_context_question=True,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Responding to MP A: Supplementary question." in result

    def test_include_context_question_false(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_context_question=False,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Responding to" not in result
        assert "Supplementary question" not in result

    def test_include_statement_true(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_statement=True,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Statement from Secretary: Opening statement." in result

    def test_include_statement_false(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_statement=False,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Statement from" not in result
        assert "Opening statement" not in result

    def test_all_flags_true(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_main_question=True,
            include_context_question=True,
            include_statement=True,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Responding to MP A: Supplementary question." in result
        assert "Main parliamentary question from MP B: Main question?" in result
        assert "Statement from Secretary: Opening statement." in result

    def test_all_flags_false(self):
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            include_main_question=False,
            include_context_question=False,
            include_statement=False,
        )
        result = fmt._format_utterance(self._answer_row())
        assert "Responding to" not in result
        assert "Main parliamentary question" not in result
        assert "Statement from" not in result
        # The utterance itself should still be present
        assert "Minister: The answer." in result


# ---------------------------------------------------------------------------
# TestTransform — full pipeline
# ---------------------------------------------------------------------------


class TestTransform:
    def test_adds_expected_columns(self, formatter):
        df = pd.DataFrame(
            [
                {
                    "utterance": "Short speech about policy.",
                    "speakername": "John Smith",
                    "speakeroffice": None,
                    "is_answer": False,
                    "context_question_text": None,
                    "context_question_speaker": None,
                    "context_question_type": None,
                    "original_context_question_text": None,
                    "question_text": None,
                    "question_speaker": None,
                    "original_question_text": None,
                    "statement_text": None,
                    "statement_speaker": None,
                    "original_statement_text": None,
                    "minor_heading": "Housing",
                    "major_heading": None,
                    "oral_heading": None,
                    "date": "2025-03-01",
                }
            ]
        )
        result = formatter.transform(df)
        assert "utterance_embedding_formatted" in result.columns
        assert "token_count" in result.columns
        assert "is_truncated" in result.columns
        assert result.iloc[0]["token_count"] > 0
        assert (
            result.iloc[0]["utterance"]
            == result.iloc[0]["utterance_embedding_formatted"]
        )

    def test_is_truncated_flag(self):
        # Use a very small max_seq_length to force truncation
        fmt = EmbeddingFormatter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_seq_length=5,
        )
        df = pd.DataFrame(
            [
                {
                    "utterance": "This is a longer sentence that should exceed five tokens.",
                    "speakername": "MP",
                    "speakeroffice": None,
                    "is_answer": False,
                    "context_question_text": None,
                    "context_question_speaker": None,
                    "context_question_type": None,
                    "original_context_question_text": None,
                    "question_text": None,
                    "question_speaker": None,
                    "original_question_text": None,
                    "statement_text": None,
                    "statement_speaker": None,
                    "original_statement_text": None,
                    "minor_heading": None,
                    "major_heading": None,
                    "oral_heading": None,
                    "date": None,
                }
            ]
        )
        result = fmt.transform(df)
        assert result.iloc[0]["is_truncated"]
