"""Transformer for formatting utterances for embedding generation."""

import logging

import pandas as pd
from transformers import AutoTokenizer

from backend.src.data.transformers.base import BaseTransformer

logger = logging.getLogger(__name__)


class EmbeddingFormatter(BaseTransformer):
    """Formatter that creates structured text representations for embeddings.

    Attributes:
        tokenizer: Tokenizer for counting tokens.
        max_seq_length: Maximum sequence length for the embedding model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 256,
        include_statement: bool = False,
        include_main_question: bool = True,
        include_context_question: bool = True,
    ) -> None:
        """Initialise the embedding formatter.

        Args:
            model_name: Name of the model to use for tokenization.
            max_seq_length: Maximum sequence length for the embedding model.
            include_statement: Whether to include statement context for answers.
            include_main_question: Whether to include main question context for answers.
            include_context_question: Whether to include supplementary/intervention context.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.include_statement = include_statement
        self.include_main_question = include_main_question
        self.include_context_question = include_context_question

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format utterances for embedding generation.

        Args:
            df: DataFrame containing utterance data.

        Returns:
            DataFrame with formatted 'utterance', 'utterance_embedding_formatted',
            'token_count', and 'is_truncated' columns.
        """
        df["utterance"] = df.apply(self._format_utterance, axis=1)
        df["utterance_embedding_formatted"] = df["utterance"]
        df["token_count"] = df["utterance_embedding_formatted"].apply(self._count_tokens)
        df["is_truncated"] = df["token_count"] > self.max_seq_length

        # Report truncation statistics
        total_utterances = len(df)
        truncated_count = df["is_truncated"].sum()
        truncation_ratio = truncated_count / total_utterances if total_utterances > 0 else 0.0
        logger.info(
            f"Truncation report: {truncated_count}/{total_utterances} utterances "
            f"({truncation_ratio:.2%}) exceed {self.max_seq_length} token limit"
        )

        return df

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens in the text.
        """
        if not text or pd.isna(text):
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _summary_label(self, current_text: str, original_text: str) -> str:
        """Return a summary label if text has been summarized.

        Args:
            current_text: The potentially summarized text.
            original_text: The original text before summarization.

        Returns:
            " (LLM summary)" if summarized, empty string otherwise.
        """
        if not current_text or pd.isna(current_text):
            return ""
        if not original_text or pd.isna(original_text):
            return ""
        return " (LLM summary)" if current_text != original_text else ""

    def _format_utterance(self, row: pd.Series) -> str:
        """Create a structured text representation of an utterance.

        Puts the utterance first with inline speaker info, followed by
        supporting context in order of importance.

        Args:
            row: DataFrame row containing utterance data.

        Returns:
            Formatted string suitable for embedding.
        """
        sections = []

        # Build speaker attribution inline
        speaker_info = ""
        if pd.notna(row.get('speakername')):
            speaker_info = row['speakername']
            if pd.notna(row.get('speakeroffice')):
                speaker_info += f" ({row['speakeroffice']})"

        # The utterance itself (at the top)
        text = row.get('utterance')
        if pd.notna(text):
            if speaker_info:
                sections.append(f"{speaker_info}: {text}")
            else:
                sections.append(text)

        # Supporting context section (in order of importance)
        context_parts = []

        # 1. Question/statement context (most important for understanding answers)
        if row.get('is_answer'):
            if self.include_context_question and pd.notna(row.get('context_question_text')):
                cq_speaker = row.get('context_question_speaker', 'Unknown')
                cq_type = row.get('context_question_type', 'supplementary')
                cq_text = row['context_question_text']
                label = self._summary_label(cq_text, row.get('original_context_question_text'))
                prefix = "Responding to intervention from" if cq_type == "intervention" else "Responding to"
                context_parts.append(f"{prefix} {cq_speaker}{label}: {cq_text}")
            if self.include_main_question and pd.notna(row.get('question_text')):
                q_speaker = row.get('question_speaker', 'Unknown')
                q_text = row['question_text']
                label = self._summary_label(q_text, row.get('original_question_text'))
                context_parts.append(f"Main parliamentary question from {q_speaker}{label}: {q_text}")
            if self.include_statement and pd.notna(row.get('statement_text')):
                s_speaker = row.get('statement_speaker', 'Unknown')
                s_text = row['statement_text']
                label = self._summary_label(s_text, row.get('original_statement_text'))
                context_parts.append(f"Statement from {s_speaker}{label}: {s_text}")

        # 2. Topic and department
        if pd.notna(row.get('minor_heading')):
            context_parts.append(f"Topic: {row['minor_heading']}")
        if pd.notna(row.get('major_heading')):
            context_parts.append(f"Department: {row['major_heading']}")

        # 3. Session info
        if pd.notna(row.get('oral_heading')):
            context_parts.append(f"Session: {row['oral_heading']}")

        # 4. Date (least important)
        if pd.notna(row.get('date')):
            context_parts.append(f"Date: {row['date']}")

        if context_parts:
            sections.append("---\nCONTEXT:\n" + "\n".join(context_parts))

        return "\n\n".join(sections)