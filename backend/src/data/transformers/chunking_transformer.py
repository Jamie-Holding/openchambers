"""Transformer for splitting long utterances into overlapping chunks."""

import logging
import re

import pandas as pd
from transformers import AutoTokenizer

from backend.src.data.transformers.base import BaseTransformer

logger = logging.getLogger(__name__)


class ChunkingTransformer(BaseTransformer):
    """Splits long utterances into overlapping chunks for embedding.

    Each chunk includes the full context (speaker info, topic, date) from the
    original EmbeddingFormatter output, plus a portion of the utterance text.

    Short utterances that fit within max_seq_length become a single chunk.
    Long utterances are split at sentence boundaries with overlap.

    Attributes:
        tokenizer: Tokenizer for counting tokens.
        max_seq_length: Maximum sequence length for the embedding model.
        chunk_size: Target token count per chunk (excluding context).
        overlap: Number of tokens to overlap between consecutive chunks.
    """

    CONTEXT_SEPARATOR = "---\nCONTEXT:"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 512,
        chunk_size: int = 400,
        overlap: int = 100,
    ) -> None:
        """Initialise the chunking transformer.

        Args:
            model_name: Name of the model to use for tokenization.
            max_seq_length: Max tokens before splitting into multiple chunks.
            chunk_size: Target tokens per chunk when splitting.
            overlap: Token overlap between consecutive chunks.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.overlap = overlap

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split utterances into chunks, exploding the DataFrame.

        Expects df to have 'utterance' and 'token_count' from EmbeddingFormatter.

        Args:
            df: DataFrame with utterance data.

        Returns:
            Exploded DataFrame where each row is a chunk, with new columns:
            - chunk_index: Position of chunk within utterance (0-indexed)
            - chunk_text: Raw text slice
            - chunk_embedding_text: Full text for embedding (context + chunk)
            - chunk_start_char: Start character offset in original_utterance
            - chunk_end_char: End character offset
        """
        chunk_records = []
        stats = {"single_chunk": 0, "multi_chunk": 0, "total_chunks": 0}

        for _, row in df.iterrows():
            utterance = row.get("utterance", "")
            original_text = row.get("original_utterance", "")
            token_count = row.get("token_count", 0)

            context, content = self._split_context(utterance)

            if token_count <= self.max_seq_length:
                # Single chunk - use full text as-is
                chunk_row = row.to_dict()
                chunk_row["chunk_index"] = 0
                chunk_row["chunk_text"] = original_text
                chunk_row["chunk_embedding_text"] = utterance
                chunk_row["chunk_start_char"] = 0
                chunk_row["chunk_end_char"] = len(original_text)
                chunk_records.append(chunk_row)
                stats["single_chunk"] += 1
                stats["total_chunks"] += 1
            else:
                # Multiple chunks needed - account for context tokens
                context_tokens = self._count_tokens(context)
                available_for_content = self.max_seq_length - context_tokens - 10  # buffer
                effective_chunk_size = min(self.chunk_size, available_for_content)

                chunks = self._create_chunks(content, original_text, effective_chunk_size)
                for chunk_idx, chunk_info in enumerate(chunks):
                    chunk_row = row.to_dict()
                    chunk_row["chunk_index"] = chunk_idx
                    chunk_row["chunk_text"] = chunk_info["text"]
                    chunk_row["chunk_start_char"] = chunk_info["start_char"]
                    chunk_row["chunk_end_char"] = chunk_info["end_char"]
                    chunk_row["chunk_embedding_text"] = self._format_chunk_embedding(
                        chunk_info["text"], context
                    )
                    chunk_records.append(chunk_row)
                stats["multi_chunk"] += 1
                stats["total_chunks"] += len(chunks)

        result_df = pd.DataFrame(chunk_records)

        logger.info(
            f"Chunking: {len(df)} utterances -> {stats['total_chunks']} chunks "
            f"({stats['single_chunk']} single, {stats['multi_chunk']} split into multiple)"
        )

        return result_df

    def _split_context(self, formatted_text: str) -> tuple[str, str]:
        """Split formatted text into context and main content.

        Args:
            formatted_text: Output from EmbeddingFormatter.

        Returns:
            Tuple of (context, content). Context may be empty string.
        """
        if self.CONTEXT_SEPARATOR in formatted_text:
            parts = formatted_text.split(self.CONTEXT_SEPARATOR, 1)
            content = parts[0].strip()
            context = self.CONTEXT_SEPARATOR + parts[1]
            return context, content
        return "", formatted_text

    def _create_chunks(
        self, content: str, original_text: str, effective_chunk_size: int
    ) -> list[dict]:
        """Create overlapping chunks from content text.

        Splits at sentence boundaries when possible.

        Args:
            content: The main utterance content (speaker: text).
            original_text: The original unformatted utterance for char offsets.
            effective_chunk_size: Max tokens per chunk (accounting for context).

        Returns:
            List of dicts with 'text', 'start_char', 'end_char' keys.
        """
        sentences = self._split_sentences(content)

        if not sentences:
            return [{"text": content, "start_char": 0, "end_char": len(original_text)}]

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if (
                current_tokens + sentence_tokens > effective_chunk_size
                and current_sentences
            ):
                chunk_text = " ".join(current_sentences)
                chunks.append({"text": chunk_text})

                # Keep overlap: retain last sentences up to overlap tokens
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({"text": chunk_text})

        # Calculate character offsets based on chunk positions
        self._calculate_char_offsets(chunks, original_text)

        return chunks

    def _calculate_char_offsets(self, chunks: list[dict], original_text: str) -> None:
        """Calculate start/end character offsets for each chunk.

        Args:
            chunks: List of chunk dicts to update in-place.
            original_text: Original utterance text.
        """
        total_len = len(original_text)
        num_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Approximate offsets based on chunk position
            chunk["start_char"] = (i * total_len) // num_chunks if num_chunks > 1 else 0
            chunk["end_char"] = (
                ((i + 1) * total_len) // num_chunks if i < num_chunks - 1 else total_len
            )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses regex that handles common abbreviations.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Split on sentence-ending punctuation followed by space and capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _format_chunk_embedding(self, chunk_text: str, context: str) -> str:
        """Format a chunk with its context for embedding.

        Args:
            chunk_text: The chunk content.
            context: The context section (may be empty).

        Returns:
            Formatted text for embedding.
        """
        if context:
            return f"{chunk_text}\n\n{context}"
        return chunk_text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
