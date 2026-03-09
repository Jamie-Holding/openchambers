"""Unit tests for ChunkingTransformer."""

import pandas as pd
import pytest

from src.data.transformers.chunking_transformer import ChunkingTransformer


@pytest.fixture
def chunker():
    """Create a ChunkingTransformer with small limits for testing."""
    return ChunkingTransformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=512,
        chunk_size=400,
        overlap=100,
    )


def _make_df(utterance: str, original: str = "", token_count: int = 0):
    """Build a single-row DataFrame matching ChunkingTransformer's expected input."""
    if not original:
        original = utterance
    return pd.DataFrame(
        [
            {
                "utterance": utterance,
                "original_utterance": original,
                "token_count": token_count,
            }
        ]
    )


class TestChunkSizeBounds:
    """All chunks must fit within max_seq_length after context is added."""

    def test_long_single_sentence_is_split(self, chunker):
        """A single sentence exceeding effective_chunk_size must be split."""
        # ~800 words ≈ ~1000+ tokens, well above 512
        long_sentence = "word " * 800
        content = f"Speaker Name: {long_sentence.strip()}"
        context = "---\nCONTEXT:\nTopic: Test\nDate: 2025-01-01"
        utterance = f"{content}\n\n{context}"
        token_count = chunker._count_tokens(utterance)

        df = _make_df(utterance, long_sentence.strip(), token_count)
        result = chunker.transform(df)

        assert len(result) > 1, "Should produce multiple chunks"
        for _, row in result.iterrows():
            embedding_tokens = chunker._count_tokens(row["chunk_embedding_text"])
            assert embedding_tokens <= chunker.max_seq_length, (
                f"Chunk {row['chunk_index']} has {embedding_tokens} tokens, "
                f"exceeds max_seq_length={chunker.max_seq_length}"
            )

    def test_text_without_sentence_punctuation(self, chunker):
        """Text lacking sentence-ending punctuation should still be chunked."""
        # Long text with commas and conjunctions but no periods
        long_text = ", and ".join(["the minister spoke about housing"] * 150)
        content = f"Speaker: {long_text}"
        context = "---\nCONTEXT:\nTopic: Housing\nDate: 2025-01-01"
        utterance = f"{content}\n\n{context}"
        token_count = chunker._count_tokens(utterance)

        df = _make_df(utterance, long_text, token_count)
        result = chunker.transform(df)

        assert len(result) > 1, "Should produce multiple chunks"
        for _, row in result.iterrows():
            embedding_tokens = chunker._count_tokens(row["chunk_embedding_text"])
            assert embedding_tokens <= chunker.max_seq_length, (
                f"Chunk {row['chunk_index']} has {embedding_tokens} tokens, "
                f"exceeds max_seq_length={chunker.max_seq_length}"
            )

    def test_short_utterance_single_chunk(self, chunker):
        """Utterances within max_seq_length remain a single chunk."""
        utterance = "Speaker: A short statement about policy."
        token_count = chunker._count_tokens(utterance)
        assert token_count <= chunker.max_seq_length

        df = _make_df(utterance, "A short statement about policy.", token_count)
        result = chunker.transform(df)

        assert len(result) == 1
        assert result.iloc[0]["chunk_index"] == 0


class TestSentenceSplitting:
    """Verify sentence splitting handles parliamentary text patterns."""

    def test_splits_on_semicolons(self, chunker):
        sentences = chunker._split_sentences(
            "The bill was read; The minister responded with amendments."
        )
        assert len(sentences) == 2

    def test_splits_on_standard_punctuation(self, chunker):
        sentences = chunker._split_sentences(
            "First sentence. Second sentence! Third sentence?"
        )
        assert len(sentences) == 3

    def test_no_split_on_lowercase_after_period(self, chunker):
        sentences = chunker._split_sentences("e.g. this should not split.")
        assert len(sentences) == 1


class TestLongSentenceSplitting:
    """Verify _split_long_sentence produces bounded pieces."""

    def test_pieces_within_limit(self, chunker):
        long_sentence = "word " * 500
        pieces = chunker._split_long_sentence(long_sentence.strip(), max_tokens=200)

        assert len(pieces) > 1
        for piece in pieces:
            tokens = chunker._count_tokens(piece)
            assert tokens <= 200, f"Piece has {tokens} tokens, exceeds 200"

    def test_preserves_all_words(self, chunker):
        words = [f"word{i}" for i in range(100)]
        sentence = " ".join(words)
        pieces = chunker._split_long_sentence(sentence, max_tokens=50)

        reassembled = " ".join(pieces)
        assert reassembled == sentence


class TestContextPreserved:
    """Every chunk must carry the full context section."""

    def test_all_chunks_contain_context(self, chunker):
        """When a long utterance is split, each chunk_embedding_text includes context."""
        context_section = (
            "---\nCONTEXT:\n"
            "Responding to Jane Doe: What about housing?\n"
            "Topic: Housing Policy\n"
            "Department: Housing, Communities and Local Government\n"
            "Date: 2025-03-01"
        )
        # Generate enough sentences to force multiple chunks
        sentences = [
            f"Sentence number {i} about the housing crisis and policy reform."
            for i in range(80)
        ]
        content = f"John Smith (Minister): {' '.join(sentences)}"
        utterance = f"{content}\n\n{context_section}"
        token_count = chunker._count_tokens(utterance)

        df = _make_df(utterance, " ".join(sentences), token_count)
        result = chunker.transform(df)

        assert len(result) > 1, "Should produce multiple chunks to test context"
        for _, row in result.iterrows():
            embedding_text = row["chunk_embedding_text"]
            assert (
                "---\nCONTEXT:" in embedding_text
            ), f"Chunk {row['chunk_index']} missing CONTEXT separator"
            assert (
                "Topic: Housing Policy" in embedding_text
            ), f"Chunk {row['chunk_index']} missing topic"
            assert (
                "Date: 2025-03-01" in embedding_text
            ), f"Chunk {row['chunk_index']} missing date"
            assert (
                "Responding to Jane Doe" in embedding_text
            ), f"Chunk {row['chunk_index']} missing question context"
