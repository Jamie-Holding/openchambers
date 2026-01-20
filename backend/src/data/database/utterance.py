"""Repository for utterance and embedding database operations."""

import pandas as pd

from .base import BaseRepository
from .models import Embedding, Utterance, UtteranceChunk


class UtteranceRepository(BaseRepository):
    """Repository for utterance and embedding database operations."""

    def _create_utterance(self, row: dict) -> Utterance:
        """Create an Utterance object from a row dict.

        Args:
            row: Dictionary containing utterance data.

        Returns:
            Utterance object (not yet added to session).
        """
        return Utterance(
            xml_path=row["xml_path"],
            date=row["date"],
            original_utterance=row["original_utterance"],
            utterance=row["utterance"],
            embedding_text=row.get("utterance_embedding_formatted"),
            speakername=row["speakername"],
            person_id=row["person_id"],
            speakeroffice=row["speakeroffice"],
            oral_heading=row["oral_heading"],
            major_heading=row["major_heading"],
            minor_heading=row["minor_heading"],
            speech_id=row["speech_id"],
            is_statement=row["is_statement"],
            is_question=row["is_question"],
            is_main_question=row["is_main_question"],
            is_supplementary_question=row["is_supplementary_question"],
            is_intervention=row["is_intervention"],
            is_answer=row["is_answer"],
            statement_text=row.get("statement_text"),
            statement_speaker=row.get("statement_speaker"),
            original_statement_text=row.get("original_statement_text"),
            question_text=row.get("question_text"),
            question_speaker=row.get("question_speaker"),
            original_question_text=row.get("original_question_text"),
            context_question_text=row.get("context_question_text"),
            context_question_speaker=row.get("context_question_speaker"),
            context_question_type=row.get("context_question_type"),
            original_context_question_text=row.get("original_context_question_text"),
            url=row["url"],
            colnum=row["colnum"],
        )

    def _create_chunk(self, row: dict, utterance_id: int) -> UtteranceChunk:
        """Create an UtteranceChunk object from a row dict.

        Args:
            row: Dictionary containing chunk data.
            utterance_id: ID of the parent utterance.

        Returns:
            UtteranceChunk object (not yet added to session).
        """
        return UtteranceChunk(
            utterance_id=utterance_id,
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            embedding_text=row["chunk_embedding_text"],
            start_char=row["chunk_start_char"],
            end_char=row["chunk_end_char"],
        )

    def _create_embedding(
        self, chunk_id: int, embedding: list, embedding_model_name: str
    ) -> Embedding:
        """Create an Embedding object.

        Args:
            chunk_id: ID of the parent chunk.
            embedding: The embedding vector.
            embedding_model_name: Name of the model used.

        Returns:
            Embedding object (not yet added to session).
        """
        return Embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            embedding_type=embedding_model_name,
        )

    def insert_batch_with_chunks(
        self, df: pd.DataFrame, embedding_model_name: str
    ) -> None:
        """Insert a batch of utterances with their chunks and embeddings.

        The DataFrame should have one row per chunk (exploded by ChunkingTransformer),
        with columns:
        - All standard utterance columns
        - chunk_index, chunk_text, chunk_embedding_text
        - chunk_start_char, chunk_end_char
        - embedding (the vector for this chunk)

        Args:
            df: DataFrame containing chunk data and embeddings.
            embedding_model_name: Name of the model used to generate embeddings.
        """
        with self.Session() as session:
            # Deduplicate utterances by speech_id (DataFrame has one row per chunk)
            utterance_df = df.drop_duplicates(subset=["speech_id"])

            # Create utterance objects
            utterance_map: dict[str, Utterance] = {}
            utterances = []

            for row in utterance_df.to_dict(orient="records"):
                utt = self._create_utterance(row)
                utterances.append(utt)
                utterance_map[row["speech_id"]] = utt

            session.add_all(utterances)
            session.flush()  # Get IDs for utterances

            # Create chunks
            chunk_embeddings: list[tuple[UtteranceChunk, list]] = []
            for row in df.to_dict(orient="records"):
                utt = utterance_map[row["speech_id"]]
                chunk = self._create_chunk(row, utt.id)
                chunk_embeddings.append((chunk, row.get("embedding")))

            chunks = [c for c, _ in chunk_embeddings]
            session.add_all(chunks)
            session.flush()  # Get IDs for chunks

            # Create embeddings linked to chunks
            embeddings = []
            for chunk, emb in chunk_embeddings:
                embeddings.append(
                    self._create_embedding(chunk.id, emb, embedding_model_name)
                )

            session.add_all(embeddings)
            session.commit()
