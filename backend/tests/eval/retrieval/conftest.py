import json


def make_candidate(
    chunk_id: int,
    *,
    vector_rank: int | None = None,
    bm25_rank: int | None = None,
    embedding_text: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": f"chunk {chunk_id}",
        "embedding_text": embedding_text or f"embedding {chunk_id}",
        "utterance_id": chunk_id * 10,
        "vector_rank": vector_rank,
        "bm25_rank": bm25_rank,
        "sources": [
            source
            for source, rank in (("vector", vector_rank), ("bm25", bm25_rank))
            if rank is not None
        ],
    }


def make_judgement(
    query_id: str,
    chunk_id: int,
    score: int,
    *,
    query_text: str = "query",
    reasoning: str = "reason",
) -> dict:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "chunk_id": chunk_id,
        "score": score,
        "reasoning": reasoning,
    }


def write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
