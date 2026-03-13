"""Stage 1: Generate chunk-level candidate pools for evaluation.

Runs vector and BM25 searches for each query at chunk level (no dedup),
truncates each to TOP_K, unions results by chunk_id, and saves with per-source ranks.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_retrieval
"""

import argparse
import json
import logging

from tqdm import tqdm

from eval.retrieval.paths import generate_run_id, run_dir
from eval.retrieval.queries import DATE_FROM, DATE_TO, EVAL_QUERIES
from src.chatbot.tools import HansardRetrievalTool

logger = logging.getLogger(__name__)

TOP_K = 200


def run(run_id: str | None = None):
    run_id = run_id or generate_run_id()
    output_path = run_dir(run_id) / "candidates.json"

    logger.info("Initialising retrieval tool (top_k=%d)...", TOP_K)
    tool = HansardRetrievalTool(top_k=TOP_K)

    all_queries = []

    for q in tqdm(EVAL_QUERIES, desc="Retrieving", unit="query"):
        query_id = q["id"]
        query_text = q["text"]

        embedding = tool.model.encode([query_text], convert_to_numpy=True)[0]

        vector_chunks = tool._vector_search_chunks(
            embedding, date_from=DATE_FROM, date_to=DATE_TO
        )[:TOP_K]
        bm25_chunks = tool._bm25_search_chunks(
            query_text, date_from=DATE_FROM, date_to=DATE_TO
        )[:TOP_K]

        # Union by chunk_id with per-source ranks
        candidates_by_id = {}

        for rank, row in enumerate(vector_chunks, start=1):
            candidates_by_id[row.id] = {
                "chunk_id": row.id,
                "chunk_text": row.chunk_text,
                "embedding_text": row.embedding_text,
                "utterance_id": row.utterance_id,
                "vector_rank": rank,
                "bm25_rank": None,
                "sources": ["vector"],
            }

        for rank, row in enumerate(bm25_chunks, start=1):
            if row.id in candidates_by_id:
                candidates_by_id[row.id]["bm25_rank"] = rank
                candidates_by_id[row.id]["sources"].append("bm25")
            else:
                candidates_by_id[row.id] = {
                    "chunk_id": row.id,
                    "chunk_text": row.chunk_text,
                    "embedding_text": row.embedding_text,
                    "utterance_id": row.utterance_id,
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "sources": ["bm25"],
                }

        candidates = list(candidates_by_id.values())

        all_queries.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "num_vector": len(vector_chunks),
                "num_bm25": len(bm25_chunks),
                "num_union": len(candidates),
                "candidates": candidates,
            }
        )

    # Save results
    output = {
        "metadata": {
            "run_id": run_id,
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "top_k": TOP_K,
            "num_queries": len(EVAL_QUERIES),
            "level": "chunk",
        },
        "queries": all_queries,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    total_candidates = sum(q["num_union"] for q in all_queries)
    logger.info(
        "Done. %d total candidates across %d queries.",
        total_candidates,
        len(all_queries),
    )
    logger.info("Saved to: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate candidate pools for evaluation"
    )
    parser.add_argument("--run-id", default=None, help="Run ID (default: generate new)")
    args = parser.parse_args()
    run(run_id=args.run_id)
