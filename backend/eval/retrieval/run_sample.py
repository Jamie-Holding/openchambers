"""Stage 4: Create stratified human review sample from judgements.

Produces a CSV with 200 examples stratified across score levels 1-5,
maximising query diversity within each stratum.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_sample
"""

import argparse
import csv
import json
import logging
from collections import defaultdict

from eval.retrieval.paths import generate_run_id, run_dir
from eval.retrieval.sampling import stratified_sample

logger = logging.getLogger(__name__)


def create_sample(run_id: str | None = None, seed: int = 42):
    """Create and save a stratified human review sample."""
    run_id = run_id or generate_run_id()
    rd = run_dir(run_id)
    candidates_path = rd / "candidates.json"
    judgements_path = rd / "judgements.json"
    output_path = rd / "human_sample.csv"

    with open(judgements_path) as f:
        judgements_data = json.load(f)
    with open(candidates_path) as f:
        candidates_data = json.load(f)

    # Build candidate lookup for context
    candidate_lookup = {}
    for query in candidates_data["queries"]:
        for c in query["candidates"]:
            candidate_lookup[(query["query_id"], c["chunk_id"])] = c

    sampled = stratified_sample(judgements_data["judgements"], seed=seed)

    # Log stratification summary
    score_counts = defaultdict(int)
    for s in sampled:
        score_counts[s["score"]] += 1
    logger.info("Sample size: %d", len(sampled))
    stratification = ", ".join(
        f"Score {level}: {score_counts[level]}" for level in range(1, 6)
    )
    logger.info("Stratification: %s", stratification)

    query_counts = len({s["query_id"] for s in sampled})
    logger.info("Queries represented: %d", query_counts)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "query_text",
                "chunk_id",
                "embedding_text",
                "judge_score",
                "judge_reasoning",
                "human_score",
            ]
        )

        for s in sampled:
            key = (s["query_id"], s["chunk_id"])
            candidate = candidate_lookup.get(key, {})

            embedding_text = candidate.get("embedding_text", "")

            writer.writerow(
                [
                    s["query_id"],
                    s["query_text"],
                    s["chunk_id"],
                    embedding_text,
                    s["score"],
                    s["reasoning"],
                    "",  # human_score — to be filled manually
                ]
            )

    logger.info("Saved to: %s", output_path)


def run():
    parser = argparse.ArgumentParser(
        description="Create stratified human review sample"
    )
    parser.add_argument("--run-id", default=None, help="Run ID (default: generate new)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    create_sample(run_id=args.run_id, seed=args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
