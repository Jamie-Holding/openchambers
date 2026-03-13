"""Stage 3: Compute NDCG@k and Recall@k from judgements.

Reconstructs vector-only, BM25-only, and RRF rankings from stored ranks,
then computes metrics using judge scores as relevance labels.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_metrics
"""

import argparse
import json
import logging
from functools import partial

from eval.retrieval.metrics import (
    build_score_lookup,
    compute_metrics_for_ranking,
    rank_candidates_by_field,
    rank_candidates_rrf,
)
from eval.retrieval.paths import generate_run_id, run_dir

logger = logging.getLogger(__name__)

K_VALUES = [5, 10, 20, 50, 100, 200]
RELEVANCE_THRESHOLD = 2  # Original score >= 3 ("Somewhat relevant"), remapped to 0-4


def compute_all_metrics(run_id: str | None = None):
    """Compute and save metrics for all ranking strategies."""
    run_id = run_id or generate_run_id()
    rd = run_dir(run_id)
    candidates_path = rd / "candidates.json"
    judgements_path = rd / "judgements.json"
    output_path = rd / "metrics.json"

    with open(judgements_path) as f:
        judgements_data = json.load(f)
    with open(candidates_path) as f:
        candidates_data = json.load(f)

    score_lookup = build_score_lookup(judgements_data["judgements"])

    # Compute per-query metrics for each ranking strategy
    strategies = {
        "vector": partial(rank_candidates_by_field, rank_field="vector_rank"),
        "bm25": partial(rank_candidates_by_field, rank_field="bm25_rank"),
        "rrf": rank_candidates_rrf,
    }

    all_metrics = {name: [] for name in strategies}

    for query in candidates_data["queries"]:
        qid = query["query_id"]
        candidates = query["candidates"]
        scores = score_lookup.get(qid, {})

        # All scores in the pool for IDCG calculation
        all_pool_scores = [scores.get(c["chunk_id"], 0) for c in candidates]

        for name, rank_fn in strategies.items():
            ranked = rank_fn(candidates)
            metrics = compute_metrics_for_ranking(
                ranked, scores, all_pool_scores, K_VALUES, RELEVANCE_THRESHOLD
            )
            all_metrics[name].append({"query_id": qid, **metrics})

    # Compute macro-averages
    summary = {}
    for name, query_metrics in all_metrics.items():
        avg = {}
        for key in query_metrics[0]:
            if key == "query_id":
                continue
            values = [m[key] for m in query_metrics]
            avg[key] = sum(values) / len(values)
        summary[name] = avg

    # Build and log summary table
    lines = []
    header = f"{'':>10}"
    for k in K_VALUES:
        header += f"  {'NDCG@' + str(k):>10}  {'Recall@' + str(k):>10}"
    lines.append(header)
    lines.append("-" * (10 + len(K_VALUES) * 24))

    for name in strategies:
        row = f"{name:>10}"
        for k in K_VALUES:
            ndcg = summary[name][f"ndcg@{k}"]
            recall = summary[name][f"recall@{k}"]
            row += f"  {ndcg:>10.4f}  {recall:>10.4f}"
        lines.append(row)

    logger.info("\n%s", "\n".join(lines))

    # Save full results
    output = {
        "metadata": {
            "run_id": run_id,
            "k_values": K_VALUES,
            "relevance_threshold": RELEVANCE_THRESHOLD,
        },
        "summary": summary,
        "per_query": all_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved to: %s", output_path)


def run():
    parser = argparse.ArgumentParser(description="Compute retrieval metrics")
    parser.add_argument("--run-id", default=None, help="Run ID (default: generate new)")
    args = parser.parse_args()
    compute_all_metrics(run_id=args.run_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
