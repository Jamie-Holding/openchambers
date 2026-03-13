"""IR metrics and ranking functions."""

import math


def dcg_at_k(relevance_scores: list[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k.

    Args:
        relevance_scores: Relevance scores in rank order (position 0 = rank 1).
        k: Cutoff rank.
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))


def ndcg_at_k(relevance_scores: list[int], k: int, all_scores: list[int]) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Args:
        relevance_scores: Relevance scores in rank order (position 0 = rank 1).
        k: Cutoff rank.
        all_scores: All relevance scores in the candidate pool (for ideal ranking).

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 if no relevant documents exist.
    """
    ideal = sorted(all_scores, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(relevance_scores, k) / idcg


def recall_at_k(
    relevance_scores: list[int],
    k: int,
    total_relevant: int,
    threshold: int = 3,
) -> float:
    """Recall at rank k.

    Args:
        relevance_scores: Relevance scores in rank order (position 0 = rank 1).
        k: Cutoff rank.
        total_relevant: Total number of relevant items in the full pool.
        threshold: Minimum score to count as relevant.

    Returns:
        Recall@k in [0, 1]. Returns 0.0 if no relevant documents exist.
    """
    if total_relevant == 0:
        return 0.0
    relevant_in_top_k = sum(1 for s in relevance_scores[:k] if s >= threshold)
    return relevant_in_top_k / total_relevant


def rrf_score(
    vector_rank: int | None,
    bm25_rank: int | None,
    k: int = 60,
) -> float:
    """Compute RRF score for a single item from its per-source ranks.

    Args:
        vector_rank: 1-based rank in vector results (None if absent).
        bm25_rank: 1-based rank in BM25 results (None if absent).
        k: RRF constant (default 60, matching HansardRetrievalTool).
    """
    score = 0.0
    if vector_rank is not None:
        score += 1.0 / (k + vector_rank)
    if bm25_rank is not None:
        score += 1.0 / (k + bm25_rank)
    return score


# --- Ranking functions ---


def build_score_lookup(judgements: list[dict]) -> dict[str, dict[int, int]]:
    """Build {query_id: {chunk_id: score}} from judgements."""
    lookup = {}
    for j in judgements:
        qid = j["query_id"]
        if qid not in lookup:
            lookup[qid] = {}
        lookup[qid][j["chunk_id"]] = j["score"]
    return lookup


def rank_candidates_by_field(candidates: list[dict], rank_field: str) -> list[dict]:
    """Sort candidates by a rank field, excluding those without a rank."""
    with_rank = [c for c in candidates if c[rank_field] is not None]
    return sorted(with_rank, key=lambda c: c[rank_field])


def rank_candidates_rrf(candidates: list[dict]) -> list[dict]:
    """Sort candidates by RRF score (same formula as HansardRetrievalTool)."""
    scored = []
    for c in candidates:
        score = rrf_score(c["vector_rank"], c["bm25_rank"])
        best_rank = min(r for r in [c["vector_rank"], c["bm25_rank"]] if r is not None)
        scored.append((c, score, best_rank))

    # Sort by: RRF score desc, best_rank asc, chunk_id asc (deterministic)
    scored.sort(key=lambda x: (-x[1], x[2], x[0]["chunk_id"]))
    return [item[0] for item in scored]


def compute_metrics_for_ranking(
    ranked_candidates: list[dict],
    scores: dict[int, int],
    all_pool_scores: list[int],
    k_values: list[int],
    relevance_threshold: int,
) -> dict:
    """Compute NDCG@k and Recall@k for a single ranking of one query."""
    # Remap 1-5 judge scores to 0-4 gain so "Not relevant" (1) contributes zero DCG
    relevance = [max(scores.get(c["chunk_id"], 0) - 1, 0) for c in ranked_candidates]
    all_pool_scores = [max(s - 1, 0) for s in all_pool_scores]
    total_relevant = sum(1 for s in all_pool_scores if s >= relevance_threshold)

    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = ndcg_at_k(relevance, k, all_pool_scores)
        results[f"recall@{k}"] = recall_at_k(
            relevance, k, total_relevant, threshold=relevance_threshold
        )
    return results
