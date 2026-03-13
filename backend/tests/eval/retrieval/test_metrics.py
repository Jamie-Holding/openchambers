import pytest
from conftest import make_candidate, make_judgement
from eval.retrieval.metrics import (
    build_score_lookup,
    compute_metrics_for_ranking,
    dcg_at_k,
    ndcg_at_k,
    rank_candidates_by_field,
    rank_candidates_rrf,
    recall_at_k,
    rrf_score,
)


class TestDcg:
    def test_matches_expected_value(self):
        result = dcg_at_k([3, 2, 1], 3)

        expected = 3 + 2 / 1.584962500721156 + 1 / 2
        assert result == pytest.approx(expected)


class TestNdcg:
    def test_is_one_for_ideal_ranking(self):
        scores = [4, 2, 1]

        assert ndcg_at_k(scores, 3, scores) == pytest.approx(1.0)

    def test_penalises_worst_ordering(self):
        worst_order = [1, 4]
        best_order = [4, 1]
        result = ndcg_at_k(worst_order, 2, best_order)

        assert result == pytest.approx(0.7609, abs=1e-4)

    def test_is_zero_when_no_relevant_documents_exist(self):
        assert ndcg_at_k([0, 0], 2, [0, 0]) == 0.0


class TestRecall:
    def test_uses_supplied_threshold(self):
        relevance_scores = [2, 1, 0]

        assert recall_at_k(relevance_scores, 2, total_relevant=1, threshold=2) == 1.0
        assert recall_at_k(relevance_scores, 2, total_relevant=1, threshold=3) == 0.0

    def test_accounts_for_relevant_docs_outside_ranking(self):
        # 2 relevant docs in top-k, but 4 relevant in the full pool
        relevance_scores = [3, 3, 1, 1]

        assert recall_at_k(relevance_scores, 4, total_relevant=4, threshold=3) == 0.5

    def test_returns_zero_when_no_relevant_docs_exist(self):
        assert recall_at_k([4, 3], 2, total_relevant=0, threshold=2) == 0.0


class TestRrf:
    def test_handles_missing_rank_inputs(self):
        assert rrf_score(1, None) == pytest.approx(1 / 61)
        assert rrf_score(None, 2) == pytest.approx(1 / 62)
        assert rrf_score(1, 2) == pytest.approx(1 / 61 + 1 / 62)


class TestRanking:
    def test_build_score_lookup_groups_by_query_and_chunk(self):
        judgements = [
            make_judgement("q1", 1, 4),
            make_judgement("q1", 2, 3),
            make_judgement("q2", 1, 5),
        ]

        assert build_score_lookup(judgements) == {
            "q1": {1: 4, 2: 3},
            "q2": {1: 5},
        }

    def test_rank_candidates_by_field_excludes_missing_ranks(self):
        candidates = [
            make_candidate(1, vector_rank=3),
            make_candidate(2, vector_rank=None),
            make_candidate(3, vector_rank=1),
        ]

        ranked = rank_candidates_by_field(candidates, rank_field="vector_rank")

        assert [candidate["chunk_id"] for candidate in ranked] == [3, 1]

    def test_rank_candidates_rrf_prefers_overlap_and_breaks_ties_deterministically(
        self,
    ):
        candidates = [
            make_candidate(10, vector_rank=1, bm25_rank=None),
            make_candidate(20, vector_rank=2, bm25_rank=1),
            make_candidate(30, vector_rank=None, bm25_rank=2),
            make_candidate(40, vector_rank=1, bm25_rank=None),
        ]

        ranked = rank_candidates_rrf(candidates)

        assert ranked[0]["chunk_id"] == 20
        assert [candidate["chunk_id"] for candidate in ranked[1:3]] == [10, 40]
        assert ranked[3]["chunk_id"] == 30


class TestComputeMetrics:
    def test_remaps_scores_and_uses_threshold(self):
        ranked = [
            make_candidate(1, vector_rank=1),
            make_candidate(2, vector_rank=2),
            make_candidate(3, vector_rank=3),
        ]
        scores = {1: 5, 2: 3, 3: 1}
        all_pool_scores = [5, 3, 1]

        metrics = compute_metrics_for_ranking(
            ranked,
            scores,
            all_pool_scores,
            k_values=[1, 3],
            relevance_threshold=2,
        )

        assert metrics["ndcg@1"] == pytest.approx(1.0)
        assert metrics["recall@1"] == pytest.approx(0.5)
        assert metrics["recall@3"] == pytest.approx(1.0)
