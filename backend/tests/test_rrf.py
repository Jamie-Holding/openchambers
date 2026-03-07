"""Unit tests for Reciprocal Rank Fusion."""

from types import SimpleNamespace

import pytest

from src.chatbot.tools import HansardRetrievalTool


def _utt(uid: int) -> SimpleNamespace:
    """Create a minimal mock Utterance with just an id."""
    return SimpleNamespace(id=uid)


@pytest.fixture
def rrf():
    """Return a bound _reciprocal_rank_fusion method without initialising
    the full HansardRetrievalTool (which needs a DB and model)."""
    instance = object.__new__(HansardRetrievalTool)
    instance.top_k = 5
    return instance._reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self, rrf):
        results = rrf([_utt(1), _utt(2), _utt(3)])
        assert [u.id for u in results] == [1, 2, 3]

    def test_identical_lists_preserves_order(self, rrf):
        a = [_utt(1), _utt(2), _utt(3)]
        b = [_utt(1), _utt(2), _utt(3)]
        results = rrf(a, b)
        assert [u.id for u in results] == [1, 2, 3]

    def test_disjoint_lists_interleaves(self, rrf):
        a = [_utt(1), _utt(2)]
        b = [_utt(3), _utt(4)]
        results = rrf(a, b)
        # All four items appear; rank-1 from each list tie on score,
        # tie-break by best_rank (both 1), then by ID ascending.
        assert set(u.id for u in results) == {1, 2, 3, 4}
        # IDs 1 and 3 are both rank-1 in their lists so they should
        # come first, with 1 before 3 (ID tie-break).
        assert results[0].id == 1
        assert results[1].id == 3

    def test_overlap_boosted(self, rrf):
        """Items in both lists get higher RRF scores than items in only one."""
        a = [_utt(1), _utt(2), _utt(3)]
        b = [_utt(2), _utt(4), _utt(5)]
        results = rrf(a, b)
        # Utterance 2 appears in both lists so should rank highest.
        assert results[0].id == 2

    def test_top_k_limits_output(self, rrf):
        big_list = [_utt(i) for i in range(20)]
        results = rrf(big_list)
        assert len(results) == 5  # top_k = 5

    def test_top_k_override(self, rrf):
        big_list = [_utt(i) for i in range(20)]
        results = rrf(big_list, top_k=3)
        assert len(results) == 3

    def test_empty_lists(self, rrf):
        assert rrf([], []) == []

    def test_one_empty_list(self, rrf):
        results = rrf([_utt(1), _utt(2)], [])
        assert [u.id for u in results] == [1, 2]

    def test_deterministic_on_ties(self, rrf):
        """Same inputs always produce the same output order."""
        a = [_utt(10), _utt(20)]
        b = [_utt(30), _utt(40)]
        first = [u.id for u in rrf(a, b)]
        for _ in range(10):
            assert [u.id for u in rrf(a, b)] == first

    def test_k_parameter(self, rrf):
        a = [_utt(1), _utt(2)]
        b = [_utt(2), _utt(1)]
        # With default k=60: both items get identical RRF scores
        # (1/(60+1) + 1/(60+2) each). Tie-break by best_rank (both 1),
        # then by ID: 1 before 2.
        results = rrf(a, b)
        assert results[0].id == 1
        assert results[1].id == 2
