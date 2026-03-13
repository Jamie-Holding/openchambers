from conftest import make_judgement
from eval.retrieval.sampling import TARGET_PER_LEVEL, TOTAL_SAMPLE, stratified_sample


class TestStratifiedSample:
    def test_is_deterministic_for_fixed_seed(self):
        judgements = [
            make_judgement("q1", chunk_id, (chunk_id % 5) + 1)
            for chunk_id in range(1, 301)
        ]

        first = stratified_sample(judgements, seed=7)
        second = stratified_sample(judgements, seed=7)

        assert first == second

    def test_skips_parse_errors_and_caps_each_bucket(self):
        judgements = [make_judgement("q1", 999, 0)]
        for score in range(1, 6):
            for index in range(TARGET_PER_LEVEL + 5):
                judgements.append(
                    make_judgement(f"q{score}", score * 1000 + index, score)
                )

        sample = stratified_sample(judgements, seed=3)

        assert len(sample) == TOTAL_SAMPLE
        assert all(item["score"] > 0 for item in sample)
        for score in range(1, 6):
            assert (
                sum(1 for item in sample if item["score"] == score) == TARGET_PER_LEVEL
            )

    def test_fills_shortfall_from_leftover_pool(self):
        judgements = []
        counts = {1: 10, 2: 10, 3: 80, 4: 80, 5: 80}
        for score, count in counts.items():
            for index in range(count):
                judgements.append(
                    make_judgement(f"q{score}", score * 1000 + index, score)
                )

        sample = stratified_sample(judgements, seed=11)

        assert len(sample) == TOTAL_SAMPLE
        assert sum(1 for item in sample if item["score"] == 1) == 10
        assert sum(1 for item in sample if item["score"] == 2) == 10
        assert len({(item["query_id"], item["chunk_id"]) for item in sample}) == len(
            sample
        )

    def test_returns_all_valid_judgements_when_under_sample_limit(self):
        judgements = [
            make_judgement("q1", 1, 1),
            make_judgement("q1", 2, 2),
            make_judgement("q1", 3, 3),
            make_judgement("q1", 4, 0),
        ]

        sample = stratified_sample(judgements, seed=1)

        assert len(sample) == 3
        assert {(s["query_id"], s["chunk_id"]) for s in sample} == {
            ("q1", 1),
            ("q1", 2),
            ("q1", 3),
        }
