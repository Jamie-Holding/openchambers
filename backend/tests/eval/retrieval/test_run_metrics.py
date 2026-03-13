import json

from conftest import make_candidate, make_judgement, write_json
from eval.retrieval import run_metrics


class TestComputeAllMetrics:
    def test_writes_expected_report(self, tmp_path, monkeypatch):
        run_id = "20260313_120000"
        run_path = tmp_path / run_id
        run_path.mkdir()

        candidates_payload = {
            "metadata": {"run_id": run_id},
            "queries": [
                {
                    "query_id": "q1",
                    "query_text": "housing",
                    "candidates": [
                        make_candidate(1, vector_rank=1, bm25_rank=2),
                        make_candidate(2, vector_rank=2, bm25_rank=1),
                        make_candidate(3, vector_rank=3, bm25_rank=None),
                    ],
                },
                {
                    "query_id": "q2",
                    "query_text": "nhs",
                    "candidates": [
                        make_candidate(4, vector_rank=1, bm25_rank=None),
                        make_candidate(5, vector_rank=None, bm25_rank=1),
                        make_candidate(6, vector_rank=2, bm25_rank=2),
                    ],
                },
            ],
        }
        judgements_payload = {
            "metadata": {"run_id": run_id},
            "judgements": [
                make_judgement("q1", 1, 5, query_text="housing"),
                make_judgement("q1", 2, 3, query_text="housing"),
                make_judgement("q1", 3, 1, query_text="housing"),
                make_judgement("q2", 4, 2, query_text="nhs"),
                make_judgement("q2", 5, 4, query_text="nhs"),
                make_judgement("q2", 6, 5, query_text="nhs"),
            ],
        }
        write_json(run_path / "candidates.json", candidates_payload)
        write_json(run_path / "judgements.json", judgements_payload)

        monkeypatch.setattr(run_metrics, "run_dir", lambda _: run_path)
        monkeypatch.setattr(run_metrics, "K_VALUES", [1, 3])

        run_metrics.compute_all_metrics(run_id=run_id)

        output = json.loads((run_path / "metrics.json").read_text())
        assert output["metadata"]["run_id"] == run_id
        assert output["metadata"]["k_values"] == [1, 3]
        assert output["metadata"]["relevance_threshold"] == 2
        assert set(output["summary"]) == {"vector", "bm25", "rrf"}
        assert output["summary"]["vector"]["ndcg@1"] > 0
        assert output["summary"]["rrf"]["recall@3"] == 1.0
        assert len(output["per_query"]["vector"]) == 2

    def test_exact_values_for_simple_fixture(self, tmp_path, monkeypatch):
        run_id = "20260313_120001"
        run_path = tmp_path / run_id
        run_path.mkdir()

        write_json(
            run_path / "candidates.json",
            {
                "metadata": {"run_id": run_id},
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "query",
                        "candidates": [
                            make_candidate(1, vector_rank=1, bm25_rank=2),
                            make_candidate(2, vector_rank=2, bm25_rank=1),
                        ],
                    }
                ],
            },
        )
        write_json(
            run_path / "judgements.json",
            {
                "metadata": {"run_id": run_id},
                "judgements": [
                    make_judgement("q1", 1, 5),
                    make_judgement("q1", 2, 3),
                ],
            },
        )

        monkeypatch.setattr(run_metrics, "run_dir", lambda _: run_path)
        monkeypatch.setattr(run_metrics, "K_VALUES", [1, 2])

        run_metrics.compute_all_metrics(run_id=run_id)

        output = json.loads((run_path / "metrics.json").read_text())
        vector = output["summary"]["vector"]
        bm25 = output["summary"]["bm25"]

        assert vector["ndcg@1"] == 1.0
        assert vector["recall@1"] == 0.5
        assert vector["recall@2"] == 1.0
        assert bm25["ndcg@1"] < 1.0
