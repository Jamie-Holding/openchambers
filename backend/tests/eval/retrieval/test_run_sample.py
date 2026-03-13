import csv

from conftest import make_candidate, make_judgement, write_json
from eval.retrieval import run_sample


class TestCreateSample:
    def test_writes_csv_with_expected_columns_and_values(self, tmp_path, monkeypatch):
        run_id = "20260313_120000"
        run_path = tmp_path / run_id
        run_path.mkdir()

        write_json(
            run_path / "candidates.json",
            {
                "metadata": {"run_id": run_id},
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "housing",
                        "candidates": [
                            make_candidate(1, embedding_text="candidate one"),
                            make_candidate(2, embedding_text="candidate two"),
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
                    make_judgement(
                        "q1", 1, 5, query_text="housing", reasoning="strong"
                    ),
                    make_judgement("q1", 2, 3, query_text="housing", reasoning="ok"),
                ],
            },
        )

        monkeypatch.setattr(run_sample, "run_dir", lambda _: run_path)

        run_sample.create_sample(run_id=run_id, seed=42)

        with open(run_path / "human_sample.csv", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0].keys() == {
            "query_id",
            "query_text",
            "chunk_id",
            "embedding_text",
            "judge_score",
            "judge_reasoning",
            "human_score",
        }
        assert {row["embedding_text"] for row in rows} == {
            "candidate one",
            "candidate two",
        }
        assert {row["judge_score"] for row in rows} == {"5", "3"}
        assert {row["judge_reasoning"] for row in rows} == {"strong", "ok"}
