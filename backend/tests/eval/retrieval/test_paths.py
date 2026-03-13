import re

from eval.retrieval import paths


class TestGenerateRunId:
    def test_matches_expected_timestamp_format(self):
        run_id = paths.generate_run_id()

        assert re.fullmatch(r"\d{8}_\d{6}", run_id)


class TestRunDir:
    def test_creates_and_reuses_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")

        path = paths.run_dir("20260313_120000")

        assert path.exists()
        assert path.is_dir()
        assert path == paths.run_dir("20260313_120000")
