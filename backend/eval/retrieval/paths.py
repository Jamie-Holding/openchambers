"""Shared paths for the evaluation harness."""

from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_CACHE_PATH = RESULTS_DIR / ".judge_cache.json"


def generate_run_id() -> str:
    """Generate a unique run ID (UTC timestamp)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_dir(run_id: str) -> Path:
    """Return the output directory for a given run, creating it if needed."""
    path = RESULTS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
