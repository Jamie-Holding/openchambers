"""Persistent JSON cache for LLM judge results."""

import json
import logging

from eval.retrieval.paths import JUDGE_CACHE_PATH

logger = logging.getLogger(__name__)


def load_cache() -> dict:
    """Load the judge cache from disk, or return empty dict if missing."""
    if JUDGE_CACHE_PATH.exists():
        with open(JUDGE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    """Write the judge cache to disk."""
    JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_CACHE_PATH, "w") as f:
        json.dump(cache, f)
