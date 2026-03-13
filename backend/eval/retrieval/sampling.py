"""Stratified sampling for human review."""

import random
from collections import defaultdict

TOTAL_SAMPLE = 200
TARGET_PER_LEVEL = TOTAL_SAMPLE // 5  # 40


def stratified_sample(judgements: list[dict], seed: int = 42) -> list[dict]:
    """Create a stratified sample across score levels 1-5.

    Samples up to TARGET_PER_LEVEL from each score bucket. If a bucket has
    fewer items, the remaining slots are filled randomly from the leftover pool.
    """
    rng = random.Random(seed)

    # Group by score level, skipping parse errors
    by_score = defaultdict(list)
    for j in judgements:
        if j["score"] > 0:
            by_score[j["score"]].append(j)

    sampled = []
    leftover = []

    for level in range(1, 6):
        items = by_score[level]
        rng.shuffle(items)
        sampled.extend(items[:TARGET_PER_LEVEL])
        leftover.extend(items[TARGET_PER_LEVEL:])

    # Fill remaining slots from the combined leftover pool
    remaining = TOTAL_SAMPLE - len(sampled)
    if remaining > 0 and leftover:
        rng.shuffle(leftover)
        sampled.extend(leftover[:remaining])

    return sampled
