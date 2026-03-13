"""Evaluate LLM judge accuracy against human labels.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_judge_eval --human eval/retrieval/human_labelled/openchambers_retreival_eval_human_labelled.csv
"""

import argparse
import csv
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def evaluate(human_path: str):
    with open(human_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to rows with human scores
    pairs = []
    for r in rows:
        human = r.get("human_score", "").strip()
        judge = r.get("judge_score", "").strip()
        if human and judge:
            pairs.append((int(judge), int(human)))

    if not pairs:
        logger.info("No labelled pairs found.")
        return

    # Overall accuracy
    exact = sum(1 for j, h in pairs if j == h)
    within_1 = sum(1 for j, h in pairs if abs(j - h) <= 1)
    mae = sum(abs(j - h) for j, h in pairs) / len(pairs)

    lines = [
        f"Labelled pairs: {len(pairs)}",
        "",
        f"Exact match:    {exact}/{len(pairs)} ({exact / len(pairs) * 100:.1f}%)",
        f"Within ±1:      {within_1}/{len(pairs)} ({within_1 / len(pairs) * 100:.1f}%)",
        f"Mean abs error: {mae:.2f}",
    ]

    # Accuracy by judge score
    by_judge = defaultdict(list)
    for j, h in pairs:
        by_judge[j].append(h)

    lines.append("")
    lines.append(
        f"{'Judge score':>12} {'Count':>6} {'Exact match':>12} "
        f"{'Within ±1':>10} {'Avg human score':>16} {'Avg error':>10}"
    )
    lines.append("-" * 70)
    for score in range(1, 6):
        items = by_judge[score]
        if not items:
            lines.append(f"{score:>12} {'0':>6}")
            continue
        n = len(items)
        ex = sum(1 for h in items if h == score)
        w1 = sum(1 for h in items if abs(h - score) <= 1)
        mean_h = sum(items) / n
        score_mae = sum(abs(h - score) for h in items) / n
        lines.append(
            f"{score:>12} {n:>6} {ex/n*100:>11.1f}% "
            f"{w1/n*100:>9.1f}% {mean_h:>16.2f} {score_mae:>10.2f}"
        )

    logger.info("\n".join(lines))


def run():
    parser = argparse.ArgumentParser(description="Evaluate LLM judge vs human labels")
    parser.add_argument("--human", required=True, help="Path to human-labelled CSV")
    args = parser.parse_args()
    evaluate(args.human)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
