"""End-to-end retrieval evaluation pipeline.

Runs all stages in sequence: retrieval → judging → metrics → human sample.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_eval
"""

import asyncio
import logging

from eval.retrieval.paths import generate_run_id
from eval.retrieval.run_judge import run_async as run_judge_async
from eval.retrieval.run_metrics import compute_all_metrics
from eval.retrieval.run_retrieval import run as run_retrieval
from eval.retrieval.run_sample import create_sample

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_id = generate_run_id()
    logger.info("Run ID: %s", run_id)

    logger.info("=" * 60)
    logger.info("STAGE 1: Candidate pool generation")
    logger.info("=" * 60)
    run_retrieval(run_id=run_id)

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: LLM judging")
    logger.info("=" * 60)
    asyncio.run(run_judge_async(run_id=run_id))

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Metrics")
    logger.info("=" * 60)
    compute_all_metrics(run_id=run_id)

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Human review sample")
    logger.info("=" * 60)
    create_sample(run_id=run_id)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
