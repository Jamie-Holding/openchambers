"""Stage 2: LLM relevance judging of candidate pools.

Scores each (query, utterance) pair 1-5 using an LLM judge.

Usage:
    docker compose exec backend python3 -m eval.retrieval.run_judge
"""

import argparse
import asyncio
import hashlib
import json
import logging

from openai import APIError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

from config.settings import FAST_LLM_MODEL_NAME
from eval.retrieval.caching import load_cache, save_cache
from eval.retrieval.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_response,
)
from eval.retrieval.paths import generate_run_id, run_dir

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 20
MAX_RETRIES = 20
REQUEST_DELAY = 0.1  # seconds between requests to stay under TPM


def _cache_key(model: str, query_text: str, embedding_text: str) -> str:
    """Hash the prompt inputs to create a stable cache key."""
    content = f"{model}\n{JUDGE_SYSTEM_PROMPT}\n{query_text}\n{embedding_text}"
    return hashlib.sha256(content.encode()).hexdigest()


async def judge_candidate(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    query_text: str,
    candidate: dict,
    cache: dict,
) -> dict:
    """Score a single (query, chunk) pair. Uses cache to avoid duplicate API calls."""
    key = _cache_key(model, query_text, candidate["embedding_text"])

    if key in cache:
        return {
            "chunk_id": candidate["chunk_id"],
            "score": cache[key]["score"],
            "reasoning": cache[key]["reasoning"],
        }

    prompt = build_judge_prompt(
        query=query_text,
        embedding_text=candidate["embedding_text"],
    )

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                await asyncio.sleep(REQUEST_DELAY)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=100,
                    temperature=0.0,
                )
            break
        except (RateLimitError, APIError) as e:
            last_error = e
            wait = 2**attempt
            logger.warning(
                "%s (attempt %d): %s. Retrying in %ds...",
                type(e).__name__,
                attempt + 1,
                e,
                wait,
            )
            await asyncio.sleep(wait)
    else:
        return {
            "chunk_id": candidate["chunk_id"],
            "score": 0,
            "reasoning": f"{type(last_error).__name__}: {last_error}",
        }

    content = response.choices[0].message.content
    score, reasoning = parse_judge_response(content)
    if score == 0:
        logger.warning("Failed to parse judge response: %s", content)

    # Cache successful results (not parse errors)
    if score > 0:
        cache[key] = {"score": score, "reasoning": reasoning}

    return {
        "chunk_id": candidate["chunk_id"],
        "score": score,
        "reasoning": reasoning,
    }


async def run_async(run_id: str | None = None):
    """Run LLM judge on all candidates."""
    run_id = run_id or generate_run_id()
    rd = run_dir(run_id)
    candidates_path = rd / "candidates.json"
    output_path = rd / "judgements.json"

    with open(candidates_path) as f:
        data = json.load(f)

    model = FAST_LLM_MODEL_NAME
    queries = data["queries"]
    total_candidates = sum(len(q["candidates"]) for q in queries)
    logger.info("Judge model: %s", model)
    logger.info("Queries: %d, candidates: %d", len(queries), total_candidates)

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    cache = load_cache()
    all_judgements = []
    progress = tqdm(total=total_candidates, desc="Judging", unit="pair")

    for query in queries:
        query_id = query["query_id"]
        tasks = [
            judge_candidate(client, semaphore, model, query["query_text"], c, cache)
            for c in query["candidates"]
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_judgements.append(
                {
                    "query_id": query_id,
                    "query_text": query["query_text"],
                    **result,
                }
            )
            progress.update(1)

        # Save cache after each query batch for resilience against interruption
        save_cache(cache)

    progress.close()
    logger.info("Cache: %d entries total", len(cache))

    # Save results
    output = {
        "metadata": {
            "run_id": run_id,
            "judge_model": model,
            "num_judged": len(all_judgements),
        },
        "judgements": all_judgements,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Log score distribution
    scores = [j["score"] for j in all_judgements if j["score"] > 0]
    logger.info("Done. %d judgements.", len(all_judgements))
    if scores:
        dist = ", ".join(
            f"{s}: {scores.count(s)} ({scores.count(s) / len(scores) * 100:.1f}%)"
            for s in range(1, 6)
        )
        logger.info("Score distribution: %s", dist)

    parse_errors = sum(1 for j in all_judgements if j["score"] == 0)
    if parse_errors:
        logger.warning("Parse errors: %d", parse_errors)

    logger.info("Saved to: %s", output_path)


def run():
    parser = argparse.ArgumentParser(description="Run LLM judge on candidate pool")
    parser.add_argument("--run-id", default=None, help="Run ID (default: generate new)")
    args = parser.parse_args()

    asyncio.run(run_async(run_id=args.run_id))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
