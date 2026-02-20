"""Run baseline evaluation queries against the current agent.

Loads test cases from backend/tests/fixtures/queries.json, runs each
through the agent, and saves input/output pairs to a timestamped
results file for before/after comparison.

Usage (from project root):
    python3 -m backend.scripts.run_agent_eval
"""

import asyncio
import json
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from backend.src.chatbot.agent import create_hansard_agent

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
QUERIES_PATH = FIXTURES_DIR / "queries.json"


async def collect_response(graph, thread_id: str, message: str) -> str:
    """Run a single message through the agent and return the full response."""
    config = {"configurable": {"thread_id": thread_id}}
    tokens = []

    async for event in graph.astream_events(
        {"messages": [("user", message)]},
        config,
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                tokens.append(content)

    return "".join(tokens)


async def main():
    print("Loading test cases...")
    with open(QUERIES_PATH) as f:
        test_cases = json.load(f)

    print(f"Found {len(test_cases)} test cases")
    print("Initialising agent (loading embedding model + DB connection)...")

    checkpointer = MemorySaver()
    graph = create_hansard_agent(checkpointer)

    results = []

    for case in test_cases:
        case_id = case["id"]
        thread_id = str(uuid.uuid4())
        print(f"\n{'=' * 60}")
        print(f"Running: {case_id} — {case['description']}")
        print(f"{'=' * 60}")

        turn_results = []
        for i, turn in enumerate(case["turns"]):
            msg = turn["message"]
            print(f"  Turn {i}: {msg}")

            start = time.perf_counter()
            try:
                response = await collect_response(graph, thread_id, msg)
                error = None
            except Exception as e:
                response = None
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"  ERROR: {e}")
            elapsed = round(time.perf_counter() - start, 2)

            if response is not None:
                print(f"  Response ({elapsed}s): {response[:120]}...")

            turn_result = {
                "message": msg,
                "response": response,
                "duration_seconds": elapsed,
            }
            if error:
                turn_result["error"] = error
            turn_results.append(turn_result)

        results.append({
            "id": case_id,
            "description": case["description"],
            "turns": turn_results,
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = FIXTURES_DIR / f"baseline_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_turns = sum(len(c["turns"]) for c in test_cases)
    print(f"\n{'=' * 60}")
    print(f"Done — {len(test_cases)} cases, {total_turns} turns")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
