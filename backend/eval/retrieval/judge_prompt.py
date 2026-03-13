"""LLM judge prompt and response parsing for scoring retrieval relevance."""

import json

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the relevance of UK parliamentary speech excerpts to search queries. You must be objective and consistent."""

JUDGE_USER_PROMPT = """Rate how relevant this parliamentary speech excerpt is to the given search query.

## Query
{query}

## Speech Excerpt
{embedding_text}

## Scoring Rubric

1 - Not relevant: The excerpt has no meaningful connection to the query topic.
2 - Marginally relevant: The excerpt mentions the topic only in passing or tangentially.
3 - Somewhat relevant: The excerpt discusses a related topic but does not directly address the query.
4 - Relevant: The excerpt directly discusses the query topic with substantive content.
5 - Highly relevant: The excerpt is centrally about the query topic with detailed, specific discussion.

Respond with ONLY a JSON object: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""


def build_judge_prompt(query: str, embedding_text: str) -> str:
    return JUDGE_USER_PROMPT.format(query=query, embedding_text=embedding_text)


def parse_judge_response(content: str) -> tuple[int, str]:
    """Parse the LLM judge JSON response into (score, reasoning).

    Returns (0, error_message) if parsing fails or score is out of range.
    """
    try:
        result = json.loads(content)
        score = int(result["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"Score {score} out of range")
        return score, result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0, f"PARSE_ERROR: {content}"
