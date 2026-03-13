from eval.retrieval.judge_prompt import build_judge_prompt, parse_judge_response


class TestBuildPrompt:
    def test_includes_query_and_embedding_text(self):
        prompt = build_judge_prompt("housing", "speaker: text")

        assert "housing" in prompt
        assert "speaker: text" in prompt


class TestParseResponse:
    def test_accepts_valid_payload(self):
        score, reasoning = parse_judge_response(
            '{"score": 4, "reasoning": "direct match"}'
        )

        assert score == 4
        assert reasoning == "direct match"

    def test_defaults_missing_reasoning_to_empty_string(self):
        score, reasoning = parse_judge_response('{"score": 3}')

        assert score == 3
        assert reasoning == ""

    def test_rejects_invalid_payloads(self):
        invalid_payloads = [
            "not-json",
            '{"reasoning": "missing score"}',
            '{"score": "bad"}',
            '{"score": 0, "reasoning": "too low"}',
            '{"score": 6, "reasoning": "too high"}',
        ]

        for payload in invalid_payloads:
            score, reasoning = parse_judge_response(payload)
            assert score == 0
            assert reasoning == f"PARSE_ERROR: {payload}"
