# Retrieval Evaluation

This is a retrieval quality evaluation harness using an LLM-judge approach. It measures the quality of vector, BM25 and RRF-fused retrieval over the UK parliamentary debate chunks.

## Results

### NDCG@k

Normalised Discounted Cumulative Gain - measures ranking quality (higher is better).

| Method | @5 | @10 | @20 | @50 | @100 | @200 |
|--------|------|------|------|------|------|------|
| Vector | 0.750 | 0.743 | 0.740 | 0.756 | 0.761 | 0.788 |
| BM25 | 0.744 | 0.752 | 0.753 | 0.775 | 0.775 | 0.784 |
| **RRF** | **0.780** | **0.781** | **0.786** | **0.805** | **0.815** | **0.842** |

### Recall@k

Percent of all relevant documents from the initial retrieved candidate pool (score >= 3) found in the top k results.

| Method | @5 | @10 | @20 | @50 | @100 | @200 |
|--------|------|------|------|------|------|------|
| Vector | 0.043 | 0.077 | 0.137 | 0.287 | 0.458 | 0.709 |
| BM25 | 0.045 | 0.085 | 0.152 | 0.314 | 0.494 | 0.731 |
| **RRF** | **0.045** | **0.085** | **0.157** | **0.327** | **0.528** | **0.804** |

**Interpretation**:

* RRF consistently outperforms both individual methods across all k values.
* BM25 edges vector on recall (especially at large k), while vector and BM25 trade leads on NDCG at small k.
* The fusion benefit grows with k. For example, at @200 RRF captures 80% of relevant documents from the candidate pool vs 73% (BM25) and 71% (vector).

## Methodology

### Summary

The retrieval pipeline is evaluated using 50 topic-based queries over Q1 2025 debates covering major UK parliamentary debate subjects (immigration, housing, NHS, defence, etc.).

For each query:

1. We build an initial candidate pool by retrieving the top 200 chunks from both vector and BM25 search - the aim here is to cast a wide net ensuring we find relevant utterances which our retrieval system isn't scoring highly.
2. We then score the deduplicated candidate pool 1-5 using an LLM judge.
3. The LLM judge results are used to calculate nDCG@k and recall@k. The judge sees the chunk's `embedding_text` which includes speaker, party, date, and debate context.
4. The LLM judge's accuracy was measured by taking a sample and writing human labels for comparison - see LLM judge section below for more details.

### LLM judge relevance scale

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Not relevant | No meaningful connection to the query topic |
| 2 | Marginally relevant | Mentions the topic only in passing or tangentially |
| 3 | Somewhat relevant | Discusses a related topic but does not directly address the query |
| 4 | Relevant | Directly discusses the query topic with substantive content |
| 5 | Highly relevant | Centrally about the query topic with detailed, specific discussion |

### Metrics

- **NDCG@k**: Normalised Discounted Cumulative Gain with judge scores remapped from 1-5 to 0-4 gain (so "Not relevant" contributes zero)
- **Recall@k**: Fraction of relevant documents (original score >= 3) in the top k results
- **k values**: 5, 10, 20, 50, 100, 200
- Rankings are computed for three strategies: vector-only, BM25-only and RRF.

## Judge calibration

### Summary

To ensure the LLM judge is accurate, 200 random examples (stratified by judge score) were manually human labelled and compared. The results showed a 70.5% exact match between the LLM judge and the human judge and, encouragingly, 92.5% within ±1.

This suggests that the LLM judge is likely good enough to be trusted on the wider eval and as a continuous evaluation measure for future retreival changes.

### Methodology

200 (query, chunk) pairs were human-labelled to measure judge accuracy. Samples were stratified across score levels 1-5 (40 per level) with query diversity maximised within each stratum.

### Overall results

| Metric | Value |
|--------|-------|
| Exact match | 70.5% (141/200) |
| Within ±1 | 92.5% (185/200) |
| Mean absolute error | 0.41 |

### Per score level

| Judge score | n | Exact match | Within ±1 | Avg human score | MAE |
|-------------|-----|-------------|-----------|-----------------|------|
| 1 | 40 | 85.0% | 92.5% | 1.30 | 0.30 |
| 2 | 40 | 57.5% | 97.5% | 1.88 | 0.47 |
| 3 | 40 | 57.5% | 85.0% | 3.23 | 0.57 |
| 4 | 40 | 62.5% | 92.5% | 3.92 | 0.47 |
| 5 | 40 | 90.0% | 95.0% | 4.78 | 0.23 |

**Interpretation**: The judge is most accurate at the extremes (scores 1 and 5) and least precise in the middle range (scores 2-3), which is expected - borderline relevance is subjective. The high within-±1 rate (92.5%) and low MAE (0.41) indicate the judge rarely makes large errors, making it reliable for aggregate metrics like nDCG and recall.

## How to re-run

All commands run from the project root with Docker Compose. Each run produces a timestamped directory under `results/<run_id>/`.

```bash
# End-to-end (all stages — generates a shared run_id automatically)
docker compose exec backend python3 -m eval.retrieval.run_eval

# Or run stages individually with a shared run_id:
RUN_ID=20260313_120000
docker compose exec backend python3 -m eval.retrieval.run_retrieval --run-id $RUN_ID
docker compose exec backend python3 -m eval.retrieval.run_judge     --run-id $RUN_ID
docker compose exec backend python3 -m eval.retrieval.run_metrics    --run-id $RUN_ID
docker compose exec backend python3 -m eval.retrieval.run_sample     --run-id $RUN_ID

# Evaluate judge accuracy against human labels
docker compose exec backend python3 -m eval.retrieval.run_judge_eval \
  --human eval/retrieval/human_labelled/openchambers_retreival_eval_human_labelled.csv
```

## File inventory

| File | Description |
|------|-------------|
| `paths.py` | Run ID generation, run directory management, shared paths |
| `caching.py` | Persistent cross-run JSON cache for LLM judge results |
| `queries.py` | 50 evaluation queries with frozen date range (Q1 2025) |
| `judge_prompt.py` | System and user prompts for the LLM relevance judge |
| `metrics.py` | NDCG@k, Recall@k, and RRF score implementations |
| `run_eval.py` | End-to-end pipeline runner (all stages in sequence) |
| `run_retrieval.py` | Stage 1: generates candidate pools via vector + BM25 search |
| `run_judge.py` | Stage 2: async LLM judging with caching and retry logic |
| `run_metrics.py` | Stage 3: computes metrics for vector, BM25, and RRF rankings |
| `run_sample.py` | Stage 4: creates stratified human review sample (200 pairs) |
| `run_judge_eval.py` | Evaluates judge accuracy against human labels |
| `human_labelled/` | Human-labelled CSV for judge calibration |
| `results/<run_id>/` | Per-run output: `candidates.json`, `judgements.json`, `metrics.json`, `human_sample.csv` |
| `results/.judge_cache.json` | Cross-run LLM judge cache (keyed by model + prompt + inputs) |
