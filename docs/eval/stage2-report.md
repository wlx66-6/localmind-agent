# Stage 2 retrieval evaluation

- generated_at: 2026-10-24T17:16:16+00:00
- collection: `localmind_chunks_v2`
- questions: 40 (30 answerable / 10 unanswerable)
- top_k: 5

## Retrieval modes

| mode | Hit@k | Recall@k | MRR@k | 拒答率 | 误拒率 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 93.3% | 82.8% | 0.776 | 20.0% | 0.0% | 22 | 27 |
| hybrid | 96.7% | 91.1% | 0.806 | 40.0% | 0.0% | 25 | 31 |
| hybrid_rerank | 93.3% | 87.2% | 0.844 | 50.0% | 0.0% | 312 | 334 |

## Threshold sweep (hybrid_rerank)

| threshold | answer recall | refusal rate | balanced acc |
| --- | --- | --- | --- |
| 0.05 | 93.3% | 70.0% | 81.7% |
| 0.10 | 90.0% | 80.0% | 85.0% |
| 0.15 | 90.0% | 80.0% | 85.0% |
| 0.20 | 90.0% | 80.0% | 85.0% |
| 0.25 | 90.0% | 80.0% | 85.0% |
| 0.30 | 86.7% | 80.0% | 83.3% |
| 0.35 | 83.3% | 80.0% | 81.7% |
| 0.40 | 80.0% | 80.0% | 80.0% |
| 0.45 | 80.0% | 80.0% | 80.0% |
| 0.50 | 80.0% | 80.0% | 80.0% |
| 0.55 | 76.7% | 80.0% | 78.3% |
| 0.60 | 76.7% | 80.0% | 78.3% |
| 0.65 | 76.7% | 80.0% | 78.3% |
| 0.70 | 73.3% | 80.0% | 76.7% |
| 0.75 | 73.3% | 80.0% | 76.7% |
| 0.80 | 73.3% | 80.0% | 76.7% |
| 0.85 | 63.3% | 80.0% | 71.7% |
| 0.90 | 60.0% | 80.0% | 70.0% |

单阈值策略的代价：balanced accuracy 最优阈值会误拒有答案的问题，因为 rerank 对“同义改写 / 需要推理”的问题给分很低（见 q12、q29）。因此生产链路采用两层拒答：

- 闸门阈值 `retrieval_gate_rerank_score` = **0.010**：只拦完全无关的检索结果；
- 生成层拒答：通过闸门的片段交给模型，由系统 prompt 按“资料不足”规则判断。

最弱有答案题 top1 = 0.0170。
闸门之下可无风险拒掉的无答案题：
- q31
- q32
- q34
- q35
- q40

## Generation (cloud, hybrid_rerank)

- provider calls: 35/40
- gate refusals: 5，generation refusals: 5
- unanswerable refused: 10/10
- true answers falsely refused: 0/30
- unanswerable falsely answered: 0
- citation precision (gold + acceptable): 89.2%
- citation precision (gold only, strict): 65.3%
- citation recall (gold-based): 93.3%
- first token p50 / p95: 899 / 1168 ms
- total p50 / p95: 1489 / 2385 ms

## Per-question retrieval failures

- vector q15: no gold chunk in top-5
- vector q20: no gold chunk in top-5
- vector q32: unanswerable, passes gate (top_score=0.534)
- vector q33: unanswerable, passes gate (top_score=0.620)
- vector q34: unanswerable, passes gate (top_score=0.542)
- vector q35: unanswerable, passes gate (top_score=0.525)
- vector q36: unanswerable, passes gate (top_score=0.617)
- vector q37: unanswerable, passes gate (top_score=0.661)
- vector q38: unanswerable, passes gate (top_score=0.556)
- vector q39: unanswerable, passes gate (top_score=0.536)
- hybrid q15: no gold chunk in top-5
- hybrid q32: unanswerable, passes gate (top_score=0.534)
- hybrid q33: unanswerable, passes gate (top_score=0.620)
- hybrid q36: unanswerable, passes gate (top_score=0.617)
- hybrid q37: unanswerable, passes gate (top_score=0.661)
- hybrid q38: unanswerable, passes gate (top_score=0.556)
- hybrid q39: unanswerable, passes gate (top_score=0.536)
- hybrid_rerank q12: no gold chunk in top-5
- hybrid_rerank q29: no gold chunk in top-5
- hybrid_rerank q33: unanswerable, passes gate (top_score=0.948)
- hybrid_rerank q36: unanswerable, passes gate (top_score=0.038)
- hybrid_rerank q37: unanswerable, passes gate (top_score=0.964)
- hybrid_rerank q38: unanswerable, passes gate (top_score=0.024)
- hybrid_rerank q39: unanswerable, passes gate (top_score=0.098)
