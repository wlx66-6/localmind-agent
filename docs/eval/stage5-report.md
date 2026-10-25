# 阶段 5 评测：同一批问题，两条轨道

- 生成时间：2026-10-25T07:49:11+00:00
- 评测集：`eval/stage2/questions.jsonl`（40 题：30 有答案 / 10 无答案）
- 集合：`localmind_chunks_v2`，top_k = 5
- 方法：每题检索一次（hybrid + rerank），同一组片段分别交给两条轨道生成，因此差值只来自生成层；检索指标见 `docs/eval/stage2-report.md`。
- 云端只接收命中的片段，不接收任何原始文件。

## 双轨对比

| 轨道 | 生成回答 | 首 token p50/p95 (ms) | 总耗时 p50/p95 (ms) | 闸门拒答 | 生成拒答 | 无答案题拒答 | 有答案题误拒 | 无答案题误答 | 引用精确率 | 引用召回 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 35 | 874 / 1174 | 1722 / 2602 | 5 | 5 | 10/10 | 0/30 | 0 | 85.6% | 93.3% |
| local | 35 | 508 / 574 | 1886 / 6045 | 5 | 5 | 10/10 | 0/30 | 0 | 87.9% | 93.3% |

本地轨道的首题包含模型加载（3～4 s），p95 会被它抬高；批处理里的后续题才代表稳态。

两条轨道共用同一次检索（CUDA 重排），而生产环境的本地轨道把重排放到 CPU 以让出显存。在 8 GB 显存的机器上，4B 生成模型与 CUDA 重排共存会互相拖慢，因此本表的本地耗时是**悲观值**；生产路径见 `docs/spikes/stage3-acceptance.md`。

## 逐题对照

| 题号 | 类别 | 有答案 | cloud 结果 | local 结果 |
| --- | --- | --- | --- | --- |
| q01 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q02 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q03 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q04 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q05 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q06 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q07 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q08 | exact | 是 | 回答，引用 1 条 | 回答，引用 2 条 |
| q09 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q10 | exact | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q11 | paraphrase | 是 | 回答，引用 3 条 | 回答，引用 2 条 |
| q12 | paraphrase | 是 | 回答，引用 3 条 | 回答，引用 2 条 |
| q13 | paraphrase | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q14 | paraphrase | 是 | 回答，引用 2 条 | 回答，引用 3 条 |
| q15 | paraphrase | 是 | 回答，引用 4 条 | 回答，引用 3 条 |
| q16 | paraphrase | 是 | 回答，引用 3 条 | 回答，引用 2 条 |
| q17 | paraphrase | 是 | 回答，引用 1 条 | 回答，引用 1 条 |
| q18 | paraphrase | 是 | 回答，引用 3 条 | 回答，引用 2 条 |
| q19 | paraphrase | 是 | 回答，引用 4 条 | 回答，引用 2 条 |
| q20 | paraphrase | 是 | 回答，引用 2 条 | 回答，引用 2 条 |
| q21 | cross_doc | 是 | 回答，引用 5 条 | 回答，引用 5 条 |
| q22 | cross_doc | 是 | 回答，引用 5 条 | 回答，引用 2 条 |
| q23 | cross_doc | 是 | 回答，引用 5 条 | 回答，引用 5 条 |
| q24 | cross_doc | 是 | 回答，引用 5 条 | 回答，引用 5 条 |
| q25 | cross_doc | 是 | 回答，引用 4 条 | 回答，引用 5 条 |
| q26 | cross_doc | 是 | 回答，引用 3 条 | 回答，引用 4 条 |
| q27 | cross_doc | 是 | 回答，引用 5 条 | 回答，引用 5 条 |
| q28 | cross_doc | 是 | 回答，引用 2 条 | 回答，引用 2 条 |
| q29 | cross_doc | 是 | 回答，引用 4 条 | 回答，引用 3 条 |
| q30 | cross_doc | 是 | 回答，引用 2 条 | 回答，引用 2 条 |
| q31 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q32 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q33 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q34 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q35 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q36 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q37 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q38 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q39 | unanswerable | 否 | 正确拒答 | 正确拒答 |
| q40 | unanswerable | 否 | 正确拒答 | 正确拒答 |

## 记忆场景

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| 跨会话召回事实 | 通过 | `{"stored": {"id": 19, "kind": "fact", "key": "所在城市", "content": "用户住在杭州。", "occurred_at": null, "created_at": "2026-10-25T07:51:09.971427+00:00", "status": "active", "superseded_by": null, "source_conversation_id": 72, "embedding_ok": true}, "recalled": ["fact/所在城市 0.623"], "answer": "你住在杭州（这是你在 202` |
| 同 key 覆盖旧值 | 通过 | `{"active": ["目前主要使用 Go 语言编写后端。"], "history": ["目前主要使用 Go 语言编写后端。 (active)", "用户平时主要用 Python 写后端。 (superseded)", "目前主要使用 Go 语言编写后端。 (superseded)", "用户平时主要用 Python 写后端。 (superseded)", "现在主要用 Go 写后端。 (superseded)", "用户平时主要用 Python 写后端。 (superseded)", "现在主要用 Go 写后端。 (superseded)", "用户平时主要用 Python 写后端。 (` |
| 无关问题不召回记忆 | 通过 | `{"memories": [], "refused": true, "answer": "资料中未找到足够依据来回答这个问题。你可以补充相关文档，或换个问法再试。"}` |
| 事件时间与记录时间分开 | 通过 | `{"stored": {"id": 18, "kind": "fact", "key": "学习Rust时间", "content": "用户于2025年8月开始学习Rust。", "occurred_at": "2025-08-01", "created_at": "2026-10-25T07:41:54.228585+00:00", "status": "active", "superseded_by": null, "source_conversation_id": 70, "embedding_ok": true}}` |
| 删除后不再召回 | 通过 | `{"deleted": "用户住在杭州。", "recalled": ["性别"], "answer": "资料中未找到足够依据，无法判断你住在哪个城市。缺少的信息是：你的居住地或所在城市。\n\n另外，我过去的对话记忆里只记录了你提到自己是男生，没有关于居住城市的信息。"}` |

通过 5/5。

## 失败样本

（无）
