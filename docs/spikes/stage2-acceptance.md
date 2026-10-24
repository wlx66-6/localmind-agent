# 阶段 2 验收记录 · 检索深度

- **日期**：2026-10-25
- **状态**：待作者验收（实现与评测已完成，引用精确率口径待确认）
- **范围**：中文稀疏索引、应用层 RRF 混合召回、cross-encoder 精排、两层拒答、40 题评测集与评测脚本、引用卡片检索轨迹

## 1. 实现清单

| 模块 | 内容 |
| --- | --- |
| 稀疏索引 | `retrieval/sparse.py`：jieba 分词 → 去停用词/标点 → 词频 → CRC32 uint32；写入时由 Qdrant `Modifier.IDF` 计算 IDF |
| 向量库 | 集合 `localmind_chunks_v2`，命名向量 `dense`（bge-m3 1024 维）与 `sparse`；布局校验失败时给出重建提示 |
| 摄取 | 每个 chunk 同时写 dense + sparse；`scan --force` 用于分块参数或向量布局变更后重建（本次已重建 47 点） |
| 融合 | `retrieval/pipeline.py` 的应用层 RRF：`score = Σ 1/(k + rank)`，`k=60`，两路各取 top-20；同分时按 `best_rank`、`chunk_id` 决定顺序，保证可复现 |
| 精排 | `retrieval/reranker.py`：`BAAI/bge-reranker-v2-m3`，懒加载 + CUDA，batch 8，sigmoid 分数；加载或推理失败自动降级为融合顺序并标记 `rerank_failed` |
| 拒答 | 两层：闸门（top1 rerank < `0.01`，不调用云端）+ 生成层（prompt 要求资料不足时输出 `REFUSAL_MARKER`） |
| 协议 | `citations.retrieval` 携带模式、降级标记、各阶段耗时；`done.refused` 区分两层拒答；每个片段带 `dense_rank / sparse_rank / fused_rank / dense_score / rerank_score` |
| 前端 | 引用卡片显示 `稠密 #2 · 稀疏 #5 · 融合 #3 · 重排 0.82`，纯向量模式回退为相似度 |
| 评测 | `scripts/eval_retrieval.py`：三模式对比 + 闸门阈值扫参 + `--full` 追加云端生成评测；与生产链路共用 `retrieve` / `should_refuse` / `build_messages` |
| 文档 | ADR-0009（评测方法）、ADR-0005（参数实测）、ADR-0008（两层拒答与检索轨迹）、README、`.env.example` |

明确未做（按阶段计划）：本地生成、记忆、pdf/docx 解析、查询改写 / HyDE、父子块、Docker/CI、UI 改版。

## 2. 验收证据

### 2.1 三种检索模式（40 题：30 有答案 / 10 无答案）

| 模式 | Hit@5 | Recall@5 | MRR@5 | 闸门拒答率 | 误拒率 | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 纯向量 | 93.3% | 82.8% | 0.776 | 20% | 0% | 19 / 22 ms |
| 混合（RRF） | **96.7%** | **91.1%** | 0.806 | 40% | 0% | 23 / 25 ms |
| 混合 + rerank | 93.3% | 87.2% | **0.844** | 50% | 0% | 312 / 341 ms |

- Recall@5 验收线 0.8：三种模式均达标；混合召回把 Recall@5 提升 8.3 个百分点。
- rerank 额外耗时 p50 288 ms / p95 313 ms（20 候选，CUDA），首 token p50 711 ms，未成为瓶颈。
- 融合方案对比（三路 RRF / Borda / 固定保留前 3）与选择理由见 ADR-0009。

### 2.2 端到端（hybrid_rerank + DeepSeek，35 次云端调用）

| 指标 | 结果 |
| --- | --- |
| 无答案题拒答 | **10/10**（闸门 5 + 生成层 5） |
| 有答案题误拒 | **0/30** |
| 无答案题误答 | 0 |
| 引用召回（gold 是否被引用） | 93.3% |
| 引用精确率（严格，只认 gold） | 64.0%（宏）/ 48.6%（按引用条数加权） |
| 首 token p50 / p95 | 711 / 981 ms |
| 总耗时 p50 / p95 | 1410 / 2196 ms |

严格精确率未达 90% 验收线，原因与口径见第 3 节。

### 2.3 单次真实请求的检索轨迹（验收时实测）

问题：`混合检索为什么要结合 BM25？`

```
retrieval: mode=hybrid_rerank, rerank_failed=false,
           embed 15ms / dense 8ms / sparse 5ms / rerank 293ms / total 321ms
1  hybrid-retrieval.md › 混合检索 > BM25 的直觉      13–15  dense #1  sparse #2  fused #2  rerank 0.960
2  hybrid-retrieval.md › 混合检索 > 稠密检索与稀疏检索 5–9   dense #2  sparse #1  fused #1  rerank 0.943
3  hybrid-retrieval.md › 混合检索 > RRF 融合          19–25  dense #3  sparse #4  fused #4  rerank 0.893
4  embeddings-zh.md    › 中文嵌入模型 > bge-m3        11–15  dense #10 sparse #7  fused #7  rerank 0.168
5  hybrid-retrieval.md › 混合检索 > 工程取舍          29–31  dense #4  sparse #3  fused #3  rerank 0.094
done: cited=[2,1,4], refused=false；首 token 1136 ms，总耗时 2255 ms
```

两点可解释、可复现的现象：

- 第 5 条原本是融合第 3 名，被 rerank 压到最后（0.094），因为它的措辞与问题字面重合少——这正是 ADR-0009 记录的"字面偏好"。
- 第 4 条从融合第 7 名被提到第 4（0.168），模型也确实引用了它，说明 rerank 引入了融合排名之外的信息，不是重复排序。

### 2.4 拒答（两层）

- 闸门：`我们公司明年的预算是多少？` 这类完全无关问题 top1 rerank < 0.01，不调用云端直接拒答（评测中 5/10 无答案题走这条路径）。
- 生成层：`bge-m3 训练数据多少条`、`LoRA 微调显存` 这类话题相关但缺少具体事实的问题通过闸门（0.948 / 0.964），由模型输出"资料中未找到足够依据"，`done.refused=true`。
- 单阈值扫参的最优点（0.25）会误拒 3/30 有答案题，因此没有采用单阈值方案，见 ADR-0009。

### 2.5 质量门禁

| 检查 | 结果 |
| --- | --- |
| `ruff check` | 通过 |
| `ruff format --check` | 68 个文件已格式化 |
| `mypy` | 35 个源文件无问题 |
| `pytest` | **30 passed**（新增 12 个检索/拒答用例） |
| `tsc --noEmit` | 通过 |
| `vite build` | 通过（JS 238.24 kB / gzip 75.05 kB） |

## 3. 已知限制与风险

1. **rerank 有字面偏好**：同一段正确内容，问法改变后分数可从 0.8 掉到 0.001。这不是封装 bug（原始 logits 与 sentence-transformers 一致），而是 cross-encoder 衡量"话题相关性"而非"是否包含答案"。应对方式是两层拒答，而非继续调阈值。
2. **严格引用精确率只有 64%**：多部分问题会额外引用同话题片段；q12/q29 的 gold 未进前 5，模型改用其他相关文档作答。该指标只作保守下界，口径需作者确认（见第 4 节）。
3. **q12/q29 是 rerank 引入的回归**：这两题在混合模式下命中、加 rerank 后掉出前 5。若后续回答质量受影响，可切换到 Borda 融合（Recall@5 92.2%，MRR 0.807）。
4. **评测语料规模小**（10 篇 / 47 chunk），指标波动敏感；40 题中单题权重 3.3 个百分点。
5. **重排模型常驻显存**：加载失败会降级而非报错，但降级只能从 `retrieval.rerank_failed` 与日志发现，前端暂无显式提示。
6. **哈希碰撞**：CRC32 为 32 位，词表规模下碰撞概率极低但非零；若未来词表显著增长，应改为确定性词表 ID。

## 4. 待作者确认：引用准确率的验收口径

验收标准原文是"引用准确率 ≥ 90%"，当前严格 gold 精确率为 64%。三种可用口径：

| 口径 | 当前结果 | 说明 |
| --- | --- | --- |
| A. 引用召回 + 引用有效性 | 93.3% / 100% | 度量"该引的有没有引""引用是否指向真实片段"。不惩罚额外引用 |
| B. 扩充 gold 标注（允许多个可接受片段） | 需补标 | 最接近人工判断，但要重标 40 题并重跑 |
| C. 句子级支撑判定（人工或 NLI） | 未做 | 最严格，需要额外模型或标注工作量，超出本阶段 |

## 5. 复现步骤

```bash
# 1. Qdrant + 后端
docker run -d --name localmind-qdrant -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v localmind_qdrant:/qdrant/storage qdrant/qdrant:latest
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# 2. 重建索引（双向量）
.venv/Scripts/python.exe scripts/ingest_cli.py scan 1 --force

# 3. 评测（离线不联网；--full 会把 top-5 片段发给 DeepSeek）
.venv/Scripts/python.exe scripts/eval_retrieval.py
.venv/Scripts/python.exe scripts/eval_retrieval.py --full

# 4. 质量门禁
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . \
  && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest
cd frontend && npm run typecheck && npm run build
```