# 0002 · 本地模型选型

- **状态**：已接受（阶段 0 实测通过；显存降级策略待阶段 2 压测）
- **日期**：2026-10-24
- **决策者**：项目作者 + 协作代理

## 背景

本地模式要在 8GB 显存的笔记本上真实跑通，并且希望生成、嵌入、rerank 尽量同时驻留 GPU，以降低延迟。RAG 的事实来自检索上下文，生成模型的主要职责是指令遵循、引用格式与拒答。

## 决策

| 角色 | 模型 | 运行方式 |
| --- | --- | --- |
| 生成 | `qwen3.5:4b`（Q4_K_M，3.3GB） | Ollama；应用层显式 `think: false` |
| 嵌入（dense） | `bge-m3`（1024 维） | Ollama `/api/embed` |
| 稀疏（BM25） | jieba 分词 + CRC32 哈希 | 自建编码器，本地计算（见 ADR-0005） |
| Rerank | `BAAI/bge-reranker-v2-m3` | sentence-transformers CrossEncoder |

## 备选与取舍

| 决策点 | 选择 | 备选 | 取舍理由 |
| --- | --- | --- | --- |
| 生成 | Qwen3.5-4B Q4_K_M | Qwen2.5-7B；Llama 3.1-8B；Qwen2.5-3B | 4B 约 3.3GB，可与嵌入/rerank 共存；新架构指令遵循更好。7B 知识更多但约 4.7GB，三模型无法共存。Llama 中文弱；3B 质量不足。 |
| 嵌入 | bge-m3 | bge-small-zh-v1.5；云端 embedding | 中英多语、1024 维、中文强。small 更快但长文本/多语弱；云端嵌入会让所有文档出站，违背本地优先。 |
| Rerank | bge-reranker-v2-m3 | bge-reranker-base；Cohere Rerank；不做 | 中英效果好、可与生成模型按需共存。base 更快但效果弱；Cohere 出站且本地模式不可用；不做会明显降低引用准确率。 |

## 阶段 0 实测（2026-10-24）

| 指标 | 实测 |
| --- | --- |
| 生成首 token | 0.26 s（预热）/ 0.38 s（首次调用） |
| 生成吞吐 | 79 tok/s |
| 嵌入延迟 | 11.1 ms/条（batch=8，100% GPU） |
| Rerank 延迟 | 6.4 ms/对（batch=20 共 128 ms） |
| Rerank 加载 | 5.4 s（进程启动一次） |
| 三模型共存显存 | 7,607 / 8,151 MiB（93%） |

结论：三者可同时驻留，但只剩约 0.5 GB 余量，属于**上限运行**。

## 后果

- `qwen3.5:4b` 的 thinking 默认开启。应用层必须显式关闭，否则首 token 延迟与输出长度不可控。
- 8GB 显存下三者同时驻留实测 7.6GB，余量约 0.5GB。长上下文或更大 batch 可能 OOM，按以下顺序降级：
  1. 卸载 rerank 模型，改 CPU 处理 top-20（20 × 6.4 ms 量级，可接受）；
  2. 嵌入改为按需加载 / CPU；
  3. 生成模型换 `qwen3.5:2b` 级别。
- 阶段 2 必须压测"最长上下文 + 最大 batch"的最坏情况，再决定 rerank 是否常驻。
- Ollama 的 `bge-m3` 只返回 dense 向量；稀疏向量由 jieba + Qdrant IDF 单独计算，二者在 Qdrant 中做混合检索。FastEmbed 的 `Qdrant/bm25` 已实测不支持中文分词，不采用（见 ADR-0005）。
- 模型文件不入库：Ollama 使用 `D:\dev\models`，HuggingFace 使用 `D:\dev\models\huggingface`。