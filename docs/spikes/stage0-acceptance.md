# 阶段 0 验收记录 · 技术验证

- **日期**：2026-10-24
- **状态**：待项目作者确认收尾
- **原始数据**：[stage0-report.json](stage0-report.json)
- **复现脚本**：[scripts/spike_stage0.py](../../scripts/spike_stage0.py)

## 1. 验收目标

1. 确定并记录技术栈（ADR-0001～0005）。
2. 验证本地生成、嵌入、rerank 三个模型能在 8GB 显存笔记本上真实跑通。
3. 建立首 token 延迟、吞吐、显存占用的基线数据。
4. 验证 Qdrant 的 dense + sparse 混合检索（RRF）可用。

## 2. 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，驱动 591.86 |
| OS | Windows 11 |
| Python | 3.12.10（`.venv`） |
| PyTorch | 2.11.0+cu128（sm_120 实测可用） |
| Ollama | 0.40.9 |
| Qdrant | 1.20.0（Docker，`127.0.0.1:6333`） |
| sentence-transformers | 6.2.0 |
| Transformers | 5.20.2 |

模型：

| 模型 | 大小 | 校验 |
| --- | --- | --- |
| `qwen3.5:4b` Q4_K_M | 3.14 GB | Ollama 正常加载，100% GPU |
| `bge-m3` | 0.69 GB（加载后） | SHA256 校验通过 |
| `BAAI/bge-reranker-v2-m3` | 2.27 GB | SHA256 `d9e3e081…b5286` 与 HF 元数据一致 |

## 3. 实测结果

| 指标 | 结果 | 目标 | 结论 |
| --- | --- | --- | --- |
| 生成首 token（预热后） | **0.26 s** | 本地 < 5 s | ✅ 余量充足 |
| 生成首 token（首次调用） | 0.38 s | 本地 < 5 s | ✅ |
| 生成吞吐 | **79 tok/s** | — | ✅ 可用 |
| 嵌入维度 | 1024 | — | ✅ 与 ADR 一致 |
| 嵌入延迟（batch=8，GPU） | **11.1 ms/条** | — | ✅ |
| Rerank 延迟（batch=20，GPU） | **6.4 ms/对**（128 ms） | — | ✅ |
| Rerank 模型加载 | 5.4 s | — | ⚠️ 只在进程启动时付一次 |
| 三模型共存显存 | **7,607 / 8,151 MiB（93%）** | ≤ 8 GB | ⚠️ 可用但余量仅约 0.5 GB |
| Qdrant 混合检索（RRF） | 正常返回，语义与关键词两路均命中 | 可用 | ✅ |
| 本地模式零出站 | rerank 用 `local_files_only=True`；Ollama/Qdrant 均走 localhost | 无外部请求 | ✅（本阶段仅覆盖模型调用） |

补测：一次长 prompt 首 token 出现过 1.63 s，复测为 0.38 s。判断为首次调用时的 CUDA/上下文初始化开销，不代表稳态；正式服务需要预热。

## 4. 关键发现

### 4.1 `bge-m3` 下载损坏（已修复）

首次 pull 得到的 blob SHA256 与 Ollama 清单不符（实际 `b8cdc66e…`，期望 `daec91ff…`），但模型仍能返回向量。重新下载并按 SHA256 校验后修复。**教训**：模型文件必须校验哈希，不能只看"能调用"。

### 4.2 FastEmbed 的 BM25 不支持中文（已改方案）

`Qdrant/bm25` 的分词器把 `混合检索结合向量与关键词` 切成 **1 个 token**，英文句子则正常切成 5 个。中文整句会退化为单个 `[UNK]`，BM25 的精确匹配优势完全失效。

改为**自建稀疏编码**：`jieba` 分词 → 去停用词/标点 → 词频 → CRC32 哈希为 uint32 索引，Qdrant 集合开启 `Modifier.IDF` 由服务端计算 IDF。已验证中文查询能正确分词并召回。该变更已写入 ADR-0005，并从依赖中移除 `fastembed`、加入 `jieba`。

### 4.3 三模型共存处于显存上限

`qwen3.5:4b` + `bge-m3` + reranker 同时驻留后显存占用 7.6/8.15 GB，只剩约 0.5 GB。当前短上下文没问题，但长上下文（8k）、更大 rerank batch、PDF 大页都可能 OOM。

**决策（待阶段 2 验证）**：生成与嵌入常驻；reranker 启动时加载并常驻，同时实现 OOM 降级——捕获异常后卸载 reranker、改用 CPU 处理 top-20（约 20 × 6.4 ms 的量级，可接受）。若仍不足，再降级嵌入。

### 4.4 Rerank 分数可用于拒答

模型对明确相关/无关样本区分度极高：相关 0.9998 / 0.9990，无关 0.0000。但对"部分相关"样本分数较低（0.08～0.15），说明**拒答阈值不能拍脑袋**，必须在阶段 2 用评测集扫参。

### 4.5 HuggingFace 直连 TLS 失败

`huggingface.co` 在本机报 `CERTIFICATE_VERIFY_FAILED`（疑似证书链/代理拦截），`hf-mirror.com` 可用。reranker 通过镜像下载，并用 HF 元数据中的 SHA256 做了完整性校验。**首次下载依赖镜像，运行时不依赖网络**。

## 5. 与计划的偏差

| 计划 | 实际 | 原因 |
| --- | --- | --- |
| 验收标准写"7～8B 模型" | 选用并通过 4B | 8GB 显存无法让 7B 与嵌入、rerank 共存；RAG 的事实来自检索上下文，4B 足够，且延迟更好（ADR-0002） |
| 依赖 `fastembed` | 改用 `jieba` 自建稀疏向量 | 见 4.2 |
| — | 新增 `jieba` 依赖 | 中文 BM25 必需 |

## 6. 已知风险与未验证项

1. **显存余量小**：长上下文、多文档、并发请求可能 OOM；阶段 2 必须实测最坏情况。
2. **jieba 分词质量**：专业术语可能切错；阶段 2 用评测集对比词级切分与字符 bigram。
3. **4B 模型能力上限**：长上下文指令遵循、引用格式稳定性未压测。
4. **真实语料规模**：本阶段只用了 8 条样例，1k～5k 文件的索引吞吐未测。
5. **云端模式**：未测（需要 DeepSeek API 密钥，属于外部调用，需单独授权）。
6. **PDF/docx 解析、断网演示**：按计划留到阶段 1/3。

## 7. 验收结论

- 技术栈可行性：✅ 三模型均在本机真实跑通。
- 延迟基线：✅ 首 token 0.26～0.38 s，生成 79 tok/s。
- 混合检索：✅ Qdrant dense + sparse + RRF 可用；中文稀疏方案已修正。
- 文档：✅ ADR-0001～0005 已记录，ADR-0002/0005 已按实测更新。

**建议**：阶段 0 通过。等待项目作者确认后收尾（commit），再进入阶段 1 的需求确认，不直接开始开发。

## 8. 复现命令

```bash
# 前置：Ollama 已 pull qwen3.5:4b、bge-m3；Qdrant 容器已启动
HF_HOME="D:\dev\models\huggingface" .venv/Scripts/python.exe scripts/spike_stage0.py
```

报告输出到 `docs/spikes/stage0-report.json`。