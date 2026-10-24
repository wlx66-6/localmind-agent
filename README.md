# LocalMind

本地优先的个人知识代理：把本机笔记建成可检索索引，用日常语言提问，回答必须给出处；资料里没有依据时明确拒答。索引与检索全部在本机完成，生成阶段默认走云端、后续可切换到本地模型。

当前处于**阶段 2 · 检索深度**：在阶段 1 的端到端闭环上加入中文稀疏检索（jieba + BM25）、应用层 RRF 混合召回、cross-encoder 精排，以及 40 题评测集与两层拒答。

## 架构

```
                 浏览器（React + Vite，仅 127.0.0.1:5173）
                                │  /api/*（Vite 开发代理）
                                ▼
                  FastAPI（仅 127.0.0.1:8000）
        ┌───────────────────────┴────────────────────────┐
        │ 摄取                                            │ 问答
        ▼                                                ▼
 scanner → parser → chunker                embed(问题) + jieba 分词
        │                                                │
        ▼                                                ▼
 embed(bge-m3, Ollama)                        Qdrant 双路召回（dense / sparse）
        │                                                │
        ├──► Qdrant（dense + sparse + 引用 payload）      ▼
        └──► SQLite（资料源/文档/分块/会话/消息）  应用层 RRF 融合（top-20）
                                                         │
                                                         ▼
                                             cross-encoder 精排 → top-5
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                        闸门拒答                生成层拒答
                                     （不调用云端）        citations / token / done
```

- **SQLite**：元数据、增量索引状态、会话历史（`data/localmind.db`）。
- **Qdrant**：同一集合内命名向量 `dense`（bge-m3，1024 维）与 `sparse`（jieba + CRC32 词频，服务端 IDF），检索在应用层融合。
- **拒答分两层**：检索闸门拦“完全无关”，生成 prompt 拦“话题相关但缺少具体事实”，见 [ADR-0009](docs/adr/0009-retrieval-evaluation.md)。
- **出站边界**：只有"问题 + 命中的 top-k 片段"会发往云端；原始文档不离机。闸门拒答路径不产生任何外部请求。

设计取舍见 [docs/adr/](docs/adr/)。

## 快速启动

### 前置条件

| 组件 | 用途 | 说明 |
| --- | --- | --- |
| Python 3.12 | 后端 | 仓库内使用 `.venv` |
| Node.js 20+ | 前端 | 阶段 1 使用 Vite 开发服务器 |
| Docker | Qdrant | 也可以直接安装 Qdrant |
| Ollama | 嵌入模型 `bge-m3` | 端口 `127.0.0.1:11434` |
| Hugging Face 缓存 | 重排模型 `BAAI/bge-reranker-v2-m3` | 约 2.3 GB；用 `HF_HOME` 指向缓存目录 |

### 1. 启动 Qdrant

```bash
docker run -d --name localmind-qdrant \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v localmind_qdrant:/qdrant/storage \
  qdrant/qdrant:latest
```

### 2. 安装依赖

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Ollama 侧拉取嵌入模型：

```bash
ollama pull bge-m3
```

重排模型只需要下载一次（联网，约 2.3 GB）。二选一：

```bash
# 方式 A：下载到自己的缓存目录，然后把该目录填进 .env 的 HF_HOME
huggingface-cli download BAAI/bge-reranker-v2-m3 --cache-dir D:/dev/models/huggingface/hub

# 方式 B：已经用默认缓存下过，则 .env 里不要设 HF_HOME，直接用 ~/.cache/huggingface
```

后端启动时按 `local_files_only=True` 加载，不会隐式联网；模型缺失或加载失败时检索自动降级为"重排前的融合顺序"，`citations` 事件里 `retrieval.rerank_failed=true`，不会让问答整体失败。

### 3. 配置

```bash
cp .env.example .env
```

- 只想看检索与引用：不需要填密钥，服务会提示"云端未配置"。
- 需要生成回答：在 `.env` 中填写 `DEEPSEEK_API_KEY`，然后**重启后端进程**（配置只在启动时读取一次）。密钥只从环境变量读取，不写入代码、不进仓库。

检索相关配置（全部有默认值，改后需重启）：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `vector` / `hybrid` / `hybrid_rerank`，用于对比三种策略 |
| `RETRIEVAL_TOP_K` | `5` | 进入上下文的片段数 |
| `RETRIEVAL_TOP_N` | `20` | 每路召回候选数，也是 rerank 的输入规模 |
| `RRF_K` | `60` | RRF 公式中的平滑常数 |
| `RETRIEVAL_MIN_SCORE` | `0.5` | 稠密余弦阈值（纯向量模式，及 rerank 不可用时） |
| `RETRIEVAL_GATE_RERANK_SCORE` | `0.01` | 第一层拒答闸门；只在 top1 重排分数低于它时拒答 |
| `RERANK_MODEL` / `RERANK_DEVICE` / `RERANK_BATCH_SIZE` | `BAAI/bge-reranker-v2-m3` / `auto` / `8` | 重排模型与运行设备 |
| `HF_HOME` | 未设置 | 重排模型缓存根目录，实际读取 `<HF_HOME>/hub` |

阈值不是拍脑袋定的：`RETRIEVAL_GATE_RERANK_SCORE=0.01` 来自 40 题评测集的扫参结果，推导过程见 [ADR-0009](docs/adr/0009-retrieval-evaluation.md)。

### 4. 建索引

启动后端后，在页面左侧「资料源」添加目录并点「扫描」；或用命令行：

```bash
.venv/Scripts/python.exe scripts/ingest_cli.py add examples/corpus
.venv/Scripts/python.exe scripts/ingest_cli.py scan 1
```

仓库自带 `examples/corpus/`（10 篇公开技术笔记）用于演示，不包含任何个人资料。

改了 `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP` 或向量布局后，旧向量会与新链路不一致，需要强制重建：

```bash
.venv/Scripts/python.exe scripts/ingest_cli.py scan 1 --force
```

### 5. 启动服务

后端：

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

打开 <http://127.0.0.1:5173>。

## 使用

- 提问后先返回命中的片段（右侧「引用片段」），再流式输出回答；回答中的 `[n]` 可点击定位。
- 每个片段下方显示检索轨迹，例如 `稠密 #2 · 稀疏 #5 · 融合 #3 · 重排 0.82`，用于解释"为什么这段被选中"。
- 拒答分两层：检索结果与问题完全无关时，不调用云端直接拒答；话题相关但资料里缺少具体事实时，由生成模型按提示词拒答。两种情况前端都显示"资料不足"。
- 修改文件后重新「扫描」：按文件哈希跳过未变化文件，删除已移除文件的向量。

## 开发

```bash
.venv/Scripts/python.exe -m pytest            # 单元 + 集成测试
.venv/Scripts/python.exe -m ruff check .      # lint
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m mypy              # 类型检查

cd frontend
npm run typecheck
npm run build
```

### 检索评测

```bash
.venv/Scripts/python.exe scripts/eval_retrieval.py          # 离线：三种模式 + 闸门阈值扫参，不联网
.venv/Scripts/python.exe scripts/eval_retrieval.py --full   # 追加云端生成评测：会把 top-5 片段发给 DeepSeek，产生真实费用
```

报告写入 `docs/eval/stage2-report.json` 与 `.md`（逐题明细 + 汇总）。评测集在 `eval/stage2/questions.jsonl`：40 题，`exact` / `paraphrase` / `cross_doc` / `unanswerable` 各 10 题，gold 用逐字引文标注，分块参数变化后仍可评测。指标定义与实测结论见 [ADR-0009](docs/adr/0009-retrieval-evaluation.md)。

## 目录结构

```
backend/app/
  api/            HTTP 路由（sources / chat / health）与 SSE
  ingestion/      扫描、解析、分块
  retrieval/      嵌入、稀疏编码（jieba+BM25）、Qdrant 封装、RRF 融合、重排
  generation/     Provider 抽象、DeepSeek 实现、提示词与拒答标记
  services/       摄取与问答编排
  config.py       环境变量与路径
  models.py       SQLAlchemy 模型
  db.py           引擎与会话
  logging.py      JSON 结构化日志
backend/tests/    单元与集成测试
frontend/src/     React 单页（对话、引用侧栏、资料源面板）
examples/corpus/  演示语料（公开技术笔记）
eval/stage2/      40 题检索评测集（gold 为逐字引文）
scripts/          ingest_cli、检索评测脚本、阶段 0 spike
docs/adr/         架构决策记录
docs/eval/        评测报告（JSON + Markdown）
docs/spikes/      阶段验证与验收记录
PROGRESS.md       阶段进度与待办
```

## 当前限制

- 只支持 `.md` / `.markdown` / `.txt`；PDF、docx、代码文件在阶段 5 接入。
- 重排模型按字面相似度打分，对"同义改写"类问题会低估正确片段（例如同一段正确内容，换一种问法分数从 0.8 掉到 0.001）。实测数据与两层拒答的设计依据见 ADR-0009。
- 引用精确率 89.2%（宏平均，按 gold + 可接受片段判定），比 90% 验收线低 0.8 个百分点；只认 gold 的严格口径为 65.3%。失分来自生成端对窄问题也给出 4～5 条引用。引用召回 93.3%、引用编号有效性 100%。口径与逐题分析见 ADR-0009。
- 分词依赖 jieba 词典，专业新词可能切错；哈希碰撞概率极低但非零。
- 只有云端生成可用；本地生成（Ollama）在阶段 3，与云端共用 Provider 接口。
- 长期记忆（阶段 4）尚未实现。
- 数据库与向量库需要分别备份：`data/localmind.db` 与 Qdrant volume。