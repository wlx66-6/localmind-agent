# LocalMind

本地优先的个人知识代理：把本机笔记建成可检索索引，用日常语言提问，回答必须给出处；资料里没有依据时明确拒答。索引与检索全部在本机完成，生成阶段默认走云端、后续可切换到本地模型。

当前处于**阶段 1 · 端到端基线**：摄取 → 分块 → 向量索引 → 检索 → 引用 → 流式回答的最小闭环已打通。

## 架构

```
                 浏览器（React + Vite，仅 127.0.0.1:5173）
                                │  /api/*（Vite 开发代理）
                                ▼
                  FastAPI（仅 127.0.0.1:8000）
        ┌───────────────────────┴────────────────────────┐
        │ 摄取                                            │ 问答
        ▼                                                ▼
 scanner → parser → chunker                    embed(问题) → Qdrant 检索
        │                                                │
        ▼                                                ├─ 低于阈值 → 直接拒答
 embed(bge-m3, Ollama)                                   │
        │                                                ▼
        ├──► Qdrant（向量 + 引用 payload）        citations 事件
        └──► SQLite（资料源/文档/分块/会话/消息）        │
                                                         ▼
                                            DeepSeek 流式生成 → token / done
```

- **SQLite**：元数据、增量索引状态、会话历史（`data/localmind.db`）。
- **Qdrant**：dense 向量与引用 payload；阶段 2 在同一集合加入 sparse 向量做混合检索。
- **出站边界**：只有"问题 + 命中的 top-k 片段"会发往云端；原始文档不离机。拒答路径不产生任何外部请求。

设计取舍见 [docs/adr/](docs/adr/)。

## 快速启动

### 前置条件

| 组件 | 用途 | 说明 |
| --- | --- | --- |
| Python 3.12 | 后端 | 仓库内使用 `.venv` |
| Node.js 20+ | 前端 | 阶段 1 使用 Vite 开发服务器 |
| Docker | Qdrant | 也可以直接安装 Qdrant |
| Ollama | 嵌入模型 `bge-m3` | 端口 `127.0.0.1:11434` |

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

### 3. 配置

```bash
cp .env.example .env
```

- 只想看检索与引用：不需要填密钥，服务会提示"云端未配置"。
- 需要生成回答：在 `.env` 中填写 `DEEPSEEK_API_KEY` 后重启后端。密钥只从环境变量读取，不写入代码、不进仓库。

### 4. 建索引

启动后端后，在页面左侧「资料源」添加目录并点「扫描」；或用命令行：

```bash
.venv/Scripts/python.exe scripts/ingest_cli.py add examples/corpus
.venv/Scripts/python.exe scripts/ingest_cli.py scan 1
```

仓库自带 `examples/corpus/`（10 篇公开技术笔记）用于演示，不包含任何个人资料。

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
- 资料中没有依据时，直接返回拒答文案，不调用生成模型。
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

## 目录结构

```
backend/app/
  api/            HTTP 路由（sources / chat / health）与 SSE
  ingestion/      扫描、解析、分块
  retrieval/      嵌入、Qdrant 封装、检索类型
  generation/     Provider 抽象、DeepSeek 实现、提示词
  services/       摄取与问答编排
  config.py       环境变量与路径
  models.py       SQLAlchemy 模型
  db.py           引擎与会话
  logging.py      JSON 结构化日志
backend/tests/    单元与集成测试
frontend/src/     React 单页（对话、引用侧栏、资料源面板）
examples/corpus/  演示语料（公开技术笔记）
scripts/          ingest_cli、阶段 0 spike
docs/adr/         架构决策记录
docs/spikes/      阶段验证与验收记录
PROGRESS.md       阶段进度与待办
```

## 当前限制

- 只支持 `.md` / `.markdown` / `.txt`；PDF、docx、代码文件在阶段 5 接入。
- 检索目前是纯向量；混合检索（BM25 + RRF）与 rerank 在阶段 2。
- 只有云端生成可用；本地生成（Ollama）在阶段 3，与云端共用 Provider 接口。
- 长期记忆（阶段 4）尚未实现。
- 数据库与向量库需要分别备份：`data/localmind.db` 与 Qdrant volume。