# 0001 · 技术栈选型

- **状态**：已接受
- **日期**：2026-10-24
- **决策者**：项目作者 + 协作代理

## 背景

LocalMind 是 6～8 周内完成的求职项目，要求本地优先、RAG 深度、双轨生成与可演示的工程化。硬件为 RTX 5060 Laptop（8GB 显存）/ 16GB 内存 / Windows 11。

## 决策

| 层 | 选择 |
| --- | --- |
| 运行时 | Python 3.12（项目内 `.venv`） |
| 后端 | FastAPI + Pydantic + Uvicorn |
| 前端 | React + Vite + TypeScript |
| 向量库 | Qdrant 1.20（Docker，绑定 `127.0.0.1`） |
| 本地推理 | Ollama 0.40 |
| 解析 | PyMuPDF + python-docx + 原生 md/txt |

## 备选与取舍

| 决策点 | 选择 | 备选 | 取舍理由 |
| --- | --- | --- | --- |
| 后端 | Python + FastAPI | TypeScript + NestJS；Go + Gin | RAG/AI 生态、解析与向量客户端同栈；异步与 SSE 流式简单。NestJS 仍需 Python 侧服务；Go 生态弱、开发慢。 |
| 前端 | React + Vite + TS | Next.js；Vue；Svelte | 单页应用不需要 SSR；Vite 轻、生态大、面试常见。Next.js 增加不必要的服务端复杂度。 |
| 向量库 | Qdrant | Chroma；pgvector；LanceDB/SQLite-vec | 原生 dense+sparse 混合检索、payload 过滤、服务化与监控。Chroma 生产化弱；pgvector 需自己拼混合检索；嵌入式方案并发/过滤弱。 |
| 本地推理 | Ollama | llama.cpp 直接集成；vLLM | Ollama 模型管理与 OpenAI 兼容 API 省时，仍可测延迟与调参。llama.cpp 性能故事更强但调试成本高；vLLM 对 8GB 单机过重。 |
| Python | 3.12 | 3.14；WSL2 | 3.14 的 PyTorch / sentence-transformers 轮子滞后；3.12 最稳。WSL2 增加 GPU 直通与文件访问成本。 |

## 后果

- 一个 Python 服务 + 一个前端构建；Qdrant 由 Docker 提供。
- 服务默认只监听 `127.0.0.1`，不引入账号与公网部署。
- `torch` 使用 CUDA 12.8 轮子（`2.11.0+cu128`），以支持 Blackwell（sm_120）。
- 后续若想换向量库，检索层需保持接口隔离（阶段 2 再落地）。