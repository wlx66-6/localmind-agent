# 阶段 1 验收记录 · 端到端基线

- **日期**：2026-10-25
- **状态**：自检通过，待项目作者验收
- **范围**：摄取 → 分块 → 向量索引 → 检索 → 引用 → 云端流式回答的最小闭环

## 1. 实现清单

| 模块 | 内容 |
| --- | --- |
| 配置 | `.env` / Pydantic Settings；密钥只读环境变量 |
| 摄取 | 白名单目录、md/txt、5MB 上限、隐藏目录跳过、sha256 增量跳过、按 `document_id` 清理旧向量 |
| 分块 | 标题感知（标题路径随块保存）、段落打包 ≤600 字符、超长硬切带 100 字符重叠、行号定位 |
| 索引 | Ollama `bge-m3`（1024 维）→ Qdrant；`Chunk.id` 内容哈希 ↔ `uuid5` point ID |
| 检索 | dense top-5，`RETRIEVAL_MIN_SCORE=0.5` |
| 生成 | DeepSeek provider，SSE：`citations → token → done / error` |
| 拒答 | 低于阈值不调用模型，固定文案落库 |
| 前端 | 单页：对话流式渲染、`[n]` 引用跳转、引用侧栏、资料源增删扫描、健康状态轮询 |
| 工程 | ruff / mypy / pytest；JSON 结构化日志；ADR-0006～0008；README |

明确未做（按阶段计划）：混合检索、rerank、本地生成、记忆、pdf/docx、Docker/CI。

## 2. 验收证据

### 2.1 索引

`examples/corpus/`（10 篇公开中文技术笔记）→ **10 个文档 / 47 个分块**。

### 2.2 检索与引用

问题：`混合检索为什么要结合 BM25？`

| 编号 | 来源 | 小节 | 行号 | 分数 |
| --- | --- | --- | --- | --- |
| 1 | hybrid-retrieval.md | 混合检索 > BM25 的直觉 | 13–15 | 0.620 |
| 2 | hybrid-retrieval.md | 混合检索 > 稠密检索与稀疏检索 | 5–9 | 0.603 |
| 3 | hybrid-retrieval.md | 混合检索 > RRF 融合 | 19–25 | 0.578 |
| 4 | hybrid-retrieval.md | 混合检索 > 工程取舍 | 29–31 | 0.566 |
| 5 | qdrant-notes.md | Qdrant 使用笔记 > 集合与向量 | 5–7 | 0.511 |

阈值依据：阶段 0/1 实测相关问题 top-1 分数 0.62～0.71，无关问题 0.40～0.47，取 0.5。

### 2.3 拒答

问题：`我们公司明年的预算是多少？` → 无召回，SSE 返回固定拒答文案并 `done.refused=true`，**不调用云端、不写 `cloud_egress` 日志**。

### 2.4 前端

用 headless Edge + CDP 实际操作页面验证：

- 拒答问题：显示「拒答」标记与拒答文案；
- 相关问题：右侧显示 5 张引用卡片（路径、小节、行号、分数、原文）；
- 资料源：UI 添加临时目录 → 扫描成功（识别 1 / 索引 1）→ 删除成功（Qdrant 点同步清理）。

### 2.5 质量门禁

| 检查 | 结果 |
| --- | --- |
| `ruff check` | 通过 |
| `ruff format --check` | 通过 |
| `mypy`（31 个文件） | 通过 |
| `pytest` | 17 passed |
| `tsc --noEmit` | 通过 |
| `vite build` | 通过（JS 237.84 kB / gzip 74.93 kB） |

## 3. 过程中发现并修复的问题

| 问题 | 原因 | 修复 |
| --- | --- | --- |
| `POST /api/chat` 返回 500，`FOREIGN KEY constraint failed` | 会话对象只 `flush` 未 `commit`，会话关闭即回滚；随后插入消息时外键不存在 | 检索成功后再建会话，会话与用户消息同事务提交；新增 `test_chat_api.py` 覆盖 |
| SSE 首次测试拿不到 `citations` | 前端测试脚本过早匹配到页面横幅中的 `DEEPSEEK_API_KEY` | 改为等待错误事件文案；确认事件顺序为 `citations → error` |
| ruff 报 `RUF100` / `SIM117` / `B905` / `E501` | 历史脚本与新增测试的细节 | 已全部修复，`ruff format` 统一格式 |

## 4. 已知限制与风险

1. **云端生成未实测**：作者尚未填写 `DEEPSEEK_API_KEY`，真实回答的引用准确率需要验收时补测。
2. **双写一致性**：向量 upsert 成功、SQLite 提交失败会留下孤儿向量；阶段 5 增加对账清理。
3. **仅 md/txt**：PDF/docx/代码解析在阶段 5。
4. **单 worker**：SQLite 写入模型只适合单进程；多 worker 需要换存储或串行化。
5. **前端无自动化测试**：当前用 CDP 手工脚本验证关键路径，CI 接入在阶段 5。

## 5. 复现步骤

```bash
# 1. Qdrant
docker run -d --name localmind-qdrant -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v localmind_qdrant:/qdrant/storage qdrant/qdrant:latest
# 2. 后端
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
# 3. 建索引（或使用页面左侧面板）
.venv/Scripts/python.exe scripts/ingest_cli.py add examples/corpus
.venv/Scripts/python.exe scripts/ingest_cli.py scan 1
# 4. 前端
cd frontend && npm install && npm run dev
# 5. 质量门禁
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest
cd frontend && npm run build
```