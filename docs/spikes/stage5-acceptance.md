# 阶段 5 验收记录 · 工程化与解析扩展

- **日期**：2026-10-25
- **状态**：**待作者验收**
- **范围**：PDF / docx / 代码解析与页码引用、Alembic 迁移、双轨评测与记忆场景、运行指标脚本、Docker Compose、GitHub Actions
- **目标回顾**：新机器按 README 能启动；CI 全绿；解析与引用扩展不破坏已有检索质量。

## 1. 实现清单

| 模块 | 内容 |
| --- | --- |
| 迁移 | `backend/alembic.ini`、`backend/migrations/`（`env.py`、`0001_stage4_baseline`、`0002_stage5_parsing_fields`）；`app/schema.py` 提供幂等补列；`Database.migrate()` 启动时 `upgrade head`，删除旧的 `create_all` + `PRAGMA/ALTER` 双轨机制 |
| 解析 | `ingestion/parser.py`：PyMuPDF（按页取块，跳过图片块）、python-docx（Word 标题转 markdown、表格按行保留）、23 种后缀的代码白名单；`ParsedDocument{text, mode, doc_type, page_starts, page_count}` |
| 分块 | `chunk_text(mode=markdown\|text\|code, page_starts=…)`：PDF 每页独立一节（不跨页），代码按顶层符号切节并用签名作 heading |
| 引用 | `RetrievalHit` / `CitationOut` 增加 `ordinal`、`page`、`page_end`；payload 与 SQLite 同步；prompt 对 PDF 写页码；前端「第 N 页 · 第 K 块」 |
| 评测 | `scripts/eval_stage5.py`（40 题双轨 + 5 个记忆场景，失败重试一次）、`scripts/eval_retrieval.py` 增加索引范围守卫 |
| 指标 | `scripts/metrics_report.py`：从 `messages.metrics_json` 汇总，Markdown 输出 |
| 容器 | `docker-compose.yml`（backend / frontend / qdrant）、`backend/Dockerfile`（CPU torch）、`frontend/Dockerfile` + `nginx.conf`（反代 `/api`、关闭 SSE 缓冲） |
| CI | `.github/workflows/ci.yml`：后端 ruff + mypy + pytest，前端 `tsc --noEmit` + `vite build` |
| 演示 | `scripts/make_demo_files.py` → `examples/formats/localmind-runbook.pdf`（3 页）、`delivery-checklist.docx` |
| 文档 | ADR-0012（格式与引用）、ADR-0013（迁移）、ADR-0014（容器与 CI）；ADR-0009 补充"评测可复现边界" |

明确未做（按阶段计划）：OCR、多用户、公网部署、发送前预览确认、记忆自动过期/批量清理。

## 2. 验收证据

### 2.1 解析与引用

`examples/formats` 作为独立资料源扫描：

```
scanned=2 indexed=2 failed=0
localmind-runbook.pdf  → page_count=3，3 个分块，page/page_end = 1-1 / 2-2 / 3-3，ordinal 0/1/2
delivery-checklist.docx → 4 个分块，heading 形如 "知识库交付清单 > 一、资料入库"，
                          表格保留为 "检查项 | 命令 | 期望结果"
```

只检索提问「备份要备份哪些东西？」：

```
[1] localmind-runbook.pdf page=1-1 ordinal=0 score=0.942
[2] qdrant-notes.md        (0.112)   ← 其余命中都是弱相关
```

扫描版 PDF（无文本层）走 `UnsupportedDocumentError("PDF 共 2 页，但没有可提取文本（可能是扫描版，当前不做 OCR）")`，`scan` 计入 `failed` 并在 `errors` 里显示原因，不产生空文档。

### 2.2 迁移

| 场景 | 结果 |
| --- | --- |
| 本机现有库（10 文档 / 47 分块 / 10 记忆）启动迁移 | `alembic_version=0002`，三张表行数不变 |
| 副本演练（`LOCALMIND_DB_URL` 指向 backup 副本） | 输出 `Running upgrade -> 0001`、`0001 -> 0002`，列与数据正确 |
| 阶段 1 结构的库 | 单测：5 个旧列 + 3 个新列齐全，旧行取默认值（`metrics_json='{}'`、`summarized_user_messages=0`） |
| 全新库 | 单测：6 张表 + `alembic_version=head` |
| 连续两次启动 | 单测：no-op，不报 `duplicate column name` |

### 2.3 双轨评测（同一组检索结果，只比较生成层）

`docs/eval/stage5-report.md`：

| 轨道 | 生成回答 | 首 token p50/p95 | 总耗时 p50/p95 | 无答案题拒答 | 有答案题误拒 | 无答案题误答 | 引用精确率 | 引用召回 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 35 | 874 / 1174 ms | 1722 / 2602 ms | 10/10 | 0/30 | 0 | 85.6% | 93.3% |
| local | 35 | 508 / 574 ms | 1886 / 6045 ms | 10/10 | 0/30 | 0 | 87.9% | 93.3% |

- 两条轨道的拒答行为一致（5 次闸门拒答 + 5 次生成拒答），且**没有任何无答案题被误答**。
- 本地轨道的引用精确率略高，但它把一部分片段原样复述得更少；这不等价于"本地更好"，样本只有 35 题。
- 本表的本地产时是**悲观值**：评测共用了云端那次 CUDA 重排，而生产环境的本地轨道用 CPU 重排以让出显存；8 GB 显卡上两者共存会互相拖慢（实测本地总耗时从 ~4.8 s 升到 ~7 s，波动很大）。
- 评测期间出现过一次 Ollama 首调 CUDA 初始化失败（`llama-server ... CUDA error`），已加"失败重试一次并计数"，最终报告中无失败样本。

### 2.4 记忆场景（5/5）

| 场景 | 证据 |
| --- | --- |
| 跨会话召回事实 | 记忆 `fact/所在城市`（0.623）被召回，回答带本地日期 |
| 同 key 覆盖旧值 | `技术栈偏好` 只有 1 条 active（Go），历史里能看到被替换的 Python 与更早的 Go |
| 无关问题不召回 | 「公司明年预算」召回 0 条记忆，两层拒答正常 |
| 事件时间与记录时间分开 | 「去年八月开始学 Rust」→ `occurred_at=2025-08-01`，`created_at=2026-10-25` |
| 删除后不再召回 | 删除 `所在城市` 后，同一问题不再召回该记忆，回答明确说明"记忆里没有城市信息" |

### 2.5 运行指标

`docs/eval/stage5-metrics.md`（本机历史数据快照）：

- 云端实际出站 205 段 / 47 653 字符；本地轨道实际出站 **0**，未出站片段 115。
- 记忆抽取 36 次中 5 次写入、31 次为空；抽取耗时 p50 951 ms。
- 云端首 token p50 1 462 ms，p95 被个别慢轮次拉到 93 s（本机开发环境、含评测并发期间的数据，不代表基线）。

### 2.6 Docker Compose

| 检查 | 结果 |
| --- | --- |
| 镜像 | backend 565 MB（CPU torch），frontend 21 MB |
| `docker compose up --build` | 三个容器起来，后端把新卷里的空库迁移到 `0002`，`/api/health` 全绿 |
| 挂载资料 | `NOTES_DIR` → `/notes:ro`，扫描 10 篇语料 → 47 分块（嵌入走宿主机 Ollama） |
| nginx | `/api/health` 经 5173 正常；SSE 逐 token 到达（间隔 ~60 ms） |
| 容器 → 宿主机 Ollama | Docker Desktop 转发到回环，Ollama 保持默认监听即可 |
| 延迟 | 重排（CPU，20 候选）稳态 2.2 s、冷启 5.7 s；云端首 token 3.1 s、本地 2.6 s |
| 端口 | 全部发布为 `127.0.0.1:*` |

结论：容器路径满足"一条命令跑起来"，**不满足"云端首 token < 2 s"**。两个数字分开陈述，不用容器数字覆盖性能验收。

### 2.7 CI

`.github/workflows/ci.yml`（push 到 main 与 PR 触发）：

- backend：CPU torch → `requirements-dev.txt` → ruff check/format → mypy → pytest（`HF_HUB_OFFLINE=1`，任何试图下载模型的测试都会失败）。
- frontend：`npm ci` → `tsc --noEmit && vite build`。
- 刻意不构建镜像、不跑评测（原因见 ADR-0014）。

### 2.8 质量门禁

| 检查 | 结果 |
| --- | --- |
| `ruff check backend scripts` | 通过 |
| `ruff format --check backend scripts` | 66 个文件已格式化 |
| `mypy` | 56 个源文件无问题 |
| `pytest backend/tests` | **115 passed**（阶段 5 新增 23 个：解析 10、分块 6、迁移改写 4、拒答标记 6） |
| 前端 `npm run build`（含 `tsc --noEmit`） | 通过（JS 250.05 kB / gzip 78.44 kB） |

## 3. 实测中发现并修复的问题

1. **出站口径错误**：`metrics_report.py` 第一版把本地轨道命中的片段也算进"出站"，得出"本地轨道出站 21 382 字符"，与隐私承诺矛盾。回执本身是对的（`enabled=false`），错的是统计口径；改为按 `egress.enabled` 过滤，并把"未出站片段"单列。
2. **拒答标记识别不全**：本地 4B 模型把「资料中未找到足够依据」说成「资料中未找到关于…的具体步骤」，`is_refusal` 认不出，前端不显示"资料不足"，评测里也算误答。改为匹配更短前缀「资料中未找到」，新增 6 个单测（含"标记出现在句中不算拒答"）。
3. **评测索引污染**：把 `examples/formats` 索引进同一集合后，40 题的 Recall@5 会静默变差。加入索引范围守卫：索引里出现语料外文档就直接退出。
4. **MRR 漂移不是回归**：重建集合后 hybrid 的 MRR 从 0.806 变成 0.772，而 Recall@5 完全一致。原因是 HNSW 近似检索的排序依赖索引构建；已写进 ADR-0009，避免把抖动当成算法变好/变差。

## 4. 已知限制与风险

1. **PDF 每页强制分块**：换来引用永远单页，代价是跨页段落被切断。`page_end` 列已预留，若评测需要改为允许跨页，只改分块、不改表。
2. **容器路径更慢**：CPU 重排 2.2 s，云端首 token 3.1 s，超出 2 s 验收线。容器定位是"能分发、能跑"，性能路径是宿主机 venv。
3. **不做 OCR**：扫描版 PDF 只能显式失败。
4. **代码切分靠正则**：宏生成的函数、没有顶层声明后缀的脚本会退化为按空行分块。
5. **`ordinal` 依赖重建**：阶段 5 之前入库的分块 payload 没有该字段；只影响"第 K 块"（目前只在 PDF 引用中出现），`--force` 重建后一致。
6. **评测样本量小**：35 个有答案回答的引用精确率在 85%～88% 之间波动（阶段 2 单次为 89.2%），单次差异不足以支撑"哪个轨道更准"的结论。
7. **指标报告的 p95 受开发环境影响**：包含评测并发期间的慢轮次，不能当基线。

## 5. 复现与走查步骤

```bash
# 0. 前置：Qdrant 容器、Ollama（bge-m3 / qwen3.5:4b）、.env 中的 DEEPSEEK_API_KEY
docker start localmind-qdrant
ollama ps

# 1. 后端（启动时自动迁移）
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
cd frontend && npm run dev        # http://127.0.0.1:5173

# 2. 质量门禁
.venv/Scripts/python.exe -m pytest backend/tests
.venv/Scripts/python.exe -m ruff check backend scripts && .venv/Scripts/python.exe -m mypy
cd frontend && npm run build

# 3. 评测与指标
.venv/Scripts/python.exe scripts/eval_retrieval.py                     # 离线，不联网
.venv/Scripts/python.exe scripts/eval_stage5.py --tracks cloud local   # 双轨 + 记忆场景
.venv/Scripts/python.exe scripts/metrics_report.py --out docs/eval/stage5-metrics.md

# 4. 容器
HF_CACHE=D:/dev/models/huggingface NOTES_DIR=D:/notes docker compose up --build
```

浏览器走查清单（作者执行）：

1. 左侧加入 `examples/formats`（或容器里的 `/notes/formats`）为资料源并扫描；引用栏应能看到 `localmind-runbook.pdf`。
2. 提问「备份要备份哪些东西？」：引用卡片显示「第 1 页 · 第 1 块」，回答里的引用编号可点。
3. 提问 `delivery-checklist.docx` 里的内容：引用显示行号 + 标题路径（如「知识库交付清单 > 一、资料入库」）。
4. 把一个代码目录加为资料源：引用显示函数签名作为小节名。
5. 上传（或用脚本生成）一个扫描版 PDF：扫描结果里出现"没有可提取文本"的错误，不产生空文档。
6. 换台机器（或删掉 `data/localmind.db` 与 Qdrant 卷）后按 README 走一遍，确认迁移与重建索引都正常。