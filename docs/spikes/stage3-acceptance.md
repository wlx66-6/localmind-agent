# 阶段 3 验收记录 · 双轨与隐私

- **日期**：2026-10-25
- **状态**：**已验收**（2026-10-25，项目作者本机跑通离线验证）
- **验收结论**：问题 6（本地轨道评测集）并入阶段 5；问题 7（发送前预览确认）不做
- **范围**：Provider 双轨抽象、Ollama 本地生成、按会话轨道切换、按资料源出站黑名单、每轮「仅检索」开关、出站回执、延迟与出站埋点、离线验证脚本

## 1. 实现清单

| 模块 | 内容 |
| --- | --- |
| 双轨协议 | `generation/base.py`：`ChatProvider` 协议（`name` / `model` / `available` / `stream_chat`）；SSE 事件结构与单轨时完全一致 |
| 云端实现 | `generation/deepseek.py`：补 `model` 与 `available()`，行为不变 |
| 本地实现 | `generation/ollama.py`：`/api/chat` 流式、`think: false`、NDJSON 逐行解析、可注入 transport 便于测试 |
| 轨道解析 | `api/chat.py`：请求 `track` > 会话 `track` > `DEFAULT_TRACK`；切换写入会话记录，不重启进程 |
| 重排分流 | `main.py`：每个轨道一个 `CrossEncoderReranker`；云端 `RERANK_DEVICE=auto`，本地 `RERANK_DEVICE_LOCAL=cpu`；启动时后台预热 |
| 出站策略 | `services/egress.py`：按 `document_id` 剔除黑名单来源、引用编号重排、出站回执（chunks/files/chars/blocked）；全拦时返回专用文案 |
| 黑名单入口 | `PATCH /api/sources/{id}`、左侧资料源开关；服务端执行，前端只是入口 |
| 埋点 | `citations.egress`、`done.ttft_ms/total_ms/track/skipped/blocked`、`messages.metrics_json`（轨道、模型、延迟、检索耗时、重排设备、出站回执） |
| 迁移 | `db.py`：启动时 `PRAGMA table_info` + `ALTER TABLE ADD COLUMN`，幂等；Alembic 留到阶段 5 |
| 前端 | 顶栏轨道开关、消息上的轨道/延迟/仅检索徽标、右侧出站回执、输入框「本次不上传（仅检索）」、资料源出站开关 |
| 验证 | `scripts/verify_offline.py`：协议断言 + 采样后端进程 TCP 连接，要求远端全为环回 |
| 文档 | ADR-0010、README、`.env.example`、本记录 |

明确未做（按阶段计划）：多设备、账号、公网、复杂权限；发送前"预览片段再确认"的两段式请求；按片段（而非按来源）的出站白名单。

## 2. 验收证据

### 2.1 本地轨道完成带引用的问答（断网可行）

问题：`混合检索为什么要结合 BM25？`，`track=local`，回答节选：

> 混合检索结合 BM25 主要是为了弥补单一检索方法的盲区，实现更稳定的结果：
> 1. **互补优势**：稠密检索擅长处理同义改写和概念关联……但可能漏掉精确字符串 [2]；而稀疏检索（典型代表是 BM25）擅长精确匹配专有名词、编号、错误码和罕见术语 [2]。
> 2. **克服各自缺陷**：……将两路结果融合，通常比任何单路都稳定 [2]。
> 3. **工程实践**：……每路各取 20 个候选后融合（如使用 RRF 算法）[3][5]。

同轮 `done`：`track=local, cited=[2,1,3,5], refused=false, ttft_ms=2712, total_ms=6028`；`citations.egress = {track: local, enabled: false, chunks: 5, files: 2}`。

### 2.2 本地轨道零出站（脚本证据）

```bash
$ python scripts/verify_offline.py --pid <backend-pid>
track        : local
egress       : enabled=False chunks=5
首 token     : 2538.9 ms
总耗时       : 4731.99 ms（墙钟 4966.5 ms）
rerank 设备  : cpu
观测到连接   : ['127.0.0.1:11434', '127.0.0.1:58641', '127.0.0.1:58642', '127.0.0.1:58727']
非环回连接   : 无
PASS: 本地轨道完成问答，出站回执为零，未观测到非环回连接。
```

`127.0.0.1:11434` 是 Ollama（嵌入 + 生成），另外三条是后端与测试客户端之间的环回连接。

### 2.3 每轮「仅检索」开关

`skip_egress=true` 时：检索与重排照常执行，返回引用片段，**不调用任何生成模型**，回答固定为"本次只做了检索，没有上传任何内容"，`done.skipped=true`，`egress.enabled=false`，总耗时 1.56s（只有检索）。开关每轮重置，不会跨轮记忆。

### 2.4 按资料源禁止出站

把唯一资料源置为 `egress_allowed=false` 后提问（云端轨道）：

- `citations.items = []`，`egress.blocked_chunks=5, blocked_files=2`
- 回答为专用文案"命中的片段全部来自标记为「禁止出站」的资料源……"，`done.blocked=true, refused=true`，DeepSeek 未被调用。
- 混合场景（部分允许）由单测覆盖：被拦片段从上下文中消失，引用编号重排为连续 `[1] [2]`。

### 2.5 轨道切换与延迟（每轨道 3 次，同一问题集）

| 指标 | 云端轨道 | 本地轨道 |
| --- | --- | --- |
| 首 token p50 / max | **1644 / 1666 ms**（本地模型驻留）/ 1353 ms（卸载后） | **2710 / 2795 ms** |
| 总耗时 p50 | 2997 ms | 5603 ms |
| 重排 p50（设备） | 745 ms（CUDA，有争用）/ 367 ms（无争用） | 2128 ms（CPU） |
| 出站 | 5 段 / 2 文件 / 918 字符 | 0 |

- 验收线：云端首 token < 2s、本地 < 5s，均达标。
- 轨道切换在同一个后端进程内完成，无需重启；会话内切换后服务端记住选择（单测 `test_track_switch_is_remembered_per_conversation`）。
- GPU 争用是实测现象，不是估算：`ollama ps` 显示 qwen3.5:4b 占 3.14GB、bge-m3 占 0.69GB，与 CUDA 重排共处 8GB 显存时重排变慢一倍；卸载本地模型后恢复。这也是本地轨道把重排放到 CPU 的直接理由。

### 2.6 人工断网演示步骤

1. 先用本地轨道问一次，确认正常（模型已在本机，无需预热）。
2. 断开网络：Windows 设置里关掉 Wi-Fi / 以太网，或在防火墙把后端进程的出站规则全部禁止。
3. 同一问题再问一次：回答与引用完全不变，右侧回执显示「本地生成 · 零出站」。
4. 切回云端轨道提问：前端出现"无法连接 DeepSeek"错误——这一步同时证明云端路径确实依赖外网，本地轨道的成功不是"网络碰巧还通"。
5. 恢复网络。

### 2.7 质量门禁

| 检查 | 结果 |
| --- | --- |
| `ruff check` / `ruff format --check` | 通过 / 77 个文件已格式化 |
| `mypy` | 40 个源文件无问题 |
| `pytest` | **48 passed**（新增 18 个：轨道、开关、黑名单、指标、Ollama 解析、迁移、出站策略） |
| `tsc --noEmit` | 通过 |
| `vite build` | 通过（JS 243.48 kB / gzip 76.90 kB） |

其中 `test_local_track_never_calls_cloud_provider` 用一个"一旦被调用就失败的假云端 provider"断言本地轨道不会触达云端；`test_db_migration.py` 用阶段 2 的旧 schema 建库后调用 `create_all()`，验证列被补齐且旧数据可用。

## 3. 已知限制与风险

1. **本地回答质量低于云端**：qwen3.5:4b 会啰嗦，实测出现过把"稠密检索擅长同义改写"说成"无法理解同义表达"这类与原意相反的句子。引用编号仍由协议解析，但事实正确性不能只靠本地模型。演示时应主动说明。
2. **GPU 争用**：本地模型驻留时云端重排从 0.37s 升到 0.75s。可选缓解：给 Ollama 设 `keep_alive` 缩短驻留，或为云端轨道关闭重排（会牺牲引用质量）。
3. **黑名单粒度为资料源**：同一目录里既想共享又不想出站的文件无法区分。若需要片段级策略，得在 ingestion 阶段加标记。
4. **「不上传」是"先勾再发"，不是发送前确认**：用户在勾选时还看不到本轮会命中哪些片段；回执是事后告知。真正的发送前预览需要两段式请求，判定为收益不足。
5. **本地轨道没有云端那样的评测集验证**：阶段 2 的 40 题评测只覆盖云端生成。本地轨道的引用质量目前只有抽查，没有量化指标。
6. **两个重排实例**：启动时两个实例都会在没有请求的情况下预热，CPU 副本占用约 2.3GB 内存。若内存紧张，应改为按需加载（当前选择预热是为了让本地首 token 稳定 < 5s）。

## 4. 复现步骤

```bash
# 1. 依赖：Qdrant + Ollama（bge-m3、qwen3.5:4b）
docker start localmind-qdrant
ollama ps

# 2. 后端（新配置生效需重启）
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# 3. 前端
cd frontend && npm run dev

# 4. 离线验证（--pid 从 netstat 取）
netstat -ano | findstr :8000
.venv/Scripts/python.exe scripts/verify_offline.py --pid <pid>

# 5. 质量门禁
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m mypy
cd frontend && npm run typecheck && npm run build
```