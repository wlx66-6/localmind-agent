# 阶段 4 验收记录 · 长期记忆

- **日期**：2026-10-25
- **状态**：实现与自测完成，**待项目作者验收**（含浏览器走查）
- **范围**：记忆抽取、会话摘要、冲突覆盖、时间语义、召回注入、出站回执计数、记忆面板与 API
- **目标回顾**："我之前说过什么"类问题能召回并给出时间；新事实能覆盖旧事实且保留历史。

## 1. 实现清单

| 模块 | 内容 |
| --- | --- |
| 数据模型 | `models.Memory`：`kind / key / content / occurred_at / created_at / source_* / status / superseded_by / embedding_ok`，`(key, status)` 联合索引；`Conversation.summarized_user_messages` 记录摘要进度 |
| 迁移 | `db.py`：沿用 `PRAGMA table_info` + `ALTER TABLE ADD COLUMN`；新表由 `create_all` 建 |
| 向量存储 | `memory/store.py`：集合 `localmind_memories`，命名向量 `dense`（bge-m3，1024 维），`upsert / search / delete / count / health` |
| 抽取 | `memory/extractor.py`：中文 prompt（今天 / 已有 kind+key / 用户发言）、`temperature=0.0`、围栏 JSON 解析、逐条校验丢弃、未来事件日期丢弃、按 key 去重 |
| 摘要 | 同一模块：合并"之前的摘要 + 最近 12 条消息"，输出 ≤200 字 |
| 编排 | `memory/service.py`：`TaskRunner` 后台任务；按会话 `asyncio.Lock`；同 key 覆盖；摘要边界条件更新抢占/失败归还；失败写 `metrics.memory` |
| 召回 | `memory/retrieval.py`：向量召回 2 倍候选 → SQLite 过滤 active → top-k；任何异常返回空 |
| 注入 | `generation/prompt.py`：`【记忆】` 独立段落、带日期与类型、禁止 `[n]`、文档优先；有记忆时解除第一层闸门 |
| 出站 | `services/egress.py`：`memories` / `memory_chars` 计入回执与 `chars` |
| API | `GET /api/memories?include_superseded=&limit=`、`DELETE /api/memories/{id}` |
| 前端 | `MemoryPanel`（列表 / 显示已替换 / 删除 / 刷新）、回答「记忆 N」徽标、引用栏「召回的记忆」分区、回执记忆计数 |
| 时间 | `timeutil.py`：SQLite naive UTC → 带 `+00:00` 的 API 时间戳与本地日期展示 |
| 文档 | ADR-0011、README、`.env.example`、本记录 |

明确未做（按阶段计划）：文档实体/事件时间线、待办与提醒、记忆编辑、按 key 精确查找、自动过期、批量清理。

## 2. 验收证据

### 2.1 一轮陈述抽取多条记忆

用户发言：「记一下：我住杭州，平时主要用 Python 写后端，偏好简洁的回答。」（云端轨道）

抽取耗时 1.27 s，写入 3 条，全部 `embedding_ok=true`：

| kind | key | content |
| --- | --- | --- |
| fact | 所在城市 | 用户住在杭州。 |
| preference | 技术栈偏好 | 用户平时主要用 Python 写后端。 |
| preference | 回答风格偏好 | 用户偏好简洁的回答。 |

同一轮 `metrics_json.memory` = `{"extraction": {"status": "stored", "stored": 3, "superseded": 0, "ms": 1273.11}, "summary": {"status": "skipped"}}`。

### 2.2 跨会话召回并给出时间

新会话提问「你还记得我住哪吗？」：

```
memories: [{"id": 1, "kind": "fact", "key": "所在城市", "content": "用户住在杭州。",
            "occurred_at": null, "created_at": "2026-10-24T18:42:43+00:00", "score": 0.632}]
egress  : {"track": "cloud", "enabled": true, "chunks": 5, "chars": 686,
           "memories": 1, "memory_chars": 7, ...}
回答    : 你之前说过你住在杭州（2026-10-25 的对话）。
```

- 时间是**本地日期**：记录发生在 UTC 18:42，即本地 10-25 凌晨。第一次实现直接截取 UTC 日期，模型回答"2026-10-24"，与用户感知不符；改为 `timeutil.local_date` 后修正。这条同时验证了"记忆不带 `[n]` 编号、与资料片段分开表达"。

### 2.3 同 key 覆盖且保留历史

先说「平时主要用 Python 写后端」，再说「我换工作了，现在主要用 Go 写后端」。

`GET /api/memories?include_superseded=true`：

```
 4 preference active     技术栈偏好   现在主要用 Go 写后端。        superseded_by=None
 3 preference active     回答风格偏好 用户偏好简洁的回答。          superseded_by=None
 2 preference superseded 技术栈偏好   用户平时主要用 Python 写后端。 superseded_by=4
 1 fact       active     所在城市     用户住在杭州。                superseded_by=None
```

- 向量集合只剩 1 条（旧向量被删除），新会话提问"我现在后端主要用什么语言"只召回 Go（0.806），回答"你之前提到过，现在主要用 Go 写后端（2026-10-25）"。
- 删除新行时会把旧行的 `superseded_by` 置空（单测 `test_delete_removes_the_memory_and_clears_the_pointer`）。

### 2.4 实测发现并修复：kind 漂移导致记忆分叉

第一版覆盖条件是 `(kind, key)`。实测中抽取器把同一个属性「技术栈偏好」一次标成 `preference`、一次标成 `fact`，于是两条都保持 active，第三轮提问时模型回答：

> 你之前提到过两种说法，彼此有冲突：2026-10-24：主要用 Go 写后端；2026-10-24：平时主要用 Python 写后端。这两条记忆同一天记录，但内容矛盾，我无法判断哪个是当前准确的。

修复：**key 是唯一身份**（覆盖与轮内去重都只看 key），同时把已有 `(kind, key)` 一起喂给抽取器，让它沿用原 kind。修复后同样的两轮输入只剩一条 active（单测 `test_kind_flip_on_the_same_key_still_supersedes`），面板标签也不再漂移。

### 2.5 实测发现并修复：摘要同一边界生成两次

第一版用 `count % 6 == 0` 触发摘要。实测一个 6 轮会话写了两条同 key 摘要（id 5 被 id 6 覆盖）：相邻两轮的后台任务重叠，两个任务都读到 `count=6`。

修复：把"已摘要到第几条用户消息"写进 `conversations.summarized_user_messages`，用 `UPDATE ... WHERE summarized_user_messages = last` 抢占；生成失败（超时/异常/向量写失败）时把边界还回去，下一轮重试；空摘要视为边界已处理。修复后同样 6 轮会话恰好 1 条 `summary:<cid>`（单测 `test_summary_is_not_generated_twice_for_the_same_boundary`、`test_failed_summary_hands_the_boundary_back`）。

### 2.6 滚动摘要内容

6 轮问完后写入 `summary:49`（节选）：

> 2026-10-25 讨论了 RAG 实现细节：分块中文技术笔记常用 400–800 字，重叠仅在强制切分时使用，参数靠评测集确定；RRF 只用排名，k 常取 60，Qdrant 用 query_points 预取融合，每路约 20 候选……

摘要属于 `kind=summary`、`key=summary:<conversation_id>`，同样参与向量召回。

### 2.7 本地轨道：记忆可用且零出站

`track=local` 提问"我主要用什么语言写后端？"：

```
recalled  : [('preference', '技术栈偏好', 0.8)]
egress    : {"track": "local", "enabled": false, "chunks": 5, "chars": 757,
             "memories": 1, "memory_chars": 13, ...}
首 token  : 3492.19 ms      总耗时: 3829.27 ms
回答      : 根据您之前的对话记录，您目前主要使用 **Go** 语言编写后端代码
metrics   : extraction {"status": "empty", "track": "local", "ms": 554.42}
```

抽取在本机完成（554 ms），没有新增任何外部请求。

`scripts/verify_offline.py` 在阶段 4 代码下重跑仍 PASS：

```
track        : local
egress       : enabled=False chunks=5
首 token     : 8114.17 ms（首次，模型冷加载）
观测到连接   : ['127.0.0.1:11434', ...]
非环回连接   : 无
```

紧接着再跑一次（模型已驻留）：首 token **4588.91 ms**，总耗时 6431.53 ms。冷启动包含约 4 s 的模型加载，见已知限制 1。

### 2.8 召回阈值抽样

`MEMORY_MIN_SCORE=0.5`，同一批记忆上的实测分数：

| 提问 | 召回 | 最高分 |
| --- | --- | --- |
| 我平时用什么编程语言写后端？ | 技术栈偏好 | 0.804 |
| 我现在后端主要用什么语言？ | 技术栈偏好（Go） | 0.806 |
| 你知道我住哪吗？ | 所在城市 | 0.632 |
| 你记得我住哪、用什么语言吗？ | 所在城市（语言那条未召回） | 0.539 |
| 分块策略有哪些取舍？ | 两条内容相关的会话摘要 | 0.586 |
| 我们公司明年的预算是多少？ | 无 | — |
| 我家猫叫什么名字？ | 无 | — |

0.5 的取舍：相关召回落在 0.54～0.81，无关问题为 0；"一句话问两件事"时只召回其中一件，是单向量查询的固有局限（见已知限制 3）。

### 2.9 质量门禁

| 检查 | 结果 |
| --- | --- |
| `ruff check backend` | 通过 |
| `ruff format --check backend` | 53 个文件已格式化 |
| `mypy` | 53 个源文件无问题 |
| `pytest backend/tests` | **92 passed**（阶段 4 新增 44 个：抽取 10、服务 12、召回 4、API 4、时区 5、回执与 SSE 注入等） |
| `tsc --noEmit`（`npm run build`） | 通过 |
| `vite build` | 通过（JS 249.85 kB / gzip 78.36 kB） |
| `scripts/verify_offline.py --pid <pid>` | PASS（零出站） |

`test_db_migration.py` 额外断言：阶段 1 的旧库经 `create_all()` 后 `memories` 表与 `conversations.summarized_user_messages` 列都存在，旧行得到默认值 0。

## 3. 已知限制与风险

1. **本地轨道冷启动超验收线**：阶段 3 记录本地首 token 2.71 s，本次冷启动 8.11 s、模型驻留后 4.59 s。差值是 Ollama 加载 3.1 GB 模型的时间，不是记忆链路引入的。若演示要求稳定 < 5s，需要在演示前先问一句预热，或给 Ollama 设 `keep_alive`。
2. **抽取质量依赖模型**：本地 4B 模型偶尔输出不合规 JSON，`parse_drafts` 逐条丢弃而不是整体失败——代价是漏记，不会污染回答。
3. **单条消息一次向量查询**：一句话里问两件事时可能只召回其中一件；按 key 精确查找未实现。
4. **记忆没有过期与批量清理**：面板只能逐条删除，说得太多会积累噪音。
5. **摘要是"模型总结的对话"**：可能压缩掉细节或写进助手自己的措辞；ADR 里已说明它只用于会话级回忆，不参与文档引用。
6. **面板刷新有 4 秒延迟**：抽取在后台跑，`done` 之后才完成；保留手动「刷新」按钮，没有做轮询。
7. **非记忆时间戳仍按 UTC 显示**：资料源"最近扫描时间"等沿用阶段 2 的 naive UTC 展示，存在最多 8 小时偏移，留到阶段 5 统一处理。

## 4. 复现与走查步骤

```bash
# 1. 依赖：Qdrant + Ollama（bge-m3、qwen3.5:4b）
docker start localmind-qdrant
ollama ps

# 2. 后端（新配置生效需重启）
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# 3. 前端
cd frontend && npm run dev      # http://127.0.0.1:5173

# 4. 质量门禁
.venv/Scripts/python.exe -m pytest backend/tests
.venv/Scripts/python.exe -m ruff check backend && .venv/Scripts/python.exe -m mypy
cd frontend && npm run typecheck && npm run build
```

浏览器走查清单（作者执行）：

1. 说一句带个人信息的陈述，等 4 秒左右：左侧「长期记忆」出现对应条目，回答顶部出现「记忆 N」。
2. 点「记忆 N」：右侧栏出现「召回的记忆」，卡片显示类型、key、日期与相似度；资料片段另起一段。
3. 点「新会话」，问"我之前告诉你我住哪"：回答给出内容与日期，回执显示"含 1 条记忆（N 字符）"。
4. 更新同一属性（例如换语言）：面板里旧条目消失，勾选「显示已被替换的旧值」后以虚线、删除线显示；向量栏只召回新值。
5. 删除一条记忆：面板刷新后消失，再问相关问题不再引用它。
6. 切到「本地 · 零出站」问一个记忆类问题：回答可用，回执显示零出站，记忆条数照常计数。