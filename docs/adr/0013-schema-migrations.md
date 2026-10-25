# 0013 · 迁移策略：没有版本表的旧库怎么接到 Alembic 上

- **状态**：已接受（阶段 5，2026-10-25）
- **日期**：2026-10-25
- **决策者**：项目作者 + 协作代理

## 背景

阶段 1～4 的 schema 变更靠两段手写代码完成：

- `Base.metadata.create_all()` 建新表；
- `_ADDED_COLUMNS` 里的 `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` 给旧表补列。

这在"只有一个开发者的本地库"上是够用的，但它有三个说不清的地方：

1. **没有顺序**：所有补列每次启动都跑一遍，靠"列是否已存在"保证幂等，变更之间没有先后关系，将来要做数据回填（backfill）时无从下手。
2. **没有记录**：数据库当前是什么结构，只能靠读代码推断。
3. **不可审计**：面试被问"你们怎么做数据库迁移"时，"if 列不存在就 ALTER"不是一个能展开讲的答案。

同时有一个现实约束：**已经存在的数据库（包括作者本机的 10 文档 / 47 分块 / 10 条记忆）没有 `alembic_version` 表**，无法判断它处于哪个版本。

## 决策

### 1. 基线用"幂等收敛"，不重放历史 DDL

`0001_stage4_baseline` 做两件事，都可以重复执行：

- 缺表就建表 —— DDL 是**冻结**的阶段 4 快照，不读 `Base.metadata`；
- 缺列就补列 —— 复用 `app/schema.py` 的 `LEGACY_COLUMNS`（阶段 3、4 加过的 5 个列）。

于是同一份 revision 对三种输入都成立：

| 输入 | 0001 的效果 |
| --- | --- |
| 全新文件 | 建 6 张表 |
| 阶段 1～2 的库（无 memories、无 track/metrics 列） | 建 memories，补 5 个列 |
| 阶段 4 的库 | 什么都不做，只记录版本号 |

为什么不"先判断 schema 再 stamp"：那需要把"阶段 3 的库长什么样"再写成一份判断代码，判断错了就静默漏列；而幂等收敛的判断标准就是**目标状态本身**，不需要知道来源版本。

为什么不"从阶段 1 的 DDL 开始逐个 revision 重放"：本项目的阶段 3、4 并没有留下对应的 DDL 快照，重放等于凭记忆重建历史，比收敛更容易出错。

### 2. 从 0002 起是普通的前向 revision

`0002_stage5_parsing_fields` 只做一件事：加 `chunks.page`、`chunks.page_end`、`documents.page_count`。加列统一走 `add_missing_columns()`，原因是 SQLite 没有 `ADD COLUMN IF NOT EXISTS`——这不是"防御性编程"，而是方言限制。

### 3. 启动时迁移，而不是单独一个部署步骤

`Database.migrate()` 在 FastAPI lifespan 里执行 `upgrade head`，失败直接终止启动。对本地单用户应用来说，这比"记得先跑 alembic"可靠：带着半旧 schema 启动只会产生更难排查的错误。

有三个配套细节：

- **不复用应用自己的 engine**：迁移用一个短生命周期的连接，结束后立即 `dispose`，不在应用要用的连接池上留事务。
- **不调用 `fileConfig`**：应用已经在 root logger 上装了 JSON formatter，alembic.ini 的 logging 段会把它们替换掉，启动日志会变成两种格式。CLI 手动执行时仍然加载 ini 的日志配置，靠一个 `attributes["app_logging_configured"]` 标记区分。
- **CLI 与程序入口读同一个路径**：`env.py` 的 URL 来自 `Settings.sqlite_path`，可用 `LOCALMIND_DB_URL` 覆盖，方便对副本做演练：

  ```bash
  LOCALMIND_DB_URL="sqlite:///D:/code/LocalMind/data/copy.db" \
    .venv/Scripts/python.exe -m alembic -c backend/alembic.ini upgrade head
  ```

### 4. 基线不可回滚

`0001.downgrade()` 直接抛错。回滚基线意味着删掉 `documents` / `chunks` / `memories`，也就是用户的索引与记忆；一个"能删库"的 downgrade 比没有 downgrade 更危险。后续 revision 的 downgrade 正常实现（`0002` 会删掉自己加的三列）。

## 备选与取舍

| 决策点 | 备选 | 为什么没选 |
| --- | --- | --- |
| 旧库接入 | 按 schema 特征判断版本后 `stamp` | 判断逻辑本身要维护，判断错会静默漏列；收敛式基线没有这个问题 |
| 基线内容 | `Base.metadata.create_all()` | 随模型漂移：将来新增的列会被"历史 revision"提前建出来，后续 revision 反而要写成幂等 |
| 基线内容 | 重放阶段 1～4 的 DDL | 历史 DDL 没有完整保存，重放等于凭记忆重建 |
| 执行时机 | 部署时单独执行 `alembic upgrade` | 本地应用没有部署流程；启动即迁移能让"结构不匹配"第一时间暴露 |
| 迁移工具 | 继续用 `create_all` + ALTER | 无顺序、无记录，数据回填类变更无法表达 |
| 迁移工具 | 手写 `schema_version` 表 | 等于自己实现一次 Alembic，收益为零 |

## 实测（2026-10-25）

| 场景 | 结果 |
| --- | --- |
| 本机现有库（10 文档 / 47 分块 / 10 记忆）启动迁移 | `alembic_version = 0002`，三张表行数不变 |
| 副本演练（`LOCALMIND_DB_URL` 指向 backup 文件） | 同上，命令输出 `Running upgrade -> 0001` / `0001 -> 0002` |
| 阶段 1 结构的库（无 memories、无 track/metrics 列） | 单测：迁移后 5 个旧列 + 3 个新列齐全，旧行取到默认值（`metrics_json='{}'`、`summarized_user_messages=0`） |
| 全新库 | 单测：迁移后 6 张表齐全、`alembic_version` 等于脚本 head |
| 连续两次启动 | 单测：第二次是 no-op，不报 `duplicate column name` |

## 后果

- 基线里的 DDL 是冻结的，**不能跟着模型改**；新增字段一律走新的 revision，否则"历史"会被改写。
- 每次 `Database()` 之后的 `migrate()` 都会跑一次 Alembic，测试里每个临时库都要多花几十毫秒；换来的是测试与生产环境用同一条迁移路径。
- 三条以上 revision 之后，可以考虑补一个"从基线到 head 的完整性测试"（新建空库 → 迁移 → 与 `Base.metadata` 比对），目前 0002 还太少，做了收益不明显。
- `_ADDED_COLUMNS` 与 `PRAGMA table_info` 那段代码已经删除：迁移逻辑只有一处，避免两套机制各自演进。