# 0006 · 存储分工：SQLite 管元数据，Qdrant 管向量

- **状态**：已接受（阶段 1）
- **日期**：2026-10-25
- **决策者**：项目作者 + 协作代理

## 背景

摄取流程要记录资料源、文档、分块、对话与消息；检索流程要在高维向量上做近邻搜索，并按 `document_id` 过滤删除。两类访问模式不同：元数据需要事务、外键、增量更新；向量需要 HNSW 索引与批量写入。单机、单人、1k～5k 文档的规模，不希望引入需要长期运维的数据库服务。

## 决策

| 数据 | 存储 | 理由 |
| --- | --- | --- |
| 资料源、文档、分块、会话、消息 | SQLite（SQLAlchemy 2.0） | 零运维、单文件、事务与外键齐全；元数据量小 |
| dense 向量 + 引用 payload | Qdrant | 原生 HNSW、payload 过滤、批量 upsert；阶段 2 还要加 sparse 向量与 RRF |
| 两边的关联 | `Chunk.id`（内容哈希）↔ `Chunk.qdrant_id`（uuid5） | 稳定 ID，重建索引不产生重复点 |

引用所需的信息（`path`、`heading`、`start_line`、`end_line`、`text`）在 Qdrant payload 中冗余一份，检索后无需回表即可展示引用；SQLite 中保留同样的字段，用于重建与审计。

**写入顺序**：先把新向量 upsert 到 Qdrant，再按 `document_id` 删除旧点（保留本次新点），最后在 SQLite 中替换分块记录并提交。删除文件时先删 Qdrant 点，再删 SQLite 行。

## 备选与取舍

| 方案 | 取舍 |
| --- | --- |
| **SQLite + Qdrant（选择）** | 元数据与向量各用擅长的引擎；都可本机跑；Qdrant 的 sparse/过滤能力直接支撑阶段 2 的混合检索。代价是两套存储，需要一致性策略。 |
| 只用 SQLite + sqlite-vec | 单一存储、事务天然一致；但向量索引能力弱，混合检索、payload 过滤要自己实现，面试可讲的内容反而变少。 |
| 只用 Qdrant（元数据也进 payload） | 少一套存储；但会话、消息、增量扫描状态用 Qdrant 表达很别扭，缺少事务。 |
| Postgres + pgvector | 工业界常见、SQL 与向量统一；需要常驻数据库服务，对单人单机项目运维成本偏高。 |

## 后果

- 双写带来一致性问题：向量写入成功、SQLite 提交失败时会留下**孤儿向量**。当前通过"按 `document_id` 全量替换"降低影响，并在阶段 5 增加对账/清理命令。
- SQLite 只适合单进程写入。当前服务是单实例 uvicorn，可接受；若将来多 worker，需要换 Postgres 或串行化写入。
- 备份需要同时复制 `data/localmind.db` 与 Qdrant volume；README 中说明。
- 数据库文件在 `.gitignore` 中排除，仓库不包含任何索引数据。