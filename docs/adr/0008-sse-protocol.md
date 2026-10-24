# 0008 · 对话接口：POST + SSE 流式协议

- **状态**：已接受（阶段 1）
- **日期**：2026-10-25
- **决策者**：项目作者 + 协作代理

## 背景

聊天需要：先返回"命中了哪些片段"，再逐 token 输出回答，最后告知哪些引用被真正使用。用户问题可能较长，不适合放在 URL 查询参数里。浏览器原生的 `EventSource` 只支持 GET，且不能带请求体。

## 决策

`POST /api/chat` 返回 `text/event-stream`，事件顺序固定：

| 事件 | 数据 | 时机 |
| --- | --- | --- |
| `citations` | `conversation_id` + 全部召回片段（编号、路径、标题、行号、分数、原文） | 检索完成、生成开始前 |
| `token` | `{ "text": "..." }` | 每收到一个增量 |
| `done` | `message_id`、`conversation_id`、`cited`（被引用的编号）、`refused` | 正常结束 |
| `error` | `{ "message": "..." }` | 云端未配置、provider 报错、生成异常 |

规则：

- **拒答分两层**：检索闸门（top1 rerank 分数低于 `retrieval_gate_rerank_score`）触发时，不调用云端，直接 `token` 输出固定拒答文案 + `done(refused=true)`；灰区问题会进入生成，模型按 prompt 输出“资料中未找到足够依据”时，`done(refused=true)` 且 `cited` 保留实际引用，前端展示“资料不足”标记。
- **检索明细随 `citations` 返回**：`retrieval` 字段包含模式、重排是否降级、各阶段耗时；每个片段带 `dense_rank / sparse_rank / fused_rank / dense_score / rerank_score`，前端引用卡片据此显示“稠密 #2 · 稀疏 #5 · 融合 #3 · 重排 0.82”。
- **引用先于回答**：前端先拿到候选片段，用户可在等待生成时查看依据；生成结束后 `cited` 标记哪些真的被引用。
- **前端不用 EventSource**：用 `fetch` + `ReadableStream` 手工解析 SSE 帧（按 `\n\n` 分帧，`event:` / `data:` 行）。
- **错误也是 SSE 事件**：流一旦开始，HTTP 状态码已发出，后续错误只能通过 `error` 事件表达；非流式错误（参数非法、会话不存在、检索服务不可用）仍用普通 4xx/5xx + JSON `detail`。

## 备选与取舍

| 方案 | 取舍 |
| --- | --- |
| **POST + SSE（选择）** | 单向推送、实现简单、可被 curl 调试、易加事件类型；代理无需特殊配置。代价是要手写解析，且不能复用 EventSource。 |
| WebSocket | 双向通信；当前没有客户端持续发消息的需求，握手与重连逻辑是额外复杂度。 |
| 轮询任务状态（提交任务 → 查进度） | 兼容性最好；延迟与请求数都更差，流式体验消失。 |
| 先返回 JSON，前端假流式 | 实现最省事；首 token 时间指标失真，无法体现本地推理性能，面试叙事也站不住。 |

## 后果

- 协议变更（新增事件类型）需要同步前端 `api.ts` 与本文档；事件名是前后端契约。
- 事件流中 `citations` 的 payload 与 Qdrant 检索结果一一对应，阶段 2 加入 rerank 后只需在发送前替换列表，协议不变。
- 生成失败时用户消息已落库，便于重试同一会话；`cloud_egress` 结构化日志记录本次发送的片段数与字符数，作为隐私审计依据。