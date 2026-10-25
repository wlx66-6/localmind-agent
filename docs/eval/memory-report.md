# 记忆场景评测

- 生成时间：2026-10-25T11:05:23+00:00
- 结果：**7/7** 通过（走真实后端 HTTP，抽取是后台任务）

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| memory-1 跨会话召回事实 | 通过 | `{"stored": {"id": 29, "kind": "fact", "key": "所在城市", "content": "用户住在杭州。", "occurred_at": null, "created_at": "2026-10-25T11:05:26.636778+00:00", "status": "active", "superseded_by": null, "source_conversation_id": 96, "embedding_ok": true}, "recalled": ["fact/所在城市 0.620"], "answer": "你住在杭州（这是你在 2026-10-25 的对话中提到的）。"}` |
| memory-2 同 key 覆盖旧值 | 通过 | `{"active": ["目前主要使用 Go 语言编写后端代码。"], "history": ["目前主要使用 Go 语言编写后端代码。 (active)", "用户平时主要用 Python 写后端。 (superseded)", "目前主要使用 Go 语言编写后端。 (superseded)", "用户平时主要用 Python 写后端。 (superseded)", "用户平时主要用 Go 写后端。 (superseded)", "目前主要使用 Go 语言编写后端。 (superseded)", "用户平时主要用 Python 写后端。 (superseded)", "目前主要使用 Go 语言编写后端。 (superseded)"…` |
| memory-3 一句问两个属性 | 通过 | `{"recalled": ["技术栈偏好 0.649", "性别 0.572", "所在城市 0.555", "summary:48 0.528", "回答风格偏好 0.515"], "answer": "记得：\n\n- 你住在杭州（2026-10-25 的对话中提到）。\n- 你主要用 Go 语言写后端（2026-10-25 的对话中提到）。"}` |
| memory-4 换个说法问同一属性 | 通过 | `{"recalled": ["技术栈偏好 0.772", "性别 0.535", "summary:48 0.527", "summary:49 0.509"], "answer": "你之前提到过，你平时主要用 Go 语言写后端代码（2026-10-25 的对话）。"}` |
| memory-5 无关问题不召回记忆 | 通过 | `{"memories": [], "refused": true, "answer": "资料中未找到足够依据来回答这个问题。你可以补充相关文档，或换个问法再试。"}` |
| memory-6 事件时间与记录时间分开 | 通过 | `{"stored": {"id": 28, "kind": "fact", "key": "学习Rust时间", "content": "用户于2025年8月开始学习Rust。", "occurred_at": "2025-08-01", "created_at": "2026-10-25T11:04:32.558511+00:00", "status": "active", "superseded_by": null, "source_conversation_id": 94, "embedding_ok": true}}` |
| memory-7 删除该条后不再被召回 | 通过 | `{"deleted": "用户住在杭州。", "recalled": ["性别", "summary:56"], "answer": "你之前提到过你住在杭州（2026-10-25 的对话）。这属于个人记忆，资料中没有相关信息。", "summary_still_mentions_city": true}` |

说明：memory-3 是阶段 6 加入的回归场景——一句话问两个属性时，单条向量查询会把两个意图平均掉，改前实测只召回 1 条（正确答案 0.448 低于 0.5 阈值），改后按子句分别召回并合并。

memory-7 的 `summary_still_mentions_city` 记录的是已知语义缺口：删除一条记忆只删那一行，会话摘要里如果曾提到同样的信息不会被改写。因此回答仍可能说出被删的事实——这一点写在 README 的已知限制里，不做"删除时连带改写摘要"这种脆弱的字符串匹配。
