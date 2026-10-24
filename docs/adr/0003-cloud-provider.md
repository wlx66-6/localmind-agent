# 0003 · 云端生成 Provider 抽象

- **状态**：已接受
- **日期**：2026-10-24
- **决策者**：项目作者 + 协作代理

## 背景

云端模式默认开启，质量优先；本地模式要求断网可用、零出站。两者必须在同一套问答流程中随时切换，不能锁死到某一家 API。

## 决策

- 定义内部 `ChatProvider` 接口：`stream_chat(messages, options) -> AsyncIterator[token]`，附带 `name`、`is_local`、`supports_streaming` 等能力声明。
- 默认云端实现：**DeepSeek**（OpenAI 兼容 `/chat/completions`）。
- 通过 OpenAI 兼容适配器，可插拔接入 Claude（Anthropic 原生适配）、OpenAI、阿里云百炼/Qwen。
- 本地实现：Ollama `/api/chat`，`think: false`。
- 所有 provider 共享：超时、重试、错误归一化、token/延迟埋点。
- 出站策略在 provider 之前独立成层：检索片段 → 出站预览 → 用户确认/关闭/来源黑名单 → 才调用云端。

## 备选与取舍

| 方案 | 取舍 |
| --- | --- |
| 直接调用各家 SDK | 最简单，但业务代码与供应商耦合，双轨切换要改多处；不利于"随时可切"的叙事。 |
| LangChain / LlamaIndex | 抽象齐全，但依赖重、行为不透明，面试追问时难以讲清每一层；与"小而深"目标冲突。 |
| LiteLLM | 统一多家 API，省适配工作；但多一个运行时依赖与代理层，延迟/错误语义不透明。可作为后期备选。 |
| **自建薄抽象（选择）** | 代码量可控，双轨切换、出站控制、埋点都能讲清；代价是要自己写 2～3 个适配器。 |

## 后果

- API 密钥只从环境变量读取，`.env` 不入库；`.env.example` 记录变量名。
- 云端调用前必须经过出站策略层；本地 provider 不经过网络。
- 供应商差异（流式格式、错误码、token 统计）在适配器内消化，上层只看到统一事件。