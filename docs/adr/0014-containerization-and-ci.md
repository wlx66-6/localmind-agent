# 0014 · 容器化与 CI：把"本机能跑"变成"别人也能跑"

- **状态**：已接受（阶段 5，2026-10-25）
- **日期**：2026-10-25
- **决策者**：项目作者 + 协作代理

## 背景

阶段 0～4 的运行方式是一串手动步骤：装 Python、装 CUDA torch、拉起 Qdrant 容器、下载三个模型、设 `HF_HOME`、跑 alembic、起后端、起前端。作者本机跑得很顺，但"新机器按 README 能启动"是阶段 5 的验收标准之一，而这串步骤里有四五个地方会踩坑（CUDA 版本、模型缓存路径、端口占用、前端代理）。

同时有一个反向约束：**本地轨道的价值在于数据不出机**，容器化不能顺手把它破坏掉——比如为了省事把 Ollama 也塞进镜像、或者让容器默认监听 `0.0.0.0` 暴露到局域网。

## 决策

### 1. 三个容器：backend / frontend / Qdrant；Ollama 留在宿主机

| 组件 | 位置 | 理由 |
| --- | --- | --- |
| backend | 容器（CPU torch） | 依赖最复杂（torch、sentence-transformers、jieba） |
| frontend | 容器（nginx） | 构建产物是静态文件，用 nginx 同时反代 `/api` |
| Qdrant | 容器 | 本来就是一个服务 |
| Ollama | **宿主机** | 模型是 3～7 GB 的独立资产，塞进镜像会让镜像不可分发；本地轨道本来就要用宿主机那套模型 |

Ollama 留在宿主机带来一个真实的连接问题：容器里的 `localhost` 是容器自己，必须走 `host.docker.internal`，而 Ollama 默认只监听 `127.0.0.1`。这一点在 README 里写明，并给出两条路（放开监听 / 不用 compose 跑本地轨道），不假装它是零配置。

### 2. 镜像里用 CPU 版 torch

宿主机装的是 CUDA 版（重排跑在显卡上）。容器如果也要求 GPU，就必须让使用者在 Windows + WSL2 上配置 NVIDIA Container Toolkit——把"一键启动"变成"先配半天驱动"。

代价是重排从 ~0.3 s 变成 ~1 s 量级（实测见下），首 token 仍在 2 s 内。README 明说：要性能就用宿主机 venv，要省心就用 compose。

### 3. 重排模型不烤进镜像，只读挂载宿主机缓存

镜像里包含 2.3 GB 的 `bge-reranker-v2-m3` 既慢又大；运行时下载又违反"不隐式联网"的承诺（应用用 `local_files_only=True` 加载）。

做法：`HF_CACHE` 指向宿主机已有的缓存目录，只读挂到容器 `/hf`，`HF_HOME=/hf`。挂载目录不存在时重排降级为融合顺序（`retrieval.rerank_failed=true`），服务照常可用——这与阶段 2 设计的降级路径一致，没有新增分支。

### 4. 前端由 nginx 提供，同时反代 `/api`

开发时 Vite 代理 `/api` 到 8000，生产环境没有 dev server，所以 nginx 承担同一职责：`location /api/ { proxy_pass http://backend:8000; }`。

两个必须写对的细节：

- `proxy_buffering off`：SSE 流式回答经过缓冲会攒到最后一次性吐出，"流式"名存实亡；
- 同源：浏览器只访问 5173，`/api` 由 nginx 转发，因此生产路径下不需要 CORS。后端仍保留 dev 用的 `127.0.0.1:5173` 白名单。

### 5. 所有 published port 绑定 127.0.0.1

`"127.0.0.1:5173:80"` 而不是 `"5173:80"`。容器内部监听 `0.0.0.0`（否则 docker 网络进不来），但宿主侧只看回环。这样"仅本机访问"这条约束在容器化之后依然成立。

### 6. CI 只做"能在 2 分钟内失败"的事

`backend`：ruff（check + format）、mypy、pytest。`frontend`：`tsc --noEmit` + `vite build`。

刻意不做：

| 不做 | 原因 |
| --- | --- |
| 构建 Docker 镜像 | 一个 CPU torch 镜像要拉几百 MB，只验证"能构建"而不发布，收益低；真出问题也是 Dockerfile 而不是代码 |
| 跑评测 | 需要 Qdrant + 模型 + DeepSeek 密钥，属于"发布前的人工步骤"，放 CI 只会变红 |
| 缓存模型 | 测试全部用 fake（`FakeEmbedder`、假 reranker），并设 `HF_HUB_OFFLINE=1`：任何试图下载模型的测试都是 bug，应该失败 |

## 备选与取舍

| 决策点 | 备选 | 为什么没选 |
| --- | --- | --- |
| Ollama 容器化 | 全部进 compose | 镜像不可分发、GPU 透传配置复杂，且本地轨道的意义是"用你能看见的那台机器" |
| torch 版本 | 容器也用 CUDA | 需要 NVIDIA Container Toolkit，Windows 上还必须 WSL2，启动成本高 |
| 前端部署 | vite preview | 不是生产服务器，且没有 `/api` 反代与 SSE 关闭缓冲的能力 |
| 前端部署 | 后端直接托管静态文件 | 少一个容器，但后端要引入静态文件挂载与 SPA 回退，职责混合 |
| 模型 | 打进镜像 | 镜像 2.3 GB+，每次改代码都要重传 |
| 迁移 | compose 里跑一次性 migrate 容器 | 应用启动即迁移（ADR-0013），多一个步骤只会让"忘了跑"更容易发生 |

## 实测（2026-10-25，Windows 11 / Docker Desktop / 8 GB 显卡 16 GB 内存）

| 场景 | 结果 |
| --- | --- |
| 镜像大小 | backend 565 MB，frontend 21 MB（CPU torch） |
| `docker compose up --build` 到健康 | 后端自动迁移新库到 `0002`，`/api/health` 全绿（Ollama / Qdrant / 云端 / 本地模型） |
| 挂载资料目录 | `NOTES_DIR` → `/notes:ro`，扫描 10 篇语料 → 47 分块，嵌入走宿主机 Ollama |
| nginx 反代 `/api` | `/api/health` 经 5173 返回正常；SSE 逐 token 到达（间隔 ~60 ms），确认 `proxy_buffering off` 生效 |
| 容器 → 宿主机 Ollama | Docker Desktop 会转发到宿主机回环，默认 `127.0.0.1:11434` 即可，无需 `OLLAMA_HOST=0.0.0.0` |
| 重排（CPU，20 候选） | 冷启动 5.7 s，稳态 2.2 s |
| 首 token（容器内） | 云端 3.1 s、本地 2.6 s；云端路径**不满足** 2 s 验收线 |
| 内存 | backend 容器约 1.3 GB 常驻（两个重排实例都在 CPU 上） |

结论：容器路径满足"新机器一条命令能跑起来"，不满足"云端首 token < 2 s"。验收记录里把两者分开陈述，不用容器数字去覆盖性能指标。

## 后果

- compose 路径下重排变慢、且本地轨道依赖宿主机 Ollama 的网络可达性；这两点都写在 README 的"已知限制"里。
- 数据落在命名卷（`localmind_data` / `qdrant_storage`），备份方式与宿主机不同：需要 `docker compose cp` 或卷挂载，README 给命令。
- CI 不覆盖"容器能否启动"，所以发布前仍需人工跑一次 `docker compose up`。
- 前端的 `/api` 代理有两个配置来源（Vite 开发、nginx 生产），改后端端口时要同时改两处。