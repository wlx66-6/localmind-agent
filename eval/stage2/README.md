# 阶段 2 评测集

40 道题，针对 `examples/corpus/` 的 10 篇公开笔记。用于比较三条检索链路：

1. 纯向量（阶段 1 基线）
2. 混合检索（dense + 稀疏 BM25 + RRF）
3. 混合检索 + rerank

## 文件

| 文件 | 用途 |
| --- | --- |
| `questions.jsonl` | **机器读取的唯一来源**，一行一题 |
| `questions.md` | 给人看的审阅表，由 JSONL 生成 |

## Schema

```json
{
  "id": "q01",
  "category": "exact | paraphrase | cross_doc | unanswerable",
  "question": "问题原文",
  "gold": [
    {"path": "hybrid-retrieval.md", "heading": "混合检索 > RRF 融合", "quote": "其中 k 通常取 60"}
  ]
}
```

- `path`：相对 `examples/corpus/` 的路径。
- `heading`：分块时的完整标题路径（`A > B`），必须与 chunker 输出一致。
- `quote`：原文中**逐字出现**的片段，用来把 gold 解析成 chunk ID。换行符与空白变化会导致匹配失败。
- `unanswerable` 的 `gold` 为空数组，期望系统拒答。

## 为什么用 quote 而不是 chunk ID

chunk ID 含全文哈希，文档改一个字就全部变化；人工标注无法维护。quote 是人类的标注单位，评测运行时再解析成当前索引里的 chunk ID。

解析规则：在 `<path>` 的 `heading` 小节中找到唯一包含 `quote` 的分块。找不到或命中多个都视为**评测集错误**，脚本直接失败，而不是静默跳过。

## 指标定义

| 指标 | 定义 |
| --- | --- |
| Recall@5 | 最终返回的 top-5 中，至少包含一个 gold 分块的题目比例（仅统计有答案题） |
| MRR@5 | 第一个 gold 分块出现在第 k 位时记 1/k，取平均 |
| 拒答正确率 | 无答案题中被正确拒答的比例 |
| 误拒率 | 有答案题中被拒答的比例 |
| 引用精确率 | 回答引用的分块中，属于该题 gold 集合的比例（回答没有任何引用时按 0 计） |
| 引用召回率 | 该题 gold 集合中被回答引用的比例（多 gold 题仅作参考，不设硬门槛） |
| 首 token 延迟 | 从请求发出到第一个 token（仅生成评测阶段） |

引用精确率的判定是**规则匹配**：引用编号对应的 chunk ID 是否落在 gold 集合里。它不判断"这句话是否真的由该片段支持"，句级语义判断留给人工抽查。

## 运行方式（阶段 2 实现后）

```bash
# 只跑检索指标：不调用云端，秒级
.venv/Scripts/python.exe scripts/eval_retrieval.py --mode retrieval

# 追加生成评测：调用云端，产出引用精确率与首 token 延迟
.venv/Scripts/python.exe scripts/eval_retrieval.py --mode full
```

报告输出到 `docs/eval/stage2-report.json` 与 `docs/eval/stage2-report.md`，记录 git commit、模型版本、参数与三组对比结果。