# PDF 控制实验：同一内容，两种格式

- 模式：`hybrid_rerank`，top_k = 5
- markdown 报告：`docs/eval/pdf-control-md/stage2-report.json`（集合 `localmind_eval_md`，2026-10-25T11:09:34+00:00）
- PDF 报告：`docs/eval/pdf-control-pdf/stage2-report.json`（集合 `localmind_eval_pdf`，2026-10-25T11:10:01+00:00）
- 内容、题目、分块参数相同，唯一变量是格式：PDF 没有标题路径、按页切分、引用以页码定位。

## 总体

| 指标 | markdown | PDF | 差值 |
| --- | --- | --- | --- |
| Hit@k | 100.0% | 100.0% | +0.0 pp |
| Recall@k | 100.0% | 100.0% | +0.0 pp |
| MRR@k | 90.0% | 100.0% | +10.0 pp |
| 无答案题拒答率 | 100.0% | 100.0% | +0.0 pp |

## 按题型

| 题型 | 题数 | markdown Recall@k | PDF Recall@k |
| --- | --- | --- | --- |
| exact | 3 | 100.0% | 100.0% |
| paraphrase | 3 | 100.0% | 100.0% |
| cross_page | 2 | 100.0% | 100.0% |

## 两套索引的形状

| | 文档 | 分块 | 平均每块字符 |
| --- | --- | --- | --- |
| markdown | 10 | 47 | 143 |
| PDF | 10 | 22 | 332 |

这一步很关键：两种格式的差异**不只是页边界**。markdown 按标题切节，小节各自成块，所以块更多、更短；PDF 没有标题信息，段落直接打包到上限，块更少、更长。下面的差值是两个因素合起来的效果，不能单独归因给"按页分块"。

## 逐题

| 题号 | 题型 | markdown 命中 | PDF 命中 | PDF 页码 |
| --- | --- | --- | --- | --- |
| pdf01 | exact | 是 | 是 | 1 |
| pdf02 | exact | 是 | 是 | 1 |
| pdf03 | exact | 是 | 是 | 1 |
| pdf04 | paraphrase | 是 | 是 | 1 |
| pdf05 | paraphrase | 是 | 是 | 1 |
| pdf06 | paraphrase | 是 | 是 | 1 |
| pdf07 | cross_page | 是 | 是 | 1,2 |
| pdf08 | cross_page | 是 | 是 | 1,2 |
| pdf09 | unanswerable | —（无答案） | —（无答案） | — |
| pdf10 | unanswerable | —（无答案） | —（无答案） | — |

## 怎么读这张表

- 这 10 道题只覆盖 control 语料的一小部分，单个题目的差异就是 10 个百分点，所以看差值的方向与题型分布，不要看小数位。
- `cross_page` 两题的 gold 分列 PDF 的两页：如果每页单独分块伤害了召回，损失会集中出现在这一行。
- PDF 丢失了 markdown 的标题路径，引用里没有「小节」这一层；这既是格式差异，也是解析器当前的行为（PDF 无标题信息）。

