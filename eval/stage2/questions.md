# 阶段 2 评测集（审阅版）

> 本文件由 `questions.jsonl` 生成，机器读取以 JSONL 为准。
> `依据片段` 参与检索指标与引用召回；`可接受补充片段` 只参与引用精确率判定。

| ID | 类别 | 问题 | 依据片段（文件 › 小节） | 可接受补充片段 |
| --- | --- | --- | --- | --- |
| q01 | 精确术语 | RRF 的公式里 k 通常取多少？ | hybrid-retrieval.md › RRF 融合 | — |
| q02 | 精确术语 | Qdrant 新版本推荐用哪个查询接口替代 search？ | qdrant-notes.md › 查询接口 | — |
| q03 | 精确术语 | bge-m3 输出的稠密向量是多少维？ | embeddings-zh.md › bge-m3 | — |
| q04 | 精确术语 | Q4_K_M 是几位量化？ | local-inference.md › 量化 | — |
| q05 | 精确术语 | 开启 IDF 修饰符后，写入稀疏向量时需要提供什么？ | qdrant-notes.md › 稀疏向量与 IDF | — |
| q06 | 精确术语 | bge-reranker-v2-m3 的分数经过哪个函数后可以理解为 0 到 1 的相关概率？ | rerank.md › bge-reranker-v2-m3 的特点 | — |
| q07 | 精确术语 | Ollama 常用的聊天和嵌入接口分别是哪两个？ | local-inference.md › Ollama 的用法 | — |
| q08 | 精确术语 | bge-m3 在本地运行时的显存占用大约是多少？ | embeddings-zh.md › bge-m3 | — |
| q09 | 精确术语 | Qdrant 的 Cosine 距离内部会做什么处理？ | embeddings-zh.md › 归一化与距离 | — |
| q10 | 精确术语 | 最小 RAG 系统五步流程里的第三步是什么？ | rag-basics.md › 标准流程 | — |
| q11 | 语义改写 | 为什么不把知识直接训练进模型参数，而要放在外部索引里？ | rag-basics.md › RAG 与微调的区别 | rag-basics.md › 什么是 RAG |
| q12 | 语义改写 | 用户提问的用词和原文不一样时，靠什么方法还能找到相关内容？ | hybrid-retrieval.md › 稠密检索与稀疏检索 | embeddings-zh.md › 嵌入模型做什么<br>rag-evaluation.md › 评测集怎么构造 |
| q13 | 语义改写 | 衡量两段文本语义接近程度时，余弦相似度看的是什么？ | embeddings-zh.md › 余弦相似度的直觉 | — |
| q14 | 语义改写 | 如果重排模型跑不动，可以按什么顺序降级？ | rerank.md › 延迟与显存 | local-inference.md › 显存共存的取舍 |
| q15 | 语义改写 | 为什么说检索没召回正确材料时，后面的模型再强也没用？ | rag-basics.md › 标准流程 | chunking-strategy.md › 分块为什么重要<br>rag-basics.md › 常见失败模式 |
| q16 | 语义改写 | 给笔记分块时保留标题有什么好处？ | embeddings-zh.md › 中文的特殊问题 | chunking-strategy.md › 标题感知 |
| q17 | 语义改写 | API 密钥应该怎么管理才不会泄露？ | privacy-local-first.md › 密钥与配置 | — |
| q18 | 语义改写 | 怎样证明本地模式真的没有联网？ | privacy-local-first.md › 出站内容的最小化 | local-inference.md › 为什么要在本地跑 |
| q19 | 语义改写 | 为什么不能拿别的模型调好的相似度阈值直接来用？ | embeddings-zh.md › 余弦相似度的直觉 | rerank.md › 重排与拒答<br>rerank.md › bge-reranker-v2-m3 的特点 |
| q20 | 语义改写 | 资料更新以后为什么不需要重新训练模型？ | rag-basics.md › 什么是 RAG | rag-basics.md › RAG 与微调的区别 |
| q21 | 跨文档 | 从召回、重排到拒答，一条完整的检索链路由哪些环节组成，各解决什么问题？ | hybrid-retrieval.md › 稠密检索与稀疏检索<br>rerank.md › 为什么需要重排<br>rerank.md › 重排与拒答 | rag-basics.md › 常见失败模式<br>hybrid-retrieval.md › 工程取舍<br>rag-basics.md › 标准流程 |
| q22 | 跨文档 | 中文语料上做 BM25 要特别注意什么？ | hybrid-retrieval.md › BM25 的直觉<br>embeddings-zh.md › 中文的特殊问题 | — |
| q23 | 跨文档 | 稠密和稀疏两路数据分别怎么产生、又怎么存在向量库里？ | embeddings-zh.md › bge-m3<br>qdrant-notes.md › 集合与向量<br>qdrant-notes.md › 稀疏向量与 IDF | hybrid-retrieval.md › 稠密检索与稀疏检索 |
| q24 | 跨文档 | 分块大小是怎么同时影响检索和生成质量的？ | chunking-strategy.md › 分块为什么重要<br>rag-basics.md › 标准流程 | chunking-strategy.md › 固定窗口<br>chunking-strategy.md › 参数怎么定 |
| q25 | 跨文档 | 为了让引用能被核对，数据层需要保存哪些信息？ | citation-design.md › 行号定位<br>qdrant-notes.md › Payload 与过滤 | citation-design.md › 引用要解决什么问题<br>citation-design.md › 编号与映射<br>citation-design.md › 稳定 ID |
| q26 | 跨文档 | 显存只有 8GB 时，怎么同时安排嵌入、重排和生成模型？ | local-inference.md › 显存共存的取舍<br>rerank.md › 延迟与显存 | local-inference.md › 量化 |
| q27 | 跨文档 | 做一套可信的 RAG 评测，需要看哪些指标、注意什么？ | rag-evaluation.md › 核心指标<br>rerank.md › 重排与拒答 | rag-evaluation.md › 评测集怎么构造<br>rag-evaluation.md › 指标之间的关系<br>rag-evaluation.md › 评测的运行方式<br>rag-evaluation.md › 为什么必须评测 |
| q28 | 跨文档 | 拒答除了保证可信，还有什么实际好处？ | rerank.md › 重排与拒答<br>rag-basics.md › 常见失败模式 | citation-design.md › 引用与拒答的配合 |
| q29 | 跨文档 | 要让原始文档不出本机，系统需要哪些机制一起配合？ | privacy-local-first.md › 出站内容的最小化<br>qdrant-notes.md › 运维注意 | privacy-local-first.md › 隐私作为工程能力<br>privacy-local-first.md › 本地优先的含义<br>privacy-local-first.md › 读取范围的显式化 |
| q30 | 跨文档 | 为什么原始文档必须备份，而向量索引可以重建？重建时顺序和 ID 怎么保证不重复？ | qdrant-notes.md › 运维注意<br>citation-design.md › 稳定 ID | — |
| q31 | 无答案 | 我们公司明年的预算是多少？ | —（应拒答） | — |
| q32 | 无答案 | 部署这套系统一共需要几张 GPU？ | —（应拒答） | — |
| q33 | 无答案 | bge-m3 的训练数据一共有多少条？ | —（应拒答） | — |
| q34 | 无答案 | Qdrant 默认的副本数是多少？ | —（应拒答） | — |
| q35 | 无答案 | DeepSeek API 每分钟的调用限额是多少？ | —（应拒答） | — |
| q36 | 无答案 | Ollama 支持 Windows 的哪些版本？ | —（应拒答） | — |
| q37 | 无答案 | 用 LoRA 微调一个 4B 模型需要多少显存？ | —（应拒答） | — |
| q38 | 无答案 | 这个项目支持哪些语言的扫描件 OCR？ | —（应拒答） | — |
| q39 | 无答案 | 怎么把索引同步到另一台电脑？ | —（应拒答） | — |
| q40 | 无答案 | 作者上一份工作在哪家公司？ | —（应拒答） | — |
