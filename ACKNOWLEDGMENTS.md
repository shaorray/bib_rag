# Acknowledgments / 致谢

bib_rag 是站在众多优秀开源项目的肩膀上建成的。本项目在架构与机制设计上广泛参考了以下项目——**所有借鉴均为机制/思路层面的 clean-room 重写**（在 bib_rag 域内用 Python 重新实现，适配 ChromaDB + llama-server + Zotero 本地栈），不含任何源码复制；各借鉴点在对应模块的文件头注中亦有标注。感谢这些项目的作者与社区。

bib_rag 本身以 [MIT License](LICENSE) 发布。

---

## 架构基础

| 项目 | 许可 | 借鉴内容 |
|---|---|---|
| [production-agentic-rag-course](https://github.com/DataTalksClub/production-agentic-rag-course)（Colt Steele / DataTalksClub，课程仓库） | MIT | Agentic RAG 的 LangGraph 骨架：State 图、guardrail → rewrite → retrieve → grade → generate 循环、interrupt 澄清、fallback 响应。`src/agent_nodes.py`、`src/agent_edges.py`、`src/agent_prompts.py` 的图结构与节点逻辑改编自该课程仓库的实现，替换为 ChromaDB + bge-m3 + llama-server 后端 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT | 工作流引擎本身（作为依赖使用） |

## 引用与检索机制

| 项目 | 许可 | 借鉴内容 |
|---|---|---|
| [paper-qa](https://github.com/Future-House/paper-qa)（FutureHouse） | Apache-2.0 | 引文白名单（answer Sources ⊆ 实际检索 parent_ids，违者确定性删除）、RetractionDataPostProcessor 的撤稿 DOI 拦截思想（`src/citation_guard.py`、`src/retraction_watch.py`）、DOI→BibTeX 导出（`src/bibtex_export.py`） |
| [Corvus](https://github.com/YS0meone/Corvus) | MIT | 引用图雪球扩展：沿参考文献列表迭代检索（`src/reference_graph.py`） |
| [production-rag-assistant](https://github.com/aieng-abdullah/production-rag-assistant) | MIT | 引用忠实度（faithfulness）评估指标的实现思路（`src/evaluate.py`） |
| [DocsGPT](https://github.com/arc53/DocsGPT) | MIT | BM25 + 稠密向量的 RRF 融合（k=60）混合检索（`src/hybrid_search.py`） |
| [zotero-rag-cli](https://github.com/Agents365-ai/zotero-rag-cli) | MIT（README 声明；其 LICENSE 文件为 CC BY-NC 4.0，二者冲突） | 本地 chunk 的 FTS5 BM25 索引模式（`src/hybrid_search.py`） |
| [RAG-Assistant-for-Zotero](https://github.com/Quiet-Signals-Lab/RAG-Assistant-for-Zotero) | Apache-2.0 | CJK 字符级 BM25 分词修正（`src/hybrid_search.py`） |
| [seerai](https://github.com/dralkh/seerai)（Zotero 插件） | MIT | 学者标识符归一化（DOI/arXiv/PMID 多形态→规范形）、RRF 共识融合、答案尾部引用的确定性剥离（`src/identifiers.py`、`src/hybrid_search.py`、`src/citation_guard.py`） |
| [zotero-redisearch-rag](https://github.com/jmiba/zotero-redisearch-rag) | Apache-2.0 | 弱结果确定性扩展策略（零 LLM 的查询放宽，`src/broaden.py`） |
| [ragent](https://github.com/nageoffer/ragent)（nageoffer） | Apache-2.0 | 证据充分性门槛：fallback 前的确定性覆盖检查 + 显式缺口报告（`src/evidence_gate.py`） |
| [haiku.rag](https://github.com/ggozad/haiku.rag) | MIT | doctor 的索引一致性检查思路：计数奇偶、parent 解析、近重复 DOI 导出（`scripts/doctor.py`） |
| [research-hub](https://github.com/WenyuChiou/research-hub) | MIT | doctor 的 CheckResult 模式：四级状态、per-check 隔离、remedy 字符串、--strict、CI 安全退出码（`scripts/doctor.py`） |
| [agentset](https://github.com/agentset-ai/agentset) | MIT | agentic search 的工具驱动检索契约参考（检索键去重账本对比基准） |
| [ragflow](https://github.com/infiniflow/ragflow) | Apache-2.0 | 检索管线的工程实践参考（深度调研对照） |
| [Agentic-RAG-R1](https://github.com/jiangxinke/Agentic-RAG-R1) | Apache-2.0 | “何时检索”的决策哲学参考（bib_rag 以 prompt 层规则实现等价行为） |
| [ragapp](https://github.com/ragapp/ragapp) | Apache-2.0 | RAG 即服务形态的参考对照 |
| [CogDoc](https://github.com/jikongabc/CogDoc) | MIT | 答案侧引用卫生的对照基准 |
| [CiteWeave-RAG](https://github.com/Youn-17/CiteWeave-RAG) | Apache-2.0 | 引用可视化思路参考 |
| [LumiCite](https://github.com/cany7/LumiCite) | AGPL-3.0 | 引用质量评估对照 |
| [citelocal-agent](https://github.com/Baldwinzc/citelocal-agent) | MIT | 本地引用 agent 对照 |
| [Graph-RAG](https://github.com/zjkhurry/Graph-RAG) | MPL-2.0 | 图增强检索对照 |
| [llm-for-zotero](https://github.com/yilewang/llm-for-zotero) | AGPL-3.0 | Zotero LLM 集成对照 |
| [Zotero-RAG](https://github.com/windfollowingheart/Zotero-RAG) | AGPL-3.0 | Zotero RAG 对照 |
| [chiken](https://github.com/yuanjua/chiken) | MIT | Zotero 检索对照 |

## 写作与元数据工作流

| 项目 | 许可 | 借鉴内容 |
|---|---|---|
| [mattpocock/skills](https://github.com/mattpocock/skills)（Matt Pocock） | MIT | /grill-me + /grill-with-docs 的两阶段质询式写作工作流（`src/bib_rag_writer.py`、`scripts/setup_library.py` 的共享词汇表） |
| [bibliometrix](https://github.com/massimoaria/bibliometrix)（Aria & Cuccurullo） | GPL-3.0+ | 元数据审计与补全技巧：Crossref 批量 DOI 验证（逗号 filter）、OpenAlex Work-ID（`id_oa`）第二查找键、行级字段溯源（`meta_provenance`，对应其 $ENRICH 列）、fill-only-if-vacant 补全语义（`scripts/metadata/meta_audit.py`） |
| [CiteRAG](https://github.com/LQgdwind/CiteRAG) | 未声明 | 大规模引用语料库的评估基准对照 |
| [CiteVerify](https://github.com/uu999/CiteVerify) | 未声明 | 引用验证工具对照 |
| [paper-rag-skill](https://github.com/GeederX/paper-rag-skill) | MIT | 论文 RAG 技能封装对照 |
| [research-co-pilot](https://github.com/Marazii/research-co-pilot) | MIT | 研究助手形态对照 |
| [ZotMeta](https://github.com/RoadToDream/ZotMeta) | MIT | Zotero 元数据管理对照 |
| [paper-tracker](https://github.com/RainerSeventeen/paper-tracker) | MIT | 论文追踪对照 |

---

## 说明

- **许可兼容性**：bib_rag 为 MIT。所借鉴项目的许可（MIT / Apache-2.0 / AGPL-3.0）均为机制思路参考、无源码复制，不受 AGPL 传染条款约束；若未来直接引用 bibliometrix 的代码或数据，需重新评估许可兼容性。
- **调研谱系**：23 份技术笔记存于 `RAG/notes/`（Agentic_RAG / citation_rag / zotero_RAG 三组 + 横向调研），对比与借鉴清单见 `RAG/notes/bib_rag_对比与借鉴.md`。
- **遗漏处理**：如发现本清单遗漏了某项目的贡献，请开 issue，我们会及时补上。