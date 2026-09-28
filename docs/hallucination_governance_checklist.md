# bib_rag 幻觉治理清单（三层框架对照）

> 来源：幻觉治理三层框架（输入层/生成层/输出层 + 回归评估），对照 bib_rag 现状审计 2026-09-28。
> 用法：改任何配置（Prompt / 模型 / 检索参数 / 加工具）→ 跑一次 L4 回归 → 看四指标趋势。

## L1 输入层 · 检索质量

| 框架要求 | bib_rag 现状 | 状态 |
|---|---|---|
| 混合检索 | hybrid_search.py（RRF 融合 dense+FTS） | ✅ |
| 重排 rerank | rerank 服务 :11436，生产路径已接 | ✅ |
| 切块优化 | chunk 参数在 kb_config；未 A/B 过 | ⚠️ 待回归 |
| 多源交叉验证 | Zotero(19805615) + 本地库 + reference graph；跨源核对未强制 | ⚠️ |
| **检索不到 → 拒答** | **只在 prompt 里说 "state what is missing"，无硬性空结果拦截** | ❌ 补 |

**行动**：evidence_gate 已存在但只在 fallback_response 里兜底——需要加**前置短路**：
进入 orchestrator LLM 之前，若本 session 所有检索 query 都返回 NO_RELEVANT_CHUNKS /
RETRIEVAL_ERROR → 直接跳过 LLM，返回「库内无文献支持该结论」（零 LLM 拒答）。

## L2 生成层 · 忠实性约束

| 框架要求 | bib_rag 现状 | 状态 |
|---|---|---|
| Prompt 写死「无答案必说不知道」 | Source Integrity 规则 ✓（agent_prompts.py:165） | ✅ |
| 强制引用溯源 | 强制 Sources 段 ✓（文件名列表） | ✅ 但只到文件级 |
| 结构化输出 schema | agent_schemas.py | ✅ |
| 不确定性估计 + 拒答 | claim_ledger 有 confidence=weak 标记；**weak 不拦截，只记录** | ⚠️ |
| temperature | agentic 查询路径未固定低 temp | ⚠️ 补 |

**行动**：引用从文件级细化到「文件 + 段落/页码」；weak 置信 claim 汇总时前置提示
（不拦截，但回答开头给「低置信」标记，用户自己决定信任度）。

## L3 输出层 · 事实校验

| 框架要求 | bib_rag 现状 | 状态 |
|---|---|---|
| 逐 claim 事实账本 | claim_ledger.jsonl ✓（45 条，每条带 evidence） | ✅ |
| evidence 自动验证 | **verified:false 全部未闭环**——账本只记不验 | ❌ 补 |
| 数字/日期/金额二次校验 | 无 | ❌ 补（数字字段过规则/计算器） |
| 高风险场景人工审核 | n/a（研究问答） | — |

**更正**：claim_ledger 已有三态验证（verified/external/unresolved）且 validate_claims 已接入
agent_nodes 流水线；citation_guard（whitelist/substring/lexical，零 LLM）也已接入 collect_answer。
真正缺口：(a) **数字/日期字段复核**——claim 中的数字必须出现在其引用的 child chunk 文本里，
否则降级 unresolved；(b) **答案内暴露未核实 claim**——unresolved 的 claim 在 Sources 前集中列出。

## L4 评估 · 回归集（最大的缺口）

| 框架要求 | bib_rag 现状 | 状态 |
|---|---|---|
| 黄金测试集 | gold_retrieval.jsonl ✓（检索级） | ✅ |
| 检索回归 | eval_retrieval.py（recall@k / MRR，镜像生产路径） | ✅ |
| **端到端忠实性回归** | **无**——换 Prompt/模型后只测检索，不测「答案是否仍忠于原文」 | ❌ 补 |

**行动**：建 `gold_faithfulness.jsonl`：≥30 条真实问题，每条标
(期望命中的 source, 关键事实句, 期望可拒答?)。端到端跑 create_agent_graph，
自动判分 4 指标：幻觉率（claim 无 evidence 支持比例）、引用准确率、
拒答率（该拒的是否拒了）、低置信暴露率。任何配置变更前后各跑一次对比。
（附：agentic 查询路径统一固定 temperature=0.1，低随机性。）

## 执行顺序（建议）

1. 空→拒答短路（半小时，防最蠢的编造）
2. ledger evidence 闭环（把已有 45 条的 verified 变成真判定）
3. gold_faithfulness.jsonl 30 条（用最近真实查询当种子）
4. eval_faithfulness.py（复用 eval_retrieval 的 harness 风格）
