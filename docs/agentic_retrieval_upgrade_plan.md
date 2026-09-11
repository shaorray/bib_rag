# Agentic Retrieval 升级计划（Top 1-3，2026-08-31 启动）

范围：① 中心性+引用元数据接线 reranker；② 引文匹配解锁 reference_graph；③ 社区标签+覆盖度自检
执行方式：两条并行工作流（子代理），文件不重叠；git 不 commit（diff 留底可回滚）

## WS-A = ①+③（reranker 接线 + 社区）
1. 读 reranker.py / hybrid_search.py，定位默认查询路径的真实排序点
2. 新增 scripts/compute_graph_metrics.py：citation_graph.json（3,002 节点/17,574 有向边）上算 pagerank、HITS(hub/authority)、精确 betweenness、Louvain(无向投影) → data/graph_metrics.json
3. chroma 元数据 upsert（复用 build_citation_graph_full.py Phase D 模式）：pagerank/authority/hub/betweenness/community_id/community_size → 497,898 chunks
4. reranker.py 最小补丁：provenance 加权（小权重、relevance 主导、env 开关、geo_rag 无字段时优雅降级），连同已有未用的 citation_count/rcr/in_corpus_cited_by 一并接上
5. agent_tools.py 新增 @tool check_evidence_coverage(sources)（簇分布 + 单簇>70% 告警）
6. 验证门槛：embedding 8081 健康检查；改前/改后同查询 top-5 对比；抽样 chunk 字段回读；coverage 工具 demo
7. 报告 → /Disk_bot/tmp/ws_a_report.md

## WS-B = ②（reference_graph 本地匹配）
1. 语料登记表（papers_meta.json + incremental_metadata.json + parent_store 兜底）：source → title/首作者姓/年份，覆盖 3,002
2. 新增 scripts/match_references.py：42,682 raw 边按唯一目标去重后本地匹配（title 相似 + surname + year±1；strong/weak/unmatched 三档；歧义标记）
3. 输出 data/reference_graph_matched.json（.md 键规范，与 citation_graph 节点 100% 可 join；不覆盖原文件）
4. 验证门槛：匹配率报告、10 对人工抽查、键 join 校验、未匹配样本清单
5. 报告 → /Disk_bot/tmp/ws_b_report.md
约束：本地 only，零 API 调用

## Phase 2（本次不做）
- 匹配边接入 snowball_search/find_papers_citing（带 provenance 标志）；Crossref API 兜底；coverage 信号进 answer 节点

## 约束
- 不 commit、不删除、不动 citation_graph.json/reference_graph.json 本体、不碰 geo_rag
