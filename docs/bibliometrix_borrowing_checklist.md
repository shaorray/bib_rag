# bibliometrix → bib_rag 借鉴清单（2026-08-31）

来源：massimoaria/bibliometrix 5.5.0（Aria & Cuccurullo，JOI 2017 + 2026 SAAS 论文），已 clone 至 /Disk_bot/github_repos/bibliometrix。

## A. 立即可落地（eph_rag 现有数据即可，天级）
- [ ] 1. **历史谱系图**（histNetwork/histPlot）— corpus 内部有向引用 + 年份 → 时间轴上的领域家谱；citation_graph.json 边 + year 元数据全有，Eph 综述直接可用
- [ ] 2. **RPYS 参考文献年份光谱**（rpys）— 被引年份分布找奠基文献；reference_graph + 年份即可算
- [ ] 3. **边权归一化**（normalizeSimilarity）— Jaccard / Salton / Association(Van Eck & Waltman) / Inclusion 四种；现在图是原始计数，归一化后才能跨网络比较与聚类
- [ ] 4. **VOSviewer / Pajek 导出**（net2VOSviewer / net2Pajek）— 现在只有 GEXF；VOSviewer 是文献计量标准可视化器
- [ ] 5. **网络统计套件**（networkStat）— degree/closeness/betweenness/eigenvector/pagerank/hub/authority + 网络级统计，networkx 直接实现

## B. 中期（1–2 周）
- [ ] 6. **通用二分共现引擎**（cocMatrix → biblioNetwork，Batagelj & Cerinšek 2013）— 一个 manuscript×field 矩阵 → 耦合/共被引/合作/共词网络全部派生；最大架构借鉴点
- [ ] 7. **主题战略图 + 演化**（thematicMap / thematicEvolution / timeslice）— centrality×density 四象限（motor/emerging/declining/basic）+ 时间切片演化 → SUMMARY.md 从静态普查升级为战略总览
- [ ] 8. **社区检测**（Louvain/Leiden 系，igraph 同款）— 引用图聚类 = 自动发现领域子结构（cis/trans 簇、cadherin 簇应自动浮现）

## C. 导入层（下批批量导入前做）
- [ ] 9. **合并+去重一体化**（mergeDbSources / duplicatedMatching）— 归一化键 + 距离匹配在 ingest 时做，防 geo_rag twins 手术复发
- [ ] 10. **本地优先引文匹配**（LTWA/ISO4 期刊归一化 + stringdist）— ltwa.rda 式权威表随库打包；先本地匹配再打 API，帮 geo_rag crossref 解析省调用

## D. 轻量补充（顺手加）
- [ ] 11. 经典定律：Bradford 期刊分区 / Lotka 作者产出 / H-index / dominance
- [ ] 12. LCS/GC 扩展：作者级本地被引 + local/global ratio（已有 in_corpus_cited_by，方向一致）
- [ ] 13. 权威表随库打包：LTWA、停用词、国家码（学 ltwa.rda / stopwords.rda 做法）
- [ ] 14. completeMetadata 式元数据补全（API 兜底填坑）

## E. 方法论叙事（可选）
- [ ] 15. SAAS 四段式（Search/Appraisal/Analysis/Synthesis，JOI 2026）— bib_rag 的 ingest→audit→classify→graph→summarize 几乎 1:1 对应，可做文档/论文骨架

## 不借鉴
- Shiny 单体 UI（agent-first 已覆盖，ROI 低）
- R 绘图栈（Python 侧导出到 VOSviewer/Gephi 更实际）

**建议顺序**：A 全做（几天）→ B 里 cocMatrix 最值 → C 赶在下批导入前 → D 顺手。

---

# F. Agentic Retrieval 视角（2026-08-31 补充）

基线：LangGraph 循环（clarification → HyDE → tool loop → compress → answer），工具 search_child_chunks / retrieve_parent(s) / snowball_search / related_papers / find_papers_citing。
⚠️ 已发现缺口：reranker 未使用任何引用元数据（citation_count/rcr/in_corpus_cited_by 已入 chroma 但查询时不读）——零新数据快赢。

## 直接进 agent loop（按价值排序）
- [ ] 1. 引文字符串匹配 → 本地匹配已完成 ✅ 2026-08-31（match_references.py → reference_graph_matched.json：40,245 边，strong 932 / weak 1,491 / 17,079 目标=14.2%，join 100%，227 篇获 iCite 未覆盖的入边；⚠️ weak 档精度仅 14-34% 不可未验证接线；snowball 接线 + Crossref 兜底 = phase 2）
- [x] 2. 中心性写 chunk 元数据 + provenance boost 接线 ✅ 2026-08-31（apply_post_fusion stage 2b，×1.08 封顶、正分守卫、BIB_RAG_PROVENANCE 开关；详见 /Disk_bot/tmp/ws_a_report.md）
- [x] 3. Louvain 社区标签（434 社区）入元数据 + check_evidence_coverage 第 7 号 agent 工具 ✅ 2026-08-31（>70% 单簇告警；query→簇路由仍待做）
- [ ] 4. 共词网络（termExtraction: LTWA+stemming+stopwords）→ 语料接地的查询扩展表，升级 broaden.py（现为纯 embedding 邻居） ⭐⭐
- [ ] 5. Salton/Jaccard 归一化 → related_papers/snowball 邻居排序去高被引偏置，多跳少重复 ⭐⭐
- [ ] 6. RPYS 奠基集 → answer 节点的充分性/停止信号（自研适配，bibliometrix 原为离线图谱） ⭐
- [ ] 7. 主题四象限（emerging/motor）标签 → 时间意图路由 ⭐

## 基座（间接）
- ingest 去重防 twins 复发（检索精度）；cocMatrix 是 3/4/5 的共同底座

## 与 agentic retrieval 无关
- Lotka/H-index/dominance、Bradford、合作网络、Shiny/绘图栈
