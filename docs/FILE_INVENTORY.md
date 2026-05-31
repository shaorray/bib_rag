# 文件清单 - 整理计划

## 1. 知识库数据文件（保留）
- chroma_db/ephrin_papers.pkl - V1原始知识库
- chroma_db_v2/ephrin_papers_v2.pkl - V2知识库
- chroma_db_v3/ephrin_papers_v3.pkl - V3知识库（当前推荐）
- paper_metadata.json - 文献元数据

## 2. 核心处理脚本（保留）
- process_v2_papers.py - V2处理脚本
- process_v3_papers.py - V3处理脚本（当前推荐）
- build_knowledge_base.py - 构建知识库
- add_new_papers.py - 添加新文献

## 3. 查询脚本（保留）
- query_v2_kb.py - V2查询
- query_v2_kb_with_citations.py - V2带引用查询
- query_v3_kb.py - V3查询（当前推荐）
- hybrid_search.py - 混合搜索
- quick_query.py - 快速查询
- batch_query.py - 批量查询
- query_interface.py - 查询接口

## 4. 学术写作辅助（保留）
- academic_writer.py - 学术写作助手
- academic_writer_with_citations.py - 带引用写作
- citation_manager.py - 引用管理
- quick_cite.py - 快速引用

## 5. Agentic RAG核心（保留）
- agentic_workflow.py - 主工作流
- agentic_workflow_v2.py - V2工作流
- rag_core.py - RAG核心
- rag_core_v2.py - V2核心
- self_rag.py - Self-RAG实现
- multi_hop_rag.py - 多跳RAG
- planner_agent.py - 规划Agent
- reflector_agent.py - 反思Agent
- guardrail_node.py - 护栏节点

## 6. 可以删除的旧文件/测试文件
- ~~demo.py~~ - 演示脚本（旧）
- ~~debug_retry.py~~ - 调试脚本
- ~~debug_routing.py~~ - 调试路由
- ~~test_agentic_workflow.py~~ - 测试文件
- ~~test_comparison.py~~ - 测试文件
- ~~test_improvements.py~~ - 测试文件
- ~~test_objective.py~~ - 测试文件
- ~~test_planner.py~~ - 测试文件
- ~~test_production.py~~ - 测试文件
- ~~test_reflector_fast.py~~ - 测试文件
- ~~test_reflector.py~~ - 测试文件
- ~~test_v2_query.py~~ - 测试文件
- ~~test_v2_retrieval.py~~ - 测试文件
- ~~verify_improvements.py~~ - 验证脚本
- ~~analyze_kb.py~~ - 分析脚本（临时）
- ~~rebuild_chroma_v2.py~~ - 重建脚本（已完成）
- ~~rebuild_chroma_v2_clean.py~~ - 重建脚本（已完成）
- ~~langfuse_monitor.py~~ - Langfuse监控（未使用）
- ~~langgraph_phase1.py~~ - LangGraph阶段1（实验）
- ~~langgraph_agentic_rag.py~~ - LangGraph RAG（实验）
- ~~redis_cache.py~~ - Redis缓存（未使用）
- ~~metrics.log~~ - 日志文件
- ~~query_history.json~~ - 查询历史
- ~~__pycache__/~~ - Python缓存

## 7. 文档文件（保留）
- README.md - 主文档
- USAGE.md - 使用说明
- QUICK_START.md - 快速开始
- ACADEMIC_USAGE.md - 学术使用
- SCHOLAR_API.md - API文档
- RAG_EMBEDDING_GUIDE.md - Embedding指南
- PMID_CITATION_GUIDE.md - PMID引用指南
- V3_IMPROVEMENT_REPORT.md - V3改进报告
- IMPROVEMENT_PLAN.md - 改进计划
- IMPROVEMENT_ANALYSIS.md - 改进分析
- AGENTIC_RAG_BEST_PRACTICES.md - 最佳实践

## 8. 可以归档的文档（旧版本/过期）
- ~~DEPLOYMENT_REPORT.md~~ - 部署报告（旧）
- ~~OPTIMIZATION_REPORT.md~~ - 优化报告（旧）
- ~~OPTIMIZATION_COMPLETE.md~~ - 优化完成（旧）
- ~~PERFORMANCE_EVALUATION.md~~ - 性能评估（旧）
- ~~PHASE1_COMPLETION.md~~ - 阶段1完成（旧）
- ~~PHASE1_IMPLEMENTATION.md~~ - 阶段1实现（旧）
- ~~PHASE1_PROGRESS.md~~ - 阶段1进度（旧）
- ~~LANGGRAPH_INTEGRATION.md~~ - LangGraph集成（实验）
- ~~LANGGRAPH_MIGRATION_GUIDE.md~~ - LangGraph迁移（实验）
- ~~LANGGRAPH_STUDY_NOTES.md~~ - LangGraph笔记（实验）
- ~~LANGGRAPH_VERIFICATION.md~~ - LangGraph验证（实验）
- ~~PRODUCTION_RAG_COURSE_SUMMARY.md~~ - 课程总结（旧）
- ~~STATUS.md~~ - 状态报告（旧）
- ~~TEST_REPORT.md~~ - 测试报告（旧）
- ~~VERSION_COMPARISON.md~~ - 版本对比（旧）
- ~~OPENDATALOADER_INSTALL_REPORT.md~~ - 安装报告（旧）
- ~~OPENDATALOADER_SEARCH_REPORT.md~~ - 搜索报告（旧）
