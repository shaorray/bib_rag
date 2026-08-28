# File Inventory - Cleanup Plan

## 1. Knowledge Base Data Files (keep)
- chroma_db/ephrin_papers.pkl - V1 original knowledge base
- chroma_db_v2/ephrin_papers_v2.pkl - V2 knowledge base
- chroma_db_v3/ephrin_papers_v3.pkl - V3 knowledge base (currently recommended)
- paper_metadata.json - Paper metadata

## 2. Core Processing Scripts (keep)
- process_v2_papers.py - V2 processing script
- process_v3_papers.py - V3 processing script (currently recommended)
- build_knowledge_base.py - Build the knowledge base
- add_new_papers.py - Add new papers

## 3. Query Scripts (keep)
- query_v2_kb.py - V2 queries
- query_v2_kb_with_citations.py - V2 queries with citations
- query_v3_kb.py - V3 queries (currently recommended)
- hybrid_search.py - Hybrid search
- quick_query.py - Quick queries
- batch_query.py - Batch queries
- query_interface.py - Query interface

## 4. Academic Writing Helpers (keep)
- academic_writer.py - Academic writing assistant
- academic_writer_with_citations.py - Writing with citations
- citation_manager.py - Citation management
- quick_cite.py - Quick citations

## 5. Agentic RAG Core (keep)
- agentic_workflow.py - Main workflow
- agentic_workflow_v2.py - V2 workflow
- rag_core.py - RAG core
- rag_core_v2.py - V2 core
- self_rag.py - Self-RAG implementation
- multi_hop_rag.py - Multi-hop RAG
- planner_agent.py - Planner agent
- reflector_agent.py - Reflector agent
- guardrail_node.py - Guardrail node

## 6. Old/Test Files That Can Be Deleted
- ~~demo.py~~ - Demo script (old)
- ~~debug_retry.py~~ - Debug script
- ~~debug_routing.py~~ - Debug routing
- ~~test_agentic_workflow.py~~ - Test file
- ~~test_comparison.py~~ - Test file
- ~~test_improvements.py~~ - Test file
- ~~test_objective.py~~ - Test file
- ~~test_planner.py~~ - Test file
- ~~test_production.py~~ - Test file
- ~~test_reflector_fast.py~~ - Test file
- ~~test_reflector.py~~ - Test file
- ~~test_v2_query.py~~ - Test file
- ~~test_v2_retrieval.py~~ - Test file
- ~~verify_improvements.py~~ - Verification script
- ~~analyze_kb.py~~ - Analysis script (temporary)
- ~~rebuild_chroma_v2.py~~ - Rebuild script (completed)
- ~~rebuild_chroma_v2_clean.py~~ - Rebuild script (completed)
- ~~langfuse_monitor.py~~ - Langfuse monitor (unused)
- ~~langgraph_phase1.py~~ - LangGraph phase 1 (experimental)
- ~~langgraph_agentic_rag.py~~ - LangGraph RAG (experimental)
- ~~redis_cache.py~~ - Redis cache (unused)
- ~~metrics.log~~ - Log file
- ~~query_history.json~~ - Query history
- ~~__pycache__/~~ - Python cache

## 7. Documentation Files (keep)
- README.md - Main doc
- USAGE.md - Usage guide
- QUICK_START.md - Quick start
- ACADEMIC_USAGE.md - Academic usage
- SCHOLAR_API.md - API docs
- RAG_EMBEDDING_GUIDE.md - Embedding guide
- PMID_CITATION_GUIDE.md - PMID citation guide
- V3_IMPROVEMENT_REPORT.md - V3 improvement report
- IMPROVEMENT_PLAN.md - Improvement plan
- IMPROVEMENT_ANALYSIS.md - Improvement analysis
- AGENTIC_RAG_BEST_PRACTICES.md - Best practices

## 8. Docs That Can Be Archived (old versions / outdated)
- ~~DEPLOYMENT_REPORT.md~~ - Deployment report (old)
- ~~OPTIMIZATION_REPORT.md~~ - Optimization report (old)
- ~~OPTIMIZATION_COMPLETE.md~~ - Optimization completion (old)
- ~~PERFORMANCE_EVALUATION.md~~ - Performance evaluation (old)
- ~~PHASE1_COMPLETION.md~~ - Phase 1 completion (old)
- ~~PHASE1_IMPLEMENTATION.md~~ - Phase 1 implementation (old)
- ~~PHASE1_PROGRESS.md~~ - Phase 1 progress (old)
- ~~LANGGRAPH_INTEGRATION.md~~ - LangGraph integration (experimental)
- ~~LANGGRAPH_MIGRATION_GUIDE.md~~ - LangGraph migration (experimental)
- ~~LANGGRAPH_STUDY_NOTES.md~~ - LangGraph notes (experimental)
- ~~LANGGRAPH_VERIFICATION.md~~ - LangGraph verification (experimental)
- ~~PRODUCTION_RAG_COURSE_SUMMARY.md~~ - Course summary (old)
- ~~STATUS.md~~ - Status report (old)
- ~~TEST_REPORT.md~~ - Test report (old)
- ~~VERSION_COMPARISON.md~~ - Version comparison (old)
- ~~OPENDATALOADER_INSTALL_REPORT.md~~ - Install report (old)
- ~~OPENDATALOADER_SEARCH_REPORT.md~~ - Search report (old)