# 文件夹整理报告

## 整理前状态

原有60+个文件混乱地放在根目录，包含：
- 多个版本的脚本（v1, v2, v3）
- 大量测试文件（test_*.py）
- 调试脚本（debug_*.py）
- 过时的文档（PHASE1*, OPTIMIZATION*）
- 实验性代码（LangGraph相关）
- Python缓存文件（__pycache__）
- 日志文件（metrics.log）

## 整理后结构

```
ephrin_agentic_rag/
├── chroma_db/          # V1知识库（遗留）
├── chroma_db_v2/       # V2知识库
├── chroma_db_v3/       # V3知识库（推荐）
├── src/                # 源代码
│   ├── agents/         # Agentic RAG组件
│   ├── queries/        # 查询脚本
│   ├── utils/          # 工具脚本
│   └── *.py            # 核心处理脚本
├── docs/               # 文档
├── data/               # 配置文件
└── archive/            # 归档文件
    ├── old_scripts/    # 旧脚本
    ├── test_files/     # 测试文件
    ├── documents/      # 旧文档
    └── experimental/   # 实验性代码
```

## 保留的核心文件

### 处理脚本（src/）
- `src/process_v3_papers.py` - V3处理（推荐）
- `src/process_v2_papers.py` - V2处理
- `src/build_knowledge_base.py` - 构建知识库
- `src/add_new_papers.py` - 添加新文献

### 查询脚本（src/queries/）
- `src/queries/query_v3_kb.py` - V3查询（推荐）
- `src/queries/query_v2_kb.py` - V2查询
- `src/queries/hybrid_search.py` - 混合搜索
- `src/queries/quick_query.py` - 快速查询

### Agent组件（src/agents/）
- `src/agents/agentic_workflow.py` - 主工作流
- `src/agents/rag_core.py` - 核心逻辑
- `src/agents/self_rag.py` - Self-RAG
- `src/agents/multi_hop_rag.py` - 多跳RAG

### 学术写作（src/utils/）
- `src/utils/academic_writer.py` - 写作助手
- `src/utils/citation_manager.py` - 引用管理

### 文档（docs/）
- `docs/README.md` - 主文档
- `docs/USAGE.md` - 使用说明
- `docs/QUICK_START.md` - 快速开始
- `docs/PMID_CITATION_GUIDE.md` - PMID引用指南
- `docs/RAG_EMBEDDING_GUIDE.md` - Embedding指南
- `docs/V3_IMPROVEMENT_REPORT.md` - V3改进报告

## 归档的文件

### 旧脚本（archive/old_scripts/）
- demo.py, debug_retry.py, debug_routing.py
- analyze_kb.py, langfuse_monitor.py
- redis_cache.py

### 测试文件（archive/test_files/）
- test_*.py (10个测试脚本)
- verify_improvements.py

### 旧文档（archive/documents/）
- DEPLOYMENT_REPORT.md
- OPTIMIZATION_*.md (3个)
- PERFORMANCE_EVALUATION.md
- PHASE1_*.md (3个)
- STATUS.md, TEST_REPORT.md
- VERSION_COMPARISON.md
- OPENDATALOADER_*.md (2个)

### 实验性代码（archive/experimental/）
- langgraph_agentic_rag.py
- langgraph_phase1.py

## 删除的文件

- `metrics.log` - 日志文件
- `query_history.json` - 查询历史
- `__pycache__/` - Python缓存

## 修复的问题

1. **导入路径**: 更新所有移动后的文件的sys.path
2. **结构清晰**: 按功能分类存放
3. **版本管理**: 明确区分V1/V2/V3
4. **文档整理**: 保留最新文档，归档旧文档

## 使用方式

### 推荐命令（V3）
```bash
# 查询
python3 src/queries/query_v3_kb.py "cis interaction" -n 5

# 带引用段落
python3 src/queries/query_v3_kb.py "Eph signaling" -n 3 --paragraph

# 混合搜索
python3 src/queries/hybrid_search.py "axon guidance"
```

### 如果需要恢复旧文件
```bash
# 从archive恢复
cp archive/old_scripts/demo.py .
cp archive/documents/STATUS.md docs/
```

## 文件数量对比

| 项目 | 整理前 | 整理后 |
|------|--------|--------|
| 根目录文件 | 60+ | 1 (README.md) |
| 源代码 | 混乱 | src/下分类存放 |
| 文档 | 混乱 | docs/下统一存放 |
| 归档文件 | 0 | archive/下4个子目录 |
