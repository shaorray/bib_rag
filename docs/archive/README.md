# Eph/Ephrin Agentic RAG Knowledge Base

🧠 基于 180+ 篇 Eph/Ephrin 论文的智能知识库系统

## Features

### Core RAG Features
- ✅ **向量检索**: Sentence-Transformers 嵌入 + 余弦相似度检索
- ✅ **智能分块**: 语义感知分块，保留上下文连贯性
- ✅ **元数据管理**: 自动提取论文标题、作者、年份

### Agentic Features
- 🔄 **Self-RAG**: 自我评估检索文档的相关性
- 🔧 **CRAG (Corrective RAG)**: 检索失败时自动重写查询并重试
- 🎯 **自适应路由**: 根据问题复杂度选择检索或直接回答
- 🕸️ **多跳推理**: 支持需要多步推理的复杂查询

## Architecture

```
User Query
    ↓
[Analyze] - 是否需要检索?
    ↓ (Yes)              ↓ (No)
[Retrieve]              [Direct Answer]
    ↓
[Grade] - 相关性评估 (Self-RAG)
    ↓
┌─────────┬──────────┐
↓ (High)  ↓ (Medium) ↓ (Low)
[Generate] [Rewrite]  [Rewrite]
              ↓
         [Retrieve]  ←  CRAG 循环 (max 2 retries)
              ↓
         [Generate]
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag
pip install -r requirements.txt
```

### 2. Build Knowledge Base

```bash
python3 build_knowledge_base.py
```

This will:
- Load 194 markdown papers from `/Disk_2/claw_working_dir/Ephrin_papers/review_output/markdown_round2`
- Generate embeddings (simple bag-of-words fallback when network unavailable)
- Create vector database at `./chroma_db`
- Run test queries
- Index ~5,700 document chunks (~1.9M words)

### 3. Interactive Query

```bash
python3 query_interface.py
```

Commands:
- Type your question directly
- `/multihop <query>` - Use multi-hop reasoning
- `/stats` - Show knowledge base statistics
- `/history` - Show query history
- `/exit` - Exit

### 4. Batch Queries

```bash
# Create queries.txt with one question per line
echo "What is cis-interaction in Eph signaling?" > queries.txt
echo "How does ephrin-B1 regulate axon guidance?" >> queries.txt

# Run batch
python3 query_interface.py --batch queries.txt
```

## File Structure

```
ephrin_agentic_rag/
├── build_knowledge_base.py    # 知识库构建脚本
├── rag_core.py                # 核心 RAG 组件（嵌入、存储、管道）
├── agentic_workflow.py        # Agentic RAG 工作流
├── query_interface.py         # 交互式查询界面
├── requirements.txt           # 依赖
├── README.md                  # 本文件
├── chroma_db/                 # 向量数据库（自动创建）
├── paper_metadata.json        # 论文元数据（自动创建）
└── query_history.json         # 查询历史（自动创建）
```

## Key Components

### 1. DocumentStore
简单的文件系统向量存储，支持：
- 持久化存储（Pickle）
- 余弦相似度检索
- 元数据管理

### 2. AgenticRAGWorkflow
基于 LangGraph 的工作流：
- **analyze**: 分析查询复杂度
- **retrieve**: 检索文档
- **grade**: 评估相关性
- **rewrite**: CRAG 查询重写
- **generate**: 生成答案

### 3. MultiHopRAG
支持复杂多步推理：
- 查询分解
- 子查询并行处理
- 结果综合

## Query Rewriting Strategy

CRAG 重写使用以下策略：
1. **关键词扩展**: 添加同义词（如 "Eph" → "Eph receptor"）
2. **布尔 OR**: 扩展检索范围
3. **上下文添加**: 添加 "Eph ephrin signaling" 等背景词

## Customization

### 使用不同的嵌入模型

在 `rag_core.py` 中修改：
```python
self.embedder = SimpleEmbedding(model_name="your-model")
```

### 调整分块大小

在 `build_knowledge_base.py` 中修改：
```python
chunks = self._smart_chunk(content, chunk_size=1000, overlap=200)
```

### 添加 LLM 集成

在 `agentic_workflow.py` 中配置：
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

## Performance

- **Embedding**: all-MiniLM-L6-v2 (384 dimensions)
- **Retrieval**: ~50ms per query (in-memory)
- **CRAG Retries**: max 2 attempts
- **Multi-hop**: Supports up to 4 sub-queries

## Troubleshooting

### No documents found
```bash
# Check if markdown files exist
ls /Disk_2/claw_working_dir/Ephrin_papers/review_output/markdown/

# Rebuild knowledge base
rm -rf /Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db
python3 build_knowledge_base.py
```

### Low relevance scores
- Check if query contains specific keywords
- Try `/multihop` for complex questions
- Use more specific terminology

## References

- **Self-RAG**: [Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- **CRAG**: [Corrective RAG](https://arxiv.org/abs/2401.15884)
- **LangGraph**: https://langchain-ai.github.io/langgraph/

## License

MIT License - Academic/Research Use
ademic/Research Use
e, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- **CRAG**: [Corrective RAG](https://arxiv.org/abs/2401.15884)
- **LangGraph**: https://langchain-ai.github.io/langgraph/

## License

MIT License - Academic/Research Use
