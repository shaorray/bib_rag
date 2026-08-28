# Agentic RAG User Manual (2026 Industry-Grade Standard)

**Version**: 2.0.0  
**Last Updated**: 2026-03-28  
**Status**: ✅ Production Ready

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Parameters](#configuration-parameters)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Already installed, no extra action needed
pip install requests langchain langchain-community
```

### 2. Load the Knowledge Base

```python
from rag_core import SimpleEmbedding, DocumentStore

doc_store = DocumentStore(
    'ephrin_papers',
    '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
)
embedder = SimpleEmbedding()

def retriever(query, k=10):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)
```

### 3. Run Agentic RAG

```python
from agentic_rag_workflow import AgenticRAGWorkflow

workflow = AgenticRAGWorkflow(retriever, model="qwen3.5:397b-cloud")

result = workflow.run("What are the functional differences between EphA2 and EphB4 in cancer?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['reflect_score']:.2f}")
print(f"Status: {result['status']}")
```

---

## 🏗️ Architecture Overview

### Complete Workflow

```
Query → Planner → Retriever → Generator → Reflector → Output
                              ↑                    ↓
                              └──── Re-retrieve (if <0.8) ──┘
```

### The Four Core Agents

| Agent | Role | Key Features |
|-------|------|---------|
| **Planner** | Decompose questions | 3-5 sub-queries, retrieval strategy |
| **Retriever** | Vector retrieval | Similarity 0.75+, top_k=10 |
| **Generator** | Generate answers | Grounded in documents, hallucination-resistant |
| **Reflector** | Validate answers | Score 0-1, output if ≥0.8 |

---

## ⚙️ Configuration Parameters

### Golden Parameters (Already Optimized)

```python
GOLDEN_PARAMS = {
    "similarity_threshold": 0.75,  # Retrieval threshold
    "reflection_threshold": 0.8,   # Reflection threshold
    "max_iterations": 3,           # Max iterations
    "top_k": 10,                   # Number of retrieved documents
    "max_sub_queries": 5,          # Max sub-queries
    "temperature": 0.1,            # LLM temperature
    "timeout": 120,                # Timeout (seconds)
    "model": "qwen3.5:397b-cloud"  # Model
}
```

### Configuration File

Edit `agentic_rag_config.json`:

```json
{
  "model": {
    "model_name": "qwen3.5:397b-cloud",
    "temperature": 0.1,
    "timeout": 120
  },
  "retrieval": {
    "similarity_threshold": 0.75,
    "top_k": 10
  },
  "reflection": {
    "threshold": 0.8,
    "max_retries": 3
  }
}
```

---

## 📖 Usage Examples

### Example 1: Simple Query

```python
from agentic_rag_workflow import AgenticRAGWorkflow

workflow = AgenticRAGWorkflow(retriever)

result = workflow.run("What is the function of Eph receptors?")

print(f"Answer: {result['answer'][:500]}")
print(f"Confidence: {result['reflect_score']:.2f}")
```

### Example 2: Complex Query (Comparison)

```python
result = workflow.run("What are the functional differences between EphA2 and EphB4 in cancer?")

# The Planner automatically decomposes this into:
# 1. Function of EphA2 in cancer
# 2. Function of EphB4 in cancer
# 3. Comparative studies of the two

print(f"Number of sub-queries: {len(result['plan']['sub_queries'])}")
print(f"Retrieval strategy: {result['plan']['search_strategy']}")
```

### Example 3: Complex Query (Causality)

```python
result = workflow.run("How does cis-interaction affect trans-signaling?")

# The Planner decomposes this into:
# 1. Molecular mechanism of cis-interaction
# 2. Activation process of trans-signaling
# 3. Regulatory relationship between the two

if result['reflect_score'] < 0.8:
    print("Re-retrieval needed")
    print(f"Suggestions: {result['suggestions']}")
```

### Example 4: Standalone Validation with the Reflector

```python
from reflector_agent import ReflectorAgent

reflector = ReflectorAgent(model="qwen3.5:397b-cloud")

result = reflector.reflect(
    question="What are Eph receptors?",
    answer="Eph receptors are receptor tyrosine kinases",
    documents=[{"text": "Eph receptors are RTKs...", "similarity": 0.85}]
)

print(f"Score: {result['score']:.2f}")
print(f"Re-retrieval needed: {result['needs_reretrieval']}")
print(f"Issues: {result['issues']}")
```

### Example 5: Question Decomposition with the Planner

```python
from planner_agent import PlannerAgent

planner = PlannerAgent(model="qwen3.5:397b-cloud")

plan = planner.plan("What are the functional differences between EphA2 and EphB4?")

print(f"Complexity: {'Complex' if plan['is_complex'] else 'Simple'}")
print(f"Sub-queries: {len(plan['sub_queries'])}")
for sq in plan['sub_queries']:
    print(f"  - {sq['query']}")
```

---

## 🔧 API Reference

### AgenticRAGWorkflow

```python
class AgenticRAGWorkflow:
    def __init__(self, retriever_fn, model="qwen3.5:397b-cloud"):
        """Initialize the workflow"""
    
    def run(self, query: str, verbose: bool = True) -> dict:
        """
        Run the full workflow
        
        Args:
            query: User query
            verbose: Whether to output detailed logs
            
        Returns:
            {
                "query": str,
                "answer": str,
                "confidence": float,
                "reflect_score": float,
                "status": "success" | "needs_reretrieval",
                "issues": List[str],
                "suggestions": List[str],
                "duration_seconds": float
            }
        """
    
    def get_stats(self) -> dict:
        """Get statistics"""
```

### PlannerAgent

```python
class PlannerAgent:
    def __init__(self, model="qwen3.5:397b-cloud", 
                 max_sub_queries=5, max_iterations=3):
        """Initialize the Planner"""
    
    def plan(self, query: str) -> dict:
        """
        Decompose the question
        
        Returns:
            {
                "original_query": str,
                "sub_queries": List[SubQuery],
                "is_complex": bool,
                "search_strategy": "parallel" | "sequential" | "direct"
            }
        """
```

### ReflectorAgent

```python
class ReflectorAgent:
    def __init__(self, model="qwen3.5:397b-cloud", 
                 reflection_threshold=0.8):
        """Initialize the Reflector"""
    
    def reflect(self, question: str, answer: str, 
                documents: List[Dict]) -> dict:
        """
        Validate the answer
        
        Returns:
            {
                "score": float,  # 0-1
                "is_sufficient": bool,
                "needs_reretrieval": bool,
                "issues": List[str],
                "suggestions": List[str]
            }
        """
```

---

## 📊 Best Practices

### 1. Query Optimization

✅ **Good queries**:
- Specific and clear: "What is the function of EphA2 in cancer?"
- Include entities: "What are the differences between EphA2 and EphB4?"
- Provide context: "What role do Eph receptors play in the tumor microenvironment?"

❌ **Queries to avoid**:
- Too broad: "Eph receptors?"
- Missing context: "What does it do?"
- Multiple negations: "Isn't it not an unimportant function?"

### 2. Parameter Tuning

**High-accuracy scenarios** (paper writing):
```python
similarity_threshold = 0.80  # Raise the threshold
reflection_threshold = 0.85  # Strict requirement
```

**High-recall scenarios** (exploratory research):
```python
similarity_threshold = 0.70  # Lower the threshold
reflection_threshold = 0.75  # Relaxed requirement
```

### 3. Performance Optimization

**Batch processing**:
```python
queries = ["Query 1", "Query 2", "Query 3"]
results = [workflow.run(q, verbose=False) for q in queries]
```

**Caching results**:
```python
# Enable caching (already in the config)
"cache_enabled": true,
"cache_ttl_seconds": 3600
```

### 4. Error Handling

```python
try:
    result = workflow.run(query)
    if result['status'] == 'needs_reretrieval':
        print(f"Re-retrieval suggested: {result['suggestions']}")
except Exception as e:
    print(f"Error: {e}")
    # Fall back to simple RAG
```

---

## 🔍 Troubleshooting

### Issue 1: Scores Are Always Low (<0.5)

**Cause**: Poor retrieval quality or hallucinated answers

**Solution**:
1. Check whether `similarity_threshold` is too high
2. Increase `top_k` to 15-20
3. Check the quality of knowledge base documents

### Issue 2: The Planner Decomposes into Too Many Sub-queries

**Cause**: The query is too complex

**Solution**:
1. Lower `max_sub_queries` to 3
2. Simplify the original query
3. Use more specific terminology

### Issue 3: Response Time Too Long (>60 seconds)

**Cause**: Too many iterations or too many documents

**Solution**:
1. Lower `max_iterations` to 2
2. Reduce `top_k` to 5-8
3. Enable caching

### Issue 4: Ollama Connection Failure

**Solution**:
```bash
# Check Ollama status
ollama list

# Restart Ollama
systemctl restart ollama

# Test the connection
curl http://localhost:11434/api/tags
```

---

## 📁 File List

| File | Description |
|------|------|
| `agentic_rag_workflow.py` | Main workflow |
| `planner_agent.py` | Planner Agent |
| `reflector_agent.py` | Reflector Agent |
| `self_rag.py` | Self-RAG (golden parameters) |
| `agentic_rag_config.json` | Configuration file |
| `USAGE.md` | This manual |
| `DEPLOYMENT_REPORT.md` | Deployment report |

---

## 📞 Support

**Docs**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/USAGE.md`  
**Config**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/agentic_rag_config.json`  
**Logs**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/logs/`

---

**Version**: 2.0.0  
**Updated**: 2026-03-28  
**Status**: ✅ Production Ready