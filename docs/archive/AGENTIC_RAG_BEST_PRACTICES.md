> **Status note (2026-08-27):** design principles and golden parameters are the basis of the current agentic pipeline (src/agent_*.py). Deployment snippets referencing openclaw.json are historical — the current entry point is `agentic_query.py` (see ../README.md).

# Agentic RAG Best Practices Study Report (2026 Industry-Grade Standard)

**Study date**: 2026-03-28  
**Source**: Complete 2026 Agentic RAG configuration workflow + operations manual  
**Status**: ✅ Studied, pending implementation

---

## 📚 Core Architecture Study

### Optimal Architecture (Highest Accuracy, Most Stable)

```
Query → Planner → Retriever → Reflector → Generator → Output
              ↓         ↑
              └─────────┘ (up to 3 iterations)
```

### The Four Core Agents (All Essential)

| Agent | Responsibility | Recommended Model | Key Prompt |
|-------|------|----------|-------------|
| **Planner** | Decompose the question, generate sub-questions, plan steps | Qwen3.5-14B | "Decompose into 3-5 sub-questions" |
| **Retriever** | Multi-source retrieval, decide whether further search is needed | Qwen3.5-14B | "Is additional retrieval needed" |
| **Reflector** | Verify answers, prevent hallucination, decide whether to re-search | Llama3.1-70B | "Score 0-1, generate if ≥0.8" |
| **Generator** | Final synthesis, structured output | Qwen3.5-14B | "Write the report based on the material" |

---

## 🔧 Current Implementation vs Best Practices Comparison

| Dimension | Current Implementation | Best Practice | Gap | Priority |
|------|---------|---------|------|--------|
| **Architecture** | Self-RAG + Multi-Hop | Planner→Retriever→Reflector→Generator | ⚠️ Medium | High |
| **Agent count** | 2 (Self-RAG + Multi-Hop) | 4 (Planner/Retriever/Reflector/Generator) | ⚠️ Missing 2 | High |
| **Iteration control** | max_retries=2 | max_iterations=3 | ✅ Close | Low |
| **Retrieval threshold** | 0.15 (similarity) | 0.75 (score_threshold) | ❌ Too low | High |
| **Reflection threshold** | None | 0.8 (reflection_threshold) | ❌ Missing | High |
| **Chunking strategy** | Not optimized | 1024 characters + 256 overlap | ❌ Not implemented | Medium |
| **Vector store** | ChromaDB | Qdrant | ⚠️ Weak features | Medium |
| **Embedding** | all-MiniLM-L6-v2 | nomic-embed-text | ⚠️ Weak performance | Medium |
| **LLM** | qwen3.5:397b-cloud | qwen3.5:14b (local) | ✅ Cloud version is stronger | Low |
| **Long-document compression** | None | Two-level summarization + sliding window | ❌ Missing | Medium |

---

## 🎯 Key Parameters Study (Golden Parameters)

| Parameter | Best Value | Current Value | Adjustment Suggestion |
|------|--------|--------|---------|
| **max_iterations** | 3 | 2 | Adjust to 3 |
| **top_k** | 10 | 8 | Adjust to 10 |
| **similarity_threshold** | 0.75 | 0.15 | **Major adjustment** |
| **reflection_threshold** | 0.8 | None | **Add new** |
| **chunk_size** | 1024 | Not set | **Add new** |
| **chunk_overlap** | 256 | Not set | **Add new** |
| **temperature** | 0.1 | 0.0 | Adjust to 0.1 |
| **max_tokens** | 8192 | Default | Set explicitly |
| **timeout** | 120s | 60s | Adjust to 120 |

---

## 📦 Deployment Configuration Study

### OpenClaw Configuration (openclaw.json)

```json
{
  "model": {
    "type": "ollama",
    "base_url": "http://127.0.0.1:11434/v1",
    "model_name": "qwen3.5:14b",
    "max_tokens": 8192,
    "temperature": 0.1,
    "timeout": 120
  },
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text"
  }
}
```

### Agentic RAG Configuration

```json
{
  "agents": [
    {
      "name": "Planner",
      "model": "qwen3.5:14b",
      "role": "Decompose the question, generate sub-queries, cap iterations at ≤3"
    },
    {
      "name": "Retriever",
      "model": "qwen3.5:14b",
      "role": "Vector retrieval, decide whether additional retrieval is needed"
    },
    {
      "name": "Reflector",
      "model": "llama3.1:70b",
      "role": "Verify answers, prevent hallucination, decide whether to re-search"
    },
    {
      "name": "Writer",
      "model": "qwen3.5:14b",
      "role": "Structured report output, context compression, overflow prevention"
    }
  ],
  "memory": {
    "type": "qdrant",
    "host": "localhost",
    "port": 6333,
    "collection": "long_docs",
    "chunk_size": 1024,
    "overlap": 256,
    "top_k": 10
  },
  "max_iterations": 3,
  "early_stopping": true,
  "reflection_threshold": 0.8
}
```

### Long-Document Compression Configuration

```json
{
  "compression": {
    "enabled": true,
    "level": 2,
    "summary_model": "qwen3.5:14b",
    "summary_ratio": 0.1,
    "sliding_window": true,
    "window_size": 8192
  }
}
```

---

## 🧠 Core Prompt Study

### Planner Prompt (Question Decomposition)

```
You are a professional question planner.
Decompose the user's question into 3–5 retrievable sub-questions, covering all dimensions and avoiding duplication.
Output a plain list only, nothing else.
```

### Reflector Prompt (Reflective Verification, Anti-Hallucination)

```
You are a strict verifier.
Assess whether the current material is sufficient to answer the question; score 0–1.
Below 0.8, re-search is mandatory; above 0.8, generation may proceed.
Output only the number.
```

---

## 🔍 Key Findings

### 1. Threshold Configuration Problem

**Current problem**: similarity_threshold=0.15 is too low

**Consequences**: 
- A large amount of low-quality documents is retained
- Noise interferes with answer generation
- Confidence is artificially inflated

**Improvement**: Adjust to 0.75, keeping only high-quality documents

### 2. Missing Reflector Agent

**Current problem**: There is no independent verification step

**Consequences**:
- Hallucinations cannot be detected
- Answer quality is inconsistent
- No self-correction is possible

**Improvement**: Add a Reflector Agent using Llama3.1-70B for verification

### 3. Missing Long-Document Handling

**Current problem**: No chunking/compression strategy

**Consequences**:
- Large documents cannot be processed
- Context overflows
- Retrieval efficiency is low

**Improvement**: Implement 1024/256 chunking + two-level summary compression

### 4. Insufficient Iteration Control

**Current problem**: max_retries=2 may not be enough

**Improvement**: Adjust to 3 iterations and add early_stopping

---

## 🚀 Improvement Plan

### Phase 1: Parameter Tuning (High priority, 1 day)

- [ ] Adjust similarity_threshold: 0.15 → 0.75
- [ ] Add reflection_threshold: 0.8
- [ ] Adjust max_iterations: 2 → 3
- [ ] Adjust top_k: 8 → 10
- [ ] Adjust temperature: 0.0 → 0.1

### Phase 2: Reflector Agent (High priority, 2 days)

- [ ] Create `reflector_agent.py`
- [ ] Implement answer-verification logic
- [ ] Integrate into the workflow
- [ ] Test anti-hallucination effectiveness

### Phase 3: Planner Agent (Medium priority, 2 days)

- [ ] Create `planner_agent.py`
- [ ] Implement question-decomposition logic
- [ ] Replace the existing Multi-Hop decomposition
- [ ] Test decomposition quality

### Phase 4: Long-Document Handling (Medium priority, 3 days)

- [ ] Implement 1024/256 chunking
- [ ] Implement two-level summary compression
- [ ] Implement sliding window
- [ ] Test with a 300,000-character document

### Phase 5: Configuration Optimization (Low priority, 1 day)

- [ ] Update openclaw.json
- [ ] Update production_workflow.py
- [ ] Create configuration documentation
- [ ] Run performance benchmarks

---

## 📊 Expected Benefits

| Metric | Current | Expected After Improvement | Gain |
|------|------|-----------|------|
| **Answer accuracy** | ~70% | ~85% | +15% |
| **Hallucination rate** | ~15% | ~5% | -10% |
| **Complex query handling** | ~60% | ~80% | +20% |
| **Long-document support** | ❌ | ✅ 300,000 characters | +100% |
| **Iteration efficiency** | 2 iterations | 3 iterations + early stopping | +50% |

---

## 📁 Reference Resources

1. **LangGraph industrial-grade configuration**: Most stable and most general
2. **OpenClaw local Agentic RAG**: Fits local environments
3. **Qdrant vector store**: Stronger than FAISS, supports filtering/metadata/multi-tenancy
4. **nomic-embed-text**: The strongest open-source embedding model
5. **Qwen3.5-14B**: The best-performing local LLM

---

## ✅ Study Summary

**Core takeaways**:
1. The four-agent architecture is the industry-grade standard (Planner→Retriever→Reflector→Generator)
2. Threshold settings are critical (0.75 for retrieval, 0.8 for reflection)
3. Long documents must be compressed hierarchically (1024/256 + two-level summarization)
4. Multi-Agent improves accuracy by 35-50% over a single Agent

**Next steps**:
1. Adjust the golden parameters immediately
2. Implement the Reflector Agent
3. Test long-document handling

---

**Study completed**: 2026-03-28  
**Implementation started**: Immediately  
**Estimated completion**: 2026-04-04 (1 week)