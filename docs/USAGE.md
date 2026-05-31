# Agentic RAG 使用手册 (2026 工业级标准)

**版本**: 2.0.0  
**更新日期**: 2026-03-28  
**状态**: ✅ 生产就绪

---

## 📚 目录

1. [快速开始](#快速开始)
2. [架构说明](#架构说明)
3. [配置参数](#配置参数)
4. [使用示例](#使用示例)
5. [API 参考](#api-参考)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 已安装，无需额外操作
pip install requests langchain langchain-community
```

### 2. 加载知识库

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

### 3. 运行 Agentic RAG

```python
from agentic_rag_workflow import AgenticRAGWorkflow

workflow = AgenticRAGWorkflow(retriever, model="qwen3.5:397b-cloud")

result = workflow.run("EphA2 与 EphB4 在癌症中的功能差异？")

print(f"答案：{result['answer']}")
print(f"置信度：{result['reflect_score']:.2f}")
print(f"状态：{result['status']}")
```

---

## 🏗️ 架构说明

### 完整工作流

```
Query → Planner → Retriever → Generator → Reflector → Output
                              ↑                    ↓
                              └──── 重查 (if <0.8) ──┘
```

### 四大核心 Agent

| Agent | 职责 | 关键功能 |
|-------|------|---------|
| **Planner** | 拆解问题 | 3-5 个子问题、检索策略 |
| **Retriever** | 向量检索 | 相似度 0.75+、top_k=10 |
| **Generator** | 生成答案 | 基于文档、防幻觉 |
| **Reflector** | 校验答案 | 打分 0-1、≥0.8 输出 |

---

## ⚙️ 配置参数

### 黄金参数 (已优化)

```python
GOLDEN_PARAMS = {
    "similarity_threshold": 0.75,  # 检索阈值
    "reflection_threshold": 0.8,   # 反思阈值
    "max_iterations": 3,           # 最大迭代
    "top_k": 10,                   # 检索数量
    "max_sub_queries": 5,          # 最大子问题
    "temperature": 0.1,            # LLM 温度
    "timeout": 120,                # 超时 (秒)
    "model": "qwen3.5:397b-cloud"  # 模型
}
```

### 配置文件

编辑 `agentic_rag_config.json`:

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

## 📖 使用示例

### 示例 1: 简单查询

```python
from agentic_rag_workflow import AgenticRAGWorkflow

workflow = AgenticRAGWorkflow(retriever)

result = workflow.run("Eph 受体的功能是什么？")

print(f"答案：{result['answer'][:500]}")
print(f"置信度：{result['reflect_score']:.2f}")
```

### 示例 2: 复杂查询 (对比)

```python
result = workflow.run("EphA2 与 EphB4 在癌症中的功能差异？")

# Planner 会自动拆解为:
# 1. EphA2 在癌症中的功能
# 2. EphB4 在癌症中的功能
# 3. 两者的对比研究

print(f"子问题数：{len(result['plan']['sub_queries'])}")
print(f"检索策略：{result['plan']['search_strategy']}")
```

### 示例 3: 复杂查询 (因果)

```python
result = workflow.run("cis-interaction 如何影响 trans-signaling？")

# Planner 会拆解为:
# 1. cis-interaction 的分子机制
# 2. trans-signaling 的激活过程
# 3. 两者的调控关系

if result['reflect_score'] < 0.8:
    print("需要重查")
    print(f"建议：{result['suggestions']}")
```

### 示例 4: 使用 Reflector 独立校验

```python
from reflector_agent import ReflectorAgent

reflector = ReflectorAgent(model="qwen3.5:397b-cloud")

result = reflector.reflect(
    question="Eph 受体的功能？",
    answer="Eph 受体是受体酪氨酸激酶",
    documents=[{"text": "Eph receptors are RTKs...", "similarity": 0.85}]
)

print(f"分数：{result['score']:.2f}")
print(f"需要重查：{result['needs_reretrieval']}")
print(f"问题：{result['issues']}")
```

### 示例 5: 使用 Planner 拆解问题

```python
from planner_agent import PlannerAgent

planner = PlannerAgent(model="qwen3.5:397b-cloud")

plan = planner.plan("EphA2 与 EphB4 的功能差异？")

print(f"复杂度：{'复杂' if plan['is_complex'] else '简单'}")
print(f"子问题：{len(plan['sub_queries'])}")
for sq in plan['sub_queries']:
    print(f"  - {sq['query']}")
```

---

## 🔧 API 参考

### AgenticRAGWorkflow

```python
class AgenticRAGWorkflow:
    def __init__(self, retriever_fn, model="qwen3.5:397b-cloud"):
        """初始化工作流"""
    
    def run(self, query: str, verbose: bool = True) -> dict:
        """
        运行完整工作流
        
        Args:
            query: 用户查询
            verbose: 是否输出详细日志
            
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
        """获取统计信息"""
```

### PlannerAgent

```python
class PlannerAgent:
    def __init__(self, model="qwen3.5:397b-cloud", 
                 max_sub_queries=5, max_iterations=3):
        """初始化 Planner"""
    
    def plan(self, query: str) -> dict:
        """
        拆解问题
        
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
        """初始化 Reflector"""
    
    def reflect(self, question: str, answer: str, 
                documents: List[Dict]) -> dict:
        """
        校验答案
        
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

## 📊 最佳实践

### 1. 查询优化

✅ **好的查询**:
- 具体明确："EphA2 在癌症中的功能？"
- 包含实体："EphA2 与 EphB4 的差异？"
- 有上下文："在肿瘤微环境中，Eph 受体的作用？"

❌ **避免的查询**:
- 太宽泛："Eph 受体？"
- 缺少上下文："它的作用？"
- 多重否定："不是不重要的功能？"

### 2. 参数调优

**高准确率场景** (论文写作):
```python
similarity_threshold = 0.80  # 提高阈值
reflection_threshold = 0.85  # 严格要求
```

**高召回率场景** (探索性研究):
```python
similarity_threshold = 0.70  # 降低阈值
reflection_threshold = 0.75  # 放宽要求
```

### 3. 性能优化

**批量处理**:
```python
queries = ["查询 1", "查询 2", "查询 3"]
results = [workflow.run(q, verbose=False) for q in queries]
```

**缓存结果**:
```python
# 启用缓存 (已在配置中)
"cache_enabled": true,
"cache_ttl_seconds": 3600
```

### 4. 错误处理

```python
try:
    result = workflow.run(query)
    if result['status'] == 'needs_reretrieval':
        print(f"建议重查：{result['suggestions']}")
except Exception as e:
    print(f"错误：{e}")
    # 回退到简单 RAG
```

---

## 🔍 故障排除

### 问题 1: 分数总是很低 (<0.5)

**原因**: 检索质量差或答案幻觉

**解决**:
1. 检查 `similarity_threshold` 是否过高
2. 增加 `top_k` 到 15-20
3. 检查知识库文档质量

### 问题 2: Planner 拆解太多子问题

**原因**: 查询太复杂

**解决**:
1. 降低 `max_sub_queries` 到 3
2. 简化原始查询
3. 使用更具体的术语

### 问题 3: 响应时间过长 (>60 秒)

**原因**: 迭代次数多或文档太多

**解决**:
1. 降低 `max_iterations` 到 2
2. 减少 `top_k` 到 5-8
3. 启用缓存

### 问题 4: Ollama 连接失败

**解决**:
```bash
# 检查 Ollama 状态
ollama list

# 重启 Ollama
systemctl restart ollama

# 测试连接
curl http://localhost:11434/api/tags
```

---

## 📁 文件清单

| 文件 | 说明 |
|------|------|
| `agentic_rag_workflow.py` | 主工作流 |
| `planner_agent.py` | Planner Agent |
| `reflector_agent.py` | Reflector Agent |
| `self_rag.py` | Self-RAG (黄金参数) |
| `agentic_rag_config.json` | 配置文件 |
| `USAGE.md` | 本手册 |
| `DEPLOYMENT_REPORT.md` | 部署报告 |

---

## 📞 支持

**文档**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/USAGE.md`  
**配置**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/agentic_rag_config.json`  
**日志**: `/Disk_2/claw_working_dir/ephrin_agentic_rag/logs/`

---

**版本**: 2.0.0  
**更新**: 2026-03-28  
**状态**: ✅ 生产就绪
