# LangGraph 实战迁移指南

**创建时间**: 2026-03-28  
**目标**: 从理论到实战，7 天掌握 LangGraph Agentic RAG  
**状态**: ✅ 生产就绪

---

## 🚀 快速开始 (5 分钟)

### 步骤 1: 导入 LangGraph 版

```python
from langgraph_agentic_rag import LangGraphAgenticRAG
from rag_core import SimpleEmbedding, DocumentStore
```

### 步骤 2: 加载知识库

```python
# 初始化 ChromaDB
doc_store = DocumentStore(
    'ephrin_papers',
    '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
)
embedder = SimpleEmbedding()

# 定义检索函数
def retriever(query, k=10):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

print(f"✓ 已加载 {doc_store.count()} 个文档")
```

### 步骤 3: 创建服务

```python
rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={
        "model": "qwen3.5:397b-cloud",
        "similarity_threshold": 0.75,
        "reflection_threshold": 0.8,
        "max_iterations": 3,
        "top_k": 10,
        "max_sub_queries": 5
    }
)

print("✅ LangGraph Agentic RAG 已初始化")
```

### 步骤 4: 运行查询

```python
result = rag.run("EphA2 与 EphB4 在癌症中的功能差异？")

print(f"答案：{result['generated_answer'][:200]}...")
print(f"耗时：{result['total_time']:.2f}秒")
print(f"Reflector 分数：{result['reflect_score']:.2f}")
print(f"检索尝试：{result['retrieval_attempts']} 次")
print(f"推理步骤：{result['reasoning_steps']}")
```

---

## 📚 7 天学习计划

### Day 1: 理解状态图思维

**学习目标**: 从线性思维转向图思维

**学习内容**:
1. 阅读 `LANGGRAPH_STUDY_NOTES.md` (前 3 章)
2. 理解 AgentState 定义
3. 理解状态流转

**实践任务**:
```python
# 任务 1: 打印初始状态
initial_state = {
    "messages": [],
    "original_query": "测试查询",
    "rewritten_query": None,
    "sub_queries": [],
    "documents": [],
    "retrieval_attempts": 0,
    "generated_answer": None,
    "reflect_score": 0.0,
    "routing_decision": None,
    "reasoning_steps": []
}

# 任务 2: 模拟状态更新
state_after_planner = {
    **initial_state,
    "sub_queries": ["子问题 1", "子问题 2"],
    "reasoning_steps": ["Planner 拆解为 2 个子问题"]
}

print(state_after_planner)
```

**验收标准**:
- ✅ 能画出状态流转图
- ✅ 能解释每个字段的作用
- ✅ 能手动模拟状态更新

---

### Day 2: 深入节点函数

**学习目标**: 理解 5 个核心节点的工作原理

**学习内容**:
1. 阅读 `langgraph_agentic_rag.py` 的节点函数
2. 理解每个节点的输入/输出
3. 理解 Runtime[Context] 依赖注入

**实践任务**:
```python
# 任务 1: 单独测试 Planner 节点
from langgraph_agentic_rag import planner_node, AgenticContext

# 创建 mock context
class MockContext:
    def __init__(self):
        self.planner = PlannerAgent()
        self.max_sub_queries = 5

class MockRuntime:
    def __init__(self):
        self.context = MockContext()

# 测试
state = {
    "original_query": "Eph 受体如何分类？",
    "rewritten_query": None
}
runtime = MockRuntime()

result = planner_node(state, runtime)
print(f"子问题：{result['sub_queries']}")
print(f"推理步骤：{result['reasoning_steps']}")
```

**验收标准**:
- ✅ 能单独测试每个节点
- ✅ 能解释节点的输入/输出
- ✅ 能修改节点逻辑

---

### Day 3: 掌握状态图构建

**学习目标**: 能独立构建 StateGraph

**学习内容**:
1. 阅读 `build_agentic_graph()` 函数
2. 理解 add_node / add_edge / add_conditional_edges
3. 理解 START / END 特殊节点

**实践任务**:
```python
from langgraph.graph import StateGraph, END, START

# 任务 1: 构建简化版图
simple_workflow = StateGraph(AgenticState)

# 只添加 3 个节点
simple_workflow.add_node("planner", planner_node)
simple_workflow.add_node("retriever", retriever_node)
simple_workflow.add_node("generator", generator_node)

# 添加边
simple_workflow.add_edge(START, "planner")
simple_workflow.add_edge("planner", "retriever")
simple_workflow.add_edge("retriever", "generator")
simple_workflow.add_edge("generator", END)

# 编译
simple_graph = simple_workflow.compile()

# 测试
result = simple_graph.invoke(initial_state)
print(f"答案：{result['generated_answer']}")
```

**验收标准**:
- ✅ 能独立构建简单 StateGraph
- ✅ 能解释条件路由原理
- ✅ 能画出完整工作流图

---

### Day 4: 理解条件路由

**学习目标**: 掌握动态路由决策

**学习内容**:
1. 阅读 `reflector_node` 和 `route_after_reflector`
2. 理解 Literal 类型在路由中的应用
3. 理解循环机制

**实践任务**:
```python
# 任务 1: 实现自定义路由函数
def custom_router(state, runtime) -> Literal["generate", "rewrite", "end"]:
    score = state.get("reflect_score", 0.0)
    attempts = state.get("retrieval_attempts", 0)
    
    # 自定义路由逻辑
    if score >= 0.9:  # 更严格
        return "end"
    elif attempts >= 2:  # 更少尝试
        return "end"
    else:
        return "rewrite"

# 任务 2: 应用到图中
workflow = StateGraph(AgenticState)
...
workflow.add_conditional_edges(
    "reflector",
    custom_router,  # 使用自定义路由
    {"generate": "generator", "rewrite": "rewrite_query", "end": END}
)
```

**验收标准**:
- ✅ 能实现自定义路由函数
- ✅ 能调整路由阈值
- ✅ 能解释循环终止条件

---

### Day 5: 添加监控和日志

**学习目标**: 实现完整的可观测性

**学习内容**:
1. 理解 logging 在节点中的应用
2. 学习 Langfuse 集成 (预留接口)
3. 实现推理步骤追踪

**实践任务**:
```python
# 任务 1: 增强日志输出
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def planner_node_verbose(state, runtime):
    logger.info(f"🔍 Planner 开始处理：{state['original_query'][:50]}...")
    
    result = planner_node(state, runtime)
    
    logger.info(f"✅ Planner 完成：{len(result['sub_queries'])} 个子问题")
    logger.info(f"📝 推理步骤：{result['reasoning_steps']}")
    
    return result

# 任务 2: 创建详细报告
def generate_report(result):
    report = f"""
=== Agentic RAG 执行报告 ===

查询：{result['original_query']}
答案：{result['generated_answer'][:200]}...

性能指标:
- 总耗时：{result['total_time']:.2f}秒
- 检索尝试：{result['retrieval_attempts']} 次
- Reflector 分数：{result['reflect_score']:.2f}

推理步骤:
"""
    for i, step in enumerate(result['reasoning_steps'], 1):
        report += f"  {i}. {step}\n"
    
    return report

print(generate_report(result))
```

**验收标准**:
- ✅ 能添加详细日志
- ✅ 能生成执行报告
- ✅ 能追踪每个节点耗时

---

### Day 6: 性能优化

**学习目标**: 优化查询性能

**学习内容**:
1. 理解性能瓶颈 (检索/生成/评估)
2. 学习参数调优
3. 实现缓存策略 (可选)

**实践任务**:
```python
# 任务 1: 参数敏感性测试
configs = [
    {"reflection_threshold": 0.7, "max_iterations": 2},  # 快速模式
    {"reflection_threshold": 0.8, "max_iterations": 3},  # 标准模式
    {"reflection_threshold": 0.9, "max_iterations": 5},  # 高质量模式
]

for i, config in enumerate(configs, 1):
    rag = LangGraphAgenticRAG(retriever_fn=retriever, config=config)
    result = rag.run("测试查询")
    
    print(f"配置 {i}:")
    print(f"  分数：{result['reflect_score']:.2f}")
    print(f"  耗时：{result['total_time']:.2f}秒")
    print(f"  尝试：{result['retrieval_attempts']} 次")

# 任务 2: 实现简单缓存 (可选)
cache = {}

def cached_retriever(query, k=10):
    if query in cache:
        return cache[query]
    result = retriever(query, k)
    cache[query] = result
    return result
```

**验收标准**:
- ✅ 能调整参数优化性能
- ✅ 能实现简单缓存
- ✅ 能分析性能瓶颈

---

### Day 7: 实战项目

**学习目标**: 完成一个完整的实战项目

**实战任务**: 构建 Eph/Ephrin 研究助手

```python
# 完整项目代码
from langgraph_agentic_rag import LangGraphAgenticRAG
from rag_core import SimpleEmbedding, DocumentStore
import json

# 1. 初始化
doc_store = DocumentStore('ephrin_papers', 'chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=10):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={
        "model": "qwen3.5:397b-cloud",
        "reflection_threshold": 0.8,
        "max_iterations": 3,
        "top_k": 10
    }
)

# 2. 研究问题列表
research_questions = [
    "Eph 受体和 ephrin 配体如何分类？",
    "cis-interaction 的分子机制是什么？",
    "EphA2 与 EphB4 在癌症中的功能差异？",
    "Eph 受体在肿瘤微环境中的作用？",
    "tetramerization inhibitor 的作用机制？"
]

# 3. 批量查询
results = []
for i, question in enumerate(research_questions, 1):
    print(f"\n{'='*80}")
    print(f"[问题 {i}/{len(research_questions)}] {question}")
    print("="*80)
    
    result = rag.run(question)
    results.append({
        "question": question,
        "answer": result['generated_answer'],
        "score": result['reflect_score'],
        "time": result['total_time'],
        "attempts": result['retrieval_attempts']
    })
    
    print(f"✅ 答案：{result['generated_answer'][:200]}...")
    print(f"⏱️  耗时：{result['total_time']:.2f}秒")
    print(f"📊 分数：{result['reflect_score']:.2f}")

# 4. 生成报告
report = {
    "timestamp": "2026-03-28",
    "total_questions": len(results),
    "avg_score": sum(r['score'] for r in results) / len(results),
    "avg_time": sum(r['time'] for r in results) / len(results),
    "results": results
}

with open('research_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n💾 报告已保存：research_report.json")
print(f"📊 平均分数：{report['avg_score']:.2f}")
print(f"⏱️  平均耗时：{report['avg_time']:.2f}秒")
```

**验收标准**:
- ✅ 完成批量查询
- ✅ 生成完整报告
- ✅ 能解释所有结果

---

## 🎯 API 参考手册

### LangGraphAgenticRAG 类

```python
class LangGraphAgenticRAG:
    """LangGraph Agentic RAG 服务封装"""
    
    def __init__(self, retriever_fn, config: Optional[Dict] = None):
        """
        初始化服务
        
        Args:
            retriever_fn: 检索函数 (query, k) -> List[Dict]
            config: 配置字典
                - model: str, LLM 模型
                - similarity_threshold: float, 检索阈值
                - reflection_threshold: float, 反思阈值
                - max_iterations: int, 最大迭代次数
                - top_k: int, 检索数量
                - max_sub_queries: int, 最大子问题数
        """
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        运行完整工作流
        
        Args:
            query: 用户查询
            verbose: 是否输出详细日志
        
        Returns:
            result: 完整结果
                - generated_answer: str, 生成的答案
                - reflect_score: float, Reflector 分数
                - retrieval_attempts: int, 检索尝试次数
                - total_time: float, 总耗时 (秒)
                - reasoning_steps: List[str], 推理步骤
                - documents: List[Dict], 来源文档
        """
    
    def get_reasoning_steps(self, result: Dict) -> List[str]:
        """提取推理步骤"""
    
    def get_sources(self, result: Dict) -> List[Dict]:
        """提取来源文档"""
```

---

## 🔧 常见问题 (FAQ)

### Q1: 如何调整 Reflector 的严格程度？

```python
# 更严格 (要求更高质量)
rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={"reflection_threshold": 0.9}  # 提高到 0.9
)

# 更宽松 (更快响应)
rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={"reflection_threshold": 0.7}  # 降低到 0.7
)
```

### Q2: 如何限制最大检索次数？

```python
rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={"max_iterations": 2}  # 最多 2 次检索
)
```

### Q3: 如何获取更多来源文档？

```python
rag = LangGraphAgenticRAG(
    retriever_fn=retriever,
    config={"top_k": 15}  # 检索 15 个文档
)
```

### Q4: 如何关闭详细日志？

```python
result = rag.run(query, verbose=False)  # 关闭日志
```

### Q5: 如何获取推理步骤？

```python
result = rag.run(query)
steps = result['reasoning_steps']
for step in steps:
    print(f"- {step}")
```

### Q6: 如何获取来源文档？

```python
result = rag.run(query)
sources = rag.get_sources(result)
for i, doc in enumerate(sources[:5], 1):
    print(f"[{i}] {doc.get('title', '无标题')}")
```

### Q7: 如何处理长查询？

```python
# LangGraph 版自动处理长查询
# Planner 会自动拆解为子问题
result = rag.run("很长的查询...")
```

### Q8: 如何自定义节点逻辑？

```python
# 创建自定义节点
def custom_planner_node(state, runtime):
    # 自定义 Planner 逻辑
    ...
    return {"sub_queries": [...]}

# 替换默认节点
workflow = StateGraph(AgenticState)
workflow.add_node("planner", custom_planner_node)  # 使用自定义节点
...
```

---

## 📊 性能基准

### 标准配置性能

```python
config = {
    "model": "qwen3.5:397b-cloud",
    "reflection_threshold": 0.8,
    "max_iterations": 3,
    "top_k": 10
}
```

**预期性能**:
- 简单查询：15-25 秒，1 次检索，分数>0.8
- 中等查询：30-45 秒，2 次检索，分数>0.8
- 复杂查询：45-90 秒，2-3 次检索，分数>0.8

### 快速配置性能

```python
config = {
    "reflection_threshold": 0.7,
    "max_iterations": 2,
    "top_k": 5
}
```

**预期性能**:
- 简单查询：10-15 秒，1 次检索，分数>0.7
- 中等查询：20-30 秒，1-2 次检索，分数>0.7
- 复杂查询：30-60 秒，2 次检索，分数>0.7

### 高质量配置性能

```python
config = {
    "reflection_threshold": 0.9,
    "max_iterations": 5,
    "top_k": 15
}
```

**预期性能**:
- 简单查询：20-30 秒，1 次检索，分数>0.9
- 中等查询：45-60 秒，2-3 次检索，分数>0.9
- 复杂查询：60-120 秒，3-5 次检索，分数>0.9

---

## 🎓 进阶学习路径

### Level 1: 基础使用者 (1 周)
- ✅ 完成 7 天学习计划
- ✅ 能使用 LangGraphAgenticRAG
- ✅ 能调整参数优化性能

### Level 2: 高级使用者 (2 周)
- 学习修改节点逻辑
- 学习添加自定义节点
- 学习实现 Guardrail

### Level 3: 开发者 (1 月)
- 学习 LangGraph 高级特性
- 学习实现多 Agent 协作
- 学习集成 Langfuse 监控

### Level 4: 专家 (持续)
- 贡献代码到项目
- 分享最佳实践
- 帮助社区解决问题

---

## 📞 资源汇总

| 资源 | 位置 | 说明 |
|------|------|------|
| **核心代码** | `langgraph_agentic_rag.py` | 19.5KB 主文件 |
| **学习笔记** | `LANGGRAPH_STUDY_NOTES.md` | 16.6KB 理论 |
| **整合指南** | `LANGGRAPH_INTEGRATION.md` | 11.0KB 手册 |
| **对比文档** | `VERSION_COMPARISON.md` | 11.0KB 对比 |
| **实战指南** | `LANGGRAPH_MIGRATION_GUIDE.md` | 本文档 |
| **测试脚本** | `test_comparison.py` | 10.6KB 测试 |
| **官方文档** | https://langchain-ai.github.io/langgraph/ | LangGraph 文档 |

---

## 🎉 开始你的 LangGraph 之旅！

**第一天任务**:
1. ✅ 运行快速开始代码
2. ✅ 阅读 `LANGGRAPH_STUDY_NOTES.md` 前 3 章
3. ✅ 理解 AgentState 定义

**记住**: 从线性思维转向图思维是关键！

---

**创建时间**: 2026-03-28 19:50  
**作者**: AI Assistant  
**状态**: ✅ 生产就绪  
**版本**: v1.0
