# LangGraph 系统学习笔记

**学习时间**: 2026-03-28  
**来源**: production-agentic-rag-course (Week 7)  
**状态**: ✅ 已完成深入分析

---

## 📚 核心概念

### 什么是 LangGraph?

LangGraph 是一个基于**状态图 (State Graph)** 的工作流引擎，专为构建复杂的 AI Agent 系统设计。

**核心理念**: 将 Agent 工作流建模为**节点 (Nodes)** 和**边 (Edges)** 的图。

---

## 🏗️ 架构对比

### 传统 RAG vs Agentic RAG

**传统 RAG (线性流程)**:
```
Query → Retrieve → Generate → Answer
```

**Agentic RAG (图流程)**:
```
Query → Guardrail → [Decision]
                     ↓
            ┌────────┴────────┐
            ↓                 ↓
       Out-of-Scope        Retrieve
            ↓                 ↓
          END          Grade Documents
                           ↓
                    ┌──────┴──────┐
                    ↓             ↓
             Generate      Rewrite Query
                    ↓             ↓
                  END         (loop back)
```

---

## 🎯 核心组件

### 1. AgentState (状态定义)

```python
from typing import Annotated, List, Optional, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """定义工作流的状态结构"""
    
    # 对话消息列表 (使用 add_messages reducer 自动追加)
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 原始查询和重写后的查询
    original_query: Optional[str]
    rewritten_query: Optional[str]
    
    # 检索尝试次数
    retrieval_attempts: int
    
    # Guardrail 评估结果
    guardrail_result: Optional[GuardrailScoring]
    
    # 路由决策
    routing_decision: Optional[RoutingDecision]
    
    # 检索到的文档和评分
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]
    grading_results: List[GradingResult]
    
    # 元数据 (用于追踪)
    metadata: Dict[str, Any]
```

**关键点**:
- ✅ 使用 `TypedDict` 而非 `BaseModel` (LangGraph 2025 最佳实践)
- ✅ 使用 `Annotated` 和 `reducer` 处理消息追加
- ✅ 所有状态字段都是**类型安全**的

---

### 2. Context (运行时上下文)

```python
@dataclass
class Context:
    """运行时依赖注入"""
    
    # 客户端
    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    
    # 追踪
    langfuse_tracer: Optional[LangfuseTracer]
    trace: Optional[Any]
    langfuse_enabled: bool
    
    # 配置
    model_name: str
    temperature: float
    top_k: int
    max_retrieval_attempts: int
    guardrail_threshold: float
```

**关键点**:
- ✅ 使用 `dataclass` 进行依赖注入
- ✅ 所有配置参数集中管理
- ✅ 支持可选的 Langfuse 追踪

---

### 3. Nodes (节点函数)

每个节点都是**纯函数**,接收 `state` 和 `runtime`,返回状态更新。

#### Guardrail Node (防护栏)

```python
async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """评估查询是否在作用域内"""
    
    # 1. 获取最新查询
    query = get_latest_query(state["messages"])
    
    # 2. 创建 Langfuse 追踪
    span = runtime.context.langfuse_tracer.create_span(...)
    
    # 3. 构建提示词
    prompt = GUARDRAIL_PROMPT.format(question=query)
    
    # 4. 调用 LLM
    llm = runtime.context.ollama_client.get_langchain_model(...)
    structured_llm = llm.with_structured_output(GuardrailScoring)
    response = await structured_llm.ainvoke(prompt)
    
    # 5. 返回结果
    return {"guardrail_result": response}
```

**GuardrailScoring 模型**:
```python
class GuardrailScoring(BaseModel):
    score: float  # 0-100 分
    reason: str   # 评估理由
```

#### Grade Documents Node (文档评分)

```python
async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """评估检索到的文档是否相关"""
    
    # 1. 获取查询和上下文
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])
    
    # 2. 如果没有上下文，直接返回重写
    if not context:
        return {"routing_decision": "rewrite_query"}
    
    # 3. 调用 LLM 评分
    prompt = GRADE_DOCUMENTS_PROMPT.format(context=context, question=question)
    structured_llm = llm.with_structured_output(GradeDocuments)
    grading_response = await structured_llm.ainvoke(prompt)
    
    # 4. 转换为布尔值
    is_relevant = grading_response.binary_score == "yes"
    
    # 5. 路由决策
    route = "generate_answer" if is_relevant else "rewrite_query"
    
    return {
        "routing_decision": route,
        "grading_results": [grading_result]
    }
```

**GradeDocuments 模型**:
```python
class GradeDocuments(BaseModel):
    binary_score: str  # "yes" or "no"
    reasoning: str     # 评分理由
```

#### Rewrite Query Node (查询重写)

```python
async def ainvoke_rewrite_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str]:
    """重写查询以提高检索质量"""
    
    # 1. 获取原始查询和当前状态
    original_query = state.get("original_query")
    
    # 2. 构建重写提示词
    prompt = REWRITE_QUERY_PROMPT.format(
        original_query=original_query,
        previous_attempts=state.get("retrieval_attempts")
    )
    
    # 3. 调用 LLM 生成新查询
    llm = runtime.context.ollama_client.get_langchain_model(...)
    new_query = await llm.ainvoke(prompt)
    
    # 4. 返回重写后的查询
    return {"rewritten_query": new_query}
```

#### Generate Answer Node (答案生成)

```python
async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, AnyMessage]:
    """从相关文档生成最终答案"""
    
    # 1. 获取查询和相关文档
    query = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])
    
    # 2. 构建生成提示词
    prompt = GENERATE_ANSWER_PROMPT.format(
        question=query,
        context=context
    )
    
    # 3. 调用 LLM 生成答案
    llm = runtime.context.ollama_client.get_langchain_model(...)
    answer = await llm.ainvoke(prompt)
    
    # 4. 添加 HumanMessage 到消息列表
    return {"messages": [HumanMessage(content=answer)]}
```

---

### 4. StateGraph (状态图构建)

```python
from langgraph.graph import StateGraph, END, START

# 1. 创建图 (指定状态类型和上下文)
workflow = StateGraph(AgentState, context_schema=Context)

# 2. 添加节点
workflow.add_node("guardrail", ainvoke_guardrail_step)
workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
workflow.add_node("retrieve", ainvoke_retrieve_step)
workflow.add_node("tool_retrieve", ToolNode(tools))
workflow.add_node("grade_documents", ainvoke_grade_documents_step)
workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
workflow.add_node("generate_answer", ainvoke_generate_answer_step)

# 3. 添加边 (条件路由)
workflow.add_edge(START, "guardrail")  # 从 START 开始

workflow.add_conditional_edges(
    "guardrail",
    continue_after_guardrail,  # 路由函数
    {
        "continue": "retrieve",
        "out_of_scope": "out_of_scope",
    }
)

workflow.add_edge("out_of_scope", END)

workflow.add_conditional_edges(
    "grade_documents",
    lambda state: state.get("routing_decision", "generate_answer"),
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
    }
)

workflow.add_edge("rewrite_query", "retrieve")  # 循环
workflow.add_edge("generate_answer", END)

# 4. 编译图
compiled_graph = workflow.compile()
```

---

### 5. 路由函数 (Conditional Edges)

```python
def continue_after_guardrail(
    state: AgentState,
    runtime: Runtime[Context]
) -> Literal["continue", "out_of_scope"]:
    """根据 Guardrail 分数决定路由"""
    
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        return "continue"  # 默认继续
    
    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold
    
    return "continue" if score >= threshold else "out_of_scope"
```

**关键点**:
- ✅ 使用 `Literal` 类型确保返回有效的边名称
- ✅ 可以访问 `runtime.context` 获取配置
- ✅ 支持复杂的条件逻辑

---

## 🚀 执行流程

### 完整调用示例

```python
# 1. 初始化服务
service = AgenticRAGService(
    opensearch_client=opensearch,
    ollama_client=ollama,
    embeddings_client=embeddings,
    langfuse_tracer=langfuse,
    graph_config=GraphConfig(
        model="qwen3.5:397b-cloud",
        top_k=10,
        use_hybrid=True,
        max_retrieval_attempts=3,
        guardrail_threshold=80.0
    )
)

# 2. 调用工作流
result = await service.ask(
    query="Eph 受体和 ephrin 配体如何分类？",
    user_id="user123"
)

# 3. 解析结果
print(f"答案：{result['answer']}")
print(f"来源：{result['sources']}")
print(f"推理步骤：{result['reasoning_steps']}")
print(f"检索尝试：{result['retrieval_attempts']}")
print(f"Guardrail 分数：{result['guardrail_score']}")
```

### 状态流转示例

```
初始状态:
{
  "messages": [HumanMessage("Eph 受体如何分类？")],
  "retrieval_attempts": 0,
  "guardrail_result": None,
  ...
}

↓ [Guardrail Node]

状态更新:
{
  "guardrail_result": GuardrailScoring(score=95, reason="学术问题"),
  ...
}

↓ [条件路由：continue]

↓ [Retrieve Node]

状态更新:
{
  "sources": {...},
  "relevant_sources": [...],
  "retrieval_attempts": 1,
  ...
}

↓ [Grade Documents Node]

状态更新:
{
  "routing_decision": "generate_answer",
  "grading_results": [GradingResult(is_relevant=True, score=0.9)],
  ...
}

↓ [条件路由：generate_answer]

↓ [Generate Answer Node]

状态更新:
{
  "messages": [..., HumanMessage("Eph 受体分为 EphA 和 EphB 两个亚家族...")],
  ...
}

↓ [END]

最终输出:
{
  "answer": "Eph 受体分为 EphA 和 EphB 两个亚家族...",
  "sources": [...],
  "reasoning_steps": [
    "Validated query scope (score: 95/100)",
    "Retrieved documents (1 attempt(s))",
    "Graded documents (1 relevant)",
    "Generated answer from context"
  ]
}
```

---

## 💡 设计模式

### 1. 依赖注入 (Dependency Injection)

```python
# 使用 Context dataclass 集中管理依赖
@dataclass
class Context:
    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    ...

# 节点函数通过 runtime.context 访问
async def node(state: AgentState, runtime: Runtime[Context]):
    client = runtime.context.ollama_client
    config = runtime.context.top_k
```

**优势**:
- ✅ 测试时可轻松 mock
- ✅ 配置集中管理
- ✅ 类型安全

### 2. 结构化输出 (Structured Output)

```python
# 定义 Pydantic 模型
class GuardrailScoring(BaseModel):
    score: float
    reason: str

# 使用 with_structured_output
structured_llm = llm.with_structured_output(GuardrailScoring)
response = await structured_llm.ainvoke(prompt)

# 直接获得类型化的对象
print(response.score)  # float
print(response.reason)  # str
```

**优势**:
- ✅ 类型安全
- ✅ 自动验证
- ✅ IDE 支持

### 3. 条件路由 (Conditional Routing)

```python
# 简单条件
workflow.add_conditional_edges(
    "node",
    lambda state: "next_a" if condition else "next_b",
    {"next_a": "node_a", "next_b": "node_b"}
)

# 复杂条件 (使用函数)
workflow.add_conditional_edges(
    "node",
    my_routing_function,
    {"option1": "node1", "option2": "node2", "option3": "node3"}
)
```

### 4. 循环重试 (Retry Loop)

```python
# 从 rewrite_query 循环回 retrieve
workflow.add_edge("rewrite_query", "retrieve")

# 在节点中跟踪尝试次数
async def rewrite_query_node(state, runtime):
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= runtime.context.max_retrieval_attempts:
        # 达到最大尝试次数，强制生成
        return {"routing_decision": "generate_answer"}
    return {"rewritten_query": new_query, "retrieval_attempts": attempts + 1}
```

### 5. Langfuse 追踪 (Observability)

```python
# 在节点中创建 span
span = runtime.context.langfuse_tracer.create_span(
    trace=runtime.context.trace,
    name="guardrail_validation",
    input_data={"query": query},
    metadata={"node": "guardrail"}
)

# 结束 span
runtime.context.langfuse_tracer.end_span(
    span,
    output={"score": response.score},
    metadata={"execution_time_ms": execution_time}
)
```

**追踪内容**:
- ✅ 每个节点的输入/输出
- ✅ 执行时间
- ✅ LLM 调用详情
- ✅ 错误信息

---

## 📊 与我们的实现对比

| 特性 | 我们的实现 | LangGraph 实现 | 改进方向 |
|------|-----------|---------------|---------|
| **架构** | 线性流程 + 手动循环 | 状态图 + 自动路由 | 学习 StateGraph |
| **状态管理** | 字典传递 | TypedDict + Reducer | 更类型安全 |
| **依赖注入** | 构造函数传参 | Context dataclass | 更清晰 |
| **路由逻辑** | if/else 硬编码 | 条件边 (Conditional Edges) | 更灵活 |
| **循环重试** | while 循环 | 图边循环 | 更优雅 |
| **可观测性** | 基础日志 | Langfuse 完整追踪 | 建议添加 |
| **测试性** | 需要 mock 整个流程 | 可单独测试节点 | 更易测试 |
| **扩展性** | 添加节点需改主流程 | 添加节点即可 | 更易扩展 |

---

## 🛠️ 整合建议

### 阶段 1: 重构状态管理 (1-2 天)

```python
# 当前实现
state = {
    "query": query,
    "documents": docs,
    "answer": answer,
    ...
}

# 改进为 TypedDict
from typing import TypedDict, Annotated, List

class AgenticState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    rewritten_query: Optional[str]
    documents: List[Document]
    reflect_score: float
    retrieval_attempts: int
```

### 阶段 2: 引入 LangGraph (2-3 天)

```python
from langgraph.graph import StateGraph, END

# 创建图
workflow = StateGraph(AgenticState)

# 添加现有节点
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("generator", generator_node)

# 添加边
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reflector")
workflow.add_conditional_edges(
    "reflector",
    lambda x: "retriever" if x['reflect_score'] < 0.8 else "generator"
)
workflow.add_edge("generator", END)

# 编译
app = workflow.compile()
```

### 阶段 3: 添加 Langfuse (1 天)

```python
# 在 .env 添加
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# 在节点中添加追踪
span = langfuse.span(
    name="reflector_node",
    input={"query": query, "answer": answer},
    metadata={"node": "reflector"}
)
```

---

## 📋 关键代码片段

### 1. 完整的 StateGraph 示例

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    documents: List[str]
    answer: str
    score: float
    attempts: int

# 定义节点
def planner_node(state: AgentState):
    # Planner 逻辑
    sub_queries = [...]  # 拆解查询
    return {"query": sub_queries[0]}

def retriever_node(state: AgentState):
    # 检索逻辑
    docs = search(state["query"])
    return {"documents": docs}

def reflector_node(state: AgentState):
    # 反思逻辑
    score = evaluate(state["query"], state["documents"], state.get("answer", ""))
    return {"score": score}

def generator_node(state: AgentState):
    # 生成逻辑
    answer = generate(state["documents"])
    return {"messages": [HumanMessage(content=answer)]}

# 构建图
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("generator", generator_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reflector")

workflow.add_conditional_edges(
    "reflector",
    lambda state: "retriever" if state["score"] < 0.8 else "generator"
)

workflow.add_edge("generator", END)

# 编译
app = workflow.compile()

# 调用
result = app.invoke({
    "messages": [],
    "query": "你的问题",
    "documents": [],
    "answer": "",
    "score": 0.0,
    "attempts": 0
})
```

### 2. 条件路由函数

```python
from typing import Literal

def route_after_reflector(
    state: AgentState,
    runtime: Runtime[Context]
) -> Literal["generate", "rewrite", "end"]:
    """根据 Reflector 分数路由"""
    
    score = state.get("score", 0)
    attempts = state.get("attempts", 0)
    threshold = runtime.context.reflection_threshold
    max_attempts = runtime.context.max_iterations
    
    # 达到最大尝试次数
    if attempts >= max_attempts:
        return "end"
    
    # 分数达标
    if score >= threshold:
        return "generate"
    
    # 需要重写查询
    return "rewrite"
```

### 3. 使用 ToolNode

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def search_papers(query: str, top_k: int = 10) -> List[str]:
    """搜索学术论文"""
    return opensearch.search(query, top_k)

tools = [search_papers]

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", ToolNode(tools))
workflow.add_conditional_edges(
    "planner",
    tools_condition,  # 自动检测是否需要调用工具
    {
        "tools": "retrieve",
        END: END
    }
)
```

---

## 🎯 学习收获

### 核心优势

1. **清晰的状态流转** - 可视化工作流，易于理解和调试
2. **类型安全** - TypedDict + Pydantic 确保类型正确
3. **依赖注入** - Context dataclass 集中管理依赖
4. **条件路由** - 灵活的决策逻辑，支持复杂场景
5. **循环重试** - 优雅的循环机制，无需手动 while
6. **可观测性** - Langfuse 完整追踪每个节点
7. **易测试** - 节点是纯函数，可单独测试
8. **易扩展** - 添加节点即可扩展功能

### 最佳实践

1. ✅ 使用 `TypedDict` 而非 `BaseModel` 定义状态
2. ✅ 使用 `Annotated` 和 `reducer` 处理消息列表
3. ✅ 使用 `dataclass` 进行依赖注入
4. ✅ 使用 `with_structured_output` 确保类型安全
5. ✅ 使用 `Literal` 类型确保路由返回值正确
6. ✅ 使用 `Runtime[Context]` 访问配置
7. ✅ 使用 Langfuse 追踪所有节点
8. ✅ 节点函数保持简洁 (<50 行)

---

## 📞 资源链接

| 资源 | 链接 |
|------|------|
| **LangGraph 官方文档** | https://langchain-ai.github.io/langgraph/ |
| **生产课程代码** | `/Disk_2/claw_working_dir/production-agentic-rag-course/` |
| **Week 7 Notebook** | `notebooks/week7/week7_agentic_rag.ipynb` |
| **核心代码** | `src/services/agents/agentic_rag.py` |
| **节点实现** | `src/services/agents/nodes/` |

---

**学习时间**: 2026-03-28 18:45  
**分析师**: AI Assistant  
**状态**: ✅ 已完成深入分析，准备整合
