#!/usr/bin/env python3
"""
Agentic RAG 工作流 - LangGraph 重构版 (2026 工业级标准)

架构：Planner → Retriever → Generator → Reflector → Output
                                      ↓
                              └── 重查 (if <0.8) ──┘

使用 LangGraph StateGraph 实现，类型安全 + 可观测性

黄金参数:
- similarity_threshold: 0.75
- reflection_threshold: 0.8
- max_iterations: 3
- top_k: 10
- max_sub_queries: 5
"""

import sys
import time
import logging
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime

# LangGraph 导入
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage

# 本地导入
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from planner_agent import PlannerAgent
from reflector_agent import ReflectorAgent
from self_rag import SelfRAGWorkflow, SelfRAGEvaluator
from rag_core import SimpleEmbedding, DocumentStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================

GOLDEN_PARAMS = {
    "similarity_threshold": 0.75,
    "reflection_threshold": 0.8,
    "max_iterations": 3,
    "top_k": 10,
    "max_sub_queries": 5,
    "temperature": 0.1,
    "timeout": 120,
    "model": "qwen3.5:397b-cloud"
}


# ==================== AgentState (状态定义) ====================

class AgenticState(TypedDict):
    """
    Agentic RAG 状态定义 (LangGraph TypedDict 模式)
    
    遵循 2025 LangGraph 最佳实践:
    - 使用 TypedDict 而非 BaseModel
    - 使用 Annotated + reducer 处理消息列表
    - 所有字段类型安全
    """
    
    # 对话消息列表 (自动追加，不会覆盖)
    messages: Annotated[List[Dict[str, str]], add_messages]
    
    # 查询相关
    original_query: Optional[str]
    rewritten_query: Optional[str]
    sub_queries: List[str]
    
    # 检索相关
    documents: List[Dict[str, Any]]
    retrieval_attempts: int
    
    # 生成相关
    generated_answer: Optional[str]
    
    # 评估相关
    reflect_score: float
    reflect_reasoning: Optional[str]
    
    # 路由决策
    routing_decision: Optional[str]  # "generate", "rewrite", "end"
    
    # 元数据
    start_time: Optional[float]
    total_time: Optional[float]
    reasoning_steps: List[str]


# ==================== Context (运行时依赖) ====================

class AgenticContext:
    """
    运行时依赖注入 (dataclass 模式)
    
    集中管理所有依赖和配置参数
    """
    
    def __init__(
        self,
        retriever_fn,
        model: str = "qwen3.5:397b-cloud",
        similarity_threshold: float = 0.75,
        reflection_threshold: float = 0.8,
        max_iterations: int = 3,
        top_k: int = 10,
        max_sub_queries: int = 5
    ):
        self.retriever_fn = retriever_fn
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.reflection_threshold = reflection_threshold
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.max_sub_queries = max_sub_queries
        
        # 初始化 Agent
        self.planner = PlannerAgent(
            model=model,
            max_sub_queries=max_sub_queries,
            max_iterations=max_iterations
        )
        
        self.reflector = ReflectorAgent(
            model=model,
            reflection_threshold=reflection_threshold
        )
        
        self.rag = SelfRAGWorkflow(
            retriever_fn,
            evaluator=SelfRAGEvaluator(model=model),
            similarity_threshold=similarity_threshold,
            reflection_threshold=reflection_threshold,
            max_retries=max_iterations,
            top_k=top_k
        )
        
        logger.info(f"✅ AgenticContext 已初始化")
        logger.info(f"   模型：{model}")
        logger.info(f"   similarity_threshold: {similarity_threshold}")
        logger.info(f"   reflection_threshold: {reflection_threshold}")
        logger.info(f"   max_iterations: {max_iterations}")
        logger.info(f"   top_k: {top_k}")


# ==================== 节点函数 (Nodes) ====================

def planner_node(state: AgenticState, runtime: Runtime[AgenticContext]) -> Dict[str, Any]:
    """
    Planner 节点：拆解复杂查询为子问题
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 状态更新 (sub_queries, reasoning_steps)
    """
    logger.info("NODE: planner")
    start_time = time.time()
    
    # 获取查询 (优先使用重写后的查询)
    query = state.get("rewritten_query") or state.get("original_query", "")
    
    if not query:
        logger.warning("No query found in state")
        return {"sub_queries": [""], "reasoning_steps": ["Error: No query provided"]}
    
    logger.info(f"Planning query: {query[:100]}...")
    
    # 调用 Planner
    plan = runtime.context.planner.plan(query)
    sub_queries = plan.get("sub_queries", [query])
    is_complex = plan.get("is_complex", False)
    
    # 记录推理步骤
    reasoning_steps = [
        f"Planner 拆解查询 ({len(sub_queries)} 个子问题)",
        f"复杂度：{'复杂' if is_complex else '简单'}"
    ]
    
    execution_time = (time.time() - start_time) * 1000
    logger.info(f"✓ Planner 完成 ({execution_time:.0f}ms): {len(sub_queries)} 个子问题")
    
    return {
        "sub_queries": sub_queries,
        "reasoning_steps": reasoning_steps,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }


def retriever_node(state: AgenticState, runtime: Runtime[AgenticContext]) -> Dict[str, Any]:
    """
    Retriever 节点：检索文档
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 状态更新 (documents)
    """
    logger.info("NODE: retriever")
    start_time = time.time()
    
    # 获取子问题
    sub_queries = state.get("sub_queries", [])
    
    if not sub_queries or sub_queries == [""]:
        logger.warning("No sub-queries to retrieve")
        return {"documents": []}
    
    # 合并所有子查询进行检索
    combined_query = " ".join(sub_queries[:3])  # 限制前 3 个
    
    logger.info(f"Retrieving for: {combined_query[:100]}...")
    
    # 调用检索函数
    try:
        documents = runtime.context.retriever_fn(combined_query, k=runtime.context.top_k)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✓ Retriever 完成 ({execution_time:.0f}ms): {len(documents)} 个文档")
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"检索到 {len(documents)} 个文档")
        
        return {
            "documents": documents,
            "reasoning_steps": reasoning_steps
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"documents": [], "reasoning_steps": [f"检索失败：{str(e)}"]}


def generator_node(state: AgenticState, runtime: Runtime[AgenticContext]) -> Dict[str, Any]:
    """
    Generator 节点：从文档生成答案
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 状态更新 (generated_answer, messages)
    """
    logger.info("NODE: generator")
    start_time = time.time()
    
    # 获取查询和文档
    query = state.get("original_query", "")
    documents = state.get("documents", [])
    
    if not documents:
        logger.warning("No documents for generation")
        return {
            "generated_answer": "未找到相关文档，无法生成答案。",
            "messages": [HumanMessage(content="未找到相关文档，无法生成答案。")]
        }
    
    logger.info(f"Generating answer from {len(documents)} documents...")
    
    # 使用 SelfRAG 的 Generator
    try:
        # 构建上下文
        context = "\n\n".join([
            f"[文档 {i+1}]\n{doc.get('content', doc.get('text', str(doc)))}"
            for i, doc in enumerate(documents[:5])  # 限制前 5 个文档
        ])
        
        # 调用 LLM 生成答案
        prompt = f"""基于以下文档回答问题。

问题：{query}

文档:
{context}

请用中文回答，确保答案准确、完整、有逻辑。"""

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = runtime.context.rag.llm
        chain = llm | StrOutputParser()
        answer = chain.invoke(prompt)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✓ Generator 完成 ({execution_time:.0f}ms): {len(answer)} 字符")
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"生成答案 ({len(answer)} 字符)")
        
        return {
            "generated_answer": answer,
            "messages": [HumanMessage(content=answer)],
            "reasoning_steps": reasoning_steps
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "generated_answer": f"生成答案时出错：{str(e)}",
            "messages": [HumanMessage(content=f"生成答案时出错：{str(e)}")]
        }


def reflector_node(state: AgenticState, runtime: Runtime[AgenticContext]) -> Dict[str, Any]:
    """
    Reflector 节点：评估答案质量并决定路由
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 状态更新 (reflect_score, routing_decision)
    """
    logger.info("NODE: reflector")
    start_time = time.time()
    
    # 获取查询、文档和答案
    query = state.get("original_query", "")
    documents = state.get("documents", [])
    answer = state.get("generated_answer", "")
    
    if not answer:
        logger.warning("No answer to reflect on")
        return {
            "reflect_score": 0.0,
            "routing_decision": "end"
        }
    
    logger.info(f"Reflecting on answer quality...")
    
    # 调用 Reflector
    try:
        reflect_result = runtime.context.reflector.reflect(query, answer, documents[:5])
        
        score = reflect_result.get("score", 0.0)
        reasoning = reflect_result.get("reasoning", "")
        is_sufficient = reflect_result.get("is_sufficient", False)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✓ Reflector 完成 ({execution_time:.0f}ms): 分数={score:.2f}")
        
        # 决定路由
        attempts = state.get("retrieval_attempts", 0)
        max_attempts = runtime.context.max_iterations
        
        if score >= runtime.context.reflection_threshold:
            routing = "end"  # 答案质量高，结束
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Reflector 评分：{score:.2f} (通过)")
            
        elif attempts >= max_attempts:
            routing = "end"  # 达到最大尝试次数，强制结束
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Reflector 评分：{score:.2f} (达到最大尝试 {attempts}/{max_attempts})")
            
        else:
            routing = "rewrite"  # 需要重写查询并重查
            reasoning_steps = state.get("reasoning_steps", [])
            reasoning_steps.append(f"Reflector 评分：{score:.2f} (需要重写)")
        
        return {
            "reflect_score": score,
            "reflect_reasoning": reasoning,
            "routing_decision": routing,
            "reasoning_steps": reasoning_steps
        }
        
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {
            "reflect_score": 0.0,
            "reflect_reasoning": f"Reflector 出错：{str(e)}",
            "routing_decision": "end"
        }


def rewrite_query_node(state: AgenticState, runtime: Runtime[AgenticContext]) -> Dict[str, Any]:
    """
    Rewrite Query 节点：重写查询以提高检索质量
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 状态更新 (rewritten_query)
    """
    logger.info("NODE: rewrite_query")
    start_time = time.time()
    
    # 获取原始查询
    original_query = state.get("original_query", "")
    current_answer = state.get("generated_answer", "")
    
    if not original_query:
        logger.warning("No query to rewrite")
        return {"rewritten_query": original_query}
    
    logger.info(f"Rewriting query to improve retrieval...")
    
    # 使用 Reflector 的建议来重写
    try:
        # 构建重写提示词
        prompt = f"""你是一个查询优化专家。请根据之前的失败尝试，重写以下查询以提高检索质量。

原始查询：{original_query}

之前生成的答案 (不够好):
{current_answer}

请重写查询，使其更具体、更清晰，能够更好地检索到相关信息。
只输出重写后的查询，不要其他内容。"""

        llm = runtime.context.rag.llm
        chain = llm | StrOutputParser()
        rewritten_query = chain.invoke(prompt)
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✓ Query Rewriter 完成 ({execution_time:.0f}ms): {rewritten_query[:50]}...")
        
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"重写查询：{rewritten_query[:50]}...")
        
        return {
            "rewritten_query": rewritten_query,
            "reasoning_steps": reasoning_steps
        }
        
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return {"rewritten_query": original_query}


# ==================== 路由函数 ====================

def route_after_reflector(state: AgenticState, runtime: Runtime[AgenticContext]) -> str:
    """
    Reflector 后的路由决策函数
    
    :param state: 当前状态
    :param runtime: 运行时上下文
    :return: 路由目标 ("generate", "rewrite", "end")
    """
    routing_decision = state.get("routing_decision", "end")
    logger.info(f"Routing decision: {routing_decision}")
    return routing_decision


# ==================== LangGraph 工作流构建 ====================

def build_agentic_graph(retriever_fn, config: Optional[Dict] = None):
    """
    构建并编译 LangGraph 工作流
    
    :param retriever_fn: 检索函数
    :param config: 可选配置覆盖
    :return: 编译后的图
    """
    logger.info("="*70)
    logger.info("构建 LangGraph Agentic RAG 工作流")
    logger.info("="*70)
    
    # 创建上下文
    ctx = AgenticContext(
        retriever_fn=retriever_fn,
        model=config.get("model", "qwen3.5:397b-cloud") if config else "qwen3.5:397b-cloud",
        similarity_threshold=config.get("similarity_threshold", 0.75) if config else 0.75,
        reflection_threshold=config.get("reflection_threshold", 0.8) if config else 0.8,
        max_iterations=config.get("max_iterations", 3) if config else 3,
        top_k=config.get("top_k", 10) if config else 10,
        max_sub_queries=config.get("max_sub_queries", 5) if config else 5
    )
    
    # 创建状态图
    workflow = StateGraph(AgenticState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    
    # 添加边
    logger.info("配置图边和路由逻辑...")
    
    # START → Planner
    workflow.add_edge(START, "planner")
    
    # Planner → Retriever
    workflow.add_edge("planner", "retriever")
    
    # Retriever → Generator
    workflow.add_edge("retriever", "generator")
    
    # Generator → Reflector
    workflow.add_edge("generator", "reflector")
    
    # Reflector → 条件路由
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {
            "generate": "generator",  # 重新生成 (使用新文档)
            "rewrite": "rewrite_query",  # 重写查询
            "end": END  # 结束
        }
    )
    
    # Rewrite Query → Planner (循环)
    workflow.add_edge("rewrite_query", "planner")
    
    # 编译图
    logger.info("编译 LangGraph 工作流...")
    compiled_graph = workflow.compile()
    logger.info("✓ LangGraph 工作流编译完成")
    
    return compiled_graph, ctx


# ==================== 服务封装 ====================

class LangGraphAgenticRAG:
    """
    LangGraph Agentic RAG 服务封装
    
    提供简洁的 API 接口，隐藏 LangGraph 复杂度
    """
    
    def __init__(self, retriever_fn, config: Optional[Dict] = None):
        """
        初始化服务
        
        :param retriever_fn: 检索函数 (query, k) -> List[Dict]
        :param config: 可选配置
        """
        self.graph, self.context = build_agentic_graph(retriever_fn, config)
        logger.info("✅ LangGraphAgenticRAG 服务已初始化")
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        运行完整工作流
        
        :param query: 用户查询
        :param verbose: 是否输出详细日志
        :return: 完整结果
        """
        logger.info("="*70)
        logger.info(f"开始 Agentic RAG 查询：{query[:100]}...")
        logger.info("="*70)
        
        # 初始化状态
        initial_state = {
            "messages": [],
            "original_query": query,
            "rewritten_query": None,
            "sub_queries": [],
            "documents": [],
            "retrieval_attempts": 0,
            "generated_answer": None,
            "reflect_score": 0.0,
            "reflect_reasoning": None,
            "routing_decision": None,
            "start_time": time.time(),
            "total_time": None,
            "reasoning_steps": []
        }
        
        # 运行图
        result = self.graph.invoke(initial_state)
        
        # 计算总时间
        result["total_time"] = time.time() - result.get("start_time", time.time())
        
        # 提取最终答案
        answer = result.get("generated_answer", "未生成答案")
        reasoning_steps = result.get("reasoning_steps", [])
        
        if verbose:
            logger.info("="*70)
            logger.info("查询完成")
            logger.info(f"答案：{answer[:200]}...")
            logger.info(f"推理步骤：{len(reasoning_steps)} 步")
            logger.info(f"总耗时：{result['total_time']:.2f}秒")
            logger.info(f"检索尝试：{result['retrieval_attempts']} 次")
            logger.info(f"Reflector 分数：{result['reflect_score']:.2f}")
            logger.info("="*70)
        
        return result
    
    def get_reasoning_steps(self, result: Dict[str, Any]) -> List[str]:
        """提取推理步骤"""
        return result.get("reasoning_steps", [])
    
    def get_sources(self, result: Dict[str, Any]) -> List[Dict]:
        """提取来源文档"""
        return result.get("documents", [])


# ==================== 快速测试 ====================

if __name__ == "__main__":
    print("="*70)
    print("LangGraph Agentic RAG 快速测试")
    print("="*70)
    
    # 加载知识库
    print("\n📂 加载知识库...")
    doc_store = DocumentStore(
        'ephrin_papers',
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    def retriever(query, k=10):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    print(f"✓ 已加载 {doc_store.count()} 个文档")
    
    # 创建服务
    print("\n🔧 初始化 LangGraph Agentic RAG...")
    rag = LangGraphAgenticRAG(
        retriever_fn=retriever,
        config={
            "model": "qwen3.5:397b-cloud",
            "similarity_threshold": 0.75,
            "reflection_threshold": 0.8,
            "max_iterations": 3,
            "top_k": 10
        }
    )
    
    # 测试查询
    print("\n" + "="*70)
    print("测试查询")
    print("="*70)
    
    test_queries = [
        "Eph 受体是什么类型的蛋白质？",
        "EphA2 与 EphB4 在癌症中的功能差异？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[测试 {i}/{len(test_queries)}]")
        print(f"查询：{query}")
        print("-"*70)
        
        try:
            result = rag.run(query, verbose=True)
            
            print(f"\n✅ 答案：{result['generated_answer'][:200]}...")
            print(f"⏱️  耗时：{result['total_time']:.2f}秒")
            print(f"📊 Reflector 分数：{result['reflect_score']:.2f}")
            print(f"🔄 检索尝试：{result['retrieval_attempts']} 次")
            
            print(f"\n📝 推理步骤:")
            for step in result['reasoning_steps']:
                print(f"  - {step}")
            
        except Exception as e:
            print(f"❌ 错误：{e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
