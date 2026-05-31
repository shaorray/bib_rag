#!/usr/bin/env python3
"""
LangGraph Agentic RAG with Phase 1 Improvements

整合内容:
1. ✅ Guardrail Node (域外检测)
2. ✅ Redis Cache (150-400x 加速)
3. ⏳ Langfuse Monitoring (可观测性)
4. ⏳ Hybrid Search (BM25 + 向量)

版本：v3.0 (Phase 1)
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from dataclasses import dataclass
import time

# 导入改进组件
from guardrail_node import GuardrailNode, guardrail_node
from redis_cache import RedisCache

# 导入原有组件
from planner_agent import PlannerAgent
from reflector_agent import ReflectorAgent
from self_rag import SelfRAGWorkflow


# ==================== 状态定义 ====================

class AgenticState(TypedDict):
    """增强版状态定义 (Phase 1)"""
    messages: Annotated[List[Dict[str, str]], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    sub_queries: List[str]
    documents: List[Dict]
    retrieval_attempts: int
    generated_answer: Optional[str]
    reflect_score: float
    routing_decision: Optional[str]
    reasoning_steps: List[str]
    
    # Phase 1 新增字段
    guardrail_decision: Optional[str]  # "allow" | "reject"
    guardrail_reason: Optional[str]  # Guardrail 判断理由
    matched_keywords: Optional[List[str]]  # 匹配的关键词
    cache_hit: Optional[bool]  # 缓存命中标志


# ==================== 依赖注入 ====================

@dataclass
class AgenticContext:
    """增强版依赖注入 (Phase 1)"""
    retriever_fn: Any
    model: str
    similarity_threshold: float
    reflection_threshold: float
    max_iterations: int
    top_k: int
    max_sub_queries: int
    temperature: float
    timeout: int
    
    # Phase 1 新增依赖
    guardrail: GuardrailNode
    cache: Optional[RedisCache] = None
    langfuse: Optional[Any] = None
    
    # 原有组件
    planner: Optional[PlannerAgent] = None
    reflector: Optional[ReflectorAgent] = None
    rag: Optional[SelfRAGWorkflow] = None
    
    def __post_init__(self):
        """初始化组件"""
        if self.planner is None:
            self.planner = PlannerAgent(model=self.model)
        
        if self.reflector is None:
            self.reflector = ReflectorAgent(model=self.model)
        
        if self.rag is None:
            self.rag = SelfRAGWorkflow(
                retriever_fn=self.retriever_fn,
                model=self.model,
                similarity_threshold=self.similarity_threshold,
                top_k=self.top_k
            )


# ==================== 节点函数 (Phase 1 增强版) ====================

def guardrail_node_enhanced(state: AgenticState, runtime) -> Dict:
    """
    Guardrail Node (增强版)
    
    集成到 LangGraph 工作流
    """
    query = state.get("original_query", "")
    guardrail = runtime.context.guardrail
    
    # 执行检查
    result = guardrail.check(query, use_llm=True)
    
    # 记录推理步骤
    reasoning = f"Guardrail 检查：{result['decision']} ({result['reason'][:50]}...)"
    
    if result["decision"] == "reject":
        return {
            "guardrail_decision": "reject",
            "guardrail_reason": result["reason"],
            "routing_decision": "end",
            "generated_answer": (
                f"⚠️ 抱歉，我无法回答这个问题。\n\n"
                f"**原因**: {result['reason']}\n\n"
                f"**本助手专注于 Eph/Ephrin 研究领域**，包括:\n"
                f"- Eph 受体 (EphA1-8, EphA10, EphB1-4, EphB6)\n"
                f"- ephrin 配体 (Efna1-5, Efnb1-3)\n"
                f"- Eph-ephrin 相互作用和信号通路\n"
                f"- Eph 在癌症、神经发育等疾病中的作用\n\n"
                f"请提问与 Eph/Ephrin 相关的问题！😊"
            ),
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning]
        }
    else:
        return {
            "guardrail_decision": "allow",
            "guardrail_reason": result["reason"],
            "routing_decision": None,
            "matched_keywords": result["matched_keywords"],
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning]
        }


def planner_node(state: AgenticState, runtime) -> Dict:
    """Planner Node (带缓存)"""
    query = state.get("rewritten_query") or state.get("original_query", "")
    cache = runtime.context.cache
    
    # 尝试缓存
    if cache:
        cached = cache.get("planner", query)
        if cached:
            return {
                "sub_queries": cached["sub_queries"],
                "reasoning_steps": state.get("reasoning_steps", []) + [
                    f"Planner 缓存命中：{len(cached['sub_queries'])} 个子问题"
                ]
            }
    
    # 实际执行
    plan_result = runtime.context.planner.plan(query)
    sub_queries = plan_result.get("sub_queries", [])
    
    # 存入缓存
    if cache:
        cache.set("planner", query, {"sub_queries": sub_queries})
    
    return {
        "sub_queries": sub_queries,
        "reasoning_steps": state.get("reasoning_steps", []) + [
            f"Planner 拆解为 {len(sub_queries)} 个子问题"
        ]
    }


def retriever_node(state: AgenticState, runtime) -> Dict:
    """Retriever Node (带缓存)"""
    query = state.get("rewritten_query") or state.get("original_query", "")
    cache = runtime.context.cache
    top_k = runtime.context.top_k
    
    # 尝试缓存
    if cache:
        cache_key = {"query": query, "k": top_k}
        cached = cache.get("retrieval", cache_key)
        if cached:
            return {
                "documents": cached,
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
                "reasoning_steps": state.get("reasoning_steps", []) + [
                    f"检索缓存命中：{len(cached)} 个文档"
                ]
            }
    
    # 实际检索
    documents = runtime.context.rag.retriever(query, k=top_k)
    
    # 存入缓存
    if cache:
        cache.set("retrieval", cache_key, documents)
    
    return {
        "documents": documents,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "reasoning_steps": state.get("reasoning_steps", []) + [
            f"检索到 {len(documents)} 个文档"
        ]
    }


def generator_node(state: AgenticState, runtime) -> Dict:
    """Generator Node"""
    query = state.get("rewritten_query") or state.get("original_query", "")
    documents = state.get("documents", [])
    
    # 生成答案
    result = runtime.context.rag.generate(query, documents)
    answer = result.get("answer", "")
    
    return {
        "generated_answer": answer,
        "reasoning_steps": state.get("reasoning_steps", []) + [
            f"生成答案 ({len(answer)} 字符)"
        ]
    }


def reflector_node(state: AgenticState, runtime) -> Dict:
    """Reflector Node"""
    query = state.get("rewritten_query") or state.get("original_query", "")
    answer = state.get("generated_answer", "")
    documents = state.get("documents", [])
    
    # 评估
    reflect_result = runtime.context.reflector.reflect(query, answer, documents)
    score = reflect_result.get("score", 0.0)
    reasoning = reflect_result.get("reasoning", "")
    
    # 路由决策
    reflection_threshold = runtime.context.reflection_threshold
    max_iterations = runtime.context.max_iterations
    attempts = state.get("retrieval_attempts", 0)
    
    if score >= reflection_threshold:
        routing = "end"
        reason = f"Reflector 评分：{score:.2f} (通过)"
    elif attempts >= max_iterations:
        routing = "end"
        reason = f"Reflector 评分：{score:.2f} (达到最大迭代次数)"
    else:
        routing = "rewrite"
        reason = f"Reflector 评分：{score:.2f} (需要改进)"
    
    return {
        "reflect_score": score,
        "reflect_reasoning": reasoning,
        "routing_decision": routing,
        "reasoning_steps": state.get("reasoning_steps", []) + [reason]
    }


def rewrite_query_node(state: AgenticState, runtime) -> Dict:
    """Rewrite Query Node"""
    original_query = state.get("original_query", "")
    reflect_reasoning = state.get("reflect_reasoning", "")
    
    # 重写查询
    prompt = f"""
原查询：{original_query}

改进建议：{reflect_reasoning}

请重写查询，使其更清晰、更具体，能够获得更好的检索结果。
只返回重写后的查询，不要其他内容。
"""
    
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": runtime.context.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        },
        timeout=runtime.context.timeout
    )
    
    if response.status_code == 200:
        rewritten = response.json().get("response", "").strip()
    else:
        rewritten = original_query
    
    return {
        "rewritten_query": rewritten,
        "reasoning_steps": state.get("reasoning_steps", []) + [
            f"重写查询：{rewritten[:50]}..."
        ]
    }


# ==================== 路由函数 ====================

def route_after_reflector(state: AgenticState, runtime) -> Literal["generate", "rewrite", "end"]:
    """Reflector 后的路由决策"""
    return state.get("routing_decision", "end")


def route_after_guardrail(state: AgenticState, runtime) -> Literal["planner", "end"]:
    """Guardrail 后的路由决策"""
    if state.get("guardrail_decision") == "reject":
        return "end"
    else:
        return "planner"


# ==================== 构建图 (Phase 1 增强版) ====================

def build_agentic_graph_phase1(retriever_fn, config: Dict):
    """
    构建 Phase 1 增强版图
    
    Args:
        retriever_fn: 检索函数
        config: 配置字典
    
    Returns:
        CompiledGraph, Context
    """
    # 创建依赖
    guardrail = GuardrailNode(model=config.get("model", "qwen3.5:397b-cloud"))
    cache = RedisCache(
        host=config.get("redis_host", "localhost"),
        port=config.get("redis_port", 6379)
    )
    
    # 创建 Context
    context = AgenticContext(
        retriever_fn=retriever_fn,
        model=config.get("model", "qwen3.5:397b-cloud"),
        similarity_threshold=config.get("similarity_threshold", 0.75),
        reflection_threshold=config.get("reflection_threshold", 0.8),
        max_iterations=config.get("max_iterations", 3),
        top_k=config.get("top_k", 10),
        max_sub_queries=config.get("max_sub_queries", 5),
        temperature=config.get("temperature", 0.1),
        timeout=config.get("timeout", 120),
        guardrail=guardrail,
        cache=cache if cache.enabled else None
    )
    
    # 创建 Runtime
    from langgraph.graph import Runtime
    runtime = Runtime(context=context)
    
    # 构建状态图
    workflow = StateGraph(AgenticState)
    
    # 添加节点
    workflow.add_node("guardrail", guardrail_node_enhanced)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    
    # 添加边
    workflow.add_edge(START, "guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"planner": "planner", "end": END}
    )
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {
            "generate": "generator",
            "rewrite": "rewrite_query",
            "end": END
        }
    )
    workflow.add_edge("rewrite_query", "planner")
    
    # 编译
    graph = workflow.compile()
    
    return graph, context


# ==================== 服务封装 ====================

class LangGraphAgenticRAGPhase1:
    """
    Phase 1 增强版服务封装
    """
    
    def __init__(self, retriever_fn, config: Dict = None):
        """
        初始化服务
        
        Args:
            retriever_fn: 检索函数
            config: 配置字典
        """
        if config is None:
            config = {}
        
        self.config = {
            "model": "qwen3.5:397b-cloud",
            "similarity_threshold": 0.75,
            "reflection_threshold": 0.8,
            "max_iterations": 3,
            "top_k": 10,
            "max_sub_queries": 5,
            "temperature": 0.1,
            "timeout": 120,
            "redis_host": "localhost",
            "redis_port": 6379,
            **config
        }
        
        # 构建图
        self.graph, self.context = build_agentic_graph_phase1(
            retriever_fn, self.config
        )
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        运行完整工作流
        
        Args:
            query: 用户查询
            verbose: 是否输出详细日志
        
        Returns:
            result: 完整结果
        """
        start_time = time.time()
        
        if verbose:
            print(f"🔍 查询：{query[:50]}...")
            print(f"🛡️  Guardrail 检查中...")
        
        # 初始状态
        initial_state = {
            "messages": [],
            "original_query": query,
            "rewritten_query": None,
            "sub_queries": [],
            "documents": [],
            "retrieval_attempts": 0,
            "generated_answer": None,
            "reflect_score": 0.0,
            "routing_decision": None,
            "reasoning_steps": [],
            "guardrail_decision": None,
            "guardrail_reason": None,
            "matched_keywords": None,
            "cache_hit": None
        }
        
        # 运行图
        result = self.graph.invoke(initial_state)
        
        # 添加元数据
        end_time = time.time()
        result["total_time"] = end_time - start_time
        result["cache_enabled"] = self.context.cache is not None
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ 完成")
            print(f"{'='*60}")
            
            # Guardrail 信息
            if result.get("guardrail_decision") == "reject":
                print(f"⚠️  Guardrail 拒绝：{result.get('guardrail_reason', '未知')}")
            else:
                print(f"✅ Guardrail 通过")
                if result.get("matched_keywords"):
                    print(f"   匹配关键词：{', '.join(result['matched_keywords'][:5])}")
            
            # 缓存信息
            if self.context.cache:
                stats = self.context.cache.get_stats()
                print(f"\n💾 缓存统计:")
                print(f"   命中率：{stats['hit_rate']*100:.1f}%")
                print(f"   命中：{stats['hits']}, 未命中：{stats['misses']}")
            
            # 性能信息
            print(f"\n⏱️  性能:")
            print(f"   总耗时：{result['total_time']:.2f}秒")
            print(f"   检索尝试：{result['retrieval_attempts']} 次")
            print(f"   Reflector 分数：{result.get('reflect_score', 0):.2f}")
            
            # 推理步骤
            if result.get("reasoning_steps"):
                print(f"\n📝 推理步骤:")
                for i, step in enumerate(result["reasoning_steps"], 1):
                    print(f"   {i}. {step}")
            
            # 答案
            if result.get("generated_answer"):
                print(f"\n💬 答案:")
                print(f"   {result['generated_answer'][:200]}...")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "guardrail": self.context.guardrail.get_stats(),
            "cache": self.context.cache.get_stats() if self.context.cache else None
        }
        return stats


# ==================== 测试函数 ====================

def test_phase1():
    """测试 Phase 1 功能"""
    print("="*60)
    print("Phase 1 功能测试")
    print("="*60)
    
    # Mock 检索函数
    def mock_retriever(query, k=10):
        return [{"title": f"Document {i}", "content": "..."} for i in range(k)]
    
    # 创建服务
    config = {
        "model": "qwen3.5:397b-cloud",
        "reflection_threshold": 0.8,
        "max_iterations": 3,
        "redis_host": "localhost",
        "redis_port": 6379
    }
    
    rag = LangGraphAgenticRAGPhase1(mock_retriever, config)
    
    # 测试 1: 域内查询
    print("\n[测试 1] 域内查询")
    result1 = rag.run("EphA2 在癌症中的功能？", verbose=True)
    
    # 测试 2: 域外查询
    print("\n[测试 2] 域外查询")
    result2 = rag.run("今天天气怎么样？", verbose=True)
    
    # 测试 3: 统计信息
    print("\n[测试 3] 统计信息")
    stats = rag.get_stats()
    print(f"Guardrail: {stats['guardrail']}")
    if stats['cache']:
        print(f"Cache: {stats['cache']}")
    
    print("\n" + "="*60)
    print("Phase 1 测试完成")
    print("="*60)


if __name__ == "__main__":
    test_phase1()
