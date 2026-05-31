#!/usr/bin/env python3
"""
Langfuse 监控集成 - 完整可观测性

功能:
1. 节点执行追踪
2. 延迟统计
3. 错误日志
4. 可视化仪表板
5. 用户反馈收集

安装:
pip install langfuse

配置:
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
"""

import os
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime


class LangfuseMonitor:
    """
    Langfuse 监控器
    
    集成到 LangGraph 工作流的每个节点
    """
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com"
    ):
        """
        初始化 Langfuse 监控器
        
        Args:
            public_key: Langfuse 公钥
            secret_key: Langfuse 私钥
            host: Langfuse 主机
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host
        self.langfuse = None
        self.enabled = False
        
        # 统计信息
        self.stats = {
            "traces": 0,
            "spans": 0,
            "errors": 0,
            "total_latency": 0.0
        }
        
        self._initialize()
    
    def _initialize(self):
        """初始化 Langfuse SDK"""
        if not self.public_key or not self.secret_key:
            print("⚠️  Langfuse 密钥未配置，监控已禁用")
            print("   获取密钥：https://cloud.langfuse.com")
            print("   设置环境变量:")
            print("   export LANGFUSE_PUBLIC_KEY=pk-lf-...")
            print("   export LANGFUSE_SECRET_KEY=sk-lf-...")
            return
        
        try:
            from langfuse import Langfuse
            
            self.langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            
            # 测试连接
            self.langfuse.ping()
            self.enabled = True
            print(f"✅ Langfuse 监控已启用")
            
        except ImportError:
            print("⚠️  Langfuse SDK 未安装：pip install langfuse")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  Langfuse 初始化失败：{e}")
            self.enabled = False
    
    def trace(self, name: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        创建 Trace
        
        Args:
            name: Trace 名称
            user_id: 用户 ID
            metadata: 元数据
        
        Returns:
            Langfuse Trace
        """
        if not self.enabled:
            return None
        
        self.stats["traces"] += 1
        
        trace = self.langfuse.trace(
            name=name,
            user_id=user_id,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return trace
    
    def span(self, trace, name: str, input_data: Optional[Dict] = None):
        """
        创建 Span (节点执行)
        
        Args:
            trace: 父 Trace
            name: Span 名称
            input_data: 输入数据
        
        Returns:
            Langfuse Span
        """
        if not self.enabled or trace is None:
            return None
        
        self.stats["spans"] += 1
        start_time = time.time()
        
        span = trace.span(
            name=name,
            input=input_data or {},
            timestamp=datetime.now()
        )
        
        # 包装 end 方法以记录延迟
        original_end = span.end
        
        def end_with_latency(output_data: Optional[Dict] = None, **kwargs):
            end_time = time.time()
            latency = end_time - start_time
            self.stats["total_latency"] += latency
            
            metadata = {
                "latency_ms": latency * 1000,
                "timestamp": datetime.now().isoformat()
            }
            
            original_end(
                output=output_data or {},
                metadata=metadata,
                **kwargs
            )
        
        span.end = end_with_latency
        return span
    
    def score(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """
        添加评分 (如 Reflector 分数)
        
        Args:
            trace: Trace
            name: 评分名称
            value: 评分值 (0-1)
            comment: 备注
        """
        if not self.enabled or trace is None:
            return
        
        self.langfuse.score(
            trace_id=trace.id,
            name=name,
            value=value,
            comment=comment
        )
    
    def error(self, trace, error_message: str, error_type: str = "Exception"):
        """
        记录错误
        
        Args:
            trace: Trace
            error_message: 错误信息
            error_type: 错误类型
        """
        if not self.enabled or trace is None:
            return
        
        self.stats["errors"] += 1
        
        trace.update(
            metadata={
                "error": {
                    "type": error_type,
                    "message": error_message,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        traces = self.stats["traces"]
        avg_latency = (
            self.stats["total_latency"] / traces if traces > 0 else 0.0
        )
        
        return {
            **self.stats,
            "avg_latency_ms": avg_latency * 1000,
            "enabled": self.enabled
        }
    
    def flush(self):
        """强制刷新数据到 Langfuse"""
        if self.enabled:
            self.langfuse.flush()


# ==================== LangGraph 节点装饰器 ====================

def monitor_node(node_func, node_name: str):
    """
    监控装饰器 - 自动追踪节点执行
    
    用法:
    @monitor_node(planner_node, "planner")
    def planner_node(state, runtime):
        ...
    """
    def wrapper(state, runtime):
        monitor = runtime.context.monitor
        
        if not monitor or not monitor.enabled:
            # 监控未启用，直接执行
            return node_func(state, runtime)
        
        # 创建或获取 Trace
        trace_id = state.get("_trace_id")
        if trace_id:
            trace = monitor.langfuse.get_trace(trace_id)
        else:
            trace = monitor.trace(
                name="agentic_rag_workflow",
                user_id=state.get("user_id"),
                metadata={"query": state.get("original_query", "")}
            )
            state["_trace_id"] = trace.id
        
        # 创建 Span
        input_data = {
            "original_query": state.get("original_query"),
            "state_keys": list(state.keys())
        }
        span = monitor.span(trace, node_name, input_data)
        
        try:
            # 执行节点
            result = node_func(state, runtime)
            
            # 记录输出
            if span:
                span.end(output_data={
                    "result_keys": list(result.keys()) if result else None,
                    "success": True
                })
            
            return result
        
        except Exception as e:
            # 记录错误
            if span:
                span.end(output_data={
                    "error": str(e),
                    "success": False
                })
            if trace:
                monitor.error(trace, str(e), type(e).__name__)
            
            raise
    
    return wrapper


# ==================== 集成到 Phase 1 工作流 ====================

def build_agentic_graph_with_monitoring(retriever_fn, config: Dict):
    """
    构建带监控的 LangGraph 工作流
    
    Args:
        retriever_fn: 检索函数
        config: 配置字典
    
    Returns:
        CompiledGraph, Context
    """
    from langgraph_phase1 import build_agentic_graph_phase1, AgenticContext
    
    # 创建监控器
    monitor = LangfuseMonitor(
        public_key=config.get("langfuse_public_key"),
        secret_key=config.get("langfuse_secret_key")
    )
    
    # 构建基础图
    graph, context = build_agentic_graph_phase1(retriever_fn, config)
    
    # 添加监控到 context
    context.monitor = monitor
    
    return graph, context


# ==================== 测试函数 ====================

def test_langfuse_monitor():
    """测试 Langfuse 监控"""
    print("="*60)
    print("Langfuse 监控测试")
    print("="*60)
    
    # 创建监控器
    monitor = LangfuseMonitor()
    
    if not monitor.enabled:
        print("\n⚠️  Langfuse 未启用，跳过测试")
        print("\n配置步骤:")
        print("  1. 访问 https://cloud.langfuse.com")
        print("  2. 创建免费账号")
        print("  3. 获取 API 密钥")
        print("  4. 设置环境变量:")
        print("     export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("     export LANGFUSE_SECRET_KEY=sk-lf-...")
        return False
    
    # 测试 1: 创建 Trace
    print("\n[测试 1] 创建 Trace")
    trace = monitor.trace(
        name="test_workflow",
        user_id="test_user",
        metadata={"test": True}
    )
    print(f"✅ Trace 创建成功：{trace.id}")
    
    # 测试 2: 创建 Span
    print("\n[测试 2] 创建 Span")
    span = monitor.span(trace, "test_node", {"input": "test"})
    time.sleep(0.1)  # 模拟执行
    span.end(output_data={"output": "result"})
    print(f"✅ Span 创建成功")
    
    # 测试 3: 添加评分
    print("\n[测试 3] 添加评分")
    monitor.score(trace, "quality_score", 0.85, "测试评分")
    print(f"✅ 评分添加成功")
    
    # 测试 4: 统计信息
    print("\n[测试 4] 统计信息")
    stats = monitor.get_stats()
    print(f"  Traces: {stats['traces']}")
    print(f"  Spans: {stats['spans']}")
    print(f"  错误：{stats['errors']}")
    print(f"  平均延迟：{stats['avg_latency_ms']:.2f}ms")
    
    # 测试 5: 错误记录
    print("\n[测试 5] 错误记录")
    try:
        raise ValueError("测试错误")
    except Exception as e:
        monitor.error(trace, str(e), type(e).__name__)
        print(f"✅ 错误记录成功")
    
    # 刷新数据
    monitor.flush()
    print(f"\n✅ 数据已刷新到 Langfuse")
    
    print("\n" + "="*60)
    print("Langfuse 监控测试完成")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_langfuse_monitor()
    if success:
        print("\n✅ Langfuse 监控测试通过！")
    else:
        print("\n⚠️  Langfuse 监控测试未完成 (可能未配置)")
