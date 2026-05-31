#!/usr/bin/env python3
"""
Planner Agent - 问题规划 Agent (2026 工业级标准)

职责:
1. 拆解复杂问题为 3-5 个子问题
2. 生成检索规划
3. 控制迭代次数 ≤3
4. 覆盖所有维度，避免重复

核心 Prompt (来自最佳实践):
"你是专业问题规划师。
将用户问题拆解为 3–5 个可检索的子问题，覆盖所有维度，避免重复。
输出纯列表，不要多余内容。"
"""

import requests
import json
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass


# ==================== 数据结构 ====================

class SubQuery(TypedDict):
    """子查询"""
    id: int
    query: str
    rationale: str  # 分解理由
    priority: int  # 优先级 (1-5)


class PlanningResult(TypedDict):
    """规划结果"""
    original_query: str
    sub_queries: List[SubQuery]
    is_complex: bool  # 是否需要拆解
    estimated_iterations: int
    search_strategy: str  # 检索策略


@dataclass
class PlanningStats:
    """规划统计"""
    total_plans: int = 0
    avg_sub_queries: float = 0.0
    complex_queries: int = 0
    simple_queries: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_plans": self.total_plans,
            "avg_sub_queries": self.avg_sub_queries,
            "complex_queries": self.complex_queries,
            "simple_queries": self.simple_queries
        }


# ==================== LLM 调用 ====================

def call_ollama(prompt: str, model: str = "qwen3.5:397b-cloud", temperature: float = 0.1) -> str:
    """调用 Ollama API (使用 generate 端点)"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)  # 黄金参数：120s
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"  ⚠️  Ollama 调用失败：{e}")
        return ""


# ==================== Planner Agent ====================

class PlannerAgent:
    """
    问题规划 Agent (2026 工业级标准)
    
    核心功能:
    1. 问题复杂度评估
    2. 智能拆解 (3-5 个子问题)
    3. 检索规划
    4. 迭代控制 (≤3 轮)
    """
    
    def __init__(self, 
                 model: str = "qwen3.5:397b-cloud",
                 max_sub_queries: int = 5,  # 黄金参数：3-5 个
                 max_iterations: int = 3):  # 黄金参数：3 轮
        """
        初始化 Planner Agent
        
        Args:
            model: LLM 模型 (推荐 qwen3.5:397b-cloud)
            max_sub_queries: 最大子问题数 (5 为最佳)
            max_iterations: 最大迭代次数 (3 为最佳)
        """
        self.model = model
        self.max_sub_queries = max_sub_queries
        self.max_iterations = max_iterations
        self.stats = PlanningStats()
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化核心 Prompts (来自最佳实践)"""
        
        # Prompt 1: 问题拆解 (核心)
        self.decompose_prompt = """你是专业问题规划师。
将用户问题拆解为 3–5 个可检索的子问题，覆盖所有维度，避免重复。
输出纯列表，不要多余内容。

**Original Query**: {query}

**拆解原则**:
1. 每个子问题应该是独立的 (可以单独检索回答)
2. 每个子问题应该是具体的 (包含明确的实体和关系)
3. 子问题应该覆盖原查询的所有关键方面
4. 避免子问题之间重复

**输出格式 (JSON)**:
{{
    "sub_queries": [
        {{
            "id": 1,
            "query": "子问题 1",
            "rationale": "为什么需要这个子问题",
            "priority": 1
        }},
        {{
            "id": 2,
            "query": "子问题 2",
            "rationale": "为什么需要这个子问题",
            "priority": 2
        }}
    ],
    "is_complex": true,
    "estimated_iterations": 2,
    "search_strategy": "parallel"  // parallel 或 sequential
}}
"""
        
        # Prompt 2: 复杂度评估
        self.complexity_prompt = """你是查询复杂度评估专家。
评估用户问题是否需要拆解为多个子问题。

**Query**: {query}

**判断标准**:
- **简单查询**: 单一事实性问题，可以直接检索回答
  - 例如："Eph 受体是什么？", "EphA2 的功能？"
  
- **复杂查询**: 需要多步推理、对比、综合分析
  - 例如："EphA2 与 EphB4 的功能差异？", "cis-interaction 如何影响癌症进展？"

**输出格式 (JSON)**:
{{
    "is_complex": true,
    "complexity_score": 0.8,
    "reason": "需要对比两个受体的功能",
    "recommended_sub_queries": 3
}}
"""
        
        # Prompt 3: 检索策略规划
        self.strategy_prompt = """你是检索策略规划专家。
为每个子问题推荐最优检索策略。

**Sub-Queries**:
{sub_queries}

**检索策略选项**:
- **parallel**: 并行检索所有子问题 (适合独立子问题)
- **sequential**: 顺序检索，后一个问题依赖前一个答案
- **hybrid**: 部分并行，部分顺序

**输出格式 (JSON)**:
{{
    "strategy": "parallel",
    "reason": "子问题相互独立",
    "top_k": 10,
    "similarity_threshold": 0.75
}}
"""
    
    def plan(self, query: str) -> PlanningResult:
        """
        核心规划方法
        
        Args:
            query: 用户原始查询
            
        Returns:
            规划结果
        """
        self.stats.total_plans += 1
        
        # Step 1: 复杂度评估
        is_complex = self._assess_complexity(query)
        
        if not is_complex:
            # 简单查询，无需拆解
            self.stats.simple_queries += 1
            return self._simple_plan(query)
        
        # 复杂查询，需要拆解
        self.stats.complex_queries += 1
        
        # Step 2: 问题拆解
        sub_queries = self._decompose_query(query)
        
        # Step 3: 检索策略规划
        search_strategy = self._plan_strategy(sub_queries)
        
        # 更新统计
        total_subs = len(sub_queries)
        self.stats.avg_sub_queries = (
            (self.stats.avg_sub_queries * (self.stats.total_plans - 1) + total_subs)
            / self.stats.total_plans
        )
        
        return PlanningResult(
            original_query=query,
            sub_queries=sub_queries,
            is_complex=True,
            estimated_iterations=min(total_subs, self.max_iterations),
            search_strategy=search_strategy
        )
    
    def _assess_complexity(self, query: str) -> bool:
        """评估查询复杂度"""
        prompt = self.complexity_prompt.format(query=query)
        
        response = call_ollama(prompt, self.model, temperature=0.0)
        
        try:
            # 提取 JSON
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            
            # 复杂度分数 > 0.5 视为复杂查询
            return data.get("is_complex", False) or data.get("complexity_score", 0) > 0.5
        except:
            # 回退：基于关键词判断
            complex_triggers = [
                "对比", "差异", "vs", "versus", "compare",
                "和...都", "both", "multiple",
                "如何影响", "how does", "mechanism",
                "为什么", "why", "reason",
                "关系", "relationship", "correlation"
            ]
            query_lower = query.lower()
            return any(t in query_lower for t in complex_triggers)
    
    def _decompose_query(self, query: str) -> List[SubQuery]:
        """拆解复杂查询为子问题"""
        prompt = self.decompose_prompt.format(query=query)
        
        response = call_ollama(prompt, self.model, temperature=0.1)
        
        try:
            # 提取 JSON
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            
            sub_queries = []
            for i, sq_data in enumerate(data.get("sub_queries", [])):
                if i >= self.max_sub_queries:
                    break
                
                sub_queries.append(SubQuery(
                    id=sq_data.get("id", i+1),
                    query=sq_data.get("query", ""),
                    rationale=sq_data.get("rationale", "覆盖查询维度"),
                    priority=sq_data.get("priority", i+1)
                ))
            
            # 如果 LLM 返回的子问题太少，补充默认拆解
            if len(sub_queries) < 2:
                sub_queries = self._fallback_decompose(query)
            
            return sub_queries
        except Exception as e:
            print(f"  ⚠️  拆解失败：{e}，使用回退策略")
            return self._fallback_decompose(query)
    
    def _fallback_decompose(self, query: str) -> List[SubQuery]:
        """回退拆解策略 (基于规则)"""
        # 对比类问题
        if any(t in query.lower() for t in ["对比", "差异", "vs", "versus", "compare"]):
            parts = query.replace("与", " ").replace("和", " ").replace("vs", " ").split()
            if len(parts) >= 2:
                entity_a = parts[0]
                entity_b = parts[1]
                aspect = "功能" if "功能" in query else "机制" if "机制" in query else "作用"
                
                return [
                    SubQuery(id=1, query=f"{entity_a} 的{aspect}是什么？", rationale=f"了解{entity_a}的{aspect}", priority=1),
                    SubQuery(id=2, query=f"{entity_b} 的{aspect}是什么？", rationale=f"了解{entity_b}的{aspect}", priority=2),
                    SubQuery(id=3, query=f"{entity_a} 与{entity_b}的{aspect}差异？", rationale="直接对比两者差异", priority=3)
                ]
        
        # 因果类问题
        if any(t in query.lower() for t in ["如何影响", "how does", "mechanism", "为什么"]):
            return [
                SubQuery(id=1, query=f"什么是 {query.split('影响')[0].strip()}？", rationale="了解原因", priority=1),
                SubQuery(id=2, query=f"什么是 {query.split('影响')[-1].strip() if '影响' in query else '结果'}？", rationale="了解结果", priority=2),
                SubQuery(id=3, query=f"{query.split('影响')[0].strip()} 影响 {query.split('影响')[-1].strip() if '影响' in query else '结果'} 的机制？", rationale="了解影响机制", priority=3)
            ]
        
        # 默认拆解
        return [
            SubQuery(id=1, query=query, rationale="单一问题", priority=1)
        ]
    
    def _plan_strategy(self, sub_queries: List[SubQuery]) -> str:
        """规划检索策略"""
        # 简单实现：如果子问题相互独立，使用并行；否则顺序
        # 可以扩展为使用 LLM 判断
        
        # 检查是否有依赖关系
        has_dependency = any(
            "差异" in sq["query"] or "对比" in sq["query"] or "vs" in sq["query"]
            for sq in sub_queries
        )
        
        if has_dependency:
            return "sequential"  # 对比类需要顺序检索
        else:
            return "parallel"  # 独立问题可以并行
    
    def _simple_plan(self, query: str) -> PlanningResult:
        """简单查询规划 (无需拆解)"""
        return PlanningResult(
            original_query=query,
            sub_queries=[
                SubQuery(id=1, query=query, rationale="简单查询，无需拆解", priority=1)
            ],
            is_complex=False,
            estimated_iterations=1,
            search_strategy="direct"
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.to_dict()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建 Planner Agent
    planner = PlannerAgent(model="qwen3.5:397b-cloud")
    
    # 测试查询
    test_queries = [
        "Eph 受体的功能是什么？",  # 简单
        "EphA2 与 EphB4 在癌症中的功能差异？",  # 复杂 - 对比
        "cis-interaction 如何影响 trans-signaling 机制？",  # 复杂 - 因果
    ]
    
    print("="*70)
    print("Planner Agent 测试")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] {query}")
        print("-" * 60)
        
        result = planner.plan(query)
        
        print(f"是否复杂：{'是' if result['is_complex'] else '否'}")
        print(f"子问题数：{len(result['sub_queries'])}")
        print(f"检索策略：{result['search_strategy']}")
        print(f"预计迭代：{result['estimated_iterations']}")
        
        if result['sub_queries']:
            print(f"\n子问题列表:")
            for sq in result['sub_queries']:
                print(f"  [{sq['id']}] {sq['query']}")
                print(f"      理由：{sq['rationale']}")
                print(f"      优先级：{sq['priority']}")
    
    # 汇总
    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)
    print(f"\nPlanner 统计:")
    print(f"  {planner.get_stats()}")
