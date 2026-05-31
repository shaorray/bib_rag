#!/usr/bin/env python3
"""
Guardrail Node - 域外查询检测

功能:
1. 检测查询是否属于 Eph/Ephrin 领域
2. 拒绝域外查询，防止幻觉
3. 提供友好提示

实现方式:
- 关键词匹配 (快速路径)
- LLM 语义判断 (慢速路径)
- 置信度评分
"""

from typing import Dict, Any, List, Optional, Literal
import re


class GuardrailNode:
    """
    Guardrail Node - 查询作用域验证
    
    检测流程:
    1. 关键词匹配 (Eph/ephrin 相关术语)
    2. LLM 语义判断 (如果关键词不确定)
    3. 返回决策：allow/reject/uncertain
    """
    
    # Eph/Ephrin 领域关键词
    DOMAIN_KEYWORDS = [
        # 受体名称
        "eph", "epha", "ephb",
        "epha1", "epha2", "epha3", "epha4", "epha5", "epha6", "epha7", "epha8", "epha10",
        "ephb1", "ephb2", "ephb3", "ephb4", "ephb6",
        
        # 配体名称
        "ephrin", "efn",
        "efna1", "efna2", "efna3", "efna4", "efna5",
        "efnb1", "efnb2", "efnb3",
        
        # 相互作用
        "cis-interaction", "trans-interaction",
        "cis interaction", "trans interaction",
        "cis-activation", "trans-activation",
        "ligand-receptor", "receptor-ligand",
        
        # 信号通路
        "signaling", "signal transduction",
        "kinase", "phosphorylation",
        "downstream", "pathway",
        
        # 生物学过程
        "cell migration", "cell adhesion",
        "axon guidance", "angiogenesis",
        "boundary formation", "segmentation",
        "cancer", "tumor", "metastasis",
        "epithelial", "barrier",
        
        # 分子机制
        "tetramerization", "clustering",
        "endocytosis", "internalization",
        "cleavage", "shedding",
        
        # 疾病相关
        "oncogene", "tumor suppressor",
        "biomarker", "therapeutic target",
        "inhibitor", "agonist", "antagonist"
    ]
    
    # 排除关键词 (明确不属于该领域)
    EXCLUDE_KEYWORDS = [
        "weather", "股票", "finance",
        "recipe", "cooking", "restaurant",
        "movie", "music", "game",
        "sports", "football", "basketball"
    ]
    
    def __init__(self, model: str = "qwen3.5:397b-cloud"):
        """
        初始化 Guardrail Node
        
        Args:
            model: LLM 模型名称
        """
        self.model = model
        self.stats = {
            "total_queries": 0,
            "allowed": 0,
            "rejected": 0,
            "uncertain": 0
        }
    
    def check_keyword_match(self, query: str) -> Dict[str, Any]:
        """
        关键词匹配检查 (快速路径)
        
        Args:
            query: 用户查询
        
        Returns:
            Dict: 匹配结果
                - matched: bool, 是否匹配
                - matched_keywords: List[str], 匹配的关键词
                - confidence: float, 置信度 (0-1)
        """
        query_lower = query.lower()
        matched = []
        
        # 检查领域关键词
        for keyword in self.DOMAIN_KEYWORDS:
            if keyword in query_lower:
                matched.append(keyword)
        
        # 检查排除关键词
        excluded = False
        for keyword in self.EXCLUDE_KEYWORDS:
            if keyword in query_lower:
                excluded = True
                break
        
        # 计算置信度
        if excluded:
            confidence = 0.0
        elif len(matched) == 0:
            confidence = 0.2  # 无关键词，低置信度
        elif len(matched) <= 2:
            confidence = 0.5  # 1-2 个关键词，中等置信度
        else:
            confidence = min(0.9, 0.3 + len(matched) * 0.15)  # 3+ 个关键词，高置信度
        
        return {
            "matched": len(matched) > 0 and not excluded,
            "matched_keywords": matched,
            "confidence": confidence,
            "excluded": excluded
        }
    
    def check_llm_semantic(self, query: str) -> Dict[str, Any]:
        """
        LLM 语义判断 (慢速路径)
        
        当关键词匹配不确定时，使用 LLM 判断
        
        Args:
            query: 用户查询
        
        Returns:
            Dict: LLM 判断结果
        """
        import requests
        
        prompt = f"""
你是一个 Eph/Ephrin 研究领域的专家助手。

请判断以下查询是否属于 Eph/Ephrin 研究领域：

**用户查询**: "{query}"

**Eph/Ephrin 领域包括**:
- Eph 受体 (EphA1-8, EphA10, EphB1-4, EphB6)
- ephrin 配体 (Efna1-5, Efnb1-3)
- Eph-ephrin 相互作用 (cis/trans interaction)
- Eph 信号通路
- Eph 在癌症、神经发育、血管生成中的作用
- Eph 靶向治疗

**判断标准**:
- 如果查询直接涉及 Eph/ephrin → allow
- 如果查询与 Eph/ephrin 相关但不直接 → allow
- 如果查询完全无关 → reject

请以 JSON 格式回答：
{{
    "decision": "allow" | "reject",
    "confidence": 0.0-1.0,
    "reason": "简短解释"
}}
"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                # 解析 JSON
                import json
                import re
                
                # 提取 JSON 部分
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return {
                        "decision": llm_result.get("decision", "reject"),
                        "confidence": llm_result.get("confidence", 0.5),
                        "reason": llm_result.get("reason", "无法解析"),
                        "method": "llm"
                    }
            
            # LLM 调用失败，保守拒绝
            return {
                "decision": "reject",
                "confidence": 0.5,
                "reason": "LLM 服务不可用",
                "method": "llm_fallback"
            }
            
        except Exception as e:
            return {
                "decision": "reject",
                "confidence": 0.5,
                "reason": f"LLM 调用错误：{str(e)}",
                "method": "llm_error"
            }
    
    def check(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        完整检查流程
        
        Args:
            query: 用户查询
            use_llm: 是否使用 LLM 辅助判断
        
        Returns:
            Dict: 检查结果
                - decision: "allow" | "reject"
                - confidence: float, 置信度
                - method: "keyword" | "llm" | "hybrid"
                - reason: str, 判断理由
                - matched_keywords: List[str], 匹配的关键词
        """
        self.stats["total_queries"] += 1
        
        # Step 1: 关键词匹配
        keyword_result = self.check_keyword_match(query)
        
        # 快速决策：高置信度允许
        if keyword_result["confidence"] >= 0.8:
            self.stats["allowed"] += 1
            return {
                "decision": "allow",
                "confidence": keyword_result["confidence"],
                "method": "keyword",
                "reason": f"匹配到 {len(keyword_result['matched_keywords'])} 个领域关键词",
                "matched_keywords": keyword_result["matched_keywords"]
            }
        
        # 快速决策：明确排除
        if keyword_result["excluded"]:
            self.stats["rejected"] += 1
            return {
                "decision": "reject",
                "confidence": 1.0,
                "method": "keyword",
                "reason": "查询明确不属于 Eph/Ephrin 领域",
                "matched_keywords": []
            }
        
        # Step 2: LLM 语义判断 (如果关键词不确定)
        if use_llm and keyword_result["confidence"] < 0.8:
            llm_result = self.check_llm_semantic(query)
            
            # 混合决策
            if keyword_result["confidence"] >= 0.5 and llm_result["decision"] == "allow":
                self.stats["allowed"] += 1
                return {
                    "decision": "allow",
                    "confidence": (keyword_result["confidence"] + llm_result["confidence"]) / 2,
                    "method": "hybrid",
                    "reason": f"关键词 +LLM 判断：{llm_result['reason']}",
                    "matched_keywords": keyword_result["matched_keywords"]
                }
            
            # LLM 决定拒绝
            if llm_result["decision"] == "reject":
                self.stats["rejected"] += 1
                return {
                    "decision": "reject",
                    "confidence": llm_result["confidence"],
                    "method": "llm",
                    "reason": llm_result["reason"],
                    "matched_keywords": keyword_result["matched_keywords"]
                }
            
            # LLM 决定允许
            self.stats["allowed"] += 1
            return {
                "decision": "allow",
                "confidence": llm_result["confidence"],
                "method": "llm",
                "reason": llm_result["reason"],
                "matched_keywords": keyword_result["matched_keywords"]
            }
        
        # 无 LLM，保守决策
        if keyword_result["confidence"] >= 0.5:
            self.stats["allowed"] += 1
            return {
                "decision": "allow",
                "confidence": keyword_result["confidence"],
                "method": "keyword",
                "reason": f"匹配到关键词：{', '.join(keyword_result['matched_keywords'])}",
                "matched_keywords": keyword_result["matched_keywords"]
            }
        else:
            self.stats["rejected"] += 1
            return {
                "decision": "reject",
                "confidence": 1.0 - keyword_result["confidence"],
                "method": "keyword",
                "reason": "未匹配到足够的领域关键词",
                "matched_keywords": keyword_result["matched_keywords"]
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "allow_rate": self.stats["allowed"] / total,
            "reject_rate": self.stats["rejected"] / total
        }


# ==================== LangGraph 节点函数 ====================

def guardrail_node(state: Dict, runtime) -> Dict:
    """
    LangGraph Guardrail 节点
    
    输入状态:
    - original_query: 用户查询
    
    输出状态:
    - guardrail_decision: "allow" | "reject"
    - guardrail_reason: 判断理由
    - routing_decision: "end" (如果拒绝) 或 None (如果允许)
    """
    query = state.get("original_query", "")
    
    # 获取 Guardrail Node 实例
    guardrail = runtime.context.guardrail
    
    # 执行检查
    result = guardrail.check(query)
    
    # 构建输出
    if result["decision"] == "reject":
        return {
            "guardrail_decision": "reject",
            "guardrail_reason": result["reason"],
            "routing_decision": "end",  # 结束工作流
            "generated_answer": f"⚠️ 抱歉，我无法回答这个问题。\n\n**原因**: {result['reason']}\n\n**本助手专注于 Eph/Ephrin 研究领域**，包括:\n"
                               f"- Eph 受体 (EphA1-8, EphA10, EphB1-4, EphB6)\n"
                               f"- ephrin 配体 (Efna1-5, Efnb1-3)\n"
                               f"- Eph-ephrin 相互作用和信号通路\n"
                               f"- Eph 在癌症、神经发育等疾病中的作用\n\n"
                               f"请提问与 Eph/Ephrin 相关的问题！😊"
        }
    else:
        return {
            "guardrail_decision": "allow",
            "guardrail_reason": result["reason"],
            "routing_decision": None,  # 继续工作流
            "matched_keywords": result["matched_keywords"]
        }


# ==================== 测试函数 ====================

def test_guardrail():
    """测试 Guardrail Node"""
    print("="*60)
    print("Guardrail Node 测试")
    print("="*60)
    
    guardrail = GuardrailNode()
    
    # 测试用例
    test_cases = [
        # 应该允许的查询
        ("EphA2 在癌症中的功能", "allow"),
        ("EphB4 与 ephrinB2 的相互作用", "allow"),
        ("cis-interaction 的分子机制", "allow"),
        ("Eph 受体如何分类？", "allow"),
        ("tetramerization inhibitor 的作用", "allow"),
        
        # 应该拒绝的查询
        ("今天天气怎么样？", "reject"),
        ("推荐几只股票", "reject"),
        ("如何做红烧肉？", "reject"),
        ("NBA 总决赛结果", "reject"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = guardrail.check(query, use_llm=False)  # 仅关键词测试
        actual = result["decision"]
        
        status = "✅" if actual == expected else "❌"
        if actual == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} 查询：{query}")
        print(f"   预期：{expected}, 实际：{actual}")
        print(f"   置信度：{result['confidence']:.2f}")
        print(f"   方法：{result['method']}")
        print(f"   理由：{result['reason'][:50]}...")
    
    print("\n" + "="*60)
    print(f"测试结果：{passed}/{len(test_cases)} 通过")
    print(f"通过率：{passed/len(test_cases)*100:.1f}%")
    print("="*60)
    
    # 统计信息
    stats = guardrail.get_stats()
    print(f"\n统计信息:")
    print(f"  总查询：{stats['total_queries']}")
    print(f"  允许：{stats['allowed']} ({stats.get('allow_rate', 0)*100:.1f}%)")
    print(f"  拒绝：{stats['rejected']} ({stats.get('reject_rate', 0)*100:.1f}%)")
    
    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_guardrail()
    if success:
        print("\n✅ Guardrail Node 测试通过！")
    else:
        print("\n❌ Guardrail Node 测试失败！")
