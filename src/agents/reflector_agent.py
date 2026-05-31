#!/usr/bin/env python3
"""
Reflector Agent - 反思校验 Agent (2026 工业级标准)

职责:
1. 校验答案是否基于文档 (防幻觉)
2. 评估答案完整性 (0-1 打分)
3. 决定是否重查 (≥0.8 生成，<0.8 重查)
4. 识别矛盾信息

核心 Prompt (来自最佳实践):
"你是严格校验官。评估当前资料是否足够回答问题，打分 0–1。
低于 0.8 必须重查；高于 0.8 可生成。只输出数字。"
"""

import requests
from typing import Dict, List, Any, Literal, TypedDict
from dataclasses import dataclass


# ==================== 数据结构 ====================

class ReflectionResult(TypedDict):
    """反思校验结果"""
    score: float  # 0-1 分
    is_sufficient: bool  # 是否足够生成答案
    issues: List[str]  # 发现的问题
    suggestions: List[str]  # 改进建议
    needs_reretrieval: bool  # 是否需要重查


@dataclass
class ReflectionStats:
    """反思统计"""
    total_reflections: int = 0
    avg_score: float = 0.0
    reretrieval_count: int = 0
    hallucination_caught: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_reflections": self.total_reflections,
            "avg_score": self.avg_score,
            "reretrieval_count": self.reretrieval_count,
            "hallucination_caught": self.hallucination_caught
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


# ==================== Reflector Agent ====================

class ReflectorAgent:
    """
    反思校验 Agent (2026 工业级标准)
    
    核心功能:
    1. 答案校验 (防幻觉)
    2. 完整性评估 (0-1 打分)
    3. 重查决策 (≥0.8 生成，<0.8 重查)
    4. 矛盾检测
    """
    
    def __init__(self, 
                 model: str = "qwen3.5:397b-cloud",
                 reflection_threshold: float = 0.8):  # 黄金参数：0.8
        """
        初始化 Reflector Agent
        
        Args:
            model: LLM 模型 (推荐 qwen3.5:397b-cloud 或 llama3.1:70b)
            reflection_threshold: 反思阈值 (0.8 为最佳)
        """
        self.model = model
        self.reflection_threshold = reflection_threshold
        self.stats = ReflectionStats()
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化核心 Prompts (来自最佳实践)"""
        
        # Prompt 1: 反思校验 (核心)
        self.reflect_prompt = """你是严格校验官。
评估当前资料是否足够回答问题，打分 0–1。
低于 0.8 必须重查；高于 0.8 可生成。
只输出数字。

**Question**: {question}

**Answer**: {answer}

**Documents**:
{documents}

**评分标准**:
- 1.0: 答案完全基于文档，信息完整，无幻觉
- 0.8-0.9: 答案基本基于文档，少量信息来自模型知识
- 0.5-0.7: 答案部分基于文档，部分信息无支持
- 0.0-0.4: 答案与文档矛盾或完全无支持

**分数**:"""
        
        # Prompt 2: 幻觉检测
        self.hallucination_prompt = """你是事实核查专家。
检测答案中是否有文档不支持的主张。

**Question**: {question}

**Documents**:
{documents}

**Answer**: {answer}

**任务**:
1. 找出答案中的关键主张
2. 检查每个主张是否在文档中有支持
3. 列出无支持的主张

**输出格式 (JSON)**:
{{
    "claims": ["主张 1", "主张 2"],
    "supported": [true, false],
    "hallucinations": ["无支持的主张"]
}}
"""
        
        # Prompt 3: 完整性评估
        self.completeness_prompt = """你是完整性评估专家。
评估答案是否完整覆盖了问题的所有方面。

**Question**: {question}

**Answer**: {answer}

**评估维度**:
1. 是否回答了问题的核心？
2. 是否覆盖了所有关键方面？
3. 是否有遗漏的重要信息？

**输出格式 (JSON)**:
{{
    "core_answered": true,
    "aspects_covered": ["方面 1", "方面 2"],
    "missing_info": ["缺失信息"],
    "completeness_score": 0.85
}}
"""
        
        # Prompt 4: 矛盾检测
        self.contradiction_prompt = """你是矛盾检测专家。
检查答案是否与文档信息矛盾。

**Question**: {question}

**Documents**:
{documents}

**Answer**: {answer}

**任务**:
找出答案中与文档矛盾的信息。

**输出格式 (JSON)**:
{{
    "contradictions": [
        {{
            "answer_claim": "答案中的主张",
            "document_fact": "文档中的事实",
            "explanation": "矛盾说明"
        }}
    ],
    "has_contradiction": false
}}
"""
    
    def reflect(self, 
                question: str, 
                answer: str, 
                documents: List[Dict]) -> ReflectionResult:
        """
        核心反思校验方法
        
        Args:
            question: 用户问题
            answer: 生成的答案
            documents: 检索到的文档
            
        Returns:
            反思校验结果
        """
        self.stats.total_reflections += 1
        
        # Step 1: 反思打分
        score = self._reflect_score(question, answer, documents)
        
        # Step 2: 幻觉检测
        hallucinations = self._detect_hallucinations(question, answer, documents)
        
        # Step 3: 完整性评估
        completeness = self._assess_completeness(question, answer)
        
        # Step 4: 矛盾检测
        contradictions = self._detect_contradictions(question, answer, documents)
        
        # Step 5: 决策 (是否重查)
        is_sufficient = score >= self.reflection_threshold
        needs_reretrieval = not is_sufficient
        
        # 更新统计
        self.stats.avg_score = (
            (self.stats.avg_score * (self.stats.total_reflections - 1) + score) 
            / self.stats.total_reflections
        )
        if needs_reretrieval:
            self.stats.reretrieval_count += 1
        if hallucinations:
            self.stats.hallucination_caught += 1
        
        # Step 6: 生成建议
        suggestions = self._generate_suggestions(
            score, hallucinations, completeness, contradictions
        )
        
        return ReflectionResult(
            score=score,
            is_sufficient=is_sufficient,
            issues=hallucinations + contradictions,
            suggestions=suggestions,
            needs_reretrieval=needs_reretrieval
        )
    
    def _reflect_score(self, question: str, answer: str, documents: List[Dict]) -> float:
        """反思打分 (0-1)"""
        docs_text = "\n\n".join([f"[Doc {i+1}]: {d.get('text', '')[:500]}" for i, d in enumerate(documents)])
        
        prompt = self.reflect_prompt.format(
            question=question,
            answer=answer,
            documents=docs_text
        )
        
        response = call_ollama(prompt, self.model, temperature=0.0)
        
        # 提取数字
        try:
            # 尝试提取 0-1 之间的数字
            import re
            numbers = re.findall(r'\d\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
        except:
            pass
        
        # 回退：基于关键词判断
        if "1.0" in response or "1" in response:
            return 1.0
        elif "0.8" in response or "0.9" in response:
            return 0.85
        elif "0.5" in response:
            return 0.5
        else:
            return 0.6  # 默认
    
    def _detect_hallucinations(self, 
                               question: str, 
                               answer: str, 
                               documents: List[Dict]) -> List[str]:
        """检测幻觉 (无支持的主张)"""
        docs_text = "\n\n".join([f"[Doc {i+1}]: {d.get('text', '')[:500]}" for i, d in enumerate(documents)])
        
        prompt = self.hallucination_prompt.format(
            question=question,
            answer=answer,
            documents=docs_text
        )
        
        response = call_ollama(prompt, self.model, temperature=0.0)
        
        try:
            import json
            # 提取 JSON
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            return data.get("hallucinations", [])
        except:
            return []
    
    def _assess_completeness(self, question: str, answer: str) -> Dict:
        """评估完整性"""
        prompt = self.completeness_prompt.format(
            question=question,
            answer=answer
        )
        
        response = call_ollama(prompt, self.model, temperature=0.0)
        
        try:
            import json
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            return data
        except:
            return {
                "core_answered": True,
                "aspects_covered": ["核心"],
                "missing_info": [],
                "completeness_score": 0.7
            }
    
    def _detect_contradictions(self,
                               question: str,
                               answer: str,
                               documents: List[Dict]) -> List[str]:
        """检测矛盾"""
        docs_text = "\n\n".join([f"[Doc {i+1}]: {d.get('text', '')[:500]}" for i, d in enumerate(documents)])
        
        prompt = self.contradiction_prompt.format(
            question=question,
            answer=answer,
            documents=docs_text
        )
        
        response = call_ollama(prompt, self.model, temperature=0.0)
        
        try:
            import json
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            
            contradictions = data.get("contradictions", [])
            return [f"矛盾：{c.get('answer_claim', '')} vs {c.get('document_fact', '')}" 
                    for c in contradictions]
        except:
            return []
    
    def _generate_suggestions(self,
                              score: float,
                              hallucinations: List[str],
                              completeness: Dict,
                              contradictions: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if score < 0.5:
            suggestions.append("⚠️ 答案质量低，建议重新检索")
        
        if hallucinations:
            suggestions.append(f"⚠️ 发现 {len(hallucinations)} 个无支持主张，请核实")
        
        if contradictions:
            suggestions.append(f"⚠️ 发现 {len(contradictions)} 个矛盾，请修正")
        
        if completeness.get("missing_info"):
            missing = completeness["missing_info"]
            suggestions.append(f"ℹ️ 缺失信息：{', '.join(missing[:3])}")
        
        if not completeness.get("core_answered"):
            suggestions.append("⚠️ 答案未回应问题核心")
        
        if not suggestions:
            suggestions.append("✅ 答案质量良好")
        
        return suggestions
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.to_dict()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建 Reflector Agent
    reflector = ReflectorAgent(model="qwen3.5:397b-cloud")
    
    # 测试数据
    question = "Eph 受体的功能是什么？"
    answer = "Eph 受体是受体酪氨酸激酶，参与细胞间信号传导。"
    documents = [
        {"text": "Eph receptors are receptor tyrosine kinases that bind ephrin ligands...", "similarity": 0.85}
    ]
    
    # 运行反思
    print("开始反思校验...")
    result = reflector.reflect(question, answer, documents)
    
    print(f"\n反思结果:")
    print(f"  分数：{result['score']:.2f}")
    print(f"  是否充分：{'✓' if result['is_sufficient'] else '✗'}")
    print(f"  需要重查：{'是' if result['needs_reretrieval'] else '否'}")
    print(f"  问题：{result['issues']}")
    print(f"  建议：{result['suggestions']}")
    print(f"\n统计：{reflector.get_stats()}")
