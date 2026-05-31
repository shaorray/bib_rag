#!/usr/bin/env python3
"""
RAGAS Evaluator - 简化版 (无需 RAGAS 包)

使用 LLM 评估 RAG 质量指标:
- Faithfulness (幻觉检测)
- Answer Relevance (问题回应)
- Context Precision (检索质量)
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# ==================== 数据结构 ====================

@dataclass
class EvaluationSample:
    """评估样本"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class EvaluationResult:
    """评估结果"""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    answer_correctness: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    samples_count: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_answer_correctness: Optional[float]


# ==================== LLM 调用 ====================

def call_ollama(prompt: str, model: str = "qwen3.5:397b-cloud", temperature: float = 0.0) -> str:
    """调用 Ollama API (使用 generate 端点)"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"  ⚠️  Ollama 调用失败：{e}")
        return "0.5"  # 默认


# ==================== RAGAS 评估器 ====================

class SimpleRAGASEvaluator:
    """简化版 RAGAS 评估器"""
    
    def __init__(self, model: str = "qwen3.5:397b-cloud"):
        self.model = model  # 使用 cloud 模型
    
    def evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """评估 Faithfulness (幻觉检测)"""
        context_text = "\n\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])
        
        prompt = f"""请评估以下答案是否完全基于提供的上下文。

**Context**:
{context_text}

**Question**: {question}

**Answer**: {answer}

**评估标准**:
- 答案的每个关键主张都能在上下文中找到支持
- 答案没有引入上下文之外的信息
- 答案没有扭曲上下文的信息

**评分 (0-1)**:
- 1.0: 完全忠于上下文，无任何幻觉
- 0.5: 部分信息有支持，部分来自模型知识
- 0.0: 完全无支持或与上下文矛盾

请只输出一个数字 (0.0-1.0): """
        
        response = call_ollama(prompt, self.model)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """评估 Answer Relevance"""
        prompt = f"""请评估以下答案在多大程度上回应了问题。

**Question**: {question}

**Answer**: {answer}

**评估标准**:
- 答案直接回应了问题的核心
- 答案覆盖了问题的所有关键方面
- 答案没有偏题或包含无关信息

**评分 (0-1)**:
- 1.0: 完美回应，完全切题
- 0.5: 部分回应，有些偏题
- 0.0: 完全偏题或未回答问题

请只输出一个数字 (0.0-1.0): """
        
        response = call_ollama(prompt, self.model)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """评估 Context Precision"""
        contexts_text = "\n\n".join([f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)])
        
        prompt = f"""请评估检索到的上下文与问题的相关性。

**Question**: {question}

**Contexts**:
{contexts_text}

**评估标准**:
- 上下文包含回答问题所需的关键信息
- 最相关的信息出现在靠前的位置
- 上下文没有大量无关内容

**评分 (0-1)**:
- 1.0: 完美检索，所有上下文都高度相关
- 0.5: 部分相关，有些上下文无用
- 0.0: 完全不相关

请只输出一个数字 (0.0-1.0): """
        
        response = call_ollama(prompt, self.model)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """评估单个样本"""
        faithfulness = self.evaluate_faithfulness(sample.question, sample.answer, sample.contexts)
        relevance = self.evaluate_relevance(sample.question, sample.answer)
        precision = self.evaluate_context_precision(sample.question, sample.contexts)
        
        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision
        )
    
    def evaluate_batch(self, samples: List[EvaluationSample]) -> BatchEvaluationResult:
        """批量评估"""
        results = [self.evaluate_sample(s) for s in samples]
        
        return BatchEvaluationResult(
            samples_count=len(samples),
            avg_faithfulness=sum(r.faithfulness for r in results) / len(results),
            avg_answer_relevance=sum(r.answer_relevance for r in results) / len(results),
            avg_context_precision=sum(r.context_precision for r in results) / len(results),
            avg_answer_correctness=None
        )


# ==================== 报告生成 ====================

class EvaluationReporter:
    """评估报告生成器"""
    
    @staticmethod
    def generate_report(result: BatchEvaluationResult, output_path: str = None) -> str:
        """生成评估报告"""
        report = f"""# RAGAS 评估报告

**生成时间**: 2026-03-28
**样本数量**: {result.samples_count}

## 核心指标

| 指标 | 分数 | 说明 |
|------|------|------|
| **Faithfulness** | {result.avg_faithfulness:.3f} | 答案忠于上下文 (幻觉检测) |
| **Answer Relevance** | {result.avg_answer_relevance:.3f} | 答案回应问题 |
| **Context Precision** | {result.avg_context_precision:.3f} | 检索质量 |

## 质量评估

"""
        
        avg_score = (result.avg_faithfulness + result.avg_answer_relevance + result.avg_context_precision) / 3
        quality = "优秀 ⭐" if avg_score >= 0.8 else "良好 ✓" if avg_score >= 0.6 else "需改进 ⚠️"
        
        report += f"**综合评分**: {avg_score:.3f} - {quality}\n\n"
        
        # 改进建议
        report += "## 改进建议\n\n"
        
        if result.avg_faithfulness < 0.7:
            report += "- ⚠️ **Faithfulness 偏低**: 增加 Self-RAG 评估，减少幻觉\n"
        if result.avg_answer_relevance < 0.7:
            report += "- ⚠️ **Answer Relevance 偏低**: 优化答案生成 prompt\n"
        if result.avg_context_precision < 0.7:
            report += "- ⚠️ **Context Precision 偏低**: 改进检索策略\n"
        
        if all([result.avg_faithfulness >= 0.7, result.avg_answer_relevance >= 0.7, result.avg_context_precision >= 0.7]):
            report += "- ✅ 所有指标良好\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 报告已保存：{output_path}")
        
        return report


# ==================== 使用示例 ====================

if __name__ == "__main__":
    evaluator = SimpleRAGASEvaluator()
    
    samples = [
        EvaluationSample(
            question="What is cis-interaction?",
            answer="Cis-interaction is when Eph receptors bind to ephrins on the same cell.",
            contexts=["Eph receptors and ephrins interact in cis on the same cell surface."]
        )
    ]
    
    result = evaluator.evaluate_batch(samples)
    report = EvaluationReporter.generate_report(result)
    print(report)
