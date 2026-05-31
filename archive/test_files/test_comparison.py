#!/usr/bin/env python3
"""
LangGraph vs 原版 对比测试

测试内容:
1. 功能正确性对比
2. 性能对比 (延迟/迭代次数)
3. 代码质量对比
4. 可维护性对比

测试查询:
- 简单事实：Eph 受体是什么类型的蛋白质？
- 复杂对比：EphA2 与 EphB4 在癌症中的功能差异？
- 机制问题：cis-interaction 的分子机制是什么？
"""

import sys
import time
import json
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

# 导入两个版本
from agentic_rag_workflow import AgenticRAGWorkflow  # 原版
from langgraph_agentic_rag import LangGraphAgenticRAG  # LangGraph 版
from rag_core import SimpleEmbedding, DocumentStore


# ==================== 测试数据集 ====================

TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "Eph 受体是什么类型的蛋白质？",
        "type": "simple",
        "expected_attempts": 1,
        "expected_min_score": 0.8
    },
    {
        "id": "Q2",
        "query": "EphA2 与 EphB4 在癌症中的功能差异？",
        "type": "complex",
        "expected_attempts": 2,
        "expected_min_score": 0.8
    },
    {
        "id": "Q3",
        "query": "cis-interaction 的分子机制是什么？",
        "type": "moderate",
        "expected_attempts": 1,
        "expected_min_score": 0.8
    }
]


# ==================== 测试函数 ====================

def test_original_workflow(retriever, query: str) -> Dict[str, Any]:
    """测试原版工作流"""
    start_time = time.time()
    
    # 创建工作流
    workflow = AgenticRAGWorkflow(retriever_fn=retriever, model="qwen3.5:397b-cloud")
    
    # 运行
    result = workflow.run(query, verbose=False)
    
    end_time = time.time()
    
    return {
        "version": "original",
        "query": query,
        "answer": result.get("answer", ""),
        "reflect_score": result.get("reflect_score", 0.0),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "total_time": end_time - start_time,
        "status": result.get("status", "unknown")
    }


def test_langgraph_workflow(retriever, query: str) -> Dict[str, Any]:
    """测试 LangGraph 版工作流"""
    start_time = time.time()
    
    # 创建工作流
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
    
    # 运行
    result = rag.run(query, verbose=False)
    
    end_time = time.time()
    
    return {
        "version": "langgraph",
        "query": query,
        "answer": result.get("generated_answer", ""),
        "reflect_score": result.get("reflect_score", 0.0),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "total_time": end_time - start_time,
        "status": "success" if result.get("reflect_score", 0) >= 0.8 else "needs_reretrieval",
        "reasoning_steps": result.get("reasoning_steps", [])
    }


def compare_results(original: Dict, langgraph: Dict) -> Dict[str, Any]:
    """对比两个版本的结果"""
    
    # 分数差异
    score_diff = langgraph["reflect_score"] - original["reflect_score"]
    
    # 时间差异
    time_diff = langgraph["total_time"] - original["total_time"]
    
    # 尝试次数差异
    attempts_diff = langgraph["retrieval_attempts"] - original["retrieval_attempts"]
    
    # 答案长度差异
    answer_len_diff = len(langgraph["answer"]) - len(original["answer"])
    
    return {
        "score_diff": score_diff,
        "time_diff_sec": time_diff,
        "attempts_diff": attempts_diff,
        "answer_len_diff": answer_len_diff,
        "winner_score": "langgraph" if score_diff > 0.05 else ("original" if score_diff < -0.05 else "tie"),
        "winner_speed": "langgraph" if time_diff < -1.0 else ("original" if time_diff > 1.0 else "tie")
    }


def run_comparison_tests():
    """运行对比测试"""
    print("="*80)
    print("LangGraph vs 原版 对比测试")
    print("="*80)
    print(f"测试时间：{datetime.now().isoformat()}")
    print()
    
    # 加载知识库
    print("📂 加载知识库...")
    doc_store = DocumentStore(
        'ephrin_papers',
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    def retriever(query, k=10):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    print(f"✓ 已加载 {doc_store.count()} 个文档")
    print()
    
    # 测试结果存储
    all_results = []
    
    # 运行测试
    print("="*80)
    print("开始测试")
    print("="*80)
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"[测试 {i}/{len(TEST_QUERIES)}] {test_case['id']} - {test_case['type']}")
        print(f"查询：{test_case['query']}")
        print(f"预期：{test_case['expected_attempts']} 次检索，分数 >= {test_case['expected_min_score']}")
        print("="*80)
        
        # 测试原版
        print("\n[1/2] 测试原版工作流...")
        try:
            original_result = test_original_workflow(retriever, test_case['query'])
            print(f"  ✓ 完成")
            print(f"    分数：{original_result['reflect_score']:.2f}")
            print(f"    尝试：{original_result['retrieval_attempts']} 次")
            print(f"    耗时：{original_result['total_time']:.2f}秒")
            print(f"    状态：{original_result['status']}")
        except Exception as e:
            print(f"  ❌ 错误：{e}")
            original_result = None
        
        # 测试 LangGraph 版
        print("\n[2/2] 测试 LangGraph 版工作流...")
        try:
            langgraph_result = test_langgraph_workflow(retriever, test_case['query'])
            print(f"  ✓ 完成")
            print(f"    分数：{langgraph_result['reflect_score']:.2f}")
            print(f"    尝试：{langgraph_result['retrieval_attempts']} 次")
            print(f"    耗时：{langgraph_result['total_time']:.2f}秒")
            print(f"    状态：{langgraph_result['status']}")
            print(f"    推理步骤：{len(langgraph_result.get('reasoning_steps', []))} 步")
        except Exception as e:
            print(f"  ❌ 错误：{e}")
            langgraph_result = None
        
        # 对比结果
        if original_result and langgraph_result:
            print("\n[对比分析]")
            comparison = compare_results(original_result, langgraph_result)
            
            print(f"  分数差异：{comparison['score_diff']:+.2f} ({'LangGraph 胜' if comparison['winner_score'] == 'langgraph' else '原版胜' if comparison['winner_score'] == 'original' else '平局'})")
            print(f"  时间差异：{comparison['time_diff_sec']:+.2f}秒 ({'LangGraph 快' if comparison['winner_speed'] == 'langgraph' else '原版快' if comparison['winner_speed'] == 'original' else '相当'})")
            print(f"  尝试差异：{comparison['attempts_diff']:+d} 次")
            print(f"  答案长度：{comparison['answer_len_diff']:+d} 字符")
            
            # 保存结果
            all_results.append({
                "test_id": test_case['id'],
                "query": test_case['query'],
                "type": test_case['type'],
                "original": original_result,
                "langgraph": langgraph_result,
                "comparison": comparison
            })
        
        print()
    
    # 汇总报告
    print("="*80)
    print("汇总报告")
    print("="*80)
    
    if all_results:
        # 统计胜负
        langgraph_wins = sum(1 for r in all_results if r['comparison']['winner_score'] == 'langgraph')
        original_wins = sum(1 for r in all_results if r['comparison']['winner_score'] == 'original')
        ties = sum(1 for r in all_results if r['comparison']['winner_score'] == 'tie')
        
        # 平均分数
        avg_original_score = sum(r['original']['reflect_score'] for r in all_results) / len(all_results)
        avg_langgraph_score = sum(r['langgraph']['reflect_score'] for r in all_results) / len(all_results)
        
        # 平均时间
        avg_original_time = sum(r['original']['total_time'] for r in all_results) / len(all_results)
        avg_langgraph_time = sum(r['langgraph']['total_time'] for r in all_results) / len(all_results)
        
        print(f"""
📊 总体统计:
  - 测试总数：{len(all_results)}
  - LangGraph 胜：{langgraph_wins} 场
  - 原版胜：{original_wins} 场
  - 平局：{ties} 场

🎯 质量对比:
  - 原版平均分数：{avg_original_score:.3f}
  - LangGraph 平均分数：{avg_langgraph_score:.3f}
  - 分数提升：{(avg_langgraph_score - avg_original_score):.3f} ({(avg_langgraph_score/avg_original_score - 1)*100:+.1f}%)

⏱️  性能对比:
  - 原版平均耗时：{avg_original_time:.2f}秒
  - LangGraph 平均耗时：{avg_langgraph_time:.2f}秒
  - 时间差异：{(avg_langgraph_time - avg_original_time):+.2f}秒 ({(avg_langgraph_time/avg_original_time - 1)*100:+.1f}%)

✅ 结论:
""")
        
        if langgraph_wins > original_wins:
            print(f"  🎉 LangGraph 版在 {langgraph_wins}/{len(all_results)} 测试中胜出！")
        elif original_wins > langgraph_wins:
            print(f"  ⚠️  原版在 {original_wins}/{len(all_results)} 测试中表现更好")
        else:
            print(f"  🤝 两个版本表现相当")
        
        if avg_langgraph_score > avg_original_score:
            print(f"  📈 LangGraph 版答案质量提升 {((avg_langgraph_score/avg_original_score - 1)*100):.1f}%")
        
        if avg_langgraph_time < avg_original_time:
            print(f"  ⚡ LangGraph 版速度提升 {((1 - avg_langgraph_time/avg_original_time)*100):.1f}%")
    
    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(all_results),
        "results": all_results,
        "summary": {
            "langgraph_wins": langgraph_wins if all_results else 0,
            "original_wins": original_wins if all_results else 0,
            "ties": ties if all_results else 0,
            "avg_original_score": avg_original_score if all_results else 0,
            "avg_langgraph_score": avg_langgraph_score if all_results else 0,
            "avg_original_time": avg_original_time if all_results else 0,
            "avg_langgraph_time": avg_langgraph_time if all_results else 0
        }
    }
    
    report_path = '/Disk_2/claw_working_dir/ephrin_agentic_rag/COMPARISON_TEST_REPORT.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告已保存：{report_path}")
    print()
    
    return report


if __name__ == "__main__":
    try:
        report = run_comparison_tests()
        
        # 最终状态
        print("="*80)
        print("测试状态")
        print("="*80)
        
        if report['summary']['langgraph_wins'] >= report['summary']['original_wins']:
            print("✅ LangGraph 版表现优秀 - 建议迁移")
        else:
            print("⚠️  原版表现更好 - 建议继续优化 LangGraph 版")
        
        print(f"\n综合评分：LangGraph {(report['summary']['avg_langgraph_score']):.3f} vs 原版 {(report['summary']['avg_original_score']):.3f}")
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
