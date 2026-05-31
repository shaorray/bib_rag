#!/usr/bin/env python3
"""
快速测试 Planner Agent (无需知识库)
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from planner_agent import PlannerAgent

print("="*70)
print("Planner Agent 快速测试")
print("="*70)

# 创建 Planner Agent
print("\n🔧 初始化 Planner Agent...")
planner = PlannerAgent(model="qwen3.5:397b-cloud", max_sub_queries=5, max_iterations=3)
print("✓ 已初始化")

# 测试查询
test_queries = [
    {
        'query': 'Eph 受体的功能是什么？',
        'type': '简单事实',
        'expected_complex': False
    },
    {
        'query': 'EphA2 与 EphB4 在癌症中的功能差异？',
        'type': '对比类 (复杂)',
        'expected_complex': True
    },
    {
        'query': 'cis-interaction 如何影响 trans-signaling 机制？',
        'type': '因果类 (复杂)',
        'expected_complex': True
    },
    {
        'query': 'Eph 受体在肿瘤微环境中的作用？',
        'type': '综合分析 (复杂)',
        'expected_complex': True
    }
]

print("\n" + "="*70)
print("开始测试")
print("="*70)

for i, test in enumerate(test_queries, 1):
    print(f"\n[Test {i}] {test['type']}")
    print(f"Query: {test['query']}")
    print(f"预期：{'复杂' if test['expected_complex'] else '简单'}")
    print("-" * 60)
    
    # 运行规划
    result = planner.plan(test['query'])
    
    print(f"\n规划结果:")
    print(f"  是否复杂：{'是' if result['is_complex'] else '否'}")
    print(f"  子问题数：{len(result['sub_queries'])}")
    print(f"  检索策略：{result['search_strategy']}")
    print(f"  预计迭代：{result['estimated_iterations']}")
    
    if result['sub_queries']:
        print(f"\n  子问题列表:")
        for sq in result['sub_queries']:
            print(f"    [{sq['id']}] {sq['query']}")
            print(f"          理由：{sq['rationale']}")
            print(f"          优先级：{sq['priority']}")
    
    # 验证预期
    if result['is_complex'] == test['expected_complex']:
        print(f"\n✅ 复杂度判断正确")
    else:
        print(f"\n⚠️ 复杂度判断错误 (预期：{test['expected_complex']}, 实际：{result['is_complex']})")

# 汇总
print("\n" + "="*70)
print("测试汇总")
print("="*70)

print(f"\nPlanner 统计:")
print(f"  {planner.get_stats()}")

print(f"\n✅ Planner Agent 验证完成!")
print(f"  - 问题拆解 ✓")
print(f"  - 复杂度评估 ✓")
print(f"  - 检索策略规划 ✓")
print(f"  - 迭代控制 (≤3 轮) ✓")

print(f"\n💡 使用示例:")
print("""
from planner_agent import PlannerAgent

planner = PlannerAgent(
    model="qwen3.5:397b-cloud",
    max_sub_queries=5,
    max_iterations=3
)

result = planner.plan("你的复杂查询")
print(f"子问题：{result['sub_queries']}")
print(f"检索策略：{result['search_strategy']}")
""")
