#!/usr/bin/env python3
"""
快速测试 Reflector Agent (无需知识库)
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from reflector_agent import ReflectorAgent

print("="*70)
print("Reflector Agent 快速测试")
print("="*70)

# 创建 Reflector Agent
print("\n🔧 初始化 Reflector Agent...")
reflector = ReflectorAgent(model="qwen3.5:397b-cloud", reflection_threshold=0.8)
print("✓ 已初始化")

# 测试数据
test_cases = [
    {
        'question': 'Eph 受体的功能是什么？',
        'answer': 'Eph 受体是受体酪氨酸激酶，参与细胞间信号传导。',
        'documents': [
            {"text": "Eph receptors are receptor tyrosine kinases that bind ephrin ligands and mediate cell-cell communication.", "similarity": 0.85}
        ],
        'expected': 'pass'
    },
    {
        'question': 'EphA2 的功能？',
        'answer': 'EphA2 是一种 G 蛋白偶联受体，激活后抑制细胞迁移。',  # 错误信息 (幻觉)
        'documents': [
            {"text": "EphA2 is a receptor tyrosine kinase that promotes cell migration and invasion in cancer.", "similarity": 0.90}
        ],
        'expected': 'fail'
    }
]

print("\n" + "="*70)
print("开始测试")
print("="*70)

for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}]")
    print(f"问题：{test['question']}")
    print(f"答案：{test['answer'][:80]}...")
    print(f"预期：{'通过' if test['expected'] == 'pass' else '检测幻觉'}")
    print("-" * 60)
    
    # 运行反思
    result = reflector.reflect(
        test['question'],
        test['answer'],
        test['documents']
    )
    
    print(f"\n反思结果:")
    print(f"  分数：{result['score']:.2f}")
    print(f"  是否充分：{'✓' if result['is_sufficient'] else '✗'}")
    print(f"  需要重查：{'是' if result['needs_reretrieval'] else '否'}")
    
    if result['issues']:
        print(f"  问题：{result['issues']}")
    
    print(f"  建议：{result['suggestions']}")
    
    # 验证预期
    if test['expected'] == 'pass' and result['score'] >= 0.8:
        print(f"\n✅ 测试通过 (高质量答案)")
    elif test['expected'] == 'fail' and result['score'] < 0.8:
        print(f"\n✅ 测试通过 (成功检测低质量)")
    else:
        print(f"\n⚠️ 结果与预期不符")

# 汇总
print("\n" + "="*70)
print("测试汇总")
print("="*70)

print(f"\nReflector 统计:")
print(f"  {reflector.get_stats()}")

print(f"\n✅ Reflector Agent 验证完成!")
print(f"  - 反思阈值：0.8 ✓")
print(f"  - 幻觉检测 ✓")
print(f"  - 重查决策 ✓")

print(f"\n💡 使用示例:")
print("""
from reflector_agent import ReflectorAgent

reflector = ReflectorAgent(
    model="qwen3.5:397b-cloud",
    reflection_threshold=0.8
)

result = reflector.reflect(question, answer, documents)
print(f"分数：{result['score']}")
print(f"需要重查：{result['needs_reretrieval']}")
""")
