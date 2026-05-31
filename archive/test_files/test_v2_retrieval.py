#!/usr/bin/env python3
"""
测试 RAG v2 检索能力优化
"""

import sys
import os
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from rag_core_v2 import DocumentStore, HybridEmbedding
from agentic_workflow_v2 import AcademicRAGWorkflow, WritingScene

def test_hybrid_retrieval():
    """测试混合检索"""
    print("="*60)
    print("测试 1: 混合检索 (Dense + BM25)")
    print("="*60)
    
    doc_store = DocumentStore(
        collection_name="ephrin_papers",
        persist_directory="/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db",
        use_hybrid=True,
        use_reranker=False  # 暂时禁用 reranker (需要下载模型)
    )
    
    print(f"\n✓ 文档库加载完成：{doc_store.count()} 篇文献")
    print(f"✓ 混合检索：{'启用' if doc_store.use_hybrid else '禁用'}")
    print(f"✓ Reranker: {'启用' if doc_store.embedder.reranker else '禁用 (使用相似度排序)'}")
    
    # 测试查询
    query = "EphA2 cis-interaction promotes cancer metastasis"
    print(f"\n查询：{query}")
    
    # Dense only
    query_emb = doc_store.embedder.embed(query)
    dense_results = doc_store.query(query_emb, n_results=5, use_reranker=False)
    print(f"\nDense only: {len(dense_results)} 篇")
    for i, doc in enumerate(dense_results[:3], 1):
        meta = doc.get('metadata', {})
        print(f"  {i}. {meta.get('paper_title', 'Unknown')[:60]}... (sim={doc.get('similarity', 0):.3f})")
    
    # Hybrid + Rerank
    hybrid_results = doc_store.query_hybrid(
        query_text=query,
        n_results=5,
        use_reranker=False  # 暂时禁用
    )
    print(f"\nHybrid (no rerank): {len(hybrid_results)} 篇")
    for i, doc in enumerate(hybrid_results[:3], 1):
        meta = doc.get('metadata', {})
        print(f"  {i}. {meta.get('paper_title', 'Unknown')[:60]}... (hybrid_score={doc.get('hybrid_score', 0):.3f})")
    
    print("\n✓ 测试完成")


def test_metadata_filter():
    """测试元数据过滤"""
    print("\n" + "="*60)
    print("测试 2: 元数据过滤")
    print("="*60)
    
    doc_store = DocumentStore(
        collection_name="ephrin_papers",
        persist_directory="/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db",
        use_hybrid=True,
        use_reranker=True
    )
    
    # 过滤：近 5 年
    print("\n过滤条件：recent_years = 5 (2021-2026)")
    query = "Eph receptor signaling"
    results = doc_store.query_hybrid(
        query_text=query,
        n_results=5,
        filter_metadata={'recent_years': 5},
        use_reranker=True
    )
    
    print(f"检索到 {len(results)} 篇")
    for i, doc in enumerate(results, 1):
        meta = doc.get('metadata', {})
        print(f"  {i}. {meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}) - {meta.get('paper_title', 'Unknown')[:50]}...")
    
    # 过滤：特定期刊
    print("\n过滤条件：journal = 'Nature'")
    results = doc_store.query_hybrid(
        query_text=query,
        n_results=5,
        filter_metadata={'journal': 'Nature'},
        use_reranker=True
    )
    
    print(f"检索到 {len(results)} 篇")
    for i, doc in enumerate(results, 1):
        meta = doc.get('metadata', {})
        print(f"  {i}. {meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}) - {meta.get('journal', 'Unknown')}")
    
    print("\n✓ 测试完成")


def test_query_rewriting():
    """测试查询重写"""
    print("\n" + "="*60)
    print("测试 3: 查询重写")
    print("="*60)
    
    doc_store = DocumentStore(
        collection_name="ephrin_papers",
        persist_directory="/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db",
        use_hybrid=True,
        use_reranker=True
    )
    
    workflow = AcademicRAGWorkflow(doc_store=doc_store, temperature=0.2)
    
    # 测试查询重写
    test_queries = [
        "EphA2 在癌症中的作用",
        "cis-interaction 机制",
        "Eph/ephrin tetramerization inhibitors",
    ]
    
    for query in test_queries:
        print(f"\n原始查询：{query}")
        rewritten = workflow.rewrite_query(query, "literature_review")
        print(f"重写后：{rewritten}")
    
    print("\n✓ 测试完成")


def test_scene_prompts():
    """测试分场景提示词"""
    print("\n" + "="*60)
    print("测试 4: 分场景提示词模板")
    print("="*60)
    
    from agentic_workflow_v2 import PROMPT_TEMPLATES
    
    print(f"\n可用场景：{len(PROMPT_TEMPLATES)} 个")
    for scene in WritingScene:
        template = PROMPT_TEMPLATES[scene.value]
        has_system = "system" in template
        has_user = "user" in template
        print(f"  ✓ {scene.value:25} - system: {has_system}, user: {has_user}")
    
    print("\n✓ 测试完成")


def test_temperature_control():
    """测试温度控制"""
    print("\n" + "="*60)
    print("测试 5: 温度控制")
    print("="*60)
    
    doc_store = DocumentStore(
        collection_name="ephrin_papers",
        persist_directory="/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db",
        use_hybrid=True,
        use_reranker=True
    )
    
    # 不同温度设置
    temps = [0.1, 0.2, 0.3]
    
    for temp in temps:
        workflow = AcademicRAGWorkflow(doc_store=doc_store, temperature=temp)
        print(f"\n温度：{temp}")
        print(f"  用途：{'方法描述/事实核查' if temp <= 0.1 else '文献综述/一般写作' if temp <= 0.2 else '讨论/推测'}")
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG v2 检索能力优化测试")
    print("="*60)
    
    test_hybrid_retrieval()
    test_metadata_filter()
    test_query_rewriting()
    test_scene_prompts()
    test_temperature_control()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
