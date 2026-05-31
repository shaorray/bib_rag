#!/usr/bin/env python3
"""
Academic Writing Assistant - 学术论文写作辅助工具
集成 Eph/Ephrin Agentic RAG 知识库

功能:
- 文献检索与引用
- 事实核查
- 段落生成辅助
- 相关研究查找
"""

import sys
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow, MultiHopRAG


@dataclass
class Citation:
    """引用信息"""
    authors: str
    year: str
    title: str
    relevance: float
    context: str


class AcademicWritingAssistant:
    """学术写作助手"""
    
    def __init__(self, kb_path: str = "/Disk_2/claw_working_dir/ephrin_agentic_rag"):
        self.kb_path = kb_path
        self.citation_history: List[Citation] = []
        
        print("📝 初始化学术写作助手...")
        
        # 初始化知识库
        self.doc_store = DocumentStore(
            'ephrin_papers',
            f'{kb_path}/chroma_db'
        )
        self.embedder = SimpleEmbedding()
        
        # 创建检索器
        def retriever(query, k=10):
            emb = self.embedder.embed(query)
            return self.doc_store.query(emb, n_results=k)
        
        self.workflow = AgenticRAGWorkflow(retriever)
        self.multi_hop = MultiHopRAG(retriever)
        
        print(f"✓ 已加载 {self.doc_store.count()} 个文档块")
    
    def find_references(self, claim: str, min_relevance: float = 0.15) -> List[Citation]:
        """
        为论文中的某个论断查找支持文献
        
        Args:
            claim: 需要支持的研究论断
            min_relevance: 最小相关性阈值
            
        Returns:
            相关引用列表
        """
        print(f"\n🔍 查找支持文献: \"{claim[:60]}...\"")
        
        result = self.workflow.run(claim)
        citations = []
        
        for doc in result['documents']:
            sim = doc.get('similarity', 0)
            if sim >= min_relevance:
                meta = doc['metadata']
                citation = Citation(
                    authors=meta.get('authors', 'Unknown'),
                    year=meta.get('year', 'N/A'),
                    title=meta.get('paper_title', 'Unknown'),
                    relevance=sim,
                    context=doc['text'][:300]
                )
                citations.append(citation)
        
        # 按相关性排序
        citations.sort(key=lambda x: x.relevance, reverse=True)
        
        # 保存到历史
        self.citation_history.extend(citations)
        
        return citations
    
    def fact_check(self, statement: str) -> Dict[str, Any]:
        """
        核查论文中的事实陈述
        
        Args:
            statement: 需要核查的陈述
            
        Returns:
            核查结果，包括支持度评分和相关文献
        """
        print(f"\n✓ 事实核查: \"{statement[:60]}...\"")
        
        result = self.workflow.run(statement)
        
        # 评估支持度
        if result['documents']:
            avg_sim = sum(d.get('similarity', 0) for d in result['documents']) / len(result['documents'])
            support_level = "strong" if avg_sim > 0.25 else "moderate" if avg_sim > 0.15 else "weak"
        else:
            avg_sim = 0
            support_level = "none"
        
        return {
            'statement': statement,
            'support_level': support_level,
            'confidence': avg_sim,
            'supporting_docs': result['documents'][:5],
            'suggestion': self._generate_fact_check_suggestion(support_level, result['documents'])
        }
    
    def _generate_fact_check_suggestion(self, support_level: str, docs: List[Dict]) -> str:
        """生成事实核查建议"""
        if support_level == "strong":
            return "该陈述有充分的文献支持，可以引用相关论文加强论证。"
        elif support_level == "moderate":
            return "该陈述有一定文献支持，建议进一步查阅相关文献或调整表述。"
        elif support_level == "weak":
            return "该陈述缺乏直接文献支持，建议：(1) 查找更多文献 (2) 使用更谨慎的措辞 (3) 添加说明这是假设/推测"
        else:
            return "未找到相关文献支持，请核实该陈述的准确性。"
    
    def find_related_work(self, topic: str, n_papers: int = 10) -> List[Dict]:
        """
        查找与某个主题相关的研究
        
        Args:
            topic: 研究主题
            n_papers: 返回论文数量
            
        Returns:
            相关论文列表
        """
        print(f"\n📚 查找相关研究: \"{topic}\"")
        
        result = self.workflow.run(topic)
        
        # 去重并格式化
        seen_titles = set()
        papers = []
        
        for doc in result['documents']:
            title = doc['metadata'].get('paper_title', 'Unknown')
            if title not in seen_titles:
                seen_titles.add(title)
                papers.append({
                    'title': title,
                    'authors': doc['metadata'].get('authors', 'Unknown'),
                    'year': doc['metadata'].get('year', 'N/A'),
                    'relevance': doc.get('similarity', 0),
                    'excerpt': doc['text'][:200]
                })
        
        papers.sort(key=lambda x: x['relevance'], reverse=True)
        return papers[:n_papers]
    
    def suggest_citation_style(self, citations: List[Citation], style: str = "APA") -> str:
        """
        生成格式化引用
        
        Args:
            citations: 引用列表
            style: 引用格式 (APA, MLA, Vancouver)
            
        Returns:
            格式化引用文本
        """
        formatted = []
        
        for cit in citations:
            if style.upper() == "APA":
                # 简化 APA 格式
                ref = f"{cit.authors} ({cit.year}). {cit.title}."
            elif style.upper() == "VANCOUVER":
                # Vancouver 格式
                ref = f"{cit.authors}. {cit.title}. {cit.year};"
            else:
                ref = f"{cit.authors} ({cit.year}) {cit.title}"
            
            formatted.append(ref)
        
        return "\n".join(formatted)
    
    def generate_paragraph_support(self, topic: str, aspect: str = "mechanism") -> str:
        """
        为论文段落生成支持材料
        
        Args:
            topic: 主题 (如 "cis-interaction")
            aspect: 方面 (mechanism, function, controversy, evidence)
            
        Returns:
            支持材料文本
        """
        query_map = {
            "mechanism": f"{topic} mechanism molecular",
            "function": f"{topic} function role",
            "controversy": f"{topic} debate controversy conflicting",
            "evidence": f"{topic} experimental evidence"
        }
        
        query = query_map.get(aspect, topic)
        
        print(f"\n📝 生成段落支持材料: {topic} ({aspect})")
        
        result = self.workflow.run(query)
        
        if not result['documents']:
            return "未找到相关材料。"
        
        # 整合材料
        materials = []
        materials.append(f"### {topic.capitalize()} - {aspect.capitalize()}\n")
        
        for i, doc in enumerate(result['documents'][:5], 1):
            meta = doc['metadata']
            materials.append(f"[{i}] {meta.get('authors', 'Unknown')} ({meta.get('year', 'N/A')}) "
                           f"reported that {doc['text'][:250]}...")
        
        materials.append(f"\n[支持度: {result['confidence']:.2f}]")
        
        return "\n\n".join(materials)
    
    def check_controversial_claim(self, claim: str) -> Dict[str, Any]:
        """
        检查论断是否存在争议
        
        Args:
            claim: 研究论断
            
        Returns:
            争议分析结果
        """
        print(f"\n⚖️  检查争议性: \"{claim[:60]}...\"")
        
        # 查找支持和反对的文献
        support_query = f"{claim} evidence supports"
        oppose_query = f"{claim} contradicts controversy debate"
        
        support_result = self.workflow.run(support_query)
        oppose_result = self.workflow.run(oppose_query)
        
        support_docs = [d for d in support_result['documents'] if d.get('similarity', 0) > 0.15]
        oppose_docs = [d for d in oppose_result['documents'] if d.get('similarity', 0) > 0.15]
        
        # 评估争议程度
        if len(oppose_docs) > len(support_docs) * 0.5:
            controversy_level = "high"
            advice = "该论断存在较大争议，建议：(1) 同时引用支持和反对的文献 (2) 使用更中性的表述"
        elif len(oppose_docs) > 0:
            controversy_level = "moderate"
            advice = "该论断存在一定争议，建议提及不同观点"
        else:
            controversy_level = "low"
            advice = "该论断争议较小，可以较确定地表述"
        
        return {
            'claim': claim,
            'controversy_level': controversy_level,
            'supporting_papers': len(support_docs),
            'opposing_papers': len(oppose_docs),
            'advice': advice,
            'support_examples': support_docs[:3],
            'oppose_examples': oppose_docs[:3]
        }
    
    def interactive_mode(self):
        """交互式写作辅助模式"""
        print("\n" + "="*70)
        print("📝 学术写作助手 - 交互模式")
        print("="*70)
        print("\n命令:")
        print("  cite <陈述>    - 为论断查找引用")
        print("  check <陈述>   - 事实核查")
        print("  related <主题> - 查找相关研究")
        print("  support <主题> - 生成段落支持材料")
        print("  controversial <论断> - 检查争议性")
        print("  export         - 导出引用历史")
        print("  quit           - 退出")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\n👋 再见!")
                    break
                
                elif user_input.lower().startswith('cite '):
                    claim = user_input[5:]
                    citations = self.find_references(claim)
                    print(f"\n找到 {len(citations)} 篇相关文献:")
                    for i, cit in enumerate(citations[:5], 1):
                        print(f"  [{i}] {cit.authors} ({cit.year}) - rel: {cit.relevance:.3f}")
                        print(f"      {cit.title[:70]}...")
                
                elif user_input.lower().startswith('check '):
                    statement = user_input[6:]
                    result = self.fact_check(statement)
                    print(f"\n✓ 支持度: {result['support_level']}")
                    print(f"  置信度: {result['confidence']:.3f}")
                    print(f"  建议: {result['suggestion']}")
                
                elif user_input.lower().startswith('related '):
                    topic = user_input[8:]
                    papers = self.find_related_work(topic, n_papers=8)
                    print(f"\n📚 相关研究:")
                    for i, paper in enumerate(papers, 1):
                        print(f"  [{i}] {paper['authors']} ({paper['year']})")
                        print(f"      {paper['title'][:60]}...")
                        print(f"      相关性: {paper['relevance']:.3f}")
                
                elif user_input.lower().startswith('support '):
                    topic = user_input[8:]
                    material = self.generate_paragraph_support(topic)
                    print(f"\n{material}")
                
                elif user_input.lower().startswith('controversial '):
                    claim = user_input[14:]
                    result = self.check_controversial_claim(claim)
                    print(f"\n⚖️  争议程度: {result['controversy_level']}")
                    print(f"  支持文献: {result['supporting_papers']}")
                    print(f"  反对文献: {result['opposing_papers']}")
                    print(f"  建议: {result['advice']}")
                
                elif user_input.lower() == 'export':
                    if self.citation_history:
                        output_file = f"citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for cit in self.citation_history:
                                f.write(f"{cit.authors} ({cit.year}). {cit.title}.\n")
                        print(f"✓ 已导出 {len(self.citation_history)} 条引用到 {output_file}")
                    else:
                        print("暂无引用历史")
                
                else:
                    print("未知命令，输入 'cite', 'check', 'related', 'support', 'controversial', 'export', 或 'quit'")
            
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='学术写作助手')
    parser.add_argument('--cite', type=str, help='为论断查找引用')
    parser.add_argument('--check', type=str, help='事实核查')
    parser.add_argument('--related', type=str, help='查找相关研究')
    parser.add_argument('--support', type=str, help='生成段落支持材料')
    parser.add_argument('--controversial', type=str, help='检查争议性')
    
    args = parser.parse_args()
    
    assistant = AcademicWritingAssistant()
    
    if args.cite:
        citations = assistant.find_references(args.cite)
        print(f"\n找到 {len(citations)} 篇相关文献:")
        print(assistant.suggest_citation_style(citations, "APA"))
    
    elif args.check:
        result = assistant.fact_check(args.check)
        print(f"\n支持度: {result['support_level']}")
        print(f"建议: {result['suggestion']}")
    
    elif args.related:
        papers = assistant.find_related_work(args.related)
        for i, paper in enumerate(papers, 1):
            print(f"[{i}] {paper['authors']} ({paper['year']}): {paper['title'][:60]}...")
    
    elif args.support:
        material = assistant.generate_paragraph_support(args.support)
        print(material)
    
    elif args.controversial:
        result = assistant.check_controversial_claim(args.controversial)
        print(f"\n争议程度: {result['controversy_level']}")
        print(f"建议: {result['advice']}")
    
    else:
        # 交互模式
        assistant.interactive_mode()


if __name__ == "__main__":
    main()
