#!/usr/bin/env python3
"""
Interactive Query Interface for Eph/Ephrin Knowledge Base
交互式查询界面
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

# Import RAG components
from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow, MultiHopRAG


class QueryInterface:
    """交互式查询界面"""
    
    def __init__(self, kb_path: str = "/Disk_2/claw_working_dir/ephrin_agentic_rag"):
        self.kb_path = kb_path
        self.chroma_path = f"{kb_path}/chroma_db"
        self.history_file = f"{kb_path}/query_history.json"
        
        print("🔧 初始化知识库...")
        
        # 初始化嵌入模型
        self.embedder = SimpleEmbedding()
        
        # 初始化文档存储
        self.doc_store = DocumentStore(
            collection_name="ephrin_papers",
            persist_directory=self.chroma_path
        )
        
        # 检查是否有数据
        doc_count = self.doc_store.count()
        if doc_count == 0:
            print(f"❌ 知识库为空！请先运行 build_knowledge_base.py")
            sys.exit(1)
        
        print(f"✓ 已加载 {doc_count} 个文档块")
        
        # 初始化 Agentic RAG
        self.workflow = AgenticRAGWorkflow(self._create_retriever())
        self.multi_hop = MultiHopRAG(self._create_retriever())
        
        # 加载查询历史
        self.query_history = self._load_history()
    
    def _create_retriever(self):
        """创建检索器"""
        def retriever_fn(query: str, k: int = 8) -> List[Dict]:
            query_embedding = self.embedder.embed(query)
            return self.doc_store.query(query_embedding, n_results=k)
        return retriever_fn
    
    def _load_history(self) -> List[Dict]:
        """加载查询历史"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """保存查询历史"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.query_history, f, indent=2, ensure_ascii=False)
    
    def _format_sources(self, documents: List[Dict], max_sources: int = 5) -> str:
        """格式化来源"""
        if not documents:
            return "No sources found."
        
        sources = []
        for i, doc in enumerate(documents[:max_sources], 1):
            title = doc['metadata'].get('paper_title', 'Unknown')
            year = doc['metadata'].get('year', 'N/A')
            similarity = doc.get('similarity', 0)
            
            sources.append(f"[{i}] {title} ({year}) - relevance: {similarity:.3f}")
        
        return "\n".join(sources)
    
    def _print_banner(self):
        """打印欢迎横幅"""
        print("\n" + "="*70)
        print("  🧠 Eph/Ephrin Agentic RAG Knowledge Base Query Interface")
        print("  🤖 Powered by Self-RAG + CRAG + Multi-hop Reasoning")
        print("="*70)
        print("\n  Commands:")
        print("    /help      - Show this help message")
        print("    /stats     - Show knowledge base statistics")
        print("    /history   - Show query history")
        print("    /multihop <query> - Use multi-hop reasoning")
        print("    /clear     - Clear screen")
        print("    /exit      - Exit the interface")
        print("-"*70)
    
    def _print_stats(self):
        """打印统计信息"""
        doc_count = self.doc_store.count()
        
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"   • Total document chunks: {doc_count:,}")
        print(f"   • Query history entries: {len(self.query_history)}")
        print(f"   • Storage path: {self.chroma_path}")
        print()
    
    def _print_history(self):
        """打印查询历史"""
        if not self.query_history:
            print("\n📜 No query history yet.\n")
            return
        
        print(f"\n📜 Query History (last 10):")
        print("-"*70)
        
        for i, entry in enumerate(self.query_history[-10:], 1):
            print(f"\n[{i}] {entry['timestamp']}")
            print(f"    Q: {entry['query']}")
            print(f"    Confidence: {entry.get('confidence', 'N/A'):.2f}")
            print(f"    Docs retrieved: {entry.get('num_docs', 'N/A')}")
            # 只显示答案的前 100 字
            answer_preview = entry.get('answer', '')[:100]
            if len(entry.get('answer', '')) > 100:
                answer_preview += "..."
            print(f"    A: {answer_preview}")
        print()
    
    def query(self, question: str, use_multihop: bool = False) -> Dict[str, Any]:
        """执行查询"""
        print(f"\n🔍 Processing: {question}")
        print("-"*70)
        
        if use_multihop:
            result = self.multi_hop.run_multi_hop(question)
        else:
            result = self.workflow.run(question)
        
        # 显示结果
        print(f"\n📋 Answer:\n{result['answer']}")
        print(f"\n📊 Confidence: {result['confidence']:.2f}")
        print(f"🔍 Documents retrieved: {len(result['documents'])}")
        
        if result.get('retries', 0) > 0:
            print(f"🔄 Query rewrites: {result['retries']}")
        
        if result.get('rewritten', False):
            print("✏️  Query was rewritten for better retrieval")
        
        # 显示来源
        print(f"\n📚 Sources:")
        print(self._format_sources(result['documents']))
        
        # 添加到历史
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'num_docs': len(result['documents']),
            'use_multihop': use_multihop
        })
        
        # 限制历史长度
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
        
        self._save_history()
        
        return result
    
    def run_interactive(self):
        """运行交互模式"""
        self._print_banner()
        
        while True:
            try:
                user_input = input("\n❓ Query: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() == '/exit':
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/help':
                    self._print_banner()
                
                elif user_input.lower() == '/stats':
                    self._print_stats()
                
                elif user_input.lower() == '/history':
                    self._print_history()
                
                elif user_input.lower() == '/clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    self._print_banner()
                
                elif user_input.lower().startswith('/multihop '):
                    query = user_input[10:].strip()
                    if query:
                        self.query(query, use_multihop=True)
                
                else:
                    # 普通查询
                    self.query(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def run_batch_queries(queries: List[str], kb_path: str):
    """批量运行查询"""
    interface = QueryInterface(kb_path)
    
    print(f"\n🚀 Running {len(queries)} queries in batch mode...\n")
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'='*70}")
        
        result = interface.query(query)
        results.append({
            'query': query,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'num_docs': len(result['documents'])
        })
    
    # 保存结果
    output_file = f"{kb_path}/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Batch complete! Results saved to: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Eph/Ephrin Agentic RAG Query Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode
  %(prog)s -q "What is cis-interaction?"    # Single query
  %(prog)s --multihop "Compare forward and reverse signaling"  # Multi-hop
  %(prog)s --batch queries.txt       # Batch mode
        """
    )
    
    parser.add_argument('-q', '--query', type=str, help='Single query mode')
    parser.add_argument('--multihop', action='store_true', help='Use multi-hop reasoning')
    parser.add_argument('--batch', type=str, help='Batch mode: file with queries (one per line)')
    parser.add_argument('--kb-path', type=str, 
                       default='/Disk_2/claw_working_dir/ephrin_agentic_rag',
                       help='Knowledge base path')
    
    args = parser.parse_args()
    
    # 批量模式
    if args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        run_batch_queries(queries, args.kb_path)
        return
    
    # 单查询模式
    if args.query:
        interface = QueryInterface(args.kb_path)
        interface.query(args.query, use_multihop=args.multihop)
        return
    
    # 交互模式
    interface = QueryInterface(args.kb_path)
    interface.run_interactive()


if __name__ == "__main__":
    main()
