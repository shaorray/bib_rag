#!/usr/bin/env python3
"""
Agentic RAG Workflow v2 - State-of-the-Art Academic Writing
增强版学术写作 RAG 工作流

新增功能:
1. Query Rewriting (查询重写)
2. Metadata Filtering (元数据过滤)
3. Re-ranking (重排序)
4. Scene-specific Prompt Templates (分场景提示词)
5. Temperature Control (温度控制)
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Literal, TypedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# 尝试导入必要库
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 本地导入
from rag_core_v2 import DocumentStore, HybridEmbedding


class WritingScene(Enum):
    """写作场景"""
    LITERATURE_REVIEW = "literature_review"  # 文献综述
    INTRODUCTION = "introduction"  # 引言
    METHODS = "methods"  # 方法描述
    RESULTS = "results"  # 结果分析
    DISCUSSION = "discussion"  # 讨论
    ARGUMENT溯源 = "argument_tracing"  # 论点溯源
    PARAGRAPH_POLISH = "paragraph_polish"  # 段落润色
    CITATION_FORMAT = "citation_format"  # 引用规范


class RAGState(TypedDict):
    """RAG 状态定义"""
    query: str
    rewritten_query: str
    writing_scene: str
    documents: List[Dict]
    reranked_documents: List[Dict]
    answer: str
    references: List[Dict]
    confidence: float
    retries: int
    errors: List[str]
    metadata_filter: Dict[str, Any]


@dataclass
class Citation:
    """引用信息 (GB/T 7714-2015)"""
    authors: str
    year: str
    title: str
    journal: str
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    doi: str = ""
    
    def format_in_text(self) -> str:
        """文内引用格式"""
        if 'et al' in self.authors.lower() or '等' in self.authors:
            return f"({self.authors}, {self.year})"
        elif ',' in self.authors:
            # 多作者
            first_author = self.authors.split(',')[0]
            return f"({first_author} 等，{self.year})"
        else:
            return f"({self.authors}, {self.year})"
    
    def format_reference(self) -> str:
        """参考文献格式 (GB/T 7714-2015)"""
        ref = f"[{self.year}] {self.authors}. {self.title}"
        
        if self.journal:
            ref += f"[J]. {self.journal}"
            if self.volume:
                ref += f", {self.volume}"
                if self.issue:
                    ref += f"({self.issue})"
                if self.pages:
                    ref += f": {self.pages}"
            elif self.pages:
                ref += f": {self.pages}"
        else:
            ref += "[D]"  # 默认学位论文格式
        
        return ref + "."


# ===== 分场景提示词模板 =====

PROMPT_TEMPLATES = {
    WritingScene.LITERATURE_REVIEW.value: {
        "system": """你是一名严谨的学术写作助手，专门撰写文献综述部分。

## 任务
基于检索到的文献，撰写{topic}的文献综述。

## 要求
1. 按时间或主题组织文献
2. 突出研究进展和争议
3. 每个论点必须标注引用 (作者，年份)
4. 使用学术中文，避免口语化
5. 若文献不足，明确说明"现有文献库中相关研究有限"

## 输出格式
```markdown
## {topic}研究进展

### 早期研究 (2020 年以前)
...

### 近期突破 (2020-2024)
...

### 争议与未解问题
...

### 参考文献
[按 GB/T 7714-2015 格式列出]
```

## 温度设置
temperature = 0.2 (降低随机性，确保准确性)""",
        
        "user": """请基于以下检索到的{doc_count}篇文献，撰写关于"{topic}"的文献综述：

{retrieved_documents}

## 写作要求
- 文献综述长度：{word_count}字
- 引用格式：(作者，年份)
- 参考文献格式：GB/T 7714-2015
- 语言：学术中文""",
    },
    
    WritingScene.INTRODUCTION.value: {
        "system": """你是一名学术论文写作助手，专门撰写引言部分。

## 任务
撰写论文引言，包含：研究背景、问题陈述、研究意义。

## 要求
1. 从广泛背景逐步聚焦到具体问题
2. 引用关键文献支持论点
3. 明确研究空白 (research gap)
4. 概述本文贡献
5. 每段 3-5 句，逻辑连贯

## 输出格式
```markdown
## 引言

第一段：研究背景和重要性
第二段：现有研究的局限
第三段：本文研究问题和贡献
```

## 温度设置
temperature = 0.2""",
        
        "user": """请基于以下{doc_count}篇文献，撰写关于"{topic}"的论文引言：

{retrieved_documents}

## 要求
- 引言长度：{word_count}字
- 包含 research gap 陈述
- 引用格式：(作者，年份)""",
    },
    
    WritingScene.METHODS.value: {
        "system": """你是一名研究方法写作助手。

## 任务
描述研究方法，确保可重复性。

## 要求
1. 详细说明实验设计、数据来源、分析方法
2. 引用方法学文献
3. 使用被动语态 (中文用"采用"、"使用")
4. 避免主观评价

## 输出格式
```markdown
## 方法

### 数据来源
...

### 实验设计
...

### 统计分析
...
```

## 温度设置
temperature = 0.1 (方法描述需要高度准确)""",
        
        "user": """请基于检索到的方法学文献，描述"{topic}"的研究方法：

{retrieved_documents}

## 要求
- 方法部分长度：{word_count}字
- 确保可重复性
- 引用方法论文献""",
    },
    
    WritingScene.RESULTS.value: {
        "system": """你是一名学术结果分析助手。

## 任务
分析研究结果，与已有文献对比。

## 要求
1. 客观陈述结果，避免过度解读
2. 与检索到的文献对比
3. 使用统计术语 (显著、相关、差异)
4. 标注图表引用

## 输出格式
```markdown
## 结果

### 主要发现
...

### 与已有研究对比
...
```

## 温度设置
temperature = 0.1""",
        
        "user": """请分析以下研究结果，并与检索到的文献对比：

{retrieved_documents}

## 要求
- 结果分析长度：{word_count}字
- 客观陈述，避免夸大
- 对比已有研究""",
    },
    
    WritingScene.DISCUSSION.value: {
        "system": """你是一名学术讨论写作助手。

## 任务
撰写讨论部分，解释结果意义。

## 要求
1. 解释结果的理论和实践意义
2. 与检索到的文献对比 (一致/矛盾)
3. 承认研究局限
4. 提出未来方向
5. 避免过度推测

## 输出格式
```markdown
## 讨论

### 主要发现的意义
...

### 与已有研究的关系
...

### 研究局限
...

### 未来方向
...
```

## 温度设置
temperature = 0.3 (讨论可适当推测，但需谨慎)""",
        
        "user": """请基于检索到的文献，撰写关于以下结果的讨论：

{retrieved_documents}

## 要求
- 讨论部分长度：{word_count}字
- 与已有文献对比
- 承认局限性""",
    },
    
    WritingScene.ARGUMENT溯源.value: {
        "system": """你是一名学术论点溯源助手。

## 任务
追溯论点的原始文献来源。

## 要求
1. 找到最早提出该论点的文献
2. 追踪后续发展和引用
3. 标注关键转折点
4. 提供完整引用链

## 输出格式
```markdown
## 论点溯源：{argument}

### 首次提出
作者 (年份) - 文献标题

### 关键发展
- 作者 (年份) - 贡献
- 作者 (年份) - 贡献

### 当前共识
...

### 完整引用链
[按时间顺序列出]
```

## 温度设置
temperature = 0.1 (溯源必须准确)""",
        
        "user": """请追溯以下论点的文献来源：

论点："{argument}"

检索到的相关文献：
{retrieved_documents}

## 要求
- 找到最早提出者
- 追踪发展脉络
- 提供完整引用""",
    },
    
    WritingScene.PARAGRAPH_POLISH.value: {
        "system": """你是一名学术论文润色助手。

## 任务
润色论文段落，提升学术性和可读性。

## 要求
1. 保持原意不变
2. 提升学术语言规范性
3. 优化句子结构
4. 检查引用格式
5. 避免口语化表达

## 修改原则
- 被动语态 → 主动语态 (中文)
- 长句拆分 → 适当短句
- 模糊词 → 精确词
- 口语 → 书面语

## 输出格式
```markdown
## 原文
...

## 润色后
...

## 修改说明
1. ...
2. ...
```

## 温度设置
temperature = 0.2""",
        
        "user": """请润色以下论文段落：

{original_text}

## 要求
- 保持原意
- 提升学术性
- 检查引用格式""",
    },
    
    WritingScene.CITATION_FORMAT.value: {
        "system": """你是一名引用格式规范助手。

## 任务
检查和规范引用格式。

## 要求
1. 文内引用：(作者，年份，页码)
2. 参考文献：GB/T 7714-2015
3. 检查完整性 (作者、年份、标题、期刊、卷期页码)
4. 标注缺失信息

## 输出格式
```markdown
## 文内引用检查
- [✓] (Zhang et al., 2026) - 完整
- [✗] (Li, 2025) - 缺页码

## 参考文献列表
[1] 完整格式
[2] 完整格式

## 缺失信息
- 文献 [2]: 缺卷号
- 文献 [3]: 缺 DOI
```

## 温度设置
temperature = 0.1 (格式必须精确)""",
        
        "user": """请检查以下内容的引用格式：

{text_to_check}

检索到的文献信息：
{retrieved_documents}

## 要求
- 文内引用：(作者，年份，页码)
- 参考文献：GB/T 7714-2015
- 标注缺失信息""",
    },
}


class AcademicRAGWorkflow:
    """
    学术写作 RAG 工作流 v2
    
    特性:
    1. Query Rewriting - LLM 重写查询
    2. Metadata Filtering - 按作者/年份/期刊过滤
    3. Re-ranking - bge-reranker 重排序
    4. Scene-specific Prompts - 分场景提示词
    5. Temperature Control - 低温度降低编造
    """
    
    def __init__(self, 
                 doc_store: DocumentStore,
                 llm_client: Optional[Any] = None,
                 temperature: float = 0.2):
        self.doc_store = doc_store
        self.llm = llm_client
        self.temperature = temperature
        
        print("✓ 学术写作 RAG 工作流 v2 初始化完成")
        print(f"  - 温度设置：{temperature}")
        print(f"  - 重排序：{'启用' if doc_store.embedder.reranker else '禁用'}")
        print(f"  - 混合检索：{'启用' if doc_store.use_hybrid else '禁用'}")
    
    def rewrite_query(self, query: str, writing_scene: str) -> str:
        """
        查询重写 - 将模糊需求转为结构化查询
        
        Args:
            query: 用户原始查询
            writing_scene: 写作场景
            
        Returns:
            重写后的查询
        """
        # 使用 LLM 重写 (如果有)
        if self.llm:
            prompt = f"""请将以下学术写作查询重写为更适合检索的结构化查询。

原始查询：{query}
写作场景：{writing_scene}

要求:
1. 提取关键词
2. 添加同义词 (用 OR 连接)
3. 指定年份范围 (如 2020-2026)
4. 指定期刊级别 (如"顶刊"→Nature/Science/Cell)

示例输出:
"EphA2 OR Eph receptor A2 AND cis-interaction OR cis binding AND cancer OR metastasis AND 2020-2026"

重写后的查询:"""
            
            try:
                # 调用 LLM (这里简化，实际应调用配置的 LLM)
                rewritten = self._call_llm(prompt, temperature=0.1)
                return rewritten.strip()
            except:
                pass
        
        # 简化版重写 (无 LLM 时)
        rewritten = query
        
        # 添加同义词
        synonym_map = {
            'Eph': 'Eph OR "Eph receptor"',
            'ephrin': 'ephrin OR "Eph ligand"',
            'cis': '"cis-interaction" OR "cis binding" OR "cis signaling"',
            'trans': '"trans-activation" OR "trans signaling"',
            '癌症': 'cancer OR tumor OR carcinoma OR metastasis',
            '转移': 'metastasis OR migration OR invasion',
        }
        
        for original, synonyms in synonym_map.items():
            if original in rewritten:
                rewritten = rewritten.replace(original, synonyms)
        
        # 添加年份范围 (默认近 5 年)
        current_year = datetime.now().year
        if not re.search(r'\d{4}', rewritten):
            rewritten += f" AND {current_year-5}-{current_year}"
        
        return rewritten
    
    def retrieve_with_filter(self, 
                            query: str,
                            metadata_filter: Optional[Dict] = None,
                            n_results: int = 10) -> List[Dict]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            metadata_filter: 元数据过滤条件
                例：{'recent_years': 5, 'journal': 'Nature'}
            n_results: 返回结果数
            
        Returns:
            检索结果
        """
        # 使用 hybrid 检索 (带 rerank)
        results = self.doc_store.query_hybrid(
            query_text=query,
            n_results=n_results,
            filter_metadata=metadata_filter,
            alpha=0.7,  # dense 权重 70%
            use_reranker=True
        )
        
        return results
    
    def generate_with_scene(self,
                           scene: WritingScene,
                           query: str,
                           retrieved_docs: List[Dict],
                           word_count: int = 500,
                           topic: str = "",
                           argument: str = "",
                           original_text: str = "",
                           text_to_check: str = "") -> Dict[str, Any]:
        """
        分场景生成内容
        
        Args:
            scene: 写作场景
            query: 原始查询
            retrieved_docs: 检索到的文献
            word_count: 目标字数
            topic: 主题 (用于综述/引言)
            argument: 论点 (用于溯源)
            original_text: 原文 (用于润色)
            text_to_check: 待检查文本 (用于引用规范)
            
        Returns:
            生成的内容和引用
        """
        # 获取场景模板
        template = PROMPT_TEMPLATES.get(scene.value)
        if not template:
            return {"error": f"Unknown scene: {scene}"}
        
        # 格式化检索结果
        docs_text = self._format_retrieved_docs(retrieved_docs)
        
        # 构建用户提示
        user_prompt = template["user"].format(
            doc_count=len(retrieved_docs),
            topic=topic or query,
            argument=argument or query,
            retrieved_documents=docs_text,
            word_count=word_count,
            original_text=original_text,
            text_to_check=text_to_check
        )
        
        # 生成内容 (使用 LLM)
        if self.llm:
            system_prompt = template["system"].format(
                topic=topic or query,
                doc_count=len(retrieved_docs),
                word_count=word_count
            )
            
            try:
                content = self._call_llm(
                    system_prompt + "\n\n" + user_prompt,
                    temperature=self.temperature
                )
                
                # 提取引用
                references = self._extract_references(retrieved_docs)
                
                return {
                    "content": content,
                    "references": references,
                    "scene": scene.value,
                    "doc_count": len(retrieved_docs)
                }
            except Exception as e:
                return {"error": str(e)}
        else:
            # 无 LLM 时，返回检索结果摘要
            return {
                "content": f"检索到 {len(retrieved_docs)} 篇相关文献:\n\n{docs_text}",
                "references": self._extract_references(retrieved_docs),
                "scene": scene.value,
                "doc_count": len(retrieved_docs)
            }
    
    def _format_retrieved_docs(self, docs: List[Dict]) -> str:
        """格式化检索结果"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get('metadata', {})
            text = doc.get('text', '')[:500]  # 截取前 500 字
            
            formatted.append(f"""[文献{i}]
作者：{meta.get('authors', 'Unknown')}
年份：{meta.get('year', 'N/A')}
期刊：{meta.get('journal', 'Unknown')}
标题：{meta.get('paper_title', 'Unknown')}
内容：{text}...
---""")
        
        return "\n".join(formatted)
    
    def _extract_references(self, docs: List[Dict]) -> List[Citation]:
        """从检索结果提取引用"""
        citations = []
        for doc in docs:
            meta = doc.get('metadata', {})
            citation = Citation(
                authors=meta.get('authors', 'Unknown'),
                year=meta.get('year', 'N/A'),
                title=meta.get('paper_title', 'Unknown'),
                journal=meta.get('journal', 'Unknown'),
                volume=meta.get('volume', ''),
                issue=meta.get('issue', ''),
                pages=meta.get('pages', ''),
                doi=meta.get('doi', '')
            )
            citations.append(citation)
        return citations
    
    def _call_llm(self, prompt: str, temperature: float = 0.2) -> str:
        """调用 LLM (简化实现)"""
        if not self.llm:
            return ""
        
        # 这里应该调用配置的 LLM
        # 示例：OpenAI API
        if OPENAI_AVAILABLE and isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        
        return ""
    
    def run(self, 
            query: str,
            scene: WritingScene = WritingScene.LITERATURE_REVIEW,
            metadata_filter: Optional[Dict] = None,
            word_count: int = 500,
            topic: str = "") -> Dict[str, Any]:
        """
        完整工作流
        
        Args:
            query: 用户查询
            scene: 写作场景
            metadata_filter: 元数据过滤
            word_count: 目标字数
            topic: 主题
            
        Returns:
            生成结果
        """
        print(f"\n📝 开始学术写作工作流")
        print(f"  场景：{scene.value}")
        print(f"  查询：{query[:50]}...")
        print(f"  温度：{self.temperature}")
        
        # Step 1: 查询重写
        print("\n1️⃣ 查询重写...")
        rewritten_query = self.rewrite_query(query, scene.value)
        print(f"   重写后：{rewritten_query[:80]}...")
        
        # Step 2: 带过滤的检索
        print("\n2️⃣ 检索文献...")
        if metadata_filter:
            print(f"   过滤条件：{metadata_filter}")
        
        retrieved_docs = self.retrieve_with_filter(
            query=rewritten_query,
            metadata_filter=metadata_filter,
            n_results=10
        )
        print(f"   检索到 {len(retrieved_docs)} 篇文献")
        
        if not retrieved_docs:
            return {
                "error": "现有文献库中未找到相关内容",
                "query": query,
                "rewritten_query": rewritten_query
            }
        
        # Step 3: 分场景生成
        print("\n3️⃣ 生成内容...")
        result = self.generate_with_scene(
            scene=scene,
            query=query,
            retrieved_docs=retrieved_docs,
            word_count=word_count,
            topic=topic
        )
        
        # Step 4: 格式化输出
        print("\n4️⃣ 格式化输出...")
        output = self._format_output(result)
        
        return output
    
    def _format_output(self, result: Dict) -> str:
        """格式化输出"""
        if "error" in result:
            return f"❌ {result['error']}"
        
        output = []
        output.append(result.get('content', ''))
        
        # 添加参考文献
        refs = result.get('references', [])
        if refs:
            output.append("\n## 参考文献\n")
            for i, ref in enumerate(refs, 1):
                output.append(f"[{i}] {ref.format_reference()}")
        
        return "\n".join(output)


# ===== 使用示例 =====

if __name__ == "__main__":
    # 初始化
    doc_store = DocumentStore(
        collection_name="ephrin_papers",
        persist_directory="/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db",
        use_hybrid=True,
        use_reranker=True
    )
    
    workflow = AcademicRAGWorkflow(
        doc_store=doc_store,
        temperature=0.2
    )
    
    # 示例：文献综述
    result = workflow.run(
        query="EphA2 cis-interaction 在癌症转移中的作用",
        scene=WritingScene.LITERATURE_REVIEW,
        metadata_filter={'recent_years': 5},  # 仅近 5 年
        word_count=800,
        topic="EphA2 cis-interaction 与癌症转移"
    )
    
    print("\n" + "="*60)
    print(result)
