# Agentic RAG 知识库改进方案

## 当前问题分析 (2026-04-30)

### 1. 重复内容问题 ⚠️
**现状**: 4844 个文档块中，有 1055 个完全重复
- 元数据前缀重复（PMID/Year/Journal）存储在每个块中
- 同一篇文献的多个块共享相同前缀
- **浪费**: 约 20% 存储空间

**解决方案**: 
- 元数据与文本分离存储
- 只在检索时拼接元数据

### 2. 空章节问题 ⚠️⚠️
**现状**: 大量章节标题后没有内容

| 章节类型 | 空比例 | 影响 |
|----------|--------|------|
| RESULTS | 78.3% | 严重 |
| Methods | 73.3% | 严重 |
| Figure Legends | 92.9% | 中等 |
| Supplementary | 90.0% | 中等 |

**根本原因**: Markdown 格式不一致
- 有些文件用 `## RESULTS` 作为标题但内容在下一级
- 有些文件内容直接跟在标题后没有空行
- 子章节（如 `### Results part 1`）被当作独立章节

**解决方案**:
- 改进章节提取逻辑
- 合并子章节到父章节
- 过滤空内容章节

### 3. 分块粒度问题
**现状**: 
- 平均 461 词/块（范围 9-1514）
- 重叠 100 词可能不够
- 有些块太短（<20词的有626个）

**解决方案**:
- 设置最小块大小（100词）
- 设置最大块大小（1000词）
- 增加重叠到 200 词

### 4. 内容质量问题
**现状**:
- 包含 "Figure 1.", "Table 2." 等无意义块
- 包含 "Acknowledgments", "References" 等非学术内容
- 包含 "Supplementary Material" 等辅助内容

**解决方案**:
- 过滤非学术章节
- 只保留核心内容（摘要、引言、结果、讨论、方法）

---

## 改进方案

### 方案 A: 修复当前知识库（推荐）

1. **重新分块**
   - 过滤空章节
   - 合并子章节
   - 设置最小/最大块大小

2. **去重**
   - 移除完全重复的块
   - 元数据与内容分离

3. **过滤**
   - 移除 Figure/Table 块
   - 移除 Acknowledgments/References
   - 只保留核心学术内容

### 方案 B: 增量优化

1. **添加引用网络**
   - 提取每篇文献的引用关系
   - 构建引用图谱
   - 支持基于引用的检索

2. **添加实体标签**
   - 用 NER 提取蛋白质、基因、细胞类型
   - 添加实体标签到文档块
   - 支持实体检索

3. **添加摘要向量**
   - 为每篇文献生成摘要向量
   - 先检索文献级别，再检索块级别
   - 提高检索效率

### 方案 C: 多知识库融合

1. **层次化检索**
   - 第一层：文献级别（基于标题/摘要）
   - 第二层：章节级别（基于章节内容）
   - 第三层：段落级别（基于具体文本）

2. **跨知识库查询**
   - v1 (199篇经典文献)
   - v2 (500篇新文献)
   - Graphify 知识图谱
   - 自动选择最优来源

---

## 具体实施步骤

### Step 1: 修复分块逻辑

```python
# 改进后的分块策略
def create_chunks_v3(text, sections, meta):
    """
    改进的分块策略：
    1. 过滤空章节（<50词）
    2. 合并子章节到父章节
    3. 设置最小块 100 词，最大块 800 词
    4. 重叠 200 词
    5. 过滤非学术章节
    """
    
    # 要过滤的章节
    skip_sections = {
        'references', 'acknowledgments', 'acknowledgements',
        'figure legends', 'tables', 'supplementary material',
        'supplementary information', 'competing interests',
        'consent for publication', 'peer review', 'footnotes',
        'figure 1.', 'figure 2.', 'figure 3.', 'figure 4.',
        'table 1.', 'table 2.', 'abbreviations', 'keywords',
        'graphical abstract', 'author contributions',
        'funding', 'ethics approval', 'data availability',
    }
    
    chunks = []
    chunk_size = 800
    overlap = 200
    min_chunk_size = 100
    
    # 元数据前缀（只存一次）
    meta_dict = {
        'pmid': meta.get('pmid', ''),
        'year': meta.get('year', ''),
        'journal': meta.get('journal', ''),
        'if': meta.get('impact_factor', ''),
        'citations': meta.get('citations', ''),
        'tier': meta.get('tier', ''),
    }
    
    # 处理核心章节
    core_sections = ['abstract', 'introduction', 'background', 
                     'results', 'discussion', 'methods', 'conclusion']
    
    for section_key in core_sections:
        if section_key not in sections:
            continue
        
        section_text = sections[section_key].strip()
        section_words = section_text.split()
        
        # 过滤空章节
        if len(section_words) < min_chunk_size:
            continue
        
        # 过滤非学术内容
        if any(skip in section_key.lower() for skip in skip_sections):
            continue
        
        # 分段
        step = chunk_size - overlap
        for i in range(0, len(section_words), step):
            chunk_words = section_words[i:i + chunk_size]
            
            # 确保最后一块不会太小
            if len(chunk_words) < min_chunk_size and i > 0:
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            # 去重检查
            chunk_hash = hash(chunk_text[:200])
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            
            chunks.append({
                'text': chunk_text,
                'meta': meta_dict,
                'section': section_key,
            })
    
    return chunks
```

### Step 2: 添加引用网络

```python
# 提取引用关系
def extract_citations(text):
    """提取文献中的引用"""
    # 匹配 (Author, Year) 格式
    citations = re.findall(r'\(([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}[a-z]?)\)', text)
    return citations

# 构建引用图
citation_graph = {}
for doc in documents:
    pmid = doc['pmid']
    citations = extract_citations(doc['text'])
    citation_graph[pmid] = citations
```

### Step 3: 添加实体标签

```python
# 使用正则表达式提取实体
Eph_PATTERN = re.compile(r'\b(Eph[A-Z]\d?|ephrin-[AB]\d?)\b', re.IGNORECASE)
CELL_PATTERN = re.compile(r'\b(neuron|astrocyte|oligodendrocyte|microglia|HEK293|COS7)\b', re.IGNORECASE)

def extract_entities(text):
    entities = {
        'proteins': Eph_PATTERN.findall(text),
        'cells': CELL_PATTERN.findall(text),
    }
    return entities
```

---

## 预期效果

| 指标 | 当前 | 改进后 |
|------|------|--------|
| 总块数 | 4844 | ~3500 (去重+过滤) |
| 重复块 | 1055 (22%) | <50 (1%) |
| 平均块大小 | 461 词 | ~600 词 |
| 空/短块 | 626 (13%) | <100 (2%) |
| 检索质量 | 中等 | 高 |

---

## 长期改进方向

1. **多模态检索**
   - 添加图片/表格描述
   - 支持图表检索

2. **时序分析**
   - 按年份分析研究趋势
   - 发现新兴研究方向

3. **引文网络分析**
   - 发现高影响力论文
   - 识别研究社区

4. **自动摘要**
   - 为每篇文献生成摘要
   - 支持快速浏览
