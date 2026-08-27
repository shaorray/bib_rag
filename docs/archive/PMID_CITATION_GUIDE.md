# PMID引用功能使用指南

## 概述

现在V2和V3知识库都支持在查询结果中自动添加PMID引用，方便学术写作时直接引用参考文献。

---

## V3 知识库 - 带PMID引用

### 文件
- `query_v3_kb.py` - 已更新支持PMID引用

### 基本查询（带引用）

```bash
# 普通查询 - 显示PMID引用
python3 query_v3_kb.py "cis interaction" -n 3

# 输出格式:
# [1] [PMID:27820703]
#     Score: 0.8437
#     PMID: 27820703 | Year: 2019 | IF: 5.99
#     Text: EphA4 expression is increased in the injured cortex...
```

### 生成学术段落（带引用）

```bash
# 使用 --paragraph 参数生成带引用的段落
python3 query_v3_kb.py "Eph receptor signaling" -n 3 --paragraph

# 输出格式:
# ============================================================
# 📝 学术写作格式（带PMID引用）
# ============================================================
# 
# Eph receptors and ephrins function as classic receptors and ligands 
# in ephrin:Eph forward signaling[PMID:30819650]. However, the roles 
# of Eph and ephrin proteins can...[PMID:31406248]
#
# 📚 参考文献:
#    [1] PMID:30819650, Year:2019, Journal:Trends in Molecular Medicine, IF:6.51
#    [2] PMID:31406248, Year:2019, Journal:Oncogene, IF:5.58
#
# 🔗 PMID列表: 30819650, 31406248
```

### Python API 使用

```python
from query_v3_kb import V3KnowledgeBase

# 初始化知识库
kb = V3KnowledgeBase()

# 查询并获取带引用的结果
results = kb.query_with_citations("cis interaction", n_results=5)

for r in results:
    print(f"{r['text'][:100]}...{r['citation']}")
    # 输出: "EphA4 expression is increased...[PMID:27820703]"

# 生成学术段落
paragraph = kb.generate_paragraph("Eph signaling", n_results=3)
print(paragraph['paragraph'])
# 输出: "Eph receptors function as...[PMID:30819650]..."

print(paragraph['pmids'])
# 输出: ['30819650', '31406248']

print(paragraph['references'])
# 输出: ['PMID:30819650, Year:2019, Journal:Trends in Molecular Medicine, IF:6.51', ...]
```

---

## V2 知识库 - 带PMID引用

### 文件
- `query_v2_kb_with_citations.py` - 新增PMID引用支持

### 基本查询

```bash
# 运行查询
python3 query_v2_kb_with_citations.py

# 输出每个结果都会显示 [PMID:xxxx]
```

### Python API 使用

```python
from query_v2_kb_with_citations import query_with_citations, generate_paragraph_with_citations
from process_v2_papers import PaperProcessor

# 初始化
processor = PaperProcessor()
# 手动加载V2知识库
processor.doc_store.db_path = Path('/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2')
processor.doc_store.name = 'ephrin_papers_v2'
processor.doc_store._load()

# 查询
results = query_with_citations(processor, "cis interaction", n_results=5)
for r in results:
    print(f"{r['text'][:100]}...{r['citation']}")

# 生成段落
paragraph = generate_paragraph_with_citations("Eph signaling", n_results=3)
print(paragraph['paragraph'])
```

---

## 学术写作中的应用

### 直接复制到论文

查询结果可以直接复制到论文中：

```
Eph受体与ephrin配体的相互作用在神经发育中起关键作用[PMID:27820703][PMID:38649412]。
研究表明，这种相互作用可以调节受体信号传导的强度和持续时间[PMID:11053419]。
```

### 生成参考文献列表

使用PMID列表自动生成参考文献：

```python
pmids = ['27820703', '38649412', '11053419']
references = get_reference_list(pmids)
for ref in references:
    print(ref)
    
# 输出:
# [1] PMID:27820703, Year:2019, Journal:Journal of Neuroscience, IF:5.99
# [2] PMID:38649412, Year:2024, Journal:Nature Microbiology, IF:13.94
# [3] PMID:11053419, Year:2000, Journal:Journal of Biological Chemistry, IF:4.01
```

---

## 技术实现

### V3实现细节

```python
def query_with_citations(self, query_text, n_results=5, **filters):
    results = self.query(query_text, n_results=n_results, **filters)
    
    cited_results = []
    for r in results:
        pmid = r.get('pmid', '')
        citation = f"[PMID:{pmid}]" if pmid else ""
        
        cited_results.append({
            **r,
            'citation': citation,
            'full_reference': f"PMID:{pmid}, Year:{year}, Journal:{journal}, IF:{if_value}"
        })
    
    return cited_results
```

### V2实现细节

```python
def query_with_citations(processor, query_text, n_results=5):
    results = processor.query(query_text, n_results=n_results)
    
    cited_results = []
    for r in results:
        meta = r['metadata']
        pmid = meta.get('pmid', '')
        citation = f"[PMID:{pmid}]" if pmid else ""
        
        cited_results.append({
            'text': r['text'],
            'pmid': pmid,
            'citation': citation,
            # ...
        })
    
    return cited_results
```

---

## 注意事项

1. **PMID去重**: 生成段落时会自动去重，同一PMID不会重复引用
2. **引用格式**: 默认使用 `[PMID:xxxx]` 格式，可根据期刊要求调整
3. **完整性**: 所有结果都包含PMID、Year、Journal、IF等元数据
4. **过滤支持**: 可在生成段落时应用年份/期刊/IF等过滤条件

---

## 示例对比

### V3查询效果
```
查询: "Eph receptor signaling"
结果:
  [PMID:30819650] Score:0.8202, Trends in Molecular Medicine, IF:6.51
  [PMID:31406248] Score:0.8081, Oncogene, IF:5.58
  [PMID:31406248] Score:0.7716, Oncogene, IF:5.58
```

### V2查询效果
```
查询: "Eph receptor signaling"
结果:
  [PMID:15537545] Score:0.1821, Cell, IF:33.6
  [PMID:18448254] Score:0.1814, Pain, IF:3.77
  [PMID:31689239] Score:0.1766, JCI, IF:7.98
```

**结论**: V3语义理解更好，分数更高；V2也能返回带PMID的结果，但相关性较弱。
