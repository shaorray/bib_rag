# 📝 学术论文写作 - Agentic RAG 调用指南

## 快速开始

```python
# 在 Python 中导入
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

# 初始化助手
assistant = AcademicWritingAssistant()
```

---

## 核心功能

### 1. 查找引用 (Find References)

**场景**: 写论文时，需要为某个论断找到支持文献

```python
# 方法 1: Python 代码
citations = assistant.find_references(
    claim="cis interaction inhibits Eph receptor signaling",
    min_relevance=0.15
)

for cit in citations:
    print(f"{cit.authors} ({cit.year}). {cit.title}")
```

**命令行**:
```bash
python3 academic_writer.py --cite "cis interaction inhibits Eph receptor signaling"
```

**输出示例**:
```
[1] Kao and Kania (2011) - rel: 0.240
    Ephrin-Mediated cis-Attenuation of Eph Receptor Signaling Is Essential...

[2] Carvalho et al. (2006) - rel: 0.193
    Silencing of EphA3 through a cis interaction with ephrinA5...
```

---

### 2. 事实核查 (Fact Check)

**场景**: 写完一段内容，想确认表述是否准确

```python
result = assistant.fact_check(
    "Eph receptors require clustering for full activation"
)

print(f"支持度: {result['support_level']}")  # strong/moderate/weak
print(f"建议: {result['suggestion']}")
```

**命令行**:
```bash
python3 academic_writer.py --check "Eph receptors require clustering"
```

**输出**:
```
✓ 支持度: moderate
  置信度: 0.23
  建议: 该陈述有一定文献支持，建议进一步查阅相关文献或调整表述。
```

---

### 3. 查找相关研究 (Related Work)

**场景**: 写 Related Work 章节，需要找类似研究的论文

```python
papers = assistant.find_related_work(
    topic="axon guidance",
    n_papers=10
)

for paper in papers:
    print(f"{paper['authors']} ({paper['year']}): {paper['title']}")
```

**命令行**:
```bash
python3 academic_writer.py --related "axon guidance"
```

---

### 4. 生成段落支持材料

**场景**: 正在写一个段落，需要找到支持该主题的文献

```python
material = assistant.generate_paragraph_support(
    topic="cis-interaction",
    aspect="mechanism"  # 可选: mechanism, function, controversy, evidence
)
print(material)
```

**输出**:
```
### Cis-interaction - Mechanism

[1] Kao and Kania (2011) reported that Ephrin-mediated cis-attenuation 
of Eph receptor signaling is essential for cell sorting...

[2] Carvalho et al. (2006) reported that silencing of EphA3 through 
a cis interaction with ephrinA5...

[支持度: 0.25]
```

---

### 5. 检查争议性

**场景**: 想确认某个论断是否 controversial，避免踩坑

```python
result = assistant.check_controversial_claim(
    "Eph-ephrin cis-interaction is inhibitory"
)

print(f"争议程度: {result['controversy_level']}")  # high/moderate/low
print(f"建议: {result['advice']}")
```

**命令行**:
```bash
python3 academic_writer.py --controversial "cis-interaction is inhibitory"
```

---

## 交互式写作模式

最方便的方式是使用交互模式：

```bash
python3 academic_writer.py
```

然后输入命令：

```
> cite cis interaction inhibits signaling
> check EphB4 promotes tumor growth
> related boundary formation
> support tetramerization controversy
> controversial forward signaling dominates
> export
> quit
```

---

## 实际写作场景示例

### 场景 1: 写 Introduction 时需要引用

```python
# 正在写: "Previous studies have shown that cis-interactions..."

# 查找支持
assistant = AcademicWritingAssistant()
citations = assistant.find_references("cis interaction inhibits Eph signaling")

# 格式化引用
citation_text = assistant.suggest_citation_style(citations, "APA")
print(citation_text)
```

**输出**:
```
Kao and Kania (2011). Ephrin-Mediated cis-Attenuation of Eph Receptor 
Signaling Is Essential...

Carvalho et al. (2006). Silencing of EphA3 through a cis interaction 
with ephrinA5...
```

**插入论文**:
```latex
Previous studies have shown that cis-interactions between Eph receptors 
and their ligands can attenuate receptor signaling (Kao \& Kania, 2011; 
Carvalho et al., 2006), suggesting a regulatory mechanism...
```

---

### 场景 2: 写 Discussion 时想提争议

```python
# 想讨论: "However, the role of cis-interaction remains debated"

result = assistant.check_controversial_claim(
    "cis-interaction is always inhibitory"
)

if result['controversy_level'] in ['high', 'moderate']:
    print("可以写争议段落！")
    print(f"支持文献: {result['supporting_papers']}")
    print(f"反对文献: {result['opposing_papers']}")
```

**论文内容**:
```latex
However, the role of cis-interaction in Eph signaling remains debated. 
While some studies suggest that cis-interactions primarily serve an 
inhibitory function (cite supporting papers), others have reported 
context-dependent effects (cite opposing papers). This discrepancy 
may reflect differences in...
```

---

### 场景 3: 写 Related Work 章节

```python
# 查找某个主题的所有相关工作
papers = assistant.find_related_work("tetramerization", n_papers=15)

# 按年份分组
by_year = {}
for p in papers:
    year = p['year']
    if year not in by_year:
        by_year[year] = []
    by_year[year].append(p)

# 按时间顺序输出
for year in sorted(by_year.keys()):
    print(f"\n{year}:")
    for p in by_year[year]:
        print(f"  - {p['authors']}: {p['title']}")
```

---

### 场景 4: 实时写作辅助

```python
# 边写边查
assistant = AcademicWritingAssistant()

paragraph = """
Eph receptor signaling requires receptor clustering for full activation.
"""

# 核查这个陈述
result = assistant.fact_check(paragraph)

if result['support_level'] == 'weak':
    print("⚠️  这个陈述可能缺乏文献支持")
    print(result['suggestion'])
    # 可能需要修改表述或添加文献
```

---

## 在 LaTeX/Word 中集成

### 方法 1: 命令行快速查询

在写作过程中打开终端：
```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag
python3 academic_writer.py --cite "your claim here"
```

复制引用到论文。

### 方法 2: 脚本批量处理

创建一个 `check_paragraphs.py`:
```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

assistant = AcademicWritingAssistant()

paragraphs = [
    "Eph receptors require clustering for activation",
    "Cis-interaction is always inhibitory",
    "Reverse signaling occurs through ephrinB",
]

for p in paragraphs:
    result = assistant.fact_check(p)
    print(f"\n[{p}]")
    print(f"  支持度: {result['support_level']}")
```

### 方法 3: Jupyter Notebook

```python
# 在 notebook 中
%run /Disk_2/claw_working_dir/ephrin_agentic_rag/academic_writer.py
assistant = AcademicWritingAssistant()

# 然后随时查询
assistant.find_references("your query")
```

---

## 引用格式

当前支持:
- **APA**: `Author (Year). Title.`
- **Vancouver**: `Author. Title. Year;`

```python
formatted = assistant.suggest_citation_style(citations, style="APA")
```

---

## 性能提示

1. **相关性阈值**: 默认 0.15，如果需要更严格的结果，设为 0.20+
2. **返回数量**: 默认 10 篇，可以增加或减少
3. **多跳查询**: 复杂比较用 `--multihop` 或 `MultiHopRAG`

---

## 完整示例脚本

```python
#!/usr/bin/env python3
"""Example: Writing a paragraph with academic support"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

def write_paragraph_with_support(topic, claim):
    """写一段带文献支持的段落"""
    
    assistant = AcademicWritingAssistant()
    
    # 1. 查找引用
    citations = assistant.find_references(claim, min_relevance=0.20)
    
    # 2. 格式化引用
    refs = "; ".join([f"{c.authors} et al., {c.year}" for c in citations[:3]])
    
    # 3. 生成段落
    paragraph = f"""
{topic} plays a critical role in Eph-ephrin signaling. 
{claim} ({refs}). 
This mechanism has been implicated in various developmental processes...
"""
    
    print(paragraph)
    
    # 4. 事实核查
    check = assistant.fact_check(claim)
    print(f"\n[支持度: {check['support_level']}, 置信度: {check['confidence']:.2f}]")

# 使用
write_paragraph_with_support(
    topic="Cis-interaction",
    claim="Cis-interaction between Eph receptors and ephrins can attenuate receptor signaling"
)
```

---

## 常见问题

**Q: 知识库包含哪些论文？**
A: 199 篇 Eph/Ephrin 相关论文，涵盖 1986-2025 年。

**Q: 相关性分数是什么意思？**
A: 0.15-0.25: 中等相关；0.25+: 高度相关；0.15以下: 弱相关

**Q: 如何更新知识库？**
A: 运行 `python3 build_knowledge_base.py` 重建（如果添加了新论文）

**Q: 支持非 Eph/Ephrin 主题吗？**
A: 当前仅支持知识库中的论文，但可以查询任何内容（只是可能找不到相关文献）
