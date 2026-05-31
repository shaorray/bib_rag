# 📚 学术写作调用指南

## 快速调用

### 方法 1: 命令行直接查询（推荐）

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag

# 查找引用
python3 quick_query.py "cis interaction inhibits Eph signaling"

# 交互式写作辅助
python3 academic_writer.py

# 示例交互:
# > cite cis interaction inhibits signaling
# > check Eph receptor requires clustering
# > related axon guidance
# > quit
```

### 方法 2: Python API 调用

```python
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow

# 初始化
doc_store = DocumentStore('ephrin_papers', 
    '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

workflow = AgenticRAGWorkflow(retriever)

# 查询
result = workflow.run("cis interaction mechanism")
print(result['answer'])
```

---

## 📝 论文写作场景

### 场景 1: 找引用

**问题**: 写了 "Cis-interaction inhibits Eph receptor signaling"，需要引用

**调用**:
```bash
python3 academic_writer.py --cite "cis interaction inhibits Eph signaling"
```

**输出**:
```
[1] Kao and Kania (2011) - rel: 0.240
    Ephrin-Mediated cis-Attenuation of Eph Receptor Signaling...

[2] Carvalho et al. (2006) - rel: 0.193
    Silencing of EphA3 through a cis interaction...
```

**论文中使用**:
```latex
Cis-interaction inhibits Eph receptor signaling 
(Kao \& Kania, 2011; Carvalho et al., 2006).
```

---

### 场景 2: 事实核查

**问题**: 不确定 "Eph receptors require clustering for activation" 是否准确

**调用**:
```bash
python3 academic_writer.py --check "Eph receptors require clustering"
```

**输出**:
```
✓ 支持度: moderate
  建议: 该陈述有一定文献支持，建议进一步查阅
```

---

### 场景 3: Related Work

**问题**: 写 Related Work 章节，找关于 "axon guidance" 的文献

**调用**:
```bash
python3 academic_writer.py --related "axon guidance"
```

**输出**:
```
[1] Bush and Soriano (2009): Ephrin-B1 regulates axon guidance...
[2] Williams et al. (2003): Ephrin-B2 and EphB1 Mediate Retinal Axon Divergence...
```

---

### 场景 4: 检查争议

**问题**: 想讨论 "cis-interaction is always inhibitory" 是否存在争议

**调用**:
```bash
python3 academic_writer.py --controversial "cis-interaction is always inhibitory"
```

**输出**:
```
⚖️ 争议程度: moderate
  支持文献: 12
  反对文献: 5
  建议: 该论断存在一定争议，建议提及不同观点
```

**论文中使用**:
```latex
However, whether cis-interaction is exclusively inhibitory 
remains debated (cite supporting; cite opposing)...
```

---

### 场景 5: 生成段落支持

**问题**: 需要为关于 "tetramerization" 的段落找支持材料

**调用**:
```bash
python3 academic_writer.py --support "tetramerization"
```

**输出**:
```
### Tetramerization - Mechanism

[1] Falivelli et al. (2013) reported that tetrameric Eph receptor 
complexes are essential for full activation...

[支持度: 0.22]
```

---

## 🔧 集成到 LaTeX 工作流

### 方法 1: Makefile 快捷命令

在论文目录创建 `Makefile`:

```makefile
# 查找引用
cite:
	@cd /Disk_2/claw_working_dir/ephrin_agentic_rag && \
	python3 quick_query.py "$(filter-out $@,$(MAKECMDGOALS))"

# 检查段落
%:
	@:
```

使用:
```bash
make cite cis interaction
```

### 方法 2: VS Code 任务

`.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Find Citation",
      "type": "shell",
      "command": "cd /Disk_2/claw_working_dir/ephrin_agentic_rag && python3 quick_query.py '${input:claim}'",
      "problemMatcher": []
    }
  ]
}
```

### 方法 3: Emacs/Vim 快捷键

Emacs (init.el):
```elisp
(defun eph-cite (claim)
  "Find citation for claim"
  (interactive "sClaim: ")
  (shell-command
   (format "cd /Disk_2/claw_working_dir/ephrin_agentic_rag && python3 quick_query.py '%s'" claim)
   "*Eph Citations*"))

(global-set-key "\C-c\C-e" 'eph-cite)
```

---

## 📊 批量写作辅助

创建 `write_helper.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from academic_writer import AcademicWritingAssistant

assistant = AcademicWritingAssistant()

# 论文中需要核查的论断
claims = [
    "Eph receptors require clustering for activation",
    "Cis-interaction is inhibitory",
    "Reverse signaling occurs through ephrinB",
    "Eph-ephrin signaling maintains tissue boundaries",
]

print("=== 论文论断核查报告 ===\n")

for claim in claims:
    result = assistant.fact_check(claim)
    print(f"✓ {claim}")
    print(f"  支持度: {result['support_level']}")
    print(f"  置信度: {result['confidence']:.2f}\n")
```

运行:
```bash
python3 write_helper.py
```

---

## 💡 使用技巧

### 1. 相关性阈值

| 阈值 | 说明 | 建议 |
|------|------|------|
| > 0.25 | 高度相关 | 可直接引用 |
| 0.15-0.25 | 中等相关 | 结合语境使用 |
| < 0.15 | 弱相关 | 建议修改查询词 |

### 2. 查询词优化

❌ 不好: `"interaction"`  
✅ 更好: `"Eph ephrin cis interaction"`

❌ 不好: `"signaling"`  
✅ 更好: `"reverse signaling ephrinB"`

### 3. 组合查询

对于复杂论断，使用多跳推理:
```bash
python3 query_interface.py --multihop "Compare cis and trans signaling"
```

---

## 📚 知识库覆盖

| 主题 | 覆盖度 |
|------|--------|
| Cis-interaction | 8 篇 |
| Reverse signaling | 12 篇 |
| Tetramerization | 5 篇 |
| Axon guidance | 16 篇 |
| Cancer/Metastasis | 24 篇 |
| Cell segregation | 14 篇 |

**总计**: 199 篇论文，388 个文档块

---

## 🐛 故障排除

**问题**: "No documents found"
**解决**: 使用更具体的专业术语，如 `"EphB4 ephrinB2"` 而非 `"protein interaction"`

**问题**: 置信度太低
**解决**: 
1. 检查查询是否包含专业术语
2. 尝试同义词，如 `"cis"` → `"cis interaction"` → `"cis attenuation"`
3. 使用多跳模式分解复杂查询

---

## 🎯 最佳实践

1. **边写边查**: 每写一个论断就查一次引用
2. **事实核查**: 不确定的陈述先用 `--check` 验证
3. **记录引用**: 用 `export` 保存找到的引用
4. **争议标注**: 有争议的论断用 `--controversial` 标记
5. **Related Work**: 定期用 `--related` 更新文献综述

---

## 📖 完整文档

- `README.md` - 系统架构说明
- `USAGE.md` - 详细使用说明
- `ACADEMIC_USAGE.md` - 学术写作专用指南
- `QUICK_START.md` - 快速开始
