# Agentic RAG 快速开始

## 知识库状态

✅ **已构建完成** - 199 篇 Eph/Ephrin 论文，388 个文档块

## 立即使用

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag

# 交互式查询
python3 query_interface.py

# 单查询示例
python3 query_interface.py -q "cis interaction Eph"

# 多跳推理
python3 query_interface.py --multihop -q "Compare forward and reverse signaling"
```

## 示例查询

```
❓ Query: Eph receptor evolution

📋 Answer:
[1] # EPH RECEPTOR SIGNALLING CASTS
   Source: Pasquale 2005

[2] # Eph receptor function is modulated by heterooligomerization
   Source: Janes et al. 2011
...

📊 Confidence: 0.32
🔍 Retrieved: 8 docs
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `build_knowledge_base.py` | 重建知识库 |
| `query_interface.py` | 交互式查询 |
| `rag_core.py` | 核心 RAG 组件 |
| `agentic_workflow.py` | Agentic RAG 逻辑 |
| `chroma_db/ephrin_papers.pkl` | 向量数据库 (3MB) |

## 特性

- ✅ Self-RAG: 自我评估相关性
- ✅ CRAG: 查询重写与重试
- ✅ Multi-hop: 复杂查询分解
- ✅ 持久化存储
- 📦 199 篇论文 / 388 文档块
