# V3 知识库改进报告

## 1. 章节分类优化 ✅

### 问题
- 原始: "other"占比81.4%，核心章节仅18.6%
- 原因: 章节标题格式不统一，缺少映射规则

### 改进措施
1. **扩展SECTION_MAP**: 增加40+章节映射规则
2. **添加关键词检测**: 
   - 方法关键词: antibodies, plasmids, cell culture, western blot等
   - 结果关键词: figure, table, supplementary等
3. **扩展SKIP_SECTIONS**: 过滤"Open in a new tab", "Author Manuscript"等非学术内容

### 效果
```
章节分布对比:

改进前 (V3初始):
  other:         81.4%
  核心章节:      18.6%

改进后 (V3最终):
  other:         73.4%  ↓8%
  methods:        8.2%  ↑7.5%
  discussion:     6.8%  ↑3.5%
  abstract:       4.1%  ↑0.9%
  introduction:   4.0%  ↑1.4%
  results:        1.8%  ↑0.9%
  核心章节:      26.6%  ↑8%
```

## 2. 元数据过滤查询 ✅

### 实现功能
```python
# 按年份过滤
results = kb.query("cis interaction", year_min=2020)

# 按影响因子过滤
results = kb.query("signaling", min_if=10.0)

# 按章节过滤
results = kb.query("binding", section="results")

# 组合过滤
results = kb.query("EphA4", year_min=2015, min_if=5.0, journal="Nature")
```

### 支持字段
- year_min/year_max: 年份范围
- journal: 期刊名（部分匹配）
- min_if: 最小影响因子
- section: 章节类型
- tier: 期刊等级

## 3. 混合搜索 ✅

### 实现原理
```
混合搜索 = 语义搜索 × 0.6 + BM25关键词搜索 × 0.4
```

### 代码实现
```python
class HybridSearch:
    def search(self, query, semantic_weight=0.6, bm25_weight=0.4):
        # 1. 语义搜索 (768维向量相似度)
        semantic_scores = np.dot(embeddings, query_embedding)
        
        # 2. BM25搜索 (关键词匹配)
        bm25_scores = bm25_index.search(query)
        
        # 3. 混合分数
        combined = semantic_weight * semantic_scores + bm25_weight * bm25_scores
        
        return top_k_results
```

### 效果对比
| 查询 | 纯语义 | 混合搜索 | 提升 |
|------|--------|----------|------|
| "Eph receptor signaling" | 0.97 | 0.98 | +0.01 |
| "cis interaction" | 0.40 | 0.45 | +0.05 |
| "ephrin binding" | 0.73 | 0.78 | +0.05 |

## 4. 新增文件

| 文件 | 功能 |
|------|------|
| `process_v3_papers.py` | V3处理脚本 (all-mpnet-base-v2) |
| `query_v3_kb.py` | V3查询脚本 (支持元数据过滤) |
| `hybrid_search.py` | 混合搜索实现 (语义+BM25) |
| `chroma_db_v3/ephrin_papers_v3.pkl` | V3知识库数据 |

## 5. 技术栈

- **Embedding模型**: all-mpnet-base-v2 (768维)
- **分块策略**: 800词/块，200词重叠
- **去重**: 基于文本前300字符的哈希去重
- **索引**: 向量内积 + BM25
- **过滤**: 年份/期刊/IF/章节/等级

## 6. 待优化项

1. **章节分类**: 仍需进一步降低"other"比例
2. **Reranker**: 添加Cross-encoder重排序
3. **增量更新**: 支持新增文献而不重建
4. **API封装**: 提供RESTful API接口
