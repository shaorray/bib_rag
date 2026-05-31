# Production Agentic RAG Course 安装报告

**安装时间**: 2026-03-28  
**来源**: GitHub - jamwithai/production-agentic-rag-course  
**Stars**: 5k+ | **Forks**: 1.2k+  
**状态**: ✅ 已下载，待安装

---

## 📊 项目信息

| 指标 | 值 |
|------|-----|
| **仓库** | jamwithai/production-agentic-rag-course |
| **Stars** | 5,000+ |
| **Forks** | 1,200+ |
| **语言** | Python 3.12+ |
| **License** | MIT |
| **最后更新** | 3 weeks ago |
| **状态** | Week 7 (Agentic RAG + Telegram) |

---

## ️ 项目架构

### 7 周完整课程

| 周次 | 主题 | 核心内容 |
|------|------|---------|
| **Week 1** | 基础设施 | Docker, FastAPI, PostgreSQL, OpenSearch, Airflow |
| **Week 2** | 数据管道 | arXiv API, PDF 解析，自动化 ingestion |
| **Week 3** | BM25 搜索 | OpenSearch, 关键词检索，相关性评分 |
| **Week 4** | 混合搜索 | 智能分块，向量 + 关键词 RRF 融合 |
| **Week 5** | 完整 RAG | 本地 LLM (Ollama), 流式响应，Gradio 界面 |
| **Week 6** | 生产监控 | Langfuse tracing, Redis 缓存 |
| **Week 7** | **Agentic RAG** | **LangGraph, Telegram Bot, 自适应检索** |

---

## 📁 已下载文件结构

```
/Disk_2/claw_working_dir/production-agentic-rag-course/
├── src/                    # 源代码
│   ├── routers/           # API 路由
│   ├── services/          # 服务层
│   ├── models/            # 数据模型
│   └── ...
├── notebooks/             # Jupyter 笔记本 (9 个)
├── airflow/               # Airflow DAGs
├── tests/                 # 测试用例
├── static/                # 架构图和静态资源
├── compose.yml            # Docker Compose 配置
├── pyproject.toml         # Python 项目配置
├── .env.example           # 环境变量模板
└── README.md              # 详细文档 (24KB)
```

**总计**: 520KB 源代码 + 文档

---

## 🔧 安装步骤

### 前置要求

- ✅ Docker Desktop (with Docker Compose)
- ✅ Python 3.12+
- ⚠️ UV Package Manager (需安装)
- ⚠️ 8GB+ RAM, 20GB+ 磁盘空间

### 步骤 1: 安装 UV (如果未安装)

```bash
# 方法 1: 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法 2: 使用 pip
pip install uv

# 验证安装
uv --version
```

### 步骤 2: 配置环境

```bash
cd /Disk_2/claw_working_dir/production-agentic-rag-course

# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加必要的 API 密钥
# - Jina Embeddings API key (免费)
# - Langfuse API key (可选，用于监控)
```

### 步骤 3: 安装依赖

```bash
# 使用 UV 安装依赖 (推荐)
uv sync

# 或使用 pip
pip install -e .
```

### 步骤 4: 启动所有服务

```bash
# 使用 Docker Compose 启动
docker compose up --build -d

# 查看日志
docker compose logs -f

# 检查健康状态
curl http://localhost:8000/api/v1/health
```

### 步骤 5: 访问服务

| 服务 | URL | 说明 |
|------|-----|------|
| **API 文档** | http://localhost:8000/docs | FastAPI 交互式文档 |
| **Gradio RAG** | http://localhost:7861 | 聊天界面 |
| **Langfuse** | http://localhost:3000 | 监控仪表板 |
| **Airflow** | http://localhost:8080 | 工作流管理 |
| **OpenSearch** | http://localhost:5601 | 搜索仪表板 |

---

## 🎯 核心特性

### Week 7: Agentic RAG (与我们当前的实现对比)

| 特性 | 我们的实现 | 此项目 | 差异 |
|------|-----------|--------|------|
| **Planner Agent** | ✅ | ✅ (LangGraph) | 相同 |
| **Reflector Agent** | ✅ | ✅ (Document Grader) | 相同 |
| **自适应检索** | ⚠️ 基础 | ✅ 完整实现 | 需学习 |
| **查询重写** | ⚠️ 基础 | ✅ 完整实现 | 需学习 |
| **Guardrails** | ⚠️ 基础 | ✅ Out-of-domain 检测 | 需学习 |
| **Telegram Bot** | ❌ | ✅ | 可选 |
| **监控** | ⚠️ 基础 | ✅ Langfuse | 需学习 |
| **缓存** | ✅ 内存 | ✅ Redis | 可升级 |

### 值得学习的特性

1. **LangGraph 工作流** - 更优雅的状态机实现
2. **Document Grading** - 语义相关性评估 (0-1 分)
3. **Query Rewriting** - 查询改写策略
4. **Out-of-Domain Detection** - 防止幻觉的防护栏
5. **Step Tracking** - 完整的推理步骤追踪
6. **Redis 缓存** - 生产级缓存策略
7. **Langfuse 监控** - 完整的可观测性

---

## 📚 学习路径建议

### 第一阶段：基础设施 (Week 1-2)

```bash
# 学习 Docker Compose 配置
cat compose.yml

# 学习 FastAPI 项目结构
ls -la src/routers/
ls -la src/services/

# 运行 Week 1 笔记本
cd notebooks/week1/
uv run jupyter notebook week1_setup.ipynb
```

### 第二阶段：搜索基础 (Week 3-4)

```bash
# 学习 OpenSearch 配置
cat src/services/opensearch/*.py

# 学习 BM25 实现
cat src/routers/search.py

# 学习混合搜索
cat src/routers/hybrid_search.py
```

### 第三阶段：完整 RAG (Week 5-6)

```bash
# 学习 RAG 实现
cat src/routers/ask.py
cat src/services/ollama/prompts/rag_system.txt

# 学习监控和缓存
cat src/services/redis_cache.py
```

### 第四阶段：Agentic RAG (Week 7)

```bash
# 学习 LangGraph 实现
cat src/services/langgraph/*.py

# 学习 Agent 决策逻辑
cat src/routers/agentic_rag.py
```

---

## 🔄 与我们当前实现的整合

### 可以整合的部分

1. **LangGraph 工作流** → 替换我们当前的 StateGraph
2. **Document Grader** → 增强我们的 Reflector Agent
3. **Query Rewriter** → 增强我们的 Planner Agent
4. **Redis 缓存** → 替换我们的内存缓存
5. **Langfuse 监控** → 添加生产监控

### 保持当前实现的部分

1. **Self-RAG 评估** - 已经实现且测试通过
2. **黄金参数** - 已经优化 (0.75/0.8/3/10)
3. **Ollama Cloud** - 使用云端模型而非本地

---

## 📋 下一步行动

### 选项 A: 完整安装 (推荐)

```bash
# 1. 安装 UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 配置环境
cd /Disk_2/claw_working_dir/production-agentic-rag-course
cp .env.example .env
# 编辑 .env 添加 API 密钥

# 3. 安装依赖
uv sync

# 4. 启动服务
docker compose up --build -d

# 5. 测试
curl http://localhost:8000/api/v1/health
```

### 选项 B: 学习特定周次

```bash
# 只克隆 Week 7 (Agentic RAG)
git clone --branch week7.0 https://github.com/jamwithai/arxiv-paper-curator
cd arxiv-paper-curator
uv sync
```

### 选项 C: 代码审查

```bash
# 查看核心代码
cat src/routers/agentic_rag.py
cat src/services/langgraph/graph.py
```

---

## 🎯 对比总结

| 维度 | 我们的实现 | 此项目 | 建议 |
|------|-----------|--------|------|
| **架构** | Planner→Reflector | LangGraph 状态机 | 学习 LangGraph |
| **Agent** | 自研 | LangGraph 内置 | 可整合 |
| **缓存** | 内存 LRU | Redis 生产级 | 可升级 |
| **监控** | 基础统计 | Langfuse 完整 | 建议添加 |
| **部署** | Python 脚本 | Docker Compose | 学习部署 |
| **文档** | 中文 | 英文 + 博客 | 互补 |
| **课程** | 无 | 7 周完整 | 值得学习 |

---

## 📞 资源链接

- **GitHub**: https://github.com/jamwithai/production-agentic-rag-course
- **Week 7 博客**: https://jamwithai.substack.com/p/agentic-rag-with-langgraph-and-telegram
- **完整课程**: https://jamwithai.substack.com/
- **文档**: `/Disk_2/claw_working_dir/production-agentic-rag-course/README.md`

---

**安装状态**: ✅ 已下载，待安装  
**建议行动**: 安装 UV → 配置环境 → 启动 Week 7  
**预计时间**: 30-60 分钟

---

**报告生成时间**: 2026-03-28  
**安装位置**: `/Disk_2/claw_working_dir/production-agentic-rag-course/`
