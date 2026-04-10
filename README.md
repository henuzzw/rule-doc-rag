# 规则文档自动生成系统

基于 RAG + LLM 的单条规则文档自动生成 demo。项目内置“消费金融风控规则”样例数据，可在离线 mock 模式跑完整闭环；生产接入时可切换为 BGE Embedding + PostgreSQL/pgvector + 百度千帆/DeepSeek。

## 能力边界

- 知识增强检索：把需求文档、历史规则文档切成知识条目，写入 PostgreSQL + pgvector，按待生成规则语义检索 top-k 上下文。
- AI 文档生成：统一 Prompt 模板约束输出 JSON，生成规则名称、业务目标、触发时机、输入输出、判断逻辑、伪代码、示例、异常处理、测试点。
- 规则文档管理：生成草稿、保存 Prompt/原始响应/RAG 上下文，支持人工校验状态流转，并提供一个轻量页面展示。

## 架构

```text
需求文档 / 历史规则文档
        │
        ├─ scripts/seed_demo_data.py
        │
        ▼
BGE/Mock Embedding ──► PostgreSQL + pgvector 知识库
        │                   │
        │                   ▼
用户选择需求 + 规则名 ─► RAG 检索 ─► Prompt 组装 ─► 千帆/DeepSeek/Mock LLM
                                                  │
                                                  ▼
                                     generated_documents 文档库
                                                  │
                                                  ▼
                                      FastAPI API + 审核页面
```

## 快速启动

```bash
cd /home/openclaw/rule-doc-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

docker compose up -d postgres
make init-db
make seed
make dev
```

浏览器打开 `http://localhost:8010`。

默认 `.env.example` 使用 `EMBEDDING_MODE=mock` 和 `LLM_MODE=mock`，不需要模型文件和 API Key。

## 向量数据库

本项目使用 `PostgreSQL + pgvector` 作为向量数据库。

- Docker 镜像：`pgvector/pgvector:pg16`
- 向量表：`knowledge_chunks`
- 向量字段：`embedding VECTOR(1024)`
- 索引：`hnsw (embedding vector_cosine_ops)`
- 检索方式：按 `embedding <=> query_vector` 计算 cosine distance，取 top-k

schema 见 `db/init/01_schema.sql`。

## 切换真实 RAG/LLM

编辑 `.env`：

```bash
EMBEDDING_MODE=bge
BGE_MODEL_NAME=BAAI/bge-large-zh-v1.5

LLM_MODE=qianfan
QIANFAN_API_KEY=你的千帆 API Key
QIANFAN_BASE_URL=https://qianfan.baidubce.com/v2/coding
QIANFAN_MODEL=qianfan-code-latest
QIANFAN_MAX_TOKENS=8192
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1
```

当前接入方式使用千帆 coding endpoint 的 OpenAI-compatible `chat/completions`。

千帆当前按 `openai-completions` 方式接入，代码中调用的是 OpenAI SDK 的 `client.completions.create()`。

首次启用 BGE 前额外安装：

```bash
pip install -r requirements-bge.txt
```

## 通过 7890 代理安装依赖

如果直连 PyPI 超时，可以临时给 pip 设置本机 HTTP 代理：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
NO_PROXY=localhost,127.0.0.1 \
pip install --timeout 120 --retries 5 -r requirements.txt
```

注意：`requirements.txt` 是默认运行依赖；`requirements-bge.txt` 会额外拉取 `sentence-transformers/torch`，体积明显更大。

如果要切回 DeepSeek：

```bash
LLM_MODE=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_MODEL=deepseek-chat
```

然后重新执行：

```bash
make seed
make dev
```

注意：当前 schema 使用 `VECTOR(1024)`，与 `BAAI/bge-large-zh-v1.5` 的维度匹配；如果换成其他维度模型，需要同步调整 `db/init/01_schema.sql` 中的向量维度并重建表。

## API 示例

生成规则文档：

```bash
curl -X POST http://localhost:8010/api/generate \
  -H 'content-type: application/json' \
  -d '{
    "requirement_id": "REQ-FRAUD-PHONE-REALNAME-002",
    "rule_name": "手机号三要素实名一致性校验",
    "rule_code": "RULE_PHONE_REALNAME",
    "top_k": 5
  }'
```

审核文档：

```bash
curl -X POST http://localhost:8010/api/documents/<document_id>/review \
  -H 'content-type: application/json' \
  -d '{
    "status": "reviewed",
    "reviewer_notes": "已确认判断逻辑和降级策略。"
  }'
```

## 目录说明

```text
app/
  main.py                 FastAPI API、页面路由
  repository.py           PostgreSQL/pgvector 数据访问
  services/
    embedding.py          BGE 与离线 hash embedding
    retrieval.py          检索编排
    generator.py          Prompt、响应解析、文档生成编排
    llm.py                百度千帆、DeepSeek 与 mock LLM
data/
  requirements.json       自定义需求数据
  history_rules.json      自定义历史规则文档
db/init/01_schema.sql     pgvector schema
scripts/
  init_db.py              初始化数据库
  seed_demo_data.py       入库样例需求和历史文档
tests/                    核心服务单测
```

## 演示讲法

这个项目解决的是“规则引擎迭代多年但规则文档缺失”的维护问题。核心流程是：先把需求和历史规则说明向量化写入 PostgreSQL/pgvector；生成某条规则时，用规则名称和需求正文检索相似历史规则/规范；再把检索结果、当前需求、固定 JSON Schema 一起放进 Prompt，让千帆/DeepSeek 输出结构化规则说明；最后落库成 draft，由规则负责人校验并流转到 reviewed/published。
