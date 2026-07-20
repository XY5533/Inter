# 🤖 IntelliRAG — 知识库问答平台

基于 **RAG（检索增强生成）** + **ReAct Agent** 架构的企业级智能问答系统。支持 PDF/Word/TXT 多格式文档导入，FAISS 向量检索，通义千问大模型对话，以及语音/图片/视频多模态输入。

## ✨ 核心特性

- **RAG 检索增强生成** — 语义检索 + LLM 生成，答案可追溯原文出处
- **ReAct 推理框架** — Agent 自主决策：搜什么、搜几次、够不够、何时说"不知道"
- **对话记忆** — 自实现 `ConversationBufferWindowMemory`，保留上下文实现多轮连贯对话
- **多模态输入** — 支持语音转文字、图片描述、视频抽帧汇总
- **未知问题兜底** — 检索相似度不足时自动礼貌回复，记录未解决问题
- **Streamlit 交互界面** — 知识库一键重建、向量库状态监控、参考来源追溯

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | 通义千问 qwen3-max（对话）/ text-embedding-v4（向量嵌入） |
| 框架 | LangChain、Streamlit |
| 向量库 | FAISS（CPU 版） |
| 文档解析 | PyPDF、python-docx |
| 多模态 | qwen-vl-max（视觉）/ paraformer-v2（语音） |

## 📁 项目结构

```
Agent/
├── main.py                  # Streamlit 入口，页面布局与交互
├── config.py                # 全局配置（模型、路径、参数）
├── rag_service.py           # RAG 核心服务，检索 + 生成
├── react_agent.py           # ReAct 推理循环（Thought→Action→Observation）
├── vector_store.py          # FAISS 向量库封装
├── knowledge_base.py        # 文档解析（PDF/Word/TXT）
├── memory_manager.py        # 对话记忆管理
├── multimodal.py            # 多模态处理（语音/图片/视频）
├── prompt_templates.py      # Prompt 模板
├── run_tests.py             # 测试入口
├── requirements.txt         # Python 依赖
└── tests/                   # 测试用例 + 评估器
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/XY5533/Inter.git
cd Inter
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

创建 `.env` 文件（已加入 `.gitignore`，不会上传）：

```bash
# Windows PowerShell
echo DASHSCOPE_API_KEY=你的通义千问APIKey > .env

# Linux / macOS
echo 'DASHSCOPE_API_KEY=你的通义千问APIKey' > .env
```

> 🔑 在[阿里云百炼平台](https://bailian.console.aliyun.com/)获取 API Key（通义千问 qwen3-max + text-embedding-v4）。

### 4. 放入知识库文档

在 `knowledge_base/` 目录下放入你的企业文档（支持 PDF / Word / TXT）：

```
knowledge_base/
├── 产品手册.pdf
├── 售后政策.docx
└── 规章制度.txt
```

### 5. 启动

```bash
streamlit run main.py
```

浏览器自动打开 `http://localhost:8501`，即可开始使用。

## ⚙️ 配置说明

在 `config.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `text-embedding-v4` | 向量嵌入模型 |
| `CHAT_MODEL` | `qwen3-max` | 对话模型 |
| `CHUNK_SIZE` | `500` | 文本分块大小（字符） |
| `RETRIEVER_K` | `3` | 每次检索返回片段数 |
| `SHORT_TERM_K` | `5` | 短期记忆保留轮数 |
| `REACT_MAX_STEPS` | `5` | ReAct 最大推理步数 |

## 📄 License

MIT
