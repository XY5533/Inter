# IntelliRAG 知识库问答平台

**某工业自动化企业 — 技术中台**

**背景**：企业积累 8 类工业机器人教材、产品手册、NLP 学术资料等非结构化文档 458 份，传统关键词检索命中率不足 40%，工程师查阅效率低。负责从零搭建基于 RAG + ReAct Agent 的多模态知识库问答平台，实现多模态输入、多步推理、持久记忆全链路闭环。知识库 504 个向量片段，评估测试通过率 93.8%，系统测试 59 项 100% 通过，单次问答平均响应 2.3s。

**技术栈**：Python, Streamlit, LangChain, FAISS, DashScope API (qwen3-max / text-embedding-v4 / qwen-vl-max / paraformer-v2), PyPDF, python-docx

---

- **ReAct 推理框架从 0 到 1 工程化落地**，替代原有 if-else 线性 RAG 管线。设计 Thought→Action→Observation 三步循环，LLM 自主决策 SEARCH_KNOWLEDGE / SEARCH_MEMORY / CLARIFY / FINAL_ANSWER 四种动作，基于 System Prompt 引导模型按标准格式输出并正则解析。三層安全机制：硬限制最大 5 步防死循环、连续重复动作检测超阈值自动降级、异常兜底回退标准 RAG 保障可用性 100%。多步推理场景下 Agent 能自动换角度检索，超出知识库范围时诚实告知而非编造。

- **两层记忆架构**：短期记忆滑动窗口（k=5）保留对话上下文，长期记忆复用 FAISS 向量库做语义检索。MemoryManager 封装 JSON + FAISS 双写存储，LLM 自动提取用户偏好、身份信息、纠正反馈、高频问题四类有价值记忆。记忆检索与 ReAct SEARCH_MEMORY 动作无缝对接，Agent 根据当前问题语义召回历史记忆注入 Prompt。k=10 超窗口时自动触发 LLM 摘要压缩，长期记忆批量写入 20 条 FAISS 同步不崩溃。

- **多模态输入统一管线**：语音 paraformer-v2 实时转写，图片 qwen-vl-max 生成技术文档描述，视频 OpenCV 抽帧后逐帧理解汇总。所有模态统一转文本后进入 ReAct+RAG 管线，检索链路零改造。MultimodalProcessor 统一入口，各模态独立 try-catch 降级不阻塞文本主流程，图片理解失败不影响纯文本问答。

- **完整评估体系**：18 条分层测试用例覆盖工业机器人基础、编程应用、NLP 知识、个人信息、兜底五类场景，关键词命中率、否定词检测、长度校验、未知兜底四维指标。支持批量执行、JSON 存储、Markdown 报告、回归对比。额外编写 59 项系统级测试覆盖配置/Prompt/向量库/ReAct/记忆/多模态/适配器/边界异常 10 模块，通过率 100%，可接入 CI 回归。

- **Windows 中文环境工程难题**：FAISS C++ 后端不支持 Unicode 路径 → 索引迁移至 ASCII TEMP 目录；ChromaDB onnxruntime C 扩展 GBK 冲突致 SIGSEGV → 选型替换 FAISS；Streamlit @st.cache_resource 同周期 clear() 不生效 → 去掉缓存装饰器；多进程端口残留静默失败 → kill+netstat 标准验证流程。全部问题已归档为技术笔记。

- **开发周期 2 周，独立完成全部模块**：Week 1 基础 RAG（文档解析→分块→FAISS 索引→Streamlit UI），Week 2 三大升级（ReAct + 记忆 + 多模态）+ 测试体系 + 评估报告 + 项目经历文档。
