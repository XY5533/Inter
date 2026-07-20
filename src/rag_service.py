"""
大模型问答 + 提示词 + 上下文记忆 + ReAct 推理
调用大模型 API，结合检索到的知识库内容回答问题
支持：ReAct 推理框架、两层记忆（短期+长期）、多模态输入
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from prompt_templates import RAG_SYSTEM_TEMPLATE


class ConversationBufferWindowMemory:
    """简易对话记忆 - 保留最近 k 轮对话（替代已移除的 langchain.memory）"""

    def __init__(self, k: int = 3, memory_key: str = "history"):
        self.k = k
        self.memory_key = memory_key
        self.chat_memory = []  # [(human, ai), ...]

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """返回对话历史消息列表"""
        messages = []
        for human, ai in self.chat_memory[-self.k :]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        return {self.memory_key: messages}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """保存一轮对话"""
        self.chat_memory.append((inputs["input"], outputs["output"]))

    def clear(self) -> None:
        """清空记忆"""
        self.chat_memory = []


class RagService:
    """RAG 问答服务（支持 ReAct 推理 + 两层记忆）

    使用方式：
        rs = RagService()
        # 标准 RAG 问答
        answer = rs.answer("什么是RV减速器？", docs)
        # ReAct 推理问答
        answer = rs.answer_with_react("什么是RV减速器？")
        # 切换模式
        rs.enable_react = False
    """

    def __init__(self):
        api_key = config.DASHSCOPE_API_KEY or None
        self.llm = ChatTongyi(
            model=config.CHAT_MODEL,
            dashscope_api_key=api_key,
        )
        # 短期对话记忆（原有）
        self.memory = ConversationBufferWindowMemory(k=config.SHORT_TERM_K)

        # ReAct 模式开关
        self.enable_react = getattr(config, "REACT_ENABLE", True)

        # 长期记忆管理器（延迟初始化，避免循环导入）
        self._memory_manager = None
        self._react_agent = None

    def _build_prompt(self) -> ChatPromptTemplate:
        """构建提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_TEMPLATE),
            MessagesPlaceholder("history"),
            ("user", "用户问题：{question}"),
        ])

    def answer(self, question: str, context_docs: List[Document]) -> str:
        """回答问题（标准 RAG 模式，带记忆上下文增强）"""
        # ---- 获取长期记忆上下文 ----
        memory_context = ""
        if self._memory_manager:
            try:
                memory_context = self._memory_manager.get_context_text(question)
            except Exception:
                pass

        # ---- 构建上下文 ----
        context = ""
        if context_docs:
            for i, doc in enumerate(context_docs, 1):
                src = doc.metadata.get("source", "未知来源")
                context += f"[参考片段 {i}]\n{doc.page_content}\n(来源：{src})\n\n"
        else:
            context = "当前知识库中没有找到与该问题直接相关的参考资料。"

        # ---- 注入记忆上下文 ----
        if memory_context:
            context = f"## 用户历史记忆\n{memory_context}\n\n## 知识库参考资料\n{context}"

        # ---- 获取对话历史 ----
        history = self.memory.load_memory_variables({})["history"]

        # ---- 调用大模型 ----
        prompt = self._build_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "history": history,
            "question": question,
        })

        answer_text = response.content

        # ---- 保存对话记忆 ----
        self.memory.save_context({"input": question}, {"output": answer_text})

        # ---- 记录长期记忆 ----
        if self._memory_manager:
            try:
                self._memory_manager.add_turn(question, answer_text)
                self._maybe_remember(question, answer_text)
            except Exception:
                pass

        return answer_text

    # ---- ReAct 推理模式 ----

    @property
    def memory_manager(self):
        """懒初始化记忆管理器"""
        if self._memory_manager is None:
            from memory_manager import MemoryManager
            self._memory_manager = MemoryManager()
            self._memory_manager.set_llm(self.llm)
        return self._memory_manager

    @property
    def react_agent(self):
        """懒初始化 ReAct Agent"""
        if self._react_agent is None:
            from react_agent import ReActAgent
            self._react_agent = ReActAgent(
                llm=self.llm,
                vector_store=None,  # 由外部注入
                memory_manager=self.memory_manager,
            )
        return self._react_agent

    def answer_with_react(self, question: str,
                          retriever_func=None,
                          context_docs: List[Document] = None):
        """使用 ReAct 推理回答问题

        Args:
            question: 用户问题
            retriever_func: 检索函数 query→List[Document]，Search知识库时调用
            context_docs: 预检索的文档（可选，如果不传则用 retriever_func）

        Returns:
            dict: {"answer": str, "steps": List[AgentStep], "fallback_used": bool}
        """
        if not self.enable_react:
            # 降级为标准 RAG
            docs = context_docs or (retriever_func(question) if retriever_func else [])
            return {
                "answer": self.answer(question, docs),
                "steps": [],
                "fallback_used": True,
                "fallback_reason": "ReAct 模式已关闭",
            }

        try:
            # 设置检索函数
            if retriever_func:
                self.react_agent.vector_store = _RetrieverAdapter(retriever_func)
            elif context_docs is not None:
                self.react_agent.vector_store = _StaticDocStore(context_docs)

            # 运行 ReAct
            result = self.react_agent.run(question)

            # 保存对话
            self.memory.save_context({"input": question}, {"output": result.answer})

            # 记录长期记忆
            if self._memory_manager:
                try:
                    self._memory_manager.add_turn(question, result.answer)
                    self._maybe_remember(question, result.answer)
                except Exception:
                    pass

            return {
                "answer": result.answer,
                "steps": result.steps,
                "fallback_used": result.fallback_used,
                "fallback_reason": result.error_message if result.fallback_used else "",
                "elapsed_ms": result.total_elapsed_ms,
            }

        except Exception as e:
            # 异常降级
            docs = context_docs or (retriever_func(question) if retriever_func else [])
            fallback_answer = self.answer(question, docs)
            return {
                "answer": fallback_answer,
                "steps": [],
                "fallback_used": True,
                "fallback_reason": f"ReAct 异常: {e}",
            }

    # ---- 记忆管理 ----

    def _maybe_remember(self, user_msg: str, ai_msg: str):
        """自动检测是否值得长期记忆"""
        if not self._memory_manager:
            return
        try:
            self._memory_manager.extract_memories_with_llm(user_msg, ai_msg)
        except Exception:
            pass

    def clear_memory(self, clear_long_term: bool = False):
        """清空对话记忆"""
        self.memory.clear()
        if clear_long_term and self._memory_manager:
            self._memory_manager.clear_all()
        elif self._memory_manager:
            self._memory_manager.clear_short_term()

    def get_memory_stats(self) -> dict:
        """获取记忆统计信息"""
        if self._memory_manager:
            return {
                "short_term_count": self._memory_manager.short_term_count,
                "long_term_count": self._memory_manager.long_term_count,
                "has_summary": self._memory_manager.has_summary,
            }
        return {"short_term_count": 0, "long_term_count": 0, "has_summary": False}

    def get_all_long_term_memories(self) -> List:
        """获取所有长期记忆列表（给 UI 展示用）"""
        if self._memory_manager:
            return self._memory_manager.get_all_memories()
        return []

    def delete_long_term_memory(self, memory_id: str) -> bool:
        """删除指定长期记忆"""
        if self._memory_manager:
            return self._memory_manager.forget(memory_id)
        return False


# ============================================================
# 适配器（桥接 ReAct Agent 与外部检索接口）
# ============================================================

class _RetrieverAdapter:
    """将 callable 检索函数适配为 VectorStore 接口"""
    def __init__(self, retriever_func):
        self._search_fn = retriever_func

    def search(self, query: str, k: int = 3):
        return self._search_fn(query, k=k) if callable(self._search_fn) else []


class _StaticDocStore:
    """将预检索的文档列表适配为 VectorStore 接口"""
    def __init__(self, docs: List[Document]):
        self._docs = docs

    def search(self, query: str, k: int = 3):
        return self._docs[:k] if self._docs else []
