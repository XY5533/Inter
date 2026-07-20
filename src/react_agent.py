"""
ReAct Agent 核心 — Thought → Action → Observation 循环

纯 Python 实现，不依赖 LangGraph/CrewAI 等重型框架。
通过 system prompt 引导 LLM 按指定格式输出 Thought/Action，
解析后执行对应工具，将 Observation 注入下一轮推理。
"""

import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

import config as global_config
from prompt_templates import (
    REACT_SYSTEM_PROMPT,
    REACT_FIRST_STEP_TEMPLATE,
    REACT_STEP_TEMPLATE,
)


# ============================================================
# 数据结构
# ============================================================

class ActionType(Enum):
    SEARCH_KNOWLEDGE = "SEARCH_KNOWLEDGE"
    SEARCH_MEMORY = "SEARCH_MEMORY"
    CLARIFY = "CLARIFY"
    FINAL_ANSWER = "FINAL_ANSWER"
    UNKNOWN = "UNKNOWN"


@dataclass
class AgentStep:
    """单步推理记录"""
    step_num: int
    thought: str = ""
    action_type: ActionType = ActionType.UNKNOWN
    action_input: str = ""
    observation: str = ""
    elapsed_ms: float = 0.0
    is_duplicate: bool = False


@dataclass
class ReActResult:
    """ReAct 运行结果"""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    success: bool = True
    fallback_used: bool = False
    total_elapsed_ms: float = 0.0
    total_tokens_estimate: int = 0
    error_message: str = ""


# ============================================================
# ReAct Agent
# ============================================================

class ReActAgent:
    """ReAct 推理-行动 Agent

    使用方式：
        agent = ReActAgent(llm, vector_store, memory_manager)
        result = agent.run("工业机器人伺服电机选型注意事项？")
        # result.answer → 最终回答
        # result.steps   → 思考步骤（给 UI 展示）
    """

    def __init__(
        self,
        llm,                       # ChatTongyi 实例
        vector_store,              # VectorStore 实例
        memory_manager=None,       # MemoryManager 实例（可选）
        max_steps: int = None,
        duplicate_threshold: int = None,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.memory_manager = memory_manager
        self.max_steps = max_steps or getattr(global_config, "REACT_MAX_STEPS", 5)
        self.duplicate_threshold = (
            duplicate_threshold
            or getattr(global_config, "REACT_DUPLICATE_THRESHOLD", 2)
        )

    # ---- 主入口 ----

    def run(self, query: str, chat_history: List = None) -> ReActResult:
        """运行 ReAct 循环，返回最终答案和思考过程"""
        start_time = time.perf_counter()
        steps: List[AgentStep] = []
        total_chars = len(query)

        try:
            # 获取记忆上下文
            memory_context = "（无历史记录）"
            if self.memory_manager:
                mem_ctx = self.memory_manager.get_context(query)
                if mem_ctx:
                    parts = []
                    if mem_ctx.get("summary"):
                        parts.append("历史摘要: " + mem_ctx["summary"])
                    if mem_ctx.get("long_term"):
                        parts.append("相关记忆: " + "; ".join(
                            m.value for m in mem_ctx["long_term"]
                        ))
                    if parts:
                        memory_context = "\n".join(parts)

            # 对话历史文本
            history_text = ""
            if chat_history:
                lines = []
                for msg in chat_history[-6:]:  # 最近3轮
                    role = "用户" if getattr(msg, "type", "") == "human" else "助手"
                    content = getattr(msg, "content", str(msg))[:200]
                    lines.append(f"{role}: {content}")
                history_text = "\n".join(lines)
            if history_text:
                memory_context = history_text + "\n\n" + memory_context

            # ReAct 循环
            previous_steps_text = ""
            for step_num in range(1, self.max_steps + 1):
                step = AgentStep(step_num=step_num)
                step_start = time.perf_counter()

                # 1) Thought: 调用 LLM
                raw_response = self._think(
                    query, memory_context, previous_steps_text, is_first=(step_num == 1)
                )
                total_chars += len(raw_response)

                # 2) 解析
                thought, action_type, action_input = self._parse_response(raw_response)
                step.thought = thought
                step.action_type = action_type
                step.action_input = action_input

                # 3) 重复检测
                if self._is_duplicate(action_type, action_input):
                    step.is_duplicate = True
                    steps.append(step)
                    # 触发降级
                    return self._fallback(query, steps, start_time, total_chars,
                                          "连续重复相同动作，触发降级")

                # 4) 如果是 FINAL_ANSWER，直接返回
                if action_type == ActionType.FINAL_ANSWER:
                    step.observation = "（回答完成）"
                    step.elapsed_ms = round((time.perf_counter() - step_start) * 1000, 1)
                    steps.append(step)
                    total_elapsed = round((time.perf_counter() - start_time) * 1000, 1)
                    return ReActResult(
                        answer=action_input,
                        steps=steps,
                        success=True,
                        total_elapsed_ms=total_elapsed,
                        total_tokens_estimate=int(total_chars * 1.5),
                    )

                # 5) 如果是 CLARIFY，也直接返回（不进入循环）
                if action_type == ActionType.CLARIFY:
                    step.observation = "（等待用户澄清）"
                    step.elapsed_ms = round((time.perf_counter() - step_start) * 1000, 1)
                    steps.append(step)
                    total_elapsed = round((time.perf_counter() - start_time) * 1000, 1)
                    return ReActResult(
                        answer=f"[澄清] {action_input}",
                        steps=steps,
                        success=True,
                        total_elapsed_ms=total_elapsed,
                        total_tokens_estimate=int(total_chars * 1.5),
                    )

                # 6) 执行动作
                observation = self._execute(action_type, action_input)
                step.observation = observation
                step.elapsed_ms = round((time.perf_counter() - step_start) * 1000, 1)
                steps.append(step)

                # 7) 构建下一轮的上下文
                previous_steps_text += (
                    f"\nStep {step_num}:\n"
                    f"THOUGHT: {thought}\n"
                    f"ACTION: [{action_type.value} {action_input}]\n"
                    f"OBSERVATION: {observation}\n"
                )

            # 超过最大步数 → 降级
            return self._fallback(query, steps, start_time, total_chars,
                                  f"超过最大步数 {self.max_steps}，触发降级")

        except Exception as e:
            return self._fallback(query, steps, start_time, total_chars,
                                  f"ReAct 异常: {str(e)}")

    # ---- 内部方法 ----

    def _think(
        self, query: str, memory_context: str, previous_steps: str, is_first: bool
    ) -> str:
        """调用 LLM 获取 Thought + Action"""
        if is_first:
            user_content = REACT_FIRST_STEP_TEMPLATE.format(
                question=query,
                memory_context=memory_context,
            )
        else:
            user_content = REACT_STEP_TEMPLATE.format(
                question=query,
                memory_context=memory_context,
                previous_steps=previous_steps,
            )

        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _parse_response(self, text: str):
        """解析 LLM 输出，提取 Thought 和 Action"""
        thought = ""
        action_type = ActionType.UNKNOWN
        action_input = ""

        # 提取 THOUGHT
        thought_match = re.search(
            r'THOUGHT[:：]\s*(.+?)(?=\n\s*(?:ACTION|OBSERVATION|FINAL|$))',
            text, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 ACTION
        # 格式: ACTION: [TOOL_NAME 参数]
        action_match = re.search(
            r'ACTION[:：]\s*\[(\w+)\s+(.+?)\]',
            text, re.IGNORECASE
        )
        if action_match:
            tool_name = action_match.group(1).strip().upper()
            action_input = action_match.group(2).strip()
            # 映射到 ActionType
            type_map = {
                "SEARCH_KNOWLEDGE": ActionType.SEARCH_KNOWLEDGE,
                "SEARCH_MEMORY": ActionType.SEARCH_MEMORY,
                "CLARIFY": ActionType.CLARIFY,
                "FINAL_ANSWER": ActionType.FINAL_ANSWER,
            }
            action_type = type_map.get(tool_name, ActionType.UNKNOWN)

        # 如果没匹配到标准格式，尝试从全文提取 FINAL_ANSWER
        if action_type == ActionType.UNKNOWN:
            final_match = re.search(
                r'FINAL_ANSWER[:：]\s*(.+)',
                text, re.DOTALL | re.IGNORECASE
            )
            if final_match:
                action_type = ActionType.FINAL_ANSWER
                action_input = final_match.group(1).strip()
                if not thought:
                    thought = "直接回答用户问题"

        # 最后兜底：整个文本作为 FINAL_ANSWER
        if action_type == ActionType.UNKNOWN:
            action_type = ActionType.FINAL_ANSWER
            action_input = text.strip()
            if not thought:
                thought = "（模型输出格式不标准，以全文作为回答）"

        return thought, action_type, action_input

    def _execute(self, action_type: ActionType, action_input: str) -> str:
        """执行工具动作，返回 Observation 文本"""
        if action_type == ActionType.SEARCH_KNOWLEDGE:
            try:
                docs = self.vector_store.search(action_input, k=3)
                if not docs:
                    return "知识库中未找到相关信息。"
                parts = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "未知")
                    content = doc.page_content[:500]  # 截断
                    parts.append(f"[{i}] ({src}) {content}")
                return "\n\n".join(parts)
            except Exception as e:
                return f"知识库搜索失败: {e}"

        elif action_type == ActionType.SEARCH_MEMORY:
            if self.memory_manager:
                try:
                    memories = self.memory_manager.recall(action_input, k=3)
                    if not memories:
                        return "没有找到相关历史记忆。"
                    parts = []
                    for m in memories:
                        parts.append(f"- [{m.category}] {m.key}: {m.value}")
                    return "\n".join(parts)
                except Exception as e:
                    return f"记忆搜索失败: {e}"
            return "记忆系统未启用。"

        elif action_type == ActionType.CLARIFY:
            return f"向用户澄清: {action_input}"

        elif action_type == ActionType.FINAL_ANSWER:
            return action_input

        else:
            return f"未知动作类型: {action_type}"

    def _is_duplicate(self, action_type: ActionType, action_input: str) -> bool:
        """检测连续重复动作"""
        # 只对 SEARCH_* 做重复检测
        if action_type not in (ActionType.SEARCH_KNOWLEDGE, ActionType.SEARCH_MEMORY):
            return False

        # 使用实例属性记录最近的动作历史
        if not hasattr(self, "_recent_actions"):
            self._recent_actions: List[tuple] = []

        current = (action_type, action_input.strip().lower())
        self._recent_actions.append(current)

        # 只保留最近 N 个
        if len(self._recent_actions) > self.duplicate_threshold:
            self._recent_actions.pop(0)

        # 检查最近 threshold 个是否都相同
        if len(self._recent_actions) >= self.duplicate_threshold:
            recent = self._recent_actions[-self.duplicate_threshold:]
            if all(a == current for a in recent):
                return True

        return False

    def _fallback(
        self,
        query: str,
        steps: List[AgentStep],
        start_time: float,
        total_chars: int,
        reason: str,
    ) -> ReActResult:
        """降级：回退到简单 RAG 问答"""
        try:
            docs = self.vector_store.search(query, k=3)
            context = ""
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "未知来源")
                context += f"[参考 {i}]\n{doc.page_content[:500]}\n(来源：{src})\n\n"
            if not context:
                context = "知识库中未找到相关信息。"

            fallback_prompt = (
                f"你是一个专业的知识库助手。请根据参考资料回答用户问题。"
                f"如果参考资料不充分，请如实告知。\n\n"
                f"参考资料：\n{context}"
            )
            messages = [
                SystemMessage(content=fallback_prompt),
                HumanMessage(content=f"用户问题：{query}"),
            ]
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            total_chars += len(answer)
        except Exception as e:
            answer = f"抱歉，系统处理您的问题时遇到错误。请稍后重试。({reason})"
            total_chars += len(answer)

        total_elapsed = round((time.perf_counter() - start_time) * 1000, 1)
        return ReActResult(
            answer=answer,
            steps=steps,
            success=True,  # 降级后仍返回结果
            fallback_used=True,
            total_elapsed_ms=total_elapsed,
            total_tokens_estimate=int(total_chars * 1.5),
            error_message=reason,
        )


# ============================================================
# 工厂函数
# ============================================================

def create_react_agent(llm, vector_store, memory_manager=None) -> ReActAgent:
    """便捷工厂：用当前 config 创建 ReActAgent"""
    return ReActAgent(
        llm=llm,
        vector_store=vector_store,
        memory_manager=memory_manager,
        max_steps=getattr(global_config, "REACT_MAX_STEPS", 5),
        duplicate_threshold=getattr(global_config, "REACT_DUPLICATE_THRESHOLD", 2),
    )
