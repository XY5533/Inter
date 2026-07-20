"""
记忆管理器 — 两层记忆系统

短期记忆：对话缓冲 + 自动摘要压缩
长期记忆：用户画像/偏好/高频问题，FAISS + JSON 混合存储
"""

import os
import json
import time
import hashlib
import shutil
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import DashScopeEmbeddings

import config as global_config
from prompt_templates import MEMORY_EXTRACT_PROMPT, SUMMARY_COMPRESS_PROMPT


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Memory:
    """一条长期记忆"""
    id: str                            # 唯一 ID (hash)
    key: str                           # 简短标题
    value: str                         # 记忆内容
    category: str = "general"          # preference | identity | correction | faq | general
    importance: int = 1                # 1-5
    access_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=d.get("id", ""),
            key=d.get("key", ""),
            value=d.get("value", ""),
            category=d.get("category", "general"),
            importance=d.get("importance", 1),
            access_count=d.get("access_count", 0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ============================================================
# 记忆管理器
# ============================================================

class MemoryManager:
    """两层记忆管理器

    使用方式：
        mm = MemoryManager(embedding_model="text-embedding-v4")
        mm.add_turn(user_msg, ai_msg)                    # 记录对话
        mm.remember("用户偏好", "关注伺服电机", "preference")  # 手动存记忆
        memories = mm.recall("伺服电机", k=3)              # 检索记忆
        ctx = mm.get_context(query)                       # 获取完整上下文

    Args:
        embedding_model: DashScope 向量模型名
        storage_dir: 记忆存储目录（默认为 config.MEMORY_DIR）
        short_term_k: 短期记忆保留轮数
        compress_threshold: 触发摘要压缩的轮数阈值
    """

    def __init__(
        self,
        embedding_model: str = None,
        storage_dir: str = None,
        short_term_k: int = None,
        compress_threshold: int = None,
    ):
        self.embedding_model = embedding_model or global_config.EMBEDDING_MODEL
        self.storage_dir = storage_dir or global_config.MEMORY_DIR
        self.short_term_k = short_term_k or global_config.SHORT_TERM_K
        self.compress_threshold = compress_threshold or global_config.COMPRESS_THRESHOLD

        # 短期记忆
        self.short_term: List[tuple] = []   # [(user_msg, ai_msg), ...]
        self.summary: str = ""              # 对话摘要

        # 长期记忆
        self._memories: Dict[str, Memory] = {}  # id → Memory
        self._embedding = DashScopeEmbeddings(model=self.embedding_model)
        self._faiss_index = None            # FAISS 索引（懒加载）
        self._faiss_texts: List[str] = []   # 与 FAISS 同步的文本列表

        # 初始化存储
        self._init_storage()

    # ================================================================
    # 短期记忆
    # ================================================================

    def add_turn(self, user_msg: str, ai_msg: str):
        """记录一轮对话"""
        self.short_term.append((user_msg, ai_msg))
        self._maybe_compress()
        self._save_short_term()

    def _maybe_compress(self):
        """如果短期记忆过长，触发摘要压缩"""
        if len(self.short_term) <= self.compress_threshold:
            return

        # 摘取前半部分（保留最近 short_term_k 轮）
        to_compress = self.short_term[: -self.short_term_k]
        self.short_term = self.short_term[-self.short_term_k :]

        # 构建压缩文本
        history_lines = []
        for user_msg, ai_msg in to_compress:
            history_lines.append(f"用户：{user_msg}")
            history_lines.append(f"助手：{ai_msg}")
        history_text = "\n".join(history_lines)

        # 用 LLM 压缩（如果 llm 已注入）
        if self._llm:
            try:
                prompt = SUMMARY_COMPRESS_PROMPT.format(history=history_text)
                from langchain_core.messages import HumanMessage
                response = self._llm.invoke([HumanMessage(content=prompt)])
                new_summary = response.content if hasattr(response, "content") else str(response)
                # 合并摘要
                if self.summary:
                    self.summary = self.summary + " | " + new_summary
                else:
                    self.summary = new_summary
            except Exception:
                # LLM 压缩失败不阻塞
                pass

    # ================================================================
    # 长期记忆
    # ================================================================

    def remember(self, key: str, value: str, category: str = "general",
                 importance: int = 1) -> Memory:
        """存储一条长期记忆"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        mem_id = hashlib.md5(f"{key}:{value}:{category}".encode()).hexdigest()[:12]

        if mem_id in self._memories:
            # 更新已有记忆
            mem = self._memories[mem_id]
            mem.value = value
            mem.importance = max(mem.importance, importance)
            mem.updated_at = ts
            mem.access_count += 1
        else:
            mem = Memory(
                id=mem_id,
                key=key,
                value=value,
                category=category,
                importance=importance,
                created_at=ts,
                updated_at=ts,
            )
            self._memories[mem_id] = mem

        # 同步到 FAISS
        self._add_to_faiss(mem)
        self._save_long_term()
        return mem

    def recall(self, query: str, k: int = None) -> List[Memory]:
        """语义检索长期记忆"""
        if k is None:
            k = global_config.LONG_TERM_K

        if not self._memories or not self._faiss_index:
            return []

        try:
            # FAISS 相似度搜索
            docs_with_scores = self._faiss_index.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in docs_with_scores:
                mem_id = doc.metadata.get("memory_id", "")
                if mem_id in self._memories:
                    mem = self._memories[mem_id]
                    mem.access_count += 1
                    results.append(mem)

            # 按重要性 + 得分排序
            results.sort(key=lambda m: (m.importance, 1.0), reverse=True)
            return results

        except Exception:
            return []

    def forget(self, memory_id: str) -> bool:
        """删除一条长期记忆"""
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._rebuild_faiss()
            self._save_long_term()
            return True
        return False

    def get_all_memories(self) -> List[Memory]:
        """列出所有长期记忆"""
        return sorted(
            self._memories.values(),
            key=lambda m: (m.importance, m.access_count),
            reverse=True,
        )

    def extract_memories_with_llm(self, user_msg: str, ai_msg: str) -> Optional[Memory]:
        """用 LLM 分析对话，自动提取值得长期记忆的内容"""
        if not self._llm:
            return None

        try:
            from langchain_core.messages import HumanMessage
            prompt = MEMORY_EXTRACT_PROMPT.format(user_msg=user_msg, ai_msg=ai_msg)
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group(0))
            if not data.get("memorable"):
                return None

            return self.remember(
                key=data.get("key", "未命名"),
                value=data.get("value", text[:50]),
                category=data.get("category", "general"),
                importance=data.get("importance", 1),
            )
        except Exception:
            return None

    # ================================================================
    # 对外接口
    # ================================================================

    def get_context(self, query: str = "") -> dict:
        """获取完整记忆上下文（用于注入 Prompt）"""
        result = {
            "summary": self.summary,
            "short_term": self.short_term[-self.short_term_k:] if self.short_term else [],
        }

        if query:
            result["long_term"] = self.recall(query)
        else:
            result["long_term"] = []

        return result

    def get_context_text(self, query: str = "") -> str:
        """获取格式化的记忆上下文文本"""
        ctx = self.get_context(query)
        parts = []

        if ctx["summary"]:
            parts.append(f"[历史摘要] {ctx['summary']}")

        if ctx["short_term"]:
            lines = []
            for user_msg, ai_msg in ctx["short_term"][-3:]:
                lines.append(f"用户: {user_msg[:100]}")
                lines.append(f"助手: {ai_msg[:100]}")
            parts.append("[最近对话]\n" + "\n".join(lines))

        if ctx["long_term"]:
            lines = []
            for m in ctx["long_term"]:
                lines.append(f"- {m.key}: {m.value}")
            parts.append("[相关记忆]\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def clear_all(self):
        """清空全部记忆"""
        self.short_term = []
        self.summary = ""
        self._memories = {}
        self._faiss_index = None
        self._faiss_texts = []

        # 清理文件
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
        self._init_storage()

    def clear_short_term(self):
        """仅清空短期记忆"""
        self.short_term = []
        self.summary = ""
        self._save_short_term()

    # ================================================================
    # 属性
    # ================================================================

    @property
    def short_term_count(self) -> int:
        return len(self.short_term)

    @property
    def long_term_count(self) -> int:
        return len(self._memories)

    @property
    def has_summary(self) -> bool:
        return bool(self.summary)

    def set_llm(self, llm):
        """注入 LLM 实例（用于摘要压缩和记忆提取）"""
        self._llm = llm

    # ================================================================
    # 内部实现
    # ================================================================

    _llm = None  # 延迟注入

    def _init_storage(self):
        """初始化存储目录和加载已有数据"""
        os.makedirs(self.storage_dir, exist_ok=True)

        # 加载短期记忆
        short_path = os.path.join(self.storage_dir, "short_term.json")
        if os.path.exists(short_path):
            try:
                with open(short_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.short_term = [tuple(pair) for pair in data.get("history", [])]
                self.summary = data.get("summary", "")
            except Exception:
                pass

        # 加载长期记忆
        long_path = os.path.join(self.storage_dir, "long_term_data.json")
        if os.path.exists(long_path):
            try:
                with open(long_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memories = {
                    mid: Memory.from_dict(md) for mid, md in data.items()
                }
            except Exception:
                pass

        # 加载 FAISS 索引
        self._load_faiss()

    def _save_short_term(self):
        """持久化短期记忆"""
        short_path = os.path.join(self.storage_dir, "short_term.json")
        try:
            with open(short_path, "w", encoding="utf-8") as f:
                json.dump({
                    "history": self.short_term,
                    "summary": self.summary,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _save_long_term(self):
        """持久化长期记忆（仅 JSON，FAISS 单独存）"""
        long_path = os.path.join(self.storage_dir, "long_term_data.json")
        try:
            with open(long_path, "w", encoding="utf-8") as f:
                json.dump(
                    {mid: m.to_dict() for mid, m in self._memories.items()},
                    f, ensure_ascii=False, indent=2,
                )
        except Exception:
            pass

    # ---- FAISS 操作 ----

    def _get_faiss_dir(self):
        """FAISS 索引目录（避开中文路径）"""
        return os.path.join(self.storage_dir, "long_term_index")

    def _load_faiss(self):
        """加载 FAISS 索引"""
        faiss_dir = self._get_faiss_dir()
        faiss_file = os.path.join(faiss_dir, "index.faiss")
        if os.path.exists(faiss_file):
            try:
                from langchain_community.vectorstores import FAISS
                self._faiss_index = FAISS.load_local(
                    faiss_dir,
                    self._embedding,
                    allow_dangerous_deserialization=True,
                )
                # 同步文本列表
                if self._faiss_index and self._faiss_index.index:
                    self._faiss_texts = [
                        f"{m.key} {m.value}" for m in self._memories.values()
                    ][:self._faiss_index.index.ntotal]
            except Exception:
                self._faiss_index = None
                self._faiss_texts = []

    def _add_to_faiss(self, mem: Memory):
        """向 FAISS 索引添加一条记忆"""
        try:
            text = f"{mem.key} {mem.value}"
            from langchain_community.vectorstores import FAISS

            if self._faiss_index is None:
                # 创建初始索引
                self._faiss_index = FAISS.from_texts(
                    [text],
                    self._embedding,
                    metadatas=[{"memory_id": mem.id}],
                )
                self._faiss_texts = [text]
            else:
                self._faiss_index.add_texts(
                    [text],
                    metadatas=[{"memory_id": mem.id}],
                )
                self._faiss_texts.append(text)

            # 持久化
            self._save_faiss()
        except Exception:
            pass

    def _rebuild_faiss(self):
        """完全重建 FAISS 索引"""
        try:
            from langchain_community.vectorstores import FAISS

            if not self._memories:
                self._faiss_index = None
                self._faiss_texts = []
            else:
                texts = []
                metadatas = []
                for mem in self._memories.values():
                    texts.append(f"{mem.key} {mem.value}")
                    metadatas.append({"memory_id": mem.id})

                self._faiss_index = FAISS.from_texts(
                    texts,
                    self._embedding,
                    metadatas=metadatas,
                )
                self._faiss_texts = texts

            self._save_faiss()
        except Exception:
            self._faiss_index = None
            self._faiss_texts = []

    def _save_faiss(self):
        """持久化 FAISS 索引"""
        if self._faiss_index is None:
            return
        faiss_dir = self._get_faiss_dir()
        try:
            if os.path.exists(faiss_dir):
                shutil.rmtree(faiss_dir)
            os.makedirs(faiss_dir, exist_ok=True)
            self._faiss_index.save_local(faiss_dir)
        except Exception:
            pass
