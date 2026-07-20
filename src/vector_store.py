"""
文本分块 + 向量库构建
将文档切块、转为向量、存入 FAISS 向量库，并提供检索接口
"""

import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

import config


class VectorStore:
    """向量存储服务（基于 FAISS）"""

    def __init__(self):
        self.embedding = DashScopeEmbeddings(model=config.EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", ".", ",", " ", ""],
        )
        self._index_dir = config.VECTOR_DB_DIR
        self.store: FAISS = None
        self._load_or_init()

    def _load_or_init(self):
        """尝试从本地加载已有索引，否则创建空索引"""
        if os.path.exists(self._index_dir) and os.path.isdir(self._index_dir):
            faiss_file = os.path.join(self._index_dir, "index.faiss")
            if os.path.exists(faiss_file):
                try:
                    self.store = FAISS.load_local(
                        self._index_dir,
                        self.embedding,
                        allow_dangerous_deserialization=True,
                    )
                    return
                except Exception as e:
                    print(f"  [WARN] FAISS 索引加载失败: {e}，将重建")

        # 创建空索引（需至少一条数据，用占位文本初始化后清空）
        self.store = FAISS.from_texts(
            ["__init_placeholder__"], self.embedding
        )
        # 删除占位数据（通过重建）
        self._clear_index()

    def _clear_index(self):
        """清空索引"""
        self.store = FAISS.from_texts(
            ["__init_placeholder__"], self.embedding
        )
        # FAISS 不支持直接删除，创建空索引即可
        self.store = None
        # 用空列表重建不行，保留 None 状态

    @property
    def count(self) -> int:
        """返回索引中的向量数量"""
        if self.store is None or self.store.index is None:
            return 0
        try:
            return self.store.index.ntotal
        except Exception:
            return 0

    def build_from_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分块后存入向量库（完全重建）"""
        # 分块
        chunks = self.splitter.split_documents(documents)
        print(f"  分块后共 {len(chunks)} 个片段")

        # 提取文本
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        # 构建新索引
        self.store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding,
            metadatas=metadatas,
        )

        # 持久化到本地
        if os.path.exists(self._index_dir):
            shutil.rmtree(self._index_dir)
        os.makedirs(self._index_dir, exist_ok=True)
        self.store.save_local(self._index_dir)
        print(f"  [OK] 已存入 {len(chunks)} 个向量片段")

        return chunks

    def search(self, query: str, k: int = None) -> List[Document]:
        """检索最相关的文档片段"""
        if k is None:
            k = config.RETRIEVER_K
        if self.store is None:
            return []
        return self.store.similarity_search(query, k=k)
