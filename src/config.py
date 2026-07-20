"""
全局配置文件
"""
import os

# ---------- API密钥 ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# ---------- 模型配置 ----------
EMBEDDING_MODEL = "text-embedding-v4"   # 向量嵌入模型
CHAT_MODEL = "qwen3-max"                # 对话模型

# ---------- 路径配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")   # 存放企业知识库

# FAISS C++ 库在 Windows 上不支持 Unicode 路径，使用不含中文的路径
_VECTOR_DIR_NAME = "vector_db"
try:
    # 优先使用 TEMP 目录（路径通常不含中文）
    VECTOR_DB_DIR = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")),
                                 "rag_agent_vector_db")
except Exception:
    VECTOR_DB_DIR = os.path.join(BASE_DIR, _VECTOR_DIR_NAME)

# ---------- 文本分块配置 ----------
CHUNK_SIZE = 500        # 每块最大字符数
CHUNK_OVERLAP = 50      # 块间重叠字符数

# ---------- 检索配置 ----------
RETRIEVER_K = 3         # 每次检索返回的片段数

# ---------- ReAct 框架配置 ----------
REACT_MAX_STEPS = 5             # ReAct 最大推理步数
REACT_ENABLE = True             # 是否默认启用 ReAct 模式
REACT_DUPLICATE_THRESHOLD = 2   # 连续重复动作阈值，超过触发降级

# ---------- 记忆系统配置 ----------
SHORT_TERM_K = 5                # 短期记忆保留的对话轮数
LONG_TERM_K = 3                 # 长期记忆每次检索条数
COMPRESS_THRESHOLD = 10         # 对话超过 N 轮时触发摘要压缩
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

# ---------- 多模态配置 ----------
VISION_MODEL = "qwen-vl-max"    # 多模态视觉模型 (DashScope)
ASR_MODEL = "paraformer-v2"     # 语音识别模型 (DashScope)
VIDEO_FPS = 1                   # 视频抽帧率（帧/秒）
AUDIO_MAX_DURATION = 60         # 音频最长时长（秒）
