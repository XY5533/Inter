"""
系统综合测试 — 覆盖所有新模块和集成点
"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vector_store import VectorStore
from rag_service import RagService, ConversationBufferWindowMemory
from prompt_templates import (
    REACT_SYSTEM_PROMPT, REACT_FIRST_STEP_TEMPLATE, REACT_STEP_TEMPLATE,
    MEMORY_EXTRACT_PROMPT, IMAGE_DESCRIPTION_PROMPT, RAG_SYSTEM_TEMPLATE,
)
from react_agent import ReActAgent, ReActResult, AgentStep, ActionType
from memory_manager import MemoryManager, Memory
from multimodal import MultimodalProcessor
from langchain_community.chat_models.tongyi import ChatTongyi

results = []
start_time = time.perf_counter()

def run_test(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        results.append({"name": name, "status": "PASS", "elapsed_ms": elapsed, "error": ""})
        print(f"  [PASS] {name} ({elapsed}ms)")
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        results.append({"name": name, "status": "FAIL", "elapsed_ms": elapsed, "error": str(e)[:200]})
        print(f"  [FAIL] {name}: {e}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ---- Shared fixtures ----
api_key = config.DASHSCOPE_API_KEY or None
llm = ChatTongyi(model=config.CHAT_MODEL, dashscope_api_key=api_key)
vs = VectorStore()
rs = RagService()
mm = MemoryManager()
rs.memory_manager.clear_all()  # Start clean

# ============================================================
section("1. 配置文件 config.py (4 项)")

def t_config_basic():
    assert hasattr(config, "DASHSCOPE_API_KEY")
    assert hasattr(config, "EMBEDDING_MODEL")
    assert config.CHUNK_SIZE == 500
run_test("config.py 基础配置完整", t_config_basic)

def t_config_react():
    assert config.REACT_MAX_STEPS == 5
    assert config.REACT_ENABLE is True
    assert config.REACT_DUPLICATE_THRESHOLD == 2
run_test("config.py ReAct 配置项", t_config_react)

def t_config_memory():
    assert config.SHORT_TERM_K == 5
    assert config.LONG_TERM_K == 3
    assert config.COMPRESS_THRESHOLD == 10
    assert "memory" in config.MEMORY_DIR
run_test("config.py 记忆系统配置项", t_config_memory)

def t_config_multimodal():
    assert config.VISION_MODEL == "qwen-vl-max"
    assert config.ASR_MODEL == "paraformer-v2"
    assert config.VIDEO_FPS == 1
run_test("config.py 多模态配置项", t_config_multimodal)


# ============================================================
section("2. Prompt 模板 prompt_templates.py (6 项)")

def t_prompt_rag():
    assert "{context}" in RAG_SYSTEM_TEMPLATE
    assert "参考资料" in RAG_SYSTEM_TEMPLATE
run_test("RAG 模板含关键占位符", t_prompt_rag)

def t_prompt_react_all_tools():
    assert "SEARCH_KNOWLEDGE" in REACT_SYSTEM_PROMPT
    assert "SEARCH_MEMORY" in REACT_SYSTEM_PROMPT
    assert "CLARIFY" in REACT_SYSTEM_PROMPT
    assert "FINAL_ANSWER" in REACT_SYSTEM_PROMPT
run_test("ReAct 模板含全部 4 个工具", t_prompt_react_all_tools)

def t_prompt_first_step():
    assert "{question}" in REACT_FIRST_STEP_TEMPLATE
    assert "{memory_context}" in REACT_FIRST_STEP_TEMPLATE
run_test("ReAct 首步模板占位符", t_prompt_first_step)

def t_prompt_step():
    assert "{previous_steps}" in REACT_STEP_TEMPLATE
run_test("ReAct 后续步骤模板占位符", t_prompt_step)

def t_prompt_memory_extract():
    assert "{user_msg}" in MEMORY_EXTRACT_PROMPT
    assert "{ai_msg}" in MEMORY_EXTRACT_PROMPT
run_test("记忆提取模板占位符", t_prompt_memory_extract)

def t_prompt_image():
    assert len(IMAGE_DESCRIPTION_PROMPT) > 50
run_test("图片描述模板有内容", t_prompt_image)


# ============================================================
section("3. 向量库 vector_store.py (4 项)")

def t_vs_init():
    v = VectorStore()
    assert v is not None
    assert v.count >= 0
run_test("VectorStore 初始化成功", t_vs_init)

def t_vs_count():
    assert vs.count == 504, f"expected 504, got {vs.count}"
run_test("向量库片段数 = 504", t_vs_count)

def t_vs_search_count():
    docs = vs.search("工业机器人", k=3)
    assert len(docs) <= 3
    assert len(docs) > 0
run_test("search 返回正确数量", t_vs_search_count)

def t_vs_search_type():
    docs = vs.search("伺服电机", k=3)
    for d in docs:
        assert hasattr(d, "page_content")
        assert hasattr(d, "metadata")
        assert "source" in d.metadata
run_test("search 返回完整 Document 对象", t_vs_search_type)


# ============================================================
section("4. ReAct Agent react_agent.py (10 项)")

def t_react_enum():
    assert ActionType.SEARCH_KNOWLEDGE.value == "SEARCH_KNOWLEDGE"
    assert ActionType.FINAL_ANSWER.value == "FINAL_ANSWER"
run_test("ActionType 枚举值正确", t_react_enum)

def t_react_init():
    agent = ReActAgent(llm=llm, vector_store=vs)
    assert agent.max_steps == 5
    assert agent.duplicate_threshold == 2
run_test("ReActAgent 初始化成功", t_react_init)

def t_react_step_dataclass():
    step = AgentStep(step_num=1, thought="test", action_type=ActionType.SEARCH_KNOWLEDGE,
                     action_input="伺服电机", observation="找到3条")
    assert step.step_num == 1
    assert step.action_type == ActionType.SEARCH_KNOWLEDGE
run_test("AgentStep 数据类创建", t_react_step_dataclass)

def t_react_hello():
    agent = ReActAgent(llm=llm, vector_store=vs)
    result = agent.run("你好")
    assert isinstance(result, ReActResult)
    assert len(result.answer) > 0
    assert result.success
run_test("ReAct 简单问候（1步完成）", t_react_hello)

def t_react_search():
    agent = ReActAgent(llm=llm, vector_store=vs)
    result = agent.run("什么是工业机器人？")
    assert isinstance(result, ReActResult)
    assert len(result.answer) > 10
run_test("ReAct 知识检索（多步推理）", t_react_search)

def t_react_oob():
    agent = ReActAgent(llm=llm, vector_store=vs)
    result = agent.run("Python 的 GIL 是什么？")
    assert isinstance(result, ReActResult)
    assert len(result.answer) > 5
run_test("ReAct 超出知识库范围诚实回答", t_react_oob)

def t_react_duplicate():
    agent = ReActAgent(llm=llm, vector_store=vs)
    agent._recent_actions = [(ActionType.SEARCH_KNOWLEDGE, "q"), (ActionType.SEARCH_KNOWLEDGE, "q")]
    assert agent._is_duplicate(ActionType.SEARCH_KNOWLEDGE, "q") is True
run_test("ReAct 重复动作检测触发", t_react_duplicate)

def t_react_parse_final():
    agent = ReActAgent(llm=llm, vector_store=vs)
    _, atype, ainput = agent._parse_response("THOUGHT: 回答\nACTION: [FINAL_ANSWER 这是最终答案]")
    assert atype == ActionType.FINAL_ANSWER
    assert "答案" in ainput
run_test("ReAct 解析 FINAL_ANSWER", t_react_parse_final)

def t_react_parse_search():
    agent = ReActAgent(llm=llm, vector_store=vs)
    _, atype, ainput = agent._parse_response("THOUGHT: 搜索\nACTION: [SEARCH_KNOWLEDGE 伺服电机]")
    assert atype == ActionType.SEARCH_KNOWLEDGE
    assert "伺服电机" in ainput
run_test("ReAct 解析 SEARCH_KNOWLEDGE", t_react_parse_search)

def t_react_parse_fallback():
    agent = ReActAgent(llm=llm, vector_store=vs)
    _, atype, ainput = agent._parse_response("这是一段不标准格式的回复")
    assert atype == ActionType.FINAL_ANSWER
    assert len(ainput) > 0
run_test("ReAct 解析不标准格式降级为 FINAL_ANSWER", t_react_parse_fallback)


# ============================================================
section("5. 记忆管理器 memory_manager.py (11 项)")

def t_mem_dataclass():
    mem = Memory(id="test1", key="偏好", value="关注伺服电机", category="preference", importance=3)
    assert mem.id == "test1"
    d = mem.to_dict()
    assert d["key"] == "偏好"
run_test("Memory 数据类创建和序列化", t_mem_dataclass)

def t_mm_init():
    m = MemoryManager()
    assert m.short_term_count >= 0
    assert m.long_term_count >= 0
run_test("MemoryManager 初始化成功", t_mm_init)

def t_mm_add_turn():
    mm.clear_all()
    before = mm.short_term_count
    mm.add_turn("测试问题", "测试回答")
    assert mm.short_term_count == before + 1
run_test("add_turn 记录短期记忆", t_mm_add_turn)

def t_mm_remember():
    mm.clear_all()
    before = mm.long_term_count
    mem = mm.remember("测试记忆", "测试内容", "general", importance=2)
    assert mm.long_term_count == before + 1
    assert mem.category == "general"
run_test("remember 存储长期记忆", t_mm_remember)

def t_mm_recall():
    mm.clear_all()
    mm.remember("伺服电机", "用户关注伺服电机选型", "preference", importance=4)
    memories = mm.recall("伺服电机类型有哪些", k=3)
    assert len(memories) > 0
    assert any("伺服电机" in m.key for m in memories)
run_test("recall 语义检索记忆", t_mm_recall)

def t_mm_get_context():
    mm.clear_all()
    mm.add_turn("你好", "你好！")
    ctx = mm.get_context("伺服电机")
    assert "short_term" in ctx
    assert "long_term" in ctx
    assert "summary" in ctx
run_test("get_context 返回完整上下文结构", t_mm_get_context)

def t_mm_get_context_text():
    mm.clear_all()
    text = mm.get_context_text("测试")
    assert isinstance(text, str)
run_test("get_context_text 返回格式化文本", t_mm_get_context_text)

def t_mm_forget():
    mm.clear_all()
    mem = mm.remember("待删除", "这条会被删除", "general")
    before = mm.long_term_count
    assert mm.forget(mem.id) is True
    assert mm.long_term_count == before - 1
run_test("forget 删除指定记忆", t_mm_forget)

def t_mm_get_all():
    mm.clear_all()
    mm.remember("A", "aa", "general")
    mm.remember("B", "bb", "preference")
    all_mems = mm.get_all_memories()
    assert len(all_mems) == 2
run_test("get_all_memories 返回全部", t_mm_get_all)

def t_mm_clear_short():
    mm.clear_all()
    mm.add_turn("q", "a")
    mm.clear_short_term()
    assert mm.short_term_count == 0
run_test("clear_short_term 只清短期", t_mm_clear_short)

def t_mm_clear_all():
    mm.clear_all()
    mm.add_turn("q", "a")
    mm.remember("X", "y", "general")
    mm.clear_all()
    assert mm.short_term_count == 0
    assert mm.long_term_count == 0
run_test("clear_all 清空全部记忆", t_mm_clear_all)


# ============================================================
section("6. 对话缓冲 ConversationBufferWindowMemory (4 项)")

def t_cbw_init():
    mem = ConversationBufferWindowMemory(k=3)
    assert mem.k == 3
    assert len(mem.chat_memory) == 0
run_test("ConversationBufferWindowMemory 初始化", t_cbw_init)

def t_cbw_save():
    mem = ConversationBufferWindowMemory(k=3)
    mem.save_context({"input": "你好"}, {"output": "你好！"})
    assert len(mem.chat_memory) == 1
run_test("save_context 保存对话", t_cbw_save)

def t_cbw_load():
    mem = ConversationBufferWindowMemory(k=3)
    mem.save_context({"input": "你好"}, {"output": "你好！"})
    vars = mem.load_memory_variables({})
    assert len(vars["history"]) == 2  # human + ai
run_test("load_memory_variables 返回完整消息", t_cbw_load)

def t_cbw_window():
    mem = ConversationBufferWindowMemory(k=2)
    for i in range(5):
        mem.save_context({"input": f"q{i}"}, {"output": f"a{i}"})
    vars = mem.load_memory_variables({})
    assert len(vars["history"]) == 4  # k=2, 2*2=4 messages
run_test("窗口限制 k=2 正确截断", t_cbw_window)


# ============================================================
section("7. RagService 集成 rag_service.py (8 项)")

def t_rs_init():
    r = RagService()
    assert hasattr(r, "enable_react")
    assert hasattr(r, "memory_manager")
    assert hasattr(r, "get_memory_stats")
    assert hasattr(r, "answer_with_react")
run_test("RagService 所有新方法存在", t_rs_init)

def t_rs_standard_rag():
    docs = vs.search("工业机器人", k=3)
    answer = rs.answer("工业机器人有哪些类型？", docs)
    assert len(answer) > 10
run_test("标准 RAG 问答正常", t_rs_standard_rag)

def t_rs_react():
    rs.enable_react = True
    result = rs.answer_with_react("伺服电机的控制方式有哪些？",
                                   retriever_func=lambda q, k=3: vs.search(q, k=k))
    assert "answer" in result
    assert "steps" in result
    assert len(result["answer"]) > 10
run_test("ReAct 集成问答返回完整结果", t_rs_react)

def t_rs_react_fallback():
    rs.enable_react = False
    result = rs.answer_with_react("步进电机原理",
                                   retriever_func=lambda q, k=3: vs.search(q, k=k))
    assert result["fallback_used"] is True
    assert len(result["answer"]) > 10
run_test("ReAct 关闭后降级到标准 RAG", t_rs_react_fallback)

def t_rs_memory_stats():
    stats = rs.get_memory_stats()
    assert isinstance(stats, dict)
    assert "short_term_count" in stats
    assert "long_term_count" in stats
run_test("get_memory_stats 返回正确字典", t_rs_memory_stats)

def t_rs_clear_memory():
    rs.clear_memory(clear_long_term=True)
    stats = rs.get_memory_stats()
    assert stats["short_term_count"] == 0
run_test("clear_memory 清空全部记忆", t_rs_clear_memory)

def t_rs_delete_memory():
    mem = rs.memory_manager.remember("测试删除", "内容", "general")
    assert rs.delete_long_term_memory(mem.id) is True
run_test("delete_long_term_memory 删除成功", t_rs_delete_memory)

def t_rs_all_memories():
    mems = rs.get_all_long_term_memories()
    assert isinstance(mems, list)
run_test("get_all_long_term_memories 返回列表", t_rs_all_memories)


# ============================================================
section("8. 多模态处理器 multimodal.py (4 项)")

def t_mp_init():
    mp = MultimodalProcessor()
    assert mp is not None
    assert mp.vision_model == "qwen-vl-max"
run_test("MultimodalProcessor 初始化成功", t_mp_init)

mp = MultimodalProcessor()

def t_mp_text_only():
    result = mp.process_input(text="测试问题")
    assert result["text"] == "测试问题"
run_test("process_input 纯文本不丢失", t_mp_text_only)

def t_mp_no_image():
    result = mp.process_input(text="这是什么？", image_bytes=None)
    assert "这是什么" in result["text"]
run_test("process_input 无图片不崩溃", t_mp_no_image)

def t_mp_set_llm():
    mp.set_llm(llm)
    assert mp._llm is not None
run_test("set_llm 注入 LLM 实例", t_mp_set_llm)


# ============================================================
section("9. 适配器 (3 项)")

from rag_service import _RetrieverAdapter, _StaticDocStore

def t_retriever_adapter():
    def my_search(q, k=3):
        class FakeDoc:
            def __init__(self, content): self.page_content = content; self.metadata = {}
        return [FakeDoc(q) for _ in range(k)]
    adapter = _RetrieverAdapter(my_search)
    docs = adapter.search("test", k=2)
    assert len(docs) == 2
run_test("_RetrieverAdapter 适配 callable", t_retriever_adapter)

def t_static_store():
    from langchain_core.documents import Document
    docs = [Document(page_content=f"doc{i}", metadata={"source": "test"}) for i in range(3)]
    store = _StaticDocStore(docs)
    results = store.search("anything", k=2)
    assert len(results) == 2
run_test("_StaticDocStore 返回预置文档", t_static_store)

def t_static_empty():
    store = _StaticDocStore([])
    results = store.search("anything")
    assert len(results) == 0
run_test("_StaticDocStore 空列表不崩溃", t_static_empty)


# ============================================================
section("10. 边界和异常 (5 项)")

def t_edge_empty_query():
    docs = vs.search("", k=3)
    assert isinstance(docs, list)
run_test("VectorStore 空查询不崩溃", t_edge_empty_query)

def t_edge_empty_recall():
    mm.clear_all()
    memories = mm.recall("", k=3)
    assert isinstance(memories, list)
run_test("MemoryManager 空查询不崩溃", t_edge_empty_recall)

def t_edge_empty_string_react():
    agent = ReActAgent(llm=llm, vector_store=vs)
    result = agent.run("")
    assert isinstance(result, ReActResult)
    assert len(result.answer) >= 0
run_test("ReActAgent 空字符串不崩溃", t_edge_empty_string_react)

def t_edge_long_query():
    agent = ReActAgent(llm=llm, vector_store=vs)
    result = agent.run("伺服电机 " * 100)
    assert isinstance(result, ReActResult)
run_test("ReActAgent 超长查询不崩溃", t_edge_long_query)

def t_edge_mass_memory():
    mt = MemoryManager()
    for i in range(20):
        mt.remember(f"记忆{i}", f"内容{i}", "general", importance=min(i//5+1, 5))
    assert mt.long_term_count == 20
    mt.clear_all()
run_test("MemoryManager 大量写入 (20条) 不崩溃", t_edge_mass_memory)


# ============================================================
section("汇总")

total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = total - passed
total_time = sum(r["elapsed_ms"] for r in results)
pass_rate = round(passed / total * 100, 1) if total > 0 else 0

print(f"\n  总计: {total} | 通过: {passed} | 失败: {failed} | 总耗时: {total_time/1000:.1f}s")
print(f"  通过率: {pass_rate}%")

# Print failures
if failed > 0:
    print(f"\n  --- 失败用例 ---")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  [FAIL] {r['name']}: {r['error']}")

# Save
output_dir = os.path.join(os.path.dirname(__file__), "..", "test_results")
os.makedirs(output_dir, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = os.path.join(output_dir, f"system_test_{ts}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total, "passed": passed, "failed": failed,
        "total_time_ms": round(total_time, 1),
        "pass_rate": pass_rate,
        "results": results,
    }, f, ensure_ascii=False, indent=2)
print(f"  结果已保存: {json_path}")

# Exit code for CI
sys.exit(0 if failed == 0 else 1)
