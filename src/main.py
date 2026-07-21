# -*- coding: utf-8 -*-
"""
IntelliRAG — 知识库问答平台
整合 RAG 检索、ReAct 推理、记忆管理、多模态输入
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ---- 页面全局配置 ----
st.set_page_config(
    page_title="IntelliRAG 知识库问答平台",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
# 防御：如果 Streamlit 缓存了旧版 config 模块，强制重新加载
if not hasattr(config, "REACT_MAX_STEPS"):
    import importlib
    importlib.reload(config)
from knowledge_base import load_knowledge
from vector_store import VectorStore
from rag_service import RagService
from multimodal import MultimodalProcessor


def _init_services():
    vs = VectorStore()
    rs = RagService()
    return vs, rs


vector_store, rag_service = _init_services()

# ---- 侧边栏 ----
with st.sidebar:
    # 品牌标识
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0;">
        <h2 style="margin:0;font-weight:700;">IntelliRAG</h2>
        <p style="color:#888;font-size:0.82rem;margin:0;">知识库问答平台</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # 页面路由
    page = st.radio(
        "工作区",
        ["💬 智能问答", "📊 评估测试"],
        key="page_nav",
        label_visibility="collapsed",
    )

    if "智能问答" in page:
        # -- Agent 引擎 --
        with st.expander("⚙️ Agent 引擎", expanded=True):
            react_enabled = st.toggle(
                "ReAct 推理模式",
                value=getattr(config, "REACT_ENABLE", True),
                help="开启后 Agent 逐步推理后回答；关闭使用标准 RAG",
            )
            rag_service.enable_react = react_enabled
            if react_enabled:
                st.caption(f"最大推理步数: {getattr(config, 'REACT_MAX_STEPS', 5)}")
                st.caption(f"重复检测阈值: {getattr(config, 'REACT_DUPLICATE_THRESHOLD', 2)}")

        # -- 知识库 --
        with st.expander("📚 知识库", expanded=False):
            try:
                count = vector_store.count
            except:
                count = 0
            st.metric("向量片段", f"{count:,}")
            if st.button("🔨 重建索引", use_container_width=True):
                with st.status("构建中...", expanded=True) as status:
                    st.write("读取文档...")
                    docs = load_knowledge()
                    if docs:
                        st.write(f"已加载 {len(docs)} 个文档，分块向量化...")
                        vector_store.build_from_documents(docs)
                        status.update(label="构建完成", state="complete")
                    else:
                        status.update(label="目录为空", state="error")

        # -- 系统状态 --
        with st.expander("📡 系统状态", expanded=False):
            mem_stats = rag_service.get_memory_stats()
            c1, c2 = st.columns(2)
            c1.metric("短期记忆", f"{mem_stats['short_term_count']} 轮")
            c2.metric("长期记忆", f"{mem_stats['long_term_count']} 条")
            if mem_stats["has_summary"]:
                st.caption("对话摘要: 已压缩")
            st.divider()
            st.caption(f"对话模型: `{config.CHAT_MODEL}`")
            st.caption(f"嵌入模型: `{config.EMBEDDING_MODEL}`")
            st.caption(f"检索配置: Top-{config.RETRIEVER_K} | Chunk-{config.CHUNK_SIZE}")

        # -- 记忆管理 --
        with st.expander("🧠 记忆管理", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("清空短期", use_container_width=True, key="clear_short"):
                    rag_service.clear_memory(clear_long_term=False)
                    st.session_state.messages = []
                    st.rerun()
            with c2:
                if st.button("清空全部", use_container_width=True, key="clear_all"):
                    rag_service.clear_memory(clear_long_term=True)
                    st.session_state.messages = []
                    st.rerun()

            all_mems = rag_service.get_all_long_term_memories()
            if all_mems:
                st.caption(f"共 {len(all_mems)} 条长期记忆")
                for mem in all_mems[:10]:
                    cat_icon = {"preference": "⭐", "identity": "👤",
                                "correction": "✏️", "faq": "📌"}.get(mem.category, "📝")
                    cols = st.columns([5, 1])
                    cols[0].caption(f"{cat_icon} **{mem.key}**  \n{mem.value[:40]}...")
                    if cols[1].button("✕", key=f"del_{mem.id}", help="删除此记忆"):
                        rag_service.delete_long_term_memory(mem.id)
                        st.rerun()
                if len(all_mems) > 10:
                    st.caption(f"… 还有 {len(all_mems)-10} 条")
            else:
                st.caption("暂无长期记忆")

        st.divider()
        st.caption("© 2026 IntelliRAG Platform")
    else:
        st.caption("📊 评估测试工作区")

# ---- 评估测试页面 ----
if "评估测试" in page:
    import datetime
    import json as json_mod
    from tests.evaluator import Evaluator
    from tests.report_generator import generate_single_report, load_results

    st.markdown("## 📊 评估测试中心")
    st.caption("RAG 问答质量评估 | 回归对比 | 系统健康检查")

    if not config.DASHSCOPE_API_KEY:
        st.error("请先配置 DASHSCOPE_API_KEY 环境变量")
        st.stop()

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = Evaluator()
        st.session_state.evaluator.load_test_cases()
    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 测试用例", "▶️ 执行测试", "📄 测试报告", "🔄 回归测试", "🔬 系统测试"
    ])

    with tab1:
        cases = st.session_state.evaluator.test_cases
        st.subheader("测试用例集 (%d 条)" % len(cases))
        cats = sorted(set(c.category for c in cases))
        sel_cat = st.selectbox("筛选分类", ["全部"] + cats)
        filtered = cases if sel_cat == "全部" else [c for c in cases if c.category == sel_cat]
        data = []
        for c in filtered:
            kw = ", ".join(c.expected_keywords[:3]) if c.expected_keywords else "(兜底)"
            data.append({
                "ID": c.id, "分类": c.category,
                "问题": c.question[:40],
                "期望关键词": kw, "说明": c.description[:25],
            })
        st.dataframe(data, use_container_width=True, hide_index=True)

        with st.expander("编辑测试用例 (JSON)"):
            tc_path = os.path.join(os.path.dirname(__file__), "tests", "test_cases.json")
            try:
                with open(tc_path, "r", encoding="utf-8") as f:
                    curr = json_mod.load(f)
                edited = st.text_area("JSON内容", json_mod.dumps(curr, ensure_ascii=False, indent=2), height=400)
                if st.button("保存修改"):
                    with open(tc_path, "w", encoding="utf-8") as f:
                        f.write(edited)
                    st.success("已保存")
            except Exception as e:
                st.error(str(e))

    with tab2:
        st.subheader("批量执行测试")
        rebuild = st.checkbox("强制重建知识库", value=False)
        st.caption(f"模型: {config.CHAT_MODEL} / {config.EMBEDDING_MODEL}")

        if st.button("开始测试", type="primary", use_container_width=True):
            progress = st.progress(0, text="准备中...")
            status_text = st.empty()
            results = []
            total = len(st.session_state.evaluator.test_cases)
            for i, case in enumerate(st.session_state.evaluator.test_cases):
                status_text.text("[%d/%d] %s: %s..." % (i+1, total, case.id, case.question[:25]))
                result = st.session_state.evaluator.evaluate_one(case, rebuild_kb=(rebuild and i == 0))
                results.append(result)
                progress.progress((i + 1) / total)
            st.session_state.test_results = results
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rdir = os.path.join(os.path.dirname(__file__), "test_results")
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, "results_%s.json" % ts), "w", encoding="utf-8") as f:
                json_mod.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
            passed = sum(1 for r in results if r.passed)
            avg_lat = sum(r.latency_ms for r in results) / total if total else 0
            st.success("测试完成! 通过 %d/%d, 平均 %.0fms" % (passed, total, avg_lat))
            progress.empty()
            status_text.empty()
            st.rerun()

        if st.session_state.test_results:
            results = st.session_state.test_results
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            st.divider()
            avg_lat = sum(r.latency_ms for r in results) / total if total else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("平均耗时", "%.0fms" % avg_lat)
            c2.metric("总耗时", "%.1fs" % (sum(r.latency_ms for r in results)/1000))
            c3.metric("通过数", passed)
            c4.metric("失败数", total - passed)
            rows = []
            for r in results:
                kw_rate = "%.0f%%" % (r.keyword_hit_rate*100) if r.expected_keywords else "-"
                rows.append({
                    "状态": "PASS" if r.passed else "FAIL",
                    "ID": r.case_id,
                    "问题": r.question[:30],
                    "耗时": "%.0fms" % r.latency_ms,
                    "命中率": kw_rate,
                    "检索": r.retrieved_docs,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab3:
        if st.session_state.test_results:
            if st.button("生成Markdown报告", use_container_width=True):
                rdir = os.path.join(os.path.dirname(__file__), "test_results")
                report = generate_single_report([r.to_dict() for r in st.session_state.test_results], output_dir=rdir)
                st.markdown(report)
        else:
            st.info("请先执行测试")

    with tab4:
        st.subheader("回归对比")
        bl = st.text_input("基线标签", "旧版本")
        cl = st.text_input("当前标签", "新版本")
        rdir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(rdir, exist_ok=True)
        existing = sorted([f for f in os.listdir(rdir) if f.startswith("results_") and f.endswith(".json")])
        if len(existing) >= 2:
            bf = st.selectbox("基线", existing, index=len(existing)-2)
            cf = st.selectbox("当前", existing, index=len(existing)-1)
            if st.button("生成对比报告", use_container_width=True):
                bd = load_results(os.path.join(rdir, bf))
                cd = load_results(os.path.join(rdir, cf))
                def mk_sum(data, label):
                    p = sum(1 for r in data if r["passed"])
                    a = sum(r["latency_ms"] for r in data) / len(data) if data else 0
                    return {"label": label, "timestamp": "now", "total": len(data),
                            "passed": p, "failed_count": len(data)-p,
                            "failed_cases": [r for r in data if not r["passed"]],
                            "pass_rate": round(p/len(data)*100,1) if data else 0,
                            "avg_latency_ms": round(a,1),
                            "total_time_ms": round(sum(r["latency_ms"] for r in data),1),
                            "results": data}
                from tests.report_generator import generate_regression_report
                report = generate_regression_report(mk_sum(bd, bl), mk_sum(cd, cl), output_dir=rdir)
                st.markdown(report)
        else:
            st.info("需要至少2次测试结果 (当前 %d 次)" % len(existing))

    with tab5:
        st.subheader("系统综合测试")
        st.caption("覆盖配置、Prompt模板、向量库、ReAct Agent、记忆管理器、对话缓冲、RagService集成、多模态处理器、适配器和边界异常共 10 个模块，59 项测试")

        if "sys_test_results" not in st.session_state:
            st.session_state.sys_test_results = None
        if "sys_test_running" not in st.session_state:
            st.session_state.sys_test_running = False

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("执行系统测试", type="primary", use_container_width=True,
                         disabled=st.session_state.sys_test_running):
                st.session_state.sys_test_running = True
                st.rerun()

        with col2:
            rdir = os.path.join(os.path.dirname(__file__), "test_results")
            os.makedirs(rdir, exist_ok=True)
            existing_sys = sorted(
                [f for f in os.listdir(rdir) if f.startswith("system_test_") and f.endswith(".json")],
                reverse=True,
            )
            if existing_sys:
                selected_sys = st.selectbox("加载历史结果", ["-- 选择 --"] + existing_sys)
                if selected_sys != "-- 选择 --":
                    with open(os.path.join(rdir, selected_sys), "r", encoding="utf-8") as f:
                        st.session_state.sys_test_results = json_mod.load(f)

        # ---- 执行测试（subprocess 方式，避免 import 副作用） ----
        if st.session_state.sys_test_running:
            st.divider()
            st.info("正在运行系统综合测试（约需 60-90 秒）...")

            test_script = os.path.join(os.path.dirname(__file__), "tests", "system_test.py")
            python_exe = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")

            import subprocess

            output_lines = []
            status_container = st.empty()

            proc = subprocess.Popen(
                [python_exe, test_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(__file__),
            )

            # 实时读取输出
            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if line:
                    output_lines.append(line)
                    # 提取 [PASS] / [FAIL] 行展示进度
                    if "[PASS]" in line or "[FAIL]" in line:
                        status_container.text(line)
            proc.wait()

            status_container.empty()

            # 从输出的 JSON 路径加载结果
            json_path = None
            for line in output_lines:
                if "__JSON_PATH__:" in line:
                    json_path = line.split("__JSON_PATH__:")[1].strip()
                    break

            if json_path and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    st.session_state.sys_test_results = json_mod.load(f)
                st.success("测试完成！")
            else:
                st.error("测试执行失败，未能生成结果文件")
                st.code("\n".join(output_lines[-30:]))

            st.session_state.sys_test_running = False
            st.rerun()

        # ---- 显示结果 ----
        if st.session_state.sys_test_results:
            data = st.session_state.sys_test_results
            st.divider()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("总用例", data["total"])
            c2.metric("通过", data["passed"])
            c3.metric("失败", data["failed"])
            c4.metric("通过率", f"{data['pass_rate']}%")
            total_time_s = round(data.get("total_time_ms", 0) / 1000, 1)
            c5.metric("总耗时", f"{total_time_s}s")

            passed_rate = data["pass_rate"] / 100
            color = "green" if passed_rate >= 0.9 else ("orange" if passed_rate >= 0.7 else "red")
            st.progress(passed_rate, text=f"通过率 {data['pass_rate']}%")

            failed_items = [r for r in data["results"] if r["status"] == "FAIL"]
            if failed_items:
                st.error(f"失败 {len(failed_items)} 项:")
                for r in failed_items:
                    st.text(f"  - {r['name']}: {r['error'][:120]}")

            with st.expander("查看全部测试结果", expanded=False):
                rows = []
                for r in data["results"]:
                    rows.append({
                        "状态": "PASS" if r["status"] == "PASS" else "FAIL",
                        "测试项": r["name"],
                        "耗时": f"{r['elapsed_ms']:.0f}ms" if r["elapsed_ms"] >= 10 else "<10ms",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

            st.caption(f"测试时间: {data.get('timestamp', '未知')}")

    st.stop()

# ---- 智能问答页面 ----
st.markdown("## 💬 智能问答")
st.caption("基于 RAG + ReAct 的知识库检索与推理")

# 初始化多模态处理器
if "multimodal_processor" not in st.session_state:
    st.session_state.multimodal_processor = MultimodalProcessor()
    st.session_state.multimodal_processor.set_llm(rag_service.llm)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.kb_checked = False
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_image_text" not in st.session_state:
    st.session_state.pending_image_text = ""
if "pending_audio_text" not in st.session_state:
    st.session_state.pending_audio_text = ""
if "pending_video_text" not in st.session_state:
    st.session_state.pending_video_text = ""

if not st.session_state.kb_checked:
    try:
        count = vector_store.count
    except:
        count = 0
    if count == 0:
        with st.spinner("首次运行，自动加载知识库..."):
            docs = load_knowledge()
            if docs:
                vector_store.build_from_documents(docs)
                st.success("知识库自动构建完成！")
            else:
                st.info("knowledge_base 目录为空")
    st.session_state.kb_checked = True

# ---- 多模态输入区域 ----
col_voice, col_image, col_video = st.columns(3)
with col_voice:
    with st.popover("🎤 语音输入"):
        audio_file = st.file_uploader("上传音频", type=["wav", "mp3", "m4a"],
                                      key="audio_upload", label_visibility="collapsed")
        if audio_file:
            with st.spinner("语音识别中..."):
                audio_bytes = audio_file.read()
                transcript = st.session_state.multimodal_processor.transcribe_audio(audio_bytes)
                st.session_state.pending_audio_text = transcript
            if transcript and not transcript.startswith("[") :
                st.success(transcript[:200])
            else:
                st.error(transcript)

with col_image:
    with st.popover("📷 图片输入"):
        image_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"],
                                      key="image_upload", label_visibility="collapsed")
        if image_file:
            st.image(image_file, use_container_width=True)
            if st.button("分析图片内容", use_container_width=True):
                with st.spinner("图片分析中..."):
                    img_bytes = image_file.read()
                    desc = st.session_state.multimodal_processor.describe_image(img_bytes)
                    st.session_state.pending_image_text = desc
                if desc and not desc.startswith("[") :
                    st.success(desc[:300])
                else:
                    st.error(desc)

with col_video:
    with st.popover("📎 视频输入"):
        video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"],
                                      key="video_upload", label_visibility="collapsed")
        if video_file:
            if st.button("分析视频内容", use_container_width=True):
                with st.spinner("视频分析中（可能需要较长时间）..."):
                    vid_bytes = video_file.read()
                    summary = st.session_state.multimodal_processor.summarize_video(vid_bytes)
                    st.session_state.pending_video_text = summary
                if summary and not summary.startswith("[") :
                    st.success(summary[:300])
                else:
                    st.error(summary)

# ---- 构建合并后的输入文本 ----
pending_text_parts = []
if st.session_state.pending_audio_text and not st.session_state.pending_audio_text.startswith("["):
    pending_text_parts.append(f"[语音输入]\n{st.session_state.pending_audio_text}")
if st.session_state.pending_image_text and not st.session_state.pending_image_text.startswith("["):
    pending_text_parts.append(f"[图片描述]\n{st.session_state.pending_image_text}")
if st.session_state.pending_video_text and not st.session_state.pending_video_text.startswith("["):
    pending_text_parts.append(f"[视频摘要]\n{st.session_state.pending_video_text}")
pending_multimodal_text = "\n\n".join(pending_text_parts)
if pending_multimodal_text:
    st.info(f"📎 多模态输入就绪 ({len(pending_multimodal_text)} 字符)", icon="📎")
    if st.button("清除多模态输入", key="clear_multimodal"):
        st.session_state.pending_audio_text = ""
        st.session_state.pending_image_text = ""
        st.session_state.pending_video_text = ""
        st.rerun()

# ---- 对话历史 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):  # ReAct 思考步骤
            with st.expander("查看推理过程", expanded=False):
                for step in msg["steps"]:
                    st.caption(f"Step {step['step_num']} ({step['elapsed_ms']}ms)")
                    st.text(f"THOUGHT: {step['thought'][:200]}")
                    st.text(f"ACTION: [{step['action_type']}] {step['action_input'][:100]}")
                    obs = step['observation'][:300]
                    if obs:
                        st.text(f"OBSERVATION: {obs}")
                    st.divider()

# ---- 聊天输入 ----
if prompt := st.chat_input("请输入您的问题..."):
    # 合并多模态文本
    full_prompt = prompt
    if pending_multimodal_text:
        full_prompt = f"{prompt}\n\n---\n多模态输入：\n{pending_multimodal_text}"

    # 显示用户消息
    display_text = prompt
    if pending_multimodal_text:
        display_text += f"\n\n> *(含图片/语音/视频输入)*"
    with st.chat_message("user"):
        st.markdown(display_text)
    st.session_state.messages.append({"role": "user", "content": display_text})

    # 清除多模态暂存
    st.session_state.pending_audio_text = ""
    st.session_state.pending_image_text = ""
    st.session_state.pending_video_text = ""

    with st.chat_message("assistant"):
        if rag_service.enable_react:
            # ---------- ReAct 模式 ----------
            react_steps_container = st.empty()
            labels = []
            step_msgs = []

            with st.status("ReAct Agent 推理中...", expanded=True) as status:
                # 定义检索函数（供 ReAct Agent 调用）
                def retriever(query_str, k=3):
                    return vector_store.search(query_str, k=k)

                # 先检索（给 ReAct 预检索上下文）
                st.write("预检索知识库...")
                pre_docs = vector_store.search(full_prompt)

                st.write("ReAct 推理中...")
                result = rag_service.answer_with_react(
                    full_prompt,
                    retriever_func=retriever,
                    context_docs=pre_docs,
                )
                answer = result["answer"]
                steps = result.get("steps", [])

                # 显示推理步骤
                if steps:
                    for step in steps:
                        step_text = (
                            f"**Step {step.step_num}** "
                            f"`[{step.action_type.value}]` "
                            f"*{step.elapsed_ms}ms*"
                        )
                        st.write(step_text)
                        if step.thought:
                            st.caption(f"💭 {step.thought[:150]}")
                        if step.observation and step.action_type.value not in ("FINAL_ANSWER", "CLARIFY"):
                            st.caption(f"📋 找到 {step.observation[:100]}...")

                if result.get("fallback_used"):
                    reason = result.get("fallback_reason", "未知")
                    st.caption(f"(已降级到标准模式: {reason})")

                status.update(label="回答完成", state="complete")

            st.markdown(answer)
            if not answer.startswith("[澄清]"):
                if pre_docs:
                    sources = list(set(d.metadata.get("source", "未知") for d in pre_docs))
                    st.caption("参考来源: " + ", ".join(sources))

            # 保存消息（含步骤）
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "steps": [{
                    "step_num": s.step_num,
                    "thought": s.thought,
                    "action_type": str(s.action_type.value) if hasattr(s.action_type, "value") else str(s.action_type),
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "elapsed_ms": s.elapsed_ms,
                } for s in steps],
            })
        else:
            # ---------- 标准 RAG 模式 ----------
            with st.status("处理中...", expanded=False) as status:
                st.write("检索知识库...")
                docs = vector_store.search(full_prompt)
                st.write("检索到", len(docs), "个片段")
                st.write("AI 正在生成回答...")
                answer = rag_service.answer(full_prompt, docs)
                status.update(label="回答完成", state="complete")
            st.markdown(answer)
            if docs:
                sources = list(set(d.metadata.get("source", "未知") for d in docs))
                st.caption("参考来源: " + ", ".join(sources))
            else:
                st.caption("知识库中未找到相关匹配内容")
            st.session_state.messages.append({"role": "assistant", "content": answer})