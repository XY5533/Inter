# 简化版 Coding Agent — 设计方案

## 一、项目定位

**一句话**：一个能读代码、搜代码、改代码、跑测试的单 Agent 编程助手。

对标 IntelliRAG 的复杂度：单 Agent + ReAct + 两层记忆 + 代码工具 + Streamlit UI + 测试体系。**不做多 Agent 协作、不做模型路由、不做 Prometheus 监控**。

---

## 二、用户场景

一个开发者在维护一个 Python 项目，遇到了 bug：

> "用户登录后 token 过期了没有自动刷新，帮我修一下 src/auth.py"

Agent 的行为：
1. 读取 `src/auth.py` 理解当前代码
2. 搜索项目中 `token`、`refresh` 相关的代码片段
3. 定位问题：`refresh_token()` 被调用了但没在 `401` 拦截器里触发
4. 生成修复代码，写入文件
5. 运行相关测试，确认不引入新问题
6. 如果测试失败，读取失败日志，修正代码，再跑一次

全程用户可以旁观 Agent 的每一个思考步骤。

---

## 三、和 IntelliRAG 的对应关系

| IntelliRAG | Coding Agent | 复杂度 |
|-----------|-------------|--------|
| 知识库检索（FAISS 搜文档） | 代码检索（FAISS 搜代码片段） | 同级 |
| ReAct Agent（4 工具） | ReAct Agent（5 工具） | 同级 |
| 两层记忆 | 两层记忆 | 完全复用 |
| 多模态输入 | 代码 Diff 可视化 | 同级 |
| 评估测试 18 条 | 编程任务测试 10+ 条 | 同级 |
| 系统测试 59 条 | 工具链测试 | 同级 |
| Streamlit UI | Streamlit UI + Diff 组件 | 稍微复杂 |

---

## 四、架构图

```
用户："src/auth.py 的 token 过期没有自动刷新，帮我修"
  │
  ▼
┌─────────────────────────────────────────────────┐
│              Streamlit UI                        │
│  ┌──────────────────────┐  ┌──────────────────┐ │
│  │  对话区（思考步骤）   │  │  代码 Diff 面板   │ │
│  └──────────────────────┘  └──────────────────┘ │
│  ┌──────────────────────────────────────────┐   │
│  │  输入框 + 文件上传                        │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              ReAct Agent                         │
│                                                  │
│  Step 1: THOUGHT "先读取目标文件"                 │
│          ACTION [READ_FILE src/auth.py]          │
│          OBSERVATION ← 文件内容                   │
│                                                  │
│  Step 2: THOUGHT "搜索 token refresh 相关代码"    │
│          ACTION [SEARCH_CODE token refresh]      │
│          OBSERVATION ← 3 个相关代码片段           │
│                                                  │
│  Step 3: THOUGHT "问题定位完毕，生成修复"          │
│          ACTION [WRITE_FILE src/auth.py <patch>] │
│          OBSERVATION ← "写入成功"                 │
│                                                  │
│  Step 4: THOUGHT "验证修复是否正确"               │
│          ACTION [RUN_TEST tests/test_auth.py]    │
│          OBSERVATION ← "3 passed, 0 failed"      │
│                                                  │
│  Step 5: THOUGHT "修复完成"                       │
│          ACTION [FINAL_ANSWER ...]               │
│                                                  │
│  安全机制：MAX_STEPS=5 / 重复检测 / 异常降级       │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │READ_FILE │ │SEARCH_CODE│ │WRITE_FILE│
    │读取任意  │ │FAISS 语义│ │生成补丁  │
    │项目文件  │ │搜索代码库│ │并写入    │
    └──────────┘ └──────────┘ └──────────┘
          │            │            │
          ▼            ▼            ▼
    ┌──────────────────────────────────────┐
    │          RUN_TEST                     │
    │  隔离执行测试 + 捕获输出 + 失败归因    │
    └──────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│              记忆层（两层）                       │
│  短期：最近 5 轮对话 + 代码修改记录               │
│  长期：项目约定、常见 bug 模式、用户偏好          │
└─────────────────────────────────────────────────┘
```

---

## 五、文件结构

```
coding_agent/
├── config.py              # API 配置、路径、模型设置
├── prompt_templates.py    # Agent System Prompt + 工具描述
├── react_agent.py         # ReAct 循环核心（~250 行）
├── code_tools.py          # 代码工具集（~200 行）
├── code_indexer.py        # 代码库索引（FAISS，~100 行）
├── memory_manager.py      # 两层记忆（复用 IntelliRAG，微调）
├── sandbox.py             # 测试执行沙箱（subprocess 隔离）
├── main.py                # Streamlit UI（~400 行）
├── tests/
│   ├── test_tools.py      # 工具链单元测试
│   ├── test_react.py      # ReAct 循环测试
│   ├── test_cases.json    # 编程任务测试用例
│   └── evaluator.py       # 评估框架
└── demo_project/          # 演示用"待修复"项目
    ├── src/
    │   ├── auth.py        # 含已知 bug
    │   ├── utils.py
    │   └── models.py
    └── tests/
        └── test_auth.py
```

**共计 8 个核心 .py 文件**，IntelliRAG 是 7 个（config, prompt_templates, react_agent, memory_manager, multimodal, rag_service, vector_store），量级持平。

---

## 六、5 个代码工具

| 工具 | 功能 | 安全约束 |
|------|------|---------|
| `READ_FILE <路径>` | 读取项目文件，返回内容 | 只能读工作目录内的文件 |
| `SEARCH_CODE <查询>` | FAISS 语义搜索代码片段 | 无（只读） |
| `WRITE_FILE <路径> <内容>` | 写入/修改文件 | 备份原文件 + 限制文件大小 |
| `RUN_TEST <测试路径>` | 隔离执行测试并返回结果 | subprocess + timeout 30s |
| `FINAL_ANSWER <回答>` | 输出最终回复 | 无 |

对比 IntelliRAG 的 4 工具（SEARCH_KNOWLEDGE / SEARCH_MEMORY / CLARIFY / FINAL_ANSWER），Coding Agent 有 5 个——同样量级。

---

## 七、和 IntelliRAG 一模一样的设计模式

### 1. ReAct 框架 → 完全复用

从 IntelliRAG 的 `react_agent.py` 改过来：
- `ActionType` 枚举：换 5 个代码工具
- `_think()` / `_parse_response()` / `_execute()` / `_is_duplicate()` → 逻辑不变
- `_fallback()` → 降级为简单代码搜索 + 提示
- MAX_STEPS=5，连续重复检测

### 2. 两层记忆 → 直接复用

`memory_manager.py` 几乎可以复制 IntelliRAG 的：
- 短期记忆：最近 k 轮对话 + 代码修改记录
- 长期记忆：FAISS 存储 + JSON 文件
- "值得记的"四类：项目约定（如"这个项目用 snake_case"）、常见 bug 模式、用户偏好、纠正反馈

### 3. FAISS 代码索引 → 和知识库索引同理

`code_indexer.py`：
- 把目标项目的 `.py` 文件分块（按函数/类边界）
- 用 text-embedding-v4 生成向量
- 存入 FAISS 索引
- 暴露 `search(query, k=3)` 接口

### 4. 测试体系 → 同样结构

- **10 个编程任务测试用例**：bug 修复、代码重构、功能新增、测试生成各 2-3 条
- **评估维度**：任务完成率 / 测试通过率 / 工具调用效率 / 是否会破坏已有代码
- **系统测试**：覆盖配置/Prompt/工具/ReAct/记忆/沙箱

---

## 八、具体砍掉了什么（vs 你之前的复杂版）

| 砍掉的模块 | 原因 |
|-----------|------|
| 三层记忆 → 两层 | 工作记忆（当前任务状态）由 ReAct 的 observation 上下文自然承担 |
| 多 Agent 协作 | 单 Agent + 5 工具足够覆盖 bug 修复/重构场景 |
| MCP 工具延迟加载 | 5 个工具不需要延迟加载 |
| 多模型动态路由 | 全用 qwen3-max，不需要路由 |
| 五层权限 → 沙箱隔离 | subprocess 隔离就够了，不需要白名单+文件范围+会话授权 |
| Prometheus + Grafana | JSON 日志 + Streamlit 面板够用 |
| SWE-bench 评测 → 自建测试集 | 10-15 条手工测试用例更可控 |
| 任务检查点 + 反馈节点 | ReAct 每步的 observation 就是天然反馈 |

---

## 九、业务需求（面试场景）

### 场景 1：Bug 修复（主打）
> 用户把 `demo_project/` 里的 `src/auth.py` 打开给 Agent："token 过期后没有自动刷新，帮我定位并修复"

Agent 行为：读文件 → 搜相关代码 → 定位 bug → 写修复 → 跑测试 → 确认通过

### 场景 2：代码重构
> "把 `src/utils.py` 里所有用 `%` 做字符串格式化的地方改成 f-string"

Agent 行为：读文件 → 搜 `%` 模式 → 生成替换 → 写入 → 跑测试

### 场景 3：功能新增
> "给 `src/models.py` 的 User 类加一个 `is_token_expired()` 方法"

Agent 行为：读现有代码 → 理解类结构 → 生成方法 → 写入 → 跑测试

### 场景 4：测试生成
> "给 `src/auth.py` 的 `refresh_token()` 写单元测试"

Agent 行为：读目标函数 → 分析分支 → 生成测试 → 写入测试文件 → 跑测试验证

---

## 十、开发计划（7 天）

| 天 | 任务 | 产出 |
|---|------|------|
| 1-2 | `code_tools.py` + `code_indexer.py` + `sandbox.py` | 5 个工具可用 + 沙箱隔离 |
| 3-4 | `react_agent.py` + `prompt_templates.py` | ReAct 循环跑通 |
| 4-5 | `memory_manager.py`（从 IntelliRAG 移植） | 两层记忆集成 |
| 6 | `main.py`（Streamlit UI + Diff 面板） | 完整可交互界面 |
| 6-7 | `demo_project/` + 测试用例 + 评估 | 可演示的完整流程 |

---

## 十一、关键风险

| 风险 | 应对 |
|------|------|
| qwen3-max 对代码任务不够强 | 必要时切 deepseek-coder 或 claude-sonnet |
| WRITE_FILE 可能写坏代码 | 写入前自动 backup，支持回滚 |
| RUN_TEST 可能无限循环 | subprocess timeout 30s 硬杀 |
| Token 消耗大（代码文件长） | 文件截断 2000 字符，提示 Agent 分次读取 |
| Streamlit 实时渲染 diff | 用 Python `difflib` 生成 HTML diff，`st.components.v1.html` 渲染 |

---

## 十二、IntelliRAG vs Coding Agent 复杂度对照

| 维度 | IntelliRAG | Coding Agent | 判定 |
|------|-----------|-------------|------|
| 核心文件数 | 7 | 8 | 同级 |
| ReAct 工具数 | 4 | 5 | 同级 |
| 记忆层 | 2 | 2 | 同级 |
| 向量库 | FAISS | FAISS | 同级 |
| UI 复杂度 | 中等（聊天+评估+多模态） | 中等（聊天+Diff+评估） | 同级 |
| 测试用例 | 59+18 | 30+10 | 略少 |
| 新增复杂度 | 无 | sandbox 隔离 + diff 渲染 | Coding Agent 稍高 |
| 总代码量 | ~1500 行核心 | ~1400 行核心 | 同级 |

---

**结论**：简化后的 Coding Agent 在架构复杂度、代码量、技术深度三个维度都和 IntelliRAG 严格对标。区别只在于：IntelliRAG 搜文档，Coding Agent 搜代码。面试时两个项目可以递进讲——先讲 RAG 基础（IntelliRAG），再讲 Agent 进阶应用（Coding Agent），形成完整叙事线。
