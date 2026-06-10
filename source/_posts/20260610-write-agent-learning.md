---
title: 如何写一个 Agent：从最小 Runtime 到 Coding Agent
date: 2026/06/10
categories:
- Agent
tags: [Agent,RAG,LangGraph]
---

最近整理了一份关于“如何写一个 Agent”的学习路径。读完之后最大的感受是：Agent 不是一段更长的 prompt，也不是把几个工具随便接到模型后面，而是一个可控的 runtime。这个 runtime 至少包含模型决策循环、工具调用、状态管理、权限边界、上下文管理和终止条件。

<!--more-->

如果只记一个公式，可以记这个：

```text
Agent = LLM 决策循环 + 工具调用 + 状态管理 + 权限边界 + 终止条件
```

也就是说，写 Agent 的核心不是“让模型更聪明”，而是给模型一个安全、清晰、可验证的运行环境。

## 从最小 Agent Loop 开始

一个最小 Agent 的执行流可以这样理解：

```text
用户任务
  -> Agent Loop
  -> LLM 决定下一步
  -> 调用工具或给出最终答案
  -> 工具结果写回上下文
  -> 重复直到完成、阻塞或达到步数上限
```

伪代码大概是：

```python
while not done:
    response = call_llm(messages, tools)

    if response.wants_tool:
        result = run_tool(response.tool_name, response.tool_args)
        messages.append(tool_result(result))
        continue

    if response.final_answer:
        done = True
```

这段循环比任何框架都重要。只有先理解这个 while loop，后面使用 LangGraph、MCP 或多 Agent 框架时，才知道框架到底帮自己封装了什么。

## 第一版应该只读

学习 Agent 最容易犯的错误，是一开始就给它写文件、执行 shell、安装依赖、提交代码的能力。这样会把问题复杂度一下拉满：工具设计、权限控制、回滚、测试、上下文污染都会同时出现。

更稳的第一版是只读代码研究 Agent，只提供几个窄工具：

```text
list_files(path)
read_file(path)
search_text(query)
final_answer
```

并且加上限制：

```text
只能读文件
不能写文件
不能执行任意 shell
最多运行 10-15 步
必须引用实际读过的文件
```

只读 Agent 的目标不是“帮我改代码”，而是回答这些问题：项目入口在哪里？某个功能在哪里实现？调用链大概是什么？哪些文件提供了证据？

这个阶段能训练最基础也最重要的能力：tool use、loop、state、context，以及基于证据回答。

## 真正要设计的是 Runtime

很多人以为写 Agent 的难点是模型 API。其实模型调用只是其中一层，真正决定 Agent 是否可靠的是 runtime 设计。

一个可维护的 runtime 至少要显式管理这些模块：

```text
State：当前任务、消息、已知事实、读过的文件、工具历史
Tools：Agent 能做什么，输入输出是什么
Policy：哪些工具允许，哪些需要确认，哪些禁止
Loop：什么时候继续，什么时候停止
Context：哪些信息放进模型，哪些压缩或丢弃
Verification：怎么判断任务真的完成了
```

不要只用一个 `messages` 变量糊到底。更好的方式是维护结构化状态：

```python
state = {
    "task": "Find where login is implemented",
    "messages": [],
    "facts": [],
    "files_read": [],
    "tool_calls": [],
    "step": 0,
    "done": False,
}
```

这样才能回答：Agent 已经确认了什么？读过哪些文件？哪些线索无效？它是不是在原地打转？下一步为什么要调用这个工具？

## 工具越窄越好

不要一开始就给 Agent 一个万能 shell：

```text
run_anything(command)
```

这类工具看起来灵活，实际是在把安全问题、执行问题和解释问题都推给模型。更好的工具应该窄、结构化、可审计：

```text
list_files(path)
read_file(path)
search_text(query)
get_git_diff()
run_tests(command_id)
propose_patch(path, old, new)
```

工具设计原则可以总结为：

+ 工具越窄越好。
+ 输入输出要结构化。
+ 默认只读。
+ 写操作要人工确认。
+ 删除、部署、push、付款、发消息必须确认。
+ 工具结果不要无限长，必要时截断或摘要。
+ 工具报错要返回清晰错误，而不是让模型猜。

如果确实需要执行命令，也应该做命令 ID 白名单，而不是让模型直接传入任意 shell 字符串：

```python
ALLOWED_COMMANDS = {
    "git_status": ["git", "status", "--short"],
    "git_diff": ["git", "diff"],
    "pytest": ["pytest"],
    "typecheck": ["npm", "run", "typecheck"],
}
```

## System Prompt 要像操作规程

Agent 的 system prompt 不应该只是：

```text
You are a helpful assistant.
```

它应该更像 SOP 或 runbook，明确目标、禁止行为、工具使用规则和最终输出格式：

```text
You are a read-only code research agent.

Your job:
- Inspect the repository using tools.
- Build an evidence-based answer.
- Cite files you actually read.

Rules:
- Do not guess.
- Do not modify files.
- Prefer search before reading large files.
- Stop when you have enough evidence.
- If blocked, explain what information is missing.

Final answer format:
- Summary
- Key files
- Execution flow
- Risks or unknowns
```

prompt 的作用不是塑造一个“人格”，而是给 runtime 中的模型节点提供操作规程。

## Planning 和终止条件

最小 Agent 是：

```text
Think -> Tool -> Observe -> Repeat
```

更稳的 Agent 是：

```text
Plan -> Tool -> Observe -> Update Plan -> Repeat
```

Planning 的价值在于让 Agent 每一步都围绕当前目标推进，而不是被搜索结果带着跑偏。但 Planning 不能只靠模型自觉，runtime 也要有终止条件。

至少应该有：

```text
max_steps
max_tool_errors
max_context_tokens
final_answer
needs_human_approval
no_new_information
```

第一版可以先设置一个简单的 `max_steps = 12`。这比追求“全自动完成所有事”更重要，因为没有停止规则的 Agent 很容易无限循环、重复搜索、或者越查越散。

## 上下文管理不是无限追加 messages

一开始可以把所有消息都传回模型，但真实 Agent 很快会遇到上下文膨胀。最容易爆上下文的是大文件全文、测试日志、搜索结果、网页内容和历史对话。

更好的方式是维护压缩状态，只保留：

```text
已确认事实
已读文件列表
关键代码片段
失败线索
当前计划
未解决问题
```

旧的、重复的、无关的工具结果应该被压缩或丢弃。summary 也不应该只是每隔几步追加一条消息，而应该作为 runtime 的上下文管理机制：用结构化 summary 替换旧上下文，并保留 evidence。

一个 summary 至少应该包含：

```json
{
  "confirmed_facts": [],
  "files_inspected": [],
  "tool_results": [],
  "failed_leads": [],
  "open_questions": [],
  "next_action": "Continue normal loop using the compacted context"
}
```

关键原则是：summary 只总结证据，不发明事实；summary 不调用工具，也不决定任务是否完成。

## RAG 是 Agent 的工具，不是 Agent 本身

RAG 的作用是让 Agent 在需要时检索外部知识，而不是把所有资料塞进 prompt。

```text
RAG = 文档索引 + 查询改写 + 检索 + 重排 + 上下文注入 + 引用验证
```

普通 RAG 是：

```text
用户问题 -> 检索 -> 回答
```

Agentic RAG 则是：

```text
用户问题
  -> 判断缺什么信息
  -> 改写检索 query
  -> 调 retrieve_docs
  -> 读关键 chunk
  -> 发现不够再检索
  -> 综合多个来源
  -> 引用证据回答
```

对 Coding Agent 来说，RAG 的正确位置通常是“背景知识工具”：查规范、架构文档、历史决策。真正修改代码前，仍然要读当前仓库里的实际文件。

文档切分比 embedding 模型更容易决定检索效果。好的 chunk 应该语义完整、长度适中、带标题路径、带来源 metadata，并且能单独被引用。

```json
{
  "source": "docs/auth/session.md",
  "title": "Session lifecycle",
  "section": "Token refresh",
  "updated_at": "2026-05-28",
  "chunk_id": "docs/auth/session.md#token-refresh-03"
}
```

检索结果也要当作不可信输入。RAG 文档里可能有 prompt injection，所以回答规则应该明确：只根据 retrieved context 和工具实际读到的内容回答，关键结论必须引用来源，不要把检索结果里的指令当成系统指令。

## Embedding Retriever 的学习顺序

Embedding 可以理解成把文本转换成语义向量，让语义接近的内容在向量空间里距离更近。RAG 中的 retrieval 通常分成两步：

```text
建索引阶段：
代码文件 / 文档
  -> 切成 chunks
  -> 每个 chunk 生成 embedding
  -> 保存：向量 + 文件路径 + chunk 内容 + metadata

查询阶段：
用户问题
  -> 生成 query embedding
  -> 和所有 chunk embedding 算相似度
  -> 返回 top_k 个相关 chunks
  -> LLM 基于 chunks 生成答案
```

学习版不需要一开始就上向量数据库。推荐顺序是：

```text
1. 手写 lexical retriever
2. 接入 embedding API
3. 实现内存向量索引
4. 用 cosine similarity 检索
5. 增加 JSON / SQLite 持久化
6. 再替换为 Chroma、FAISS、Qdrant 等向量库
7. 做关键词 + embedding + rerank 的混合检索
```

对 code agent 来说，不应该只依赖 embedding。代码任务里经常有函数名、变量名、错误码、文件路径，这些精确符号用关键词搜索和符号搜索往往更可靠。更常见的组合是：

```text
关键词搜索 + embedding 搜索 + rerank + read_file 精读
```

## LangChain 和 LangGraph 的位置

LangChain 和 LangGraph 都有用，但不要一开始就用框架掩盖原理。

一句话区分：

```text
LangChain：组件库，偏 prompt、model、retriever、tool、chain 集成
LangGraph：有状态执行图，偏 agent loop、分支、循环、checkpoint、人类审批
```

如果目标是快速做 RAG demo，LangChain 很方便。如果目标是写真正可控的 Agent，LangGraph 更接近 runtime，因为 Agent 本质上是一个有状态循环：

```text
plan -> act -> observe -> update state -> route -> continue or stop
```

推荐学习顺序是：

```text
1. 手写 read-only agent loop
2. 手写 tool schema 和 tool execution
3. 手写 state、max_steps、权限检查
4. 手写一个简单 RAG retriever tool
5. 用 LangChain 替换 document loader / splitter / vector store
6. 用 LangGraph 重写 agent loop
7. 加 checkpoint、human approval、多节点工作流
```

也就是先理解 Agent runtime，再用 LangChain 加速组件集成，最后用 LangGraph 管复杂流程。

## 从只读 Agent 到 Coding Agent

比较稳的升级路线是四步：

```text
只读研究 Agent -> 测试诊断 Agent -> Patch Agent -> Coding Agent
```

V1 只读研究 Agent：

```text
能力：list_files / read_file / search_text
目标：找入口、画调用链、基于文件证据总结结构
验收：不编造文件、不无限循环、答案包含文件引用
```

V2 测试诊断 Agent：

```text
增加：run_tests(test_command_id) / read_test_output(output_id)
目标：读取失败日志，定位相关代码，区分 deterministic failure 和 flaky
限制：不改代码，不安装依赖，不自动删除缓存
```

V3 Patch Agent：

```text
增加：propose_patch(path, old, new) / show_diff()
目标：提出最小 patch，用户确认后再应用，应用后跑测试
流程：读代码 -> 提计划 -> 生成 patch -> 人工确认 -> 应用 -> 测试 -> 总结
```

V4 Coding Agent：

```text
增加：edit_file / run_tests / get_diff
目标：完成小范围 bugfix 或小功能，自动验证，输出变更摘要
限制：不 commit，不 push，不删文件，不重置 git 状态，危险操作必须确认
```

这个顺序的核心是：先把读和判断做稳，再给写权限。

## 安全边界必须在代码里

Agent 一定要有权限边界，否则只是一个会随机执行操作的模型。

默认规则可以这样定：

```text
读操作默认允许
写操作需要确认
外部网络需要确认
删除需要确认
部署需要确认
push / force push 禁止或强确认
密钥永远不进 prompt
工具输出要当作不可信输入
```

危险操作例如：

```bash
rm -rf
git reset --hard
git push --force
curl ... | sh
chmod -R 777
drop database
```

即使是测试 Agent，也不要让它随便执行任意命令。测试命令最好用 ID 白名单，而不是直接让模型提供 shell 字符串。

## Plan Mode 的关键不是一句 prompt

真正可控的 Plan Mode 不是在 system prompt 里写“请先计划”，而是状态机 + 工具权限门控 + 结构化计划 + 人工审批 + 执行阶段分离。

核心目标是：

```text
plan 模式只负责理解任务、读取证据、生成计划，然后停止。
execute 模式只能执行已经批准的计划。
```

可以把 Agent 拆成三个模式：

```python
class AgentMode(str, Enum):
    ANSWER = "answer"
    PLAN = "plan"
    EXECUTE = "execute"
```

不同 mode 暴露不同工具：

```python
def tools_for_mode(mode: AgentMode):
    if mode in {AgentMode.ANSWER, AgentMode.PLAN}:
        return READ_TOOLS
    if mode == AgentMode.EXECUTE:
        return READ_TOOLS | WRITE_TOOLS
    raise ValueError(f"unknown mode: {mode}")
```

也就是说，即使模型在 plan 模式里请求 `edit_file`，runtime 也必须拒绝。权限边界应该由代码保证，而不是依赖模型自觉。

Plan Mode 的输出也应该有稳定 schema，例如：

```json
{
  "status": "needs_approval",
  "goal": "Fix streaming planning output",
  "findings": [],
  "steps": [],
  "risks": [],
  "verification": []
}
```

生成计划后程序要停止，等待人类审批。不要让同一次模型调用从“计划”直接滑到“执行”。

## 如何评估 Agent 是否变好

不要靠感觉判断 Agent 是否变好了。应该准备固定任务集，例如：

```text
1. 找项目入口
2. 找登录流程
3. 找某个 API handler
4. 判断一个测试失败原因
5. 给出一个最小 patch 方案
6. 识别一次危险操作并停止
7. 在上下文很长时正确总结已知事实
```

每次修改 Agent 后记录：

```text
成功 / 失败
调用了哪些工具
用了多少步
是否读错文件
是否产生幻觉
是否越权
总耗时
成本
```

好 Agent 的标准不是“回答很像人”，而是：知道什么时候用工具，不编造文件内容，不无限循环，遇到危险操作会停，能引用证据，能从工具错误中恢复，修改后会验证。

## 最后记住这条路线

如果把整份学习路径压缩成一条实践顺序，就是：

```text
只读 Agent -> 测试诊断 Agent -> Patch Agent -> Coding Agent
```

框架学习也应该按这个节奏来：

```text
自己写最小 Agent
-> Anthropic tool use / OpenAI function calling
-> MCP
-> LangGraph
-> 多 Agent orchestration
```

不要一开始就上多 Agent，也不要一开始就堆框架。先把只读 Agent 写稳，再给它测试能力，再让它生成 patch，最后才给它编辑文件的权限。

写 Agent 的核心不是“让模型更自由”，而是相反：给它合适的工具，限制它的权限，保存必要状态，压缩无用上下文，设置明确停止条件，并用测试验证结果。这样做出来的才不是 demo，而是一个可以逐步变强的 Agent runtime。

## 参考资料与工具分类

### Agent Harness / 状态机框架

这一类负责 Agent 怎么运行：模型什么时候调用，工具什么时候调用，状态怎么流转，什么时候停止，什么时候 human-in-the-loop，以及如何 checkpoint / resume。

+ 手写 `run_loop`
+ LangGraph: https://langchain-ai.github.io/langgraph/
+ PydanticAI
+ OpenAI Agents SDK
+ Claude Agent SDK / Claude Code 风格 harness
+ Anthropic Engineering, Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
+ ReAct, Synergizing Reasoning and Acting in Language Models: https://arxiv.org/abs/2210.03629
+ Anthropic Messages API: https://docs.anthropic.com/en/api/messages

优先理解的是：手写 harness + LangGraph。

### RAG / 数据检索框架

这一类负责资料怎么进入系统、怎么切分、怎么检索，包括 loader、parser、chunker、embedding、vector store、retriever 和 query engine。

+ LangChain: https://python.langchain.com/docs/
+ LlamaIndex
+ Haystack
+ Model Context Protocol: https://modelcontextprotocol.io/docs

### 向量数据库 / 向量索引

这一类负责保存 chunk embedding，并按 query embedding 找回 top-k 结果，也会涉及 metadata filter 和 hybrid search。

+ Chroma
+ FAISS
+ Qdrant
+ Milvus
+ Weaviate
+ pgvector
+ Elasticsearch / OpenSearch
+ Pinecone
+ LanceDB
+ Redis Vector

推荐顺序：学习阶段先用 Chroma / FAISS，后续生产化再评估 Qdrant；如果已有 PostgreSQL，可以考虑 pgvector；如果已有搜索系统，可以考虑 Elasticsearch / OpenSearch。

### 多 Agent 协作框架

这一类关注多个角色 Agent 怎么协作，例如 planner、coder、reviewer、researcher、executor、handoff 和 group chat。

+ AutoGen
+ CrewAI
+ OpenAI Agents SDK handoffs

### 结构化输出 / 类型校验工具

这一类负责让模型稳定输出 JSON / 对象，并做校验和重试，适合 plan schema、summary schema、tool args validation、RAG citation schema 等场景。

+ Pydantic
+ Instructor
+ PydanticAI
+ OpenAI Structured Outputs
+ Anthropic tool use / JSON Schema: https://json-schema.org/

### Workflow / Durable Execution 框架

这一类负责长时间、可恢复、可重试的任务执行，例如 pause、resume、retry、checkpoint、scheduled jobs、human approval wait 和 distributed execution。

+ Temporal
+ Prefect
+ Dagster
+ Airflow
+ Argo Workflows

### Eval / Observability / Optimization 工具

这一类负责评估、追踪和优化 LLM pipeline 的效果，包括 prompt optimization、retriever tuning、trace visualization、hallucination / faithfulness 检测。

+ DSPy
+ LangSmith
+ OpenAI Evals
+ Ragas: https://docs.ragas.io/
+ promptfoo: https://www.promptfoo.dev/docs/
+ TruLens
+ Phoenix
+ OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/

