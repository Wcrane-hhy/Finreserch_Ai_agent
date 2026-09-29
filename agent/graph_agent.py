"""
LangGraph v2 版 Agent。

这个文件做的事只有一件：
把 agent/financial_agent.py 里已经跑通的手写流程，
翻译成一张 LangGraph 图。

业务逻辑一行不改，全部复用 financial_agent.py 里现成的函数。
financial_agent.py 本身没有任何改动。

v2 的图长这样：

    START
      ↓
    planner              生成研究计划 + 初始化对话
      ↓
    agent  ⇄  tools      有 tool_calls 就去 tools，没有就去 evidence
      ↓
    evidence             证据融合
      ↓
    reviewer             质量审核
      ↓
    ├─ PASS ──────────────────────────→ final → save → END
    │
    └─ FAIL 且 retry_count < 1
         ↓
       prepare_retry     把 Reviewer 的意见交回给模型
         ↓
       agent ⇄ tools → evidence → reviewer     ← 回到 reviewer，再判一次
                                                 （最多补 1 次，不会无限循环）

v2 相比 v1 只多了一件事：Reviewer FAIL 之后的补充研究。
多出来的就是 1 个节点（prepare_retry）+ 1 条条件边，其余与 v1 完全一样。
"""

import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agent.financial_agent import (
    SYSTEM_PROMPT,
    MAX_TOOL_ROUNDS,
    MAX_REVIEW_RETRIES,
    build_research_plan,
    build_tool_schemas,
    call_deepseek,
    build_evidence,
    review_research,
    save_run_outputs,
)
from agent.tool_executor import execute_tool


# ============================================================
# 1. State
# ============================================================
#
# State 就是原来 run_agent() 里的那一堆局部变量。
# 之所以要拎出来，是因为 LangGraph 的每个节点是独立的函数，
# 节点之间只能通过 State 传数据。
#
# 每个节点只返回自己改动的那几个字段，不用把整个 State 返回。

class AgentState(TypedDict):

    question: str           # 用户问题
    research_plan: dict     # Planner 的计划
    messages: list          # 和模型对话的消息列表
    tool_results: list      # 所有 Tool 的原始调用记录
    stop_reasons: list      # 每一轮 Tool Calling 是怎么结束的
    tool_round: int         # 已经调了几轮 Tool，用来判断有没有到上限
    evidence: dict          # Evidence Fusion 结果
    review: dict            # Reviewer 结果
    retry_count: int        # 已经补充研究了几次，用来判断有没有到上限
    final_answer: str       # 最终报告
    run_dir: str            # 本次结果落盘的目录


# ============================================================
# 2. 节点
# ============================================================

def planner_node(state):
    """
    对应原代码 Step 1 + Step 2。

    做两件事：生成 research_plan，然后按 plan 拼出初始 messages。
    这两件事放在一起，是因为 messages 完全由 plan 决定。
    """

    question = state["question"]

    research_plan = build_research_plan(question)

    print("\n============================================================")
    print("Research Plan")
    print("============================================================")

    print(
        json.dumps(
            research_plan,
            ensure_ascii=False,
            indent=2
        )
    )

    planner_context = (
        "\n\nResearch Planner 已生成以下研究计划：\n"
        + json.dumps(
            research_plan,
            ensure_ascii=False
        )
        + "\n\n请根据该研究计划完成研究任务。"
        + "\n如果执行过程中发现还需要其他 Tool，可以继续调用可用 Tool。"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question + planner_context
        }
    ]

    return {
        "research_plan": research_plan,
        "messages": messages
    }


def agent_node(state):
    """
    对应原 run_tool_calling() 循环的前半段：调模型，看它要不要调 Tool。
    """

    messages = state["messages"]
    stop_reasons = state["stop_reasons"]
    tool_round = state["tool_round"]

    # --------------------------------------------------------
    # 已经跑满最大轮数，不再调模型。
    # 对应原代码 for round_index in range(1, MAX_TOOL_ROUNDS + 1) 跑到尽头。
    # --------------------------------------------------------

    if tool_round >= MAX_TOOL_ROUNDS:

        print(
            f"\n已达到最大 Tool 调用轮数（{MAX_TOOL_ROUNDS} 轮），停止调用 Tool。"
        )

        return {
            "stop_reasons": stop_reasons + ["max_rounds_reached"]
        }

    response = call_deepseek(
        messages,
        tools=build_tool_schemas()
    )

    message = response["choices"][0]["message"]

    tool_calls = message.get("tool_calls")

    # --------------------------------------------------------
    # 没有 tool_calls：模型认为研究做完了。
    # 这条 assistant 消息不用存，因为后面没用了。
    # --------------------------------------------------------

    if not tool_calls:

        return {
            "stop_reasons": stop_reasons + ["completed"]
        }

    # --------------------------------------------------------
    # 有 tool_calls：把这条 assistant 消息存进 messages，
    # 然后由后面的条件边把它送去 tools 节点。
    # --------------------------------------------------------

    messages.append(message)

    return {
        "messages": messages,
        "tool_round": tool_round + 1
    }


def route_after_agent(state):
    """
    agent 之后的分岔口。

    只看最后一条消息有没有 tool_calls：
        有 → tools       （继续调 Tool）
        没有 → evidence  （研究结束，去融合证据）
    """

    last_message = state["messages"][-1]

    if last_message.get("tool_calls"):
        return "tools"

    return "evidence"


def tools_node(state):
    """
    对应原 run_tool_calling() 循环的后半段：逐个执行 Tool，
    把结果记进 tool_results，并回灌给模型。
    """

    messages = state["messages"]
    tool_results = state["tool_results"]

    round_index = state["tool_round"]

    last_message = messages[-1]

    tool_calls = last_message.get("tool_calls", [])

    for tool_call in tool_calls:

        tool_name = tool_call["function"]["name"]

        raw_arguments = tool_call["function"]["arguments"]

        print(
            "\n============================================================"
        )
        print(f"Agent 调用 Tool（第 {round_index} 轮）")
        print(
            "============================================================"
        )

        print(f"Tool： {tool_name}")
        print(f"参数： {raw_arguments}")

        # ------------------------------------------------
        # 模型偶尔会给出不合法的 JSON 参数。
        # 这一步必须自己接住，否则整个 Agent 会直接崩掉。
        # ------------------------------------------------

        try:

            arguments = json.loads(raw_arguments)

        except json.JSONDecodeError as error:

            print("\nTool 参数解析失败：")
            print(error)

            error_result = {
                "error": "Tool 参数不是合法 JSON：" + str(error)
            }

            tool_results.append(
                {
                    "tool": tool_name,
                    "arguments": raw_arguments,
                    "status": "failed",
                    "result": error_result
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        error_result,
                        ensure_ascii=False
                    )
                }
            )

            # 跳过这个 Tool，继续处理下一个
            continue

        # ------------------------------------------------
        # 执行 Tool
        # ------------------------------------------------

        try:

            result = execute_tool(
                tool_name,
                arguments
            )

            print("\nTool 返回结果：")
            print(result)

            tool_status = "success"

        except Exception as e:

            result = {
                "error": str(e)
            }

            print("\nTool 执行失败：")
            print(e)

            tool_status = "failed"

        # ------------------------------------------------
        # 保存 Tool 原始记录
        # ------------------------------------------------

        tool_results.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "status": tool_status,
                "result": result
            }
        )

        # ------------------------------------------------
        # 把 Tool 结果回灌给模型
        # ------------------------------------------------

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str
                )
            }
        )

    return {
        "messages": messages,
        "tool_results": tool_results
    }


def evidence_node(state):
    """
    对应原代码 Step 5：Evidence Fusion。
    """

    print("\n============================================================")
    print("Evidence Fusion")
    print("============================================================")

    evidence = build_evidence(
        state["question"],
        state["research_plan"],
        state["tool_results"]
    )

    print(
        json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2
        )
    )

    return {"evidence": evidence}


def reviewer_node(state):
    """
    对应原代码 Step 6：Reviewer。

    v1 只是把审核结果记下来，不管 PASS 还是 FAIL 都直接去 final。
    按 FAIL 触发补充研究是 v2 的事。
    """

    print("\n============================================================")
    print("Research Reviewer")
    print("============================================================")

    review = review_research(
        state["question"],
        state["research_plan"],
        state["tool_results"],
        state["evidence"]
    )

    print(
        json.dumps(
            review,
            ensure_ascii=False,
            indent=2
        )
    )

    return {"review": review}


def route_after_reviewer(state):
    """
    reviewer 之后的分岔口。

    对应原代码 Step 7 的 while 条件：

        while (review.get("status") == "FAIL"
               and retry_count < MAX_REVIEW_RETRIES):

    这里只回答“下一站去哪”，不改 State。
    """

    if (
        state["review"].get("status") == "FAIL"
        and state["retry_count"] < MAX_REVIEW_RETRIES
    ):
        return "prepare_retry"

    return "final"


def prepare_retry_node(state):
    """
    对应原代码 Step 7 里 while 循环体的前半段：
    计数 + 1，然后把 Reviewer 的意见作为一条 user 消息交给模型。

    agent_node / tools_node 都不用改，
    它们只是被这条回边再叫了一次。
    """

    retry_count = state["retry_count"] + 1

    review = state["review"]

    messages = state["messages"]

    print("\n============================================================")
    print(f"Reviewer 未通过，进行第 {retry_count} 次补充研究")
    print("============================================================")

    print(
        json.dumps(
            review,
            ensure_ascii=False,
            indent=2
        )
    )

    # 把 Reviewer 的意见交回给模型，让它补充调用 Tool
    messages.append(
        {
            "role": "user",
            "content": (
                "Reviewer 认为当前研究还存在以下问题：\n"
                + json.dumps(
                    review,
                    ensure_ascii=False,
                    default=str
                )
                + "\n\n请针对这些问题补充调用 Tool，"
                + "把缺失的数据补上，然后结束研究。\n"
                + "如果确实没有任何可用 Tool 能提供这些数据，"
                + "请直接说明这一点，不要再调用 Tool。"
            )
        }
    )

    # tool_round 归零：原代码里补充研究是重新调用 run_tool_calling()，
    # 等于重新拿到 MAX_TOOL_ROUNDS 轮预算，这里保持一致。
    return {
        "messages": messages,
        "retry_count": retry_count,
        "tool_round": 0
    }


def final_node(state):
    """
    对应原代码 Step 8：最终报告。

    final_prompt 是从 financial_agent.py 的 run_agent() 里原样复制过来的，
    因为它原本是函数内的局部变量，import 不到。
    这也是本轮唯一一处内容重复。
    """

    retry_count = state["retry_count"]

    # 对应原代码 while 循环之后那一段：
    # 补过研究、Reviewer 仍然 FAIL，就带着问题进入最终报告。
    if state["review"].get("status") == "FAIL":

        print(
            f"\n补充研究 {retry_count} 次后 Reviewer 仍未通过，"
            "将带着未解决的问题进入最终报告。"
        )

    final_prompt = """
现在请根据以下研究结果生成最终回答。

你必须严格遵守：

1. 只能使用 Tool 返回的数据。
2. 必须参考 Research Plan。
3. 必须参考 Evidence Fusion。
4. 必须遵守 Reviewer 的审核结果。
5. 不得补充未经验证的事实。
6. 如果 Reviewer 指出证据不足，必须在最终回答中明确说明。
7. 如果某个问题无法通过当前 Tool 判断，必须明确说明“当前数据不足以判断”。
8. 可以对 Tool 数据进行合理计算和比较，但不能虚构数据。
9. 回答应该结构清晰。
10. 金融数据优先使用表格展示。
11. 对“数据事实”和“基于数据的解释”进行区分。
12. 不要输出 Research Plan、Evidence Fusion 或 Reviewer 的内部 JSON。
13. tool_results 是各个 Tool 返回的原始数据，evidence 是对它的整理。
    最终报告里出现的每一个数字，都应该能在 tool_results 里找到出处。
    如果两者不一致，以 tool_results 为准。
14. 如果 tool_calling_stop_reasons 里出现了 "max_rounds_reached"，
    说明 Tool 调用轮数达到上限、研究被提前终止，
    必须在报告中说明研究未完成的原因。
15. 如果 review_retry_count 大于 0，说明经过补充研究后 Reviewer 仍未通过，
    必须明确列出哪些证据仍然不足。
16. 如果目标公司是银行，毛利率、自由现金流、经营现金流/净利润
    要标注为不适用于银行；资产负债率不能按普通企业标准解读。
    银行的净息差、不良贷款率、拨备覆盖率没有数据来源，不能编造。
"""

    final_input = {
        "user_question": state["question"],
        "research_plan": state["research_plan"],
        "tool_results": state["tool_results"],
        "evidence": state["evidence"],
        "review": state["review"],
        "review_retry_count": retry_count,
        "tool_calling_stop_reasons": state["stop_reasons"]
    }

    final_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "system",
            "content": final_prompt
        },
        {
            "role": "user",
            "content": json.dumps(
                final_input,
                ensure_ascii=False,
                default=str
            )
        }
    ]

    print("\n============================================================")
    print("最终报告生成中（Final Report 已带上原始 tool_results）")
    print("============================================================")

    final_response = call_deepseek(
        final_messages
    )

    final_answer = final_response["choices"][0]["message"].get(
        "content",
        ""
    )

    return {"final_answer": final_answer}


def save_node(state):
    """
    对应原代码 Step 9：结果落盘。
    """

    run_dir = save_run_outputs(
        state["research_plan"],
        state["tool_results"],
        state["evidence"],
        state["review"],
        state["final_answer"]
    )

    print("\n本次运行结果已保存到：")
    print(run_dir)

    return {"run_dir": run_dir}


# ============================================================
# 3. 拼图
# ============================================================

def build_graph():
    """
    把上面的节点和边拼成一张图，然后编译。
    """

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("final", final_node)
    graph.add_node("save", save_node)

    # 顺序边：一步接一步
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_edge("tools", "agent")
    graph.add_edge("evidence", "reviewer")
    # 这条边是 v2 的回边：补充研究后回到 agent，再走一遍
    graph.add_edge("prepare_retry", "agent")
    graph.add_edge("final", "save")
    graph.add_edge("save", END)

    # 条件边①：agent 之后看有没有 tool_calls
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "evidence": "evidence"
        }
    )

    # 条件边②：reviewer 之后看 PASS / FAIL
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "prepare_retry": "prepare_retry",
            "final": "final"
        }
    )

    return graph.compile()


# ============================================================
# 4. 入口
# ============================================================

def run_graph(question):
    """
    运行 LangGraph 版 Agent，用法和 run_agent(question) 一样。

    返回运行结束后的完整 State。
    """

    app = build_graph()

    initial_state = {
        "question": question,
        "research_plan": {},
        "messages": [],
        "tool_results": [],
        "stop_reasons": [],
        "tool_round": 0,
        "evidence": {},
        "review": {},
        "retry_count": 0,
        "final_answer": "",
        "run_dir": ""
    }

    # recursion_limit 是 LangGraph 的安全阀：
    # 每一步算 1，agent ⇄ tools 最多转 20 步，
    # 再加上一次补充研究，默认的 25 会不够，所以调大一点。
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 100}
    )

    return final_state
