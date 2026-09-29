import os
import json
import requests

from datetime import datetime

from agent.tool_registry import TOOLS
from agent.tool_executor import execute_tool


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


# Tool Calling 最多跑多少轮，防止模型一直不收敛，无限调用下去
MAX_TOOL_ROUNDS = 10

# Reviewer 判 FAIL 后，最多允许补充研究几次
MAX_REVIEW_RETRIES = 1


# ============================================================
# 1. 主 Agent Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一个金融研究 Agent。

你的任务是根据用户提出的研究问题，合理选择并调用可用的金融数据 Tool，
然后基于 Tool 返回的真实数据进行分析。

你必须遵守以下规则：

1. 只能使用 Tool 返回的数据作为事实依据。
2. 不得凭借模型自身知识补充未经 Tool 验证的公司信息、行业信息、新闻、客户信息或市场观点。
   新闻、公告、行业动态必须来自 search_web 返回的结果，
   引用时必须标明来源网址，不得凭记忆写出任何一条新闻。
3. 如果当前 Tool 无法提供某项数据，必须明确说明“当前数据不足以判断”。
4. 不得虚构股价、估值、新闻、行业数据或其他市场数据。
5. 行情类 Tool 已经接入：get_stock_quote 是最新行情快照，
   get_price_history 是历史日 K 线，get_stock_valuation 是 PE、PB 等估值指标。
   用户询问股价、涨跌幅、成交量、估值时，应调用这些 Tool 获取真实数据，
   不要再说“系统暂未接入行情数据”。
6. 财务指标应优先使用 calculate_financial_indicators 返回的结果。
   这个 Tool 只需要传股票代码，它自己会去取财务报表，
   不要把财务数据复制到 Tool 参数里。
   如果已知公司名称，把 company_name 一起传给这个 Tool。
7. 对数据进行解释时，应区分：
   - Tool 直接提供的数据
   - 根据数据计算得到的结论
   - 当前无法验证的信息
8. 回答应简洁、结构清晰，优先使用表格。
9. 不要为了让回答完整而编造信息。
10. 如果目标公司属于银行业，不要把毛利率、自由现金流、
    经营现金流/净利润当成普通企业指标来解读；资产负债率对银行
    天然接近 90%，也不能按普通企业标准解读。这些指标要明确标注为
    “不适用于银行”。银行的净息差、不良贷款率、拨备覆盖率等专用指标，
    当前系统没有数据来源，属于缺失证据，不得估算。
11. 公司年报原文只能来自 rag_search。
    当问题涉及年报里的具体数字或原文表述（营业收入、归母净利润、
    分季度财务数据、研发投入、经营活动现金流、主营业务构成、
    主要会计数据、管理层讨论与分析、股东情况等）时，
    应调用 rag_search 检索已入库的年报原文。
    引用年报内容时必须写明来源文件名和页码，
    例如“中际旭创 2024 年年报第 26 页”。
    不得凭模型自身记忆写出任何年报数字。
12. rag_search 只能查到已经入库的年报，
    查不到实时股价、最新估值和最近几天的新闻，
    这些应该用行情类 Tool 或 search_web。
    如果 rag_search 返回失败信息，说明年报原文检索当前不可用，
    此时应改用其他 Tool，并在报告中说明这一点，
    绝不能因此凭记忆补上年报数字。
"""


# ============================================================
# 2. Planner Prompt
# ============================================================

def build_planner_prompt():
    """
    根据 TOOLS 注册表动态生成 Planner 的 System Prompt。

    Tool 清单直接从 tool_registry.py 读出来，
    以后增删 Tool 只要改注册表就行，
    不会再出现 Prompt 里写了一份、注册表里是另一份的冲突。
    """

    tool_lines = []

    for name, tool in TOOLS.items():
        tool_lines.append(
            "- " + name + "：" + tool["description"]
        )

    tool_text = "\n".join(tool_lines)

    return f"""
你是一个金融研究任务规划器。

你的任务不是回答用户问题，而是根据用户的研究问题，
生成一个简洁、明确的金融研究计划。

当前可用 Tool：

{tool_text}

请根据用户问题判断需要哪些研究步骤。

要求：

1. 只规划真正需要的步骤，不要无意义调用 Tool。
2. 如果用户询问具体公司，通常需要先通过 search_stock 确认股票代码。
3. 如果涉及盈利能力、成长性、现金流、财务质量等分析，
   应调用 calculate_financial_indicators，
   这个 Tool 只需要股票代码，它内部会自己取最近 4 个年度的财务报表。
   只有在需要查看报表原始科目明细时，才另外调用 get_financial_statements。
4. 如果涉及当前股价、涨跌幅、成交量等，应调用 get_stock_quote。
5. 如果涉及历史走势、K 线，应调用 get_price_history。
6. 如果涉及 PE、PB、PS、PCF 等估值指标，应调用 get_stock_valuation。
7. 如果涉及公司年报里的具体数字或原文表述（营业收入、归母净利润、
   分季度财务数据、研发投入、经营活动现金流、主营业务构成、
   主要会计数据、管理层讨论与分析等），
   应调用 rag_search 检索已入库的年报原文。
8. 可以规划多个 Tool。
9. 不要执行 Tool。
10. 不要编造任何数据。
11. 最终只输出 JSON，不要输出 Markdown。

JSON 格式：

{{
    "research_goal": "研究目标",
    "steps": [
        {{
            "step": 1,
            "tool": "search_stock",
            "purpose": "确认目标公司及股票代码"
        }}
    ]
}}
"""


# ============================================================
# 3. Evidence Prompt
# ============================================================

EVIDENCE_PROMPT = """
你是一个金融研究 Evidence Fusion 模块。

你的任务是根据：

1. 用户原始研究问题
2. Research Planner 生成的研究计划
3. Agent 实际执行的 Tool 结果

整理出结构化的研究证据。

注意：

- 你不是最终回答模块。
- 不要进行投资建议。
- 不要补充 Tool 没有提供的数据。
- 不要使用模型自身知识补充事实。
- 每条证据必须能够追溯到具体 Tool。
- 如果 Planner 中某一步没有执行，必须明确标记。
- 如果 Tool 返回的数据不足以支持某个判断，也必须明确标记。
- 如果多个 Tool 的数据存在明显冲突，需要标记出来。
- 如果使用了 rag_search，它返回的是公司年报的原文片段，
  每段都带 source（年报文件名）和 page（页码）。
  整理这类证据时必须保留来源和页码，
  不能只写“年报显示”而不写是第几页 ——
  没有页码的年报数字，人和 Reviewer 都无法核对。
- 如果 rag_search 返回的是失败信息（例如年报未入库、检索不可用），
  必须把“年报原文检索不可用”写进 data_limitations，
  不得用模型自身记忆补上年报数字。
- 如果目标公司属于银行业，要检查是否把毛利率、自由现金流、
  经营现金流/净利润当成了普通企业指标；资产负债率对银行天然很高，
  这些指标应标记为不适用于银行。
- 银行的净息差、不良贷款率、拨备覆盖率等专用指标，如果 Tool 没有提供，
  必须放进 data_limitations，不要估算。

请输出 JSON。

格式：

{
    "research_goal": "...",
    "completed_steps": [
        {
            "step": 1,
            "tool": "search_stock",
            "status": "completed",
            "evidence": "..."
        }
    ],
    "missing_steps": [],
    "data_limitations": [],
    "conflicts": []
}
"""


# ============================================================
# 4. Reviewer Prompt
# ============================================================

REVIEWER_PROMPT = """
你是一个金融研究质量审核 Reviewer。

你的任务是检查一次金融研究任务是否可以进入最终回答阶段。

你需要根据：

1. 用户原始问题
2. Research Planner
3. Agent 实际执行的 Tool
4. Evidence Fusion 结果

进行审核。

重点检查：

一、计划完成度
- Planner 中规划的步骤是否已经执行。
- 如果没有执行，必须指出缺失步骤。

二、证据完整性
- 每一个重要事实是否有 Tool 数据支持。
- 是否存在没有证据支持的结论。
- 如果结论涉及公司年报里的具体数字或原文表述，
  检查它是否能对应到 rag_search 返回的证据
  （要能看到来源文件名和页码）。
  只有模型自身记忆、没有 rag_search 证据支撑的年报数字，
  应写进 unsupported_claims。

三、数据一致性
- Tool 之间是否存在明显冲突。
- 数值是否前后一致。

四、回答边界
- 是否存在模型凭借自身知识补充的信息。
- 是否存在未经验证的行业、新闻、客户、竞争格局等信息。
- 如果数据不足，是否明确说明。
- 如果使用了 search_web，检查新闻类结论是否都能对应到具体来源网址，
  没有来源网址的新闻结论应写进 unsupported_claims。
  如果来源网址存在、只是来源是自媒体或聚合平台（权威性一般），
  写进 issues 或 data_limitations 说明即可，不要仅因此判 FAIL，
  因为当前系统没有获取官方公告原文的能力，这类问题补研究也解决不了。
- 如果 rag_search 失败、或者目标公司年报没有入库，
  检查最终是否明确说明了“年报原文检索不可用”。
  这种情况下如果仍然出现了具体的年报数字，应写进 unsupported_claims。
- 反过来说，如果年报已经入库、问题也确实需要年报原文，
  但 Agent 一次都没有调用 rag_search，
  应写进 issues 并说明缺少年报原文证据。

五、计算合理性
- 如果结论来自多个 Tool 数据的计算或比较，
  检查计算逻辑是否合理。

六、行业适配性
- 如果目标公司是银行，检查是否错误地把毛利率、自由现金流、
  经营现金流/净利润、资产负债率当成普通企业指标来下结论。
- 银行的净息差、不良贷款率、拨备覆盖率，如果 Tool 没有提供
  却被当成重要判断依据，应写进 missing_evidence。

Reviewer 不负责重新研究，也不负责补充数据。

只需要判断当前证据是否足够支持最终回答。

请只输出 JSON：

{
    "status": "PASS",
    "issues": [],
    "missing_evidence": [],
    "unsupported_claims": [],
    "review_summary": "..."
}

如果存在影响最终回答的重要问题：

{
    "status": "FAIL",
    ...
}
"""


# ============================================================
# 5. DeepSeek API
# ============================================================

def call_deepseek(messages, tools=None):
    """
    调用 DeepSeek API。
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "未找到 DEEPSEEK_API_KEY"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.2
    }

    if tools:
        payload["tools"] = tools

    response = requests.post(
        DEEPSEEK_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# 6. Tool Schema
# ============================================================

def build_tool_schemas():
    """
    将 Tool Registry 转换成 DeepSeek Function Calling 格式。
    """

    tool_schemas = []

    for name, tool in TOOLS.items():

        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
        )

    return tool_schemas


# ============================================================
# 7. Research Planner
# ============================================================

def build_research_plan(question):
    """
    根据用户问题生成 Research Plan。
    """

    messages = [
        {
            "role": "system",
            "content": build_planner_prompt()
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = call_deepseek(messages)

    content = response["choices"][0]["message"].get(
        "content",
        ""
    )

    try:

        plan = json.loads(content)

    except json.JSONDecodeError:

        print("\nPlanner 返回结果无法解析为 JSON：")
        print(content)

        plan = {
            "research_goal": question,
            "steps": []
        }

    return plan


# ============================================================
# 8. Evidence Fusion
# ============================================================

def build_evidence(
    question,
    research_plan,
    tool_results
):
    """
    根据 Planner 和实际 Tool 结果生成 Evidence。
    """

    evidence_input = {
        "user_question": question,
        "research_plan": research_plan,
        "tool_results": tool_results
    }

    messages = [
        {
            "role": "system",
            "content": EVIDENCE_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps(
                evidence_input,
                ensure_ascii=False,
                default=str
            )
        }
    ]

    response = call_deepseek(messages)

    content = response["choices"][0]["message"].get(
        "content",
        ""
    )

    try:

        evidence = json.loads(content)

    except json.JSONDecodeError:

        print("\nEvidence 返回结果无法解析为 JSON：")
        print(content)

        evidence = {
            "research_goal": research_plan.get(
                "research_goal",
                question
            ),
            "completed_steps": [],
            "missing_steps": [],
            "data_limitations": [
                "Evidence 模块返回结果无法解析"
            ],
            "conflicts": []
        }

    return evidence


# ============================================================
# 9. Reviewer
# ============================================================

def review_research(
    question,
    research_plan,
    tool_results,
    evidence
):
    """
    根据 Planner、Tool 结果和 Evidence
    审核当前研究是否可以进入最终回答。
    """

    review_input = {
        "user_question": question,
        "research_plan": research_plan,
        "tool_results": tool_results,
        "evidence": evidence
    }

    messages = [
        {
            "role": "system",
            "content": REVIEWER_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps(
                review_input,
                ensure_ascii=False,
                default=str
            )
        }
    ]

    response = call_deepseek(messages)

    content = response["choices"][0]["message"].get(
        "content",
        ""
    )

    try:

        review = json.loads(content)

    except json.JSONDecodeError:

        print("\nReviewer 返回结果无法解析为 JSON：")
        print(content)

        review = {
            "status": "FAIL",
            "issues": [
                "Reviewer 返回结果无法解析"
            ],
            "missing_evidence": [],
            "unsupported_claims": [],
            "review_summary": "无法完成研究质量审核"
        }

    return review


# ============================================================
# 10. Tool Calling
# ============================================================

def run_tool_calling(messages, tool_results):
    """
    反复调用模型，直到模型不再要求调用 Tool，或者达到最大轮数。

    messages 和 tool_results 都是 list，是可变对象，
    函数内部直接往里加内容，调用方拿到的就是同一份，不需要再传回来。

    返回：
        结束原因字符串。
        "completed"          模型自己认为研究做完了
        "max_rounds_reached" 达到最大轮数被迫停下
    """

    tool_schemas = build_tool_schemas()

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):

        response = call_deepseek(
            messages,
            tools=tool_schemas
        )

        message = response["choices"][0]["message"]

        tool_calls = message.get("tool_calls")

        # ====================================================
        # 没有 Tool 调用，研究结束
        # ====================================================

        if not tool_calls:
            return "completed"

        # ====================================================
        # 保存 Assistant Tool Call
        # ====================================================

        messages.append(message)

        # ====================================================
        # 逐个执行 Tool
        # ====================================================

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
            # 保存 Tool Evidence 原始记录
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
            # 将 Tool 结果返回给模型
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

    # ========================================================
    # 跑满最大轮数还没结束
    # ========================================================

    print(
        f"\n已达到最大 Tool 调用轮数（{MAX_TOOL_ROUNDS} 轮），停止调用 Tool。"
    )

    return "max_rounds_reached"


# ============================================================
# 11. 运行结果落盘
# ============================================================

# runs/ 放在项目根目录下。
# 用绝对路径而不是 "runs/" 这种相对路径，
# 这样无论从哪里启动脚本，结果都写到同一个地方。
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def write_json_file(path, data):
    """
    把一个 Python 对象写成 JSON 文件。

    就是 open + json.dump，没有引入任何日志框架。
    """

    with open(path, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )


def save_run_outputs(
    research_plan,
    tool_results,
    evidence,
    review,
    final_answer
):
    """
    把一次运行的中间结果保存到 runs/<时间戳>/ 目录。

    目的只有一个：以后出问题时能回头看清
    Agent 到底调用了哪些 Tool、拿到了什么数据、
    Reviewer 为什么 PASS 或者 FAIL。

    参数：
        research_plan: Planner 生成的计划
        tool_results:  所有 Tool 的原始调用记录
        evidence:      Evidence Fusion 结果
        review:        Reviewer 结果
        final_answer:  最终报告文本

    返回：
        本次运行的目录路径（用来打印给用户看）。
    """

    folder_name = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S"
    )

    run_dir = os.path.join(
        RUNS_DIR,
        folder_name
    )

    # exist_ok=True：目录已存在也不报错
    os.makedirs(run_dir, exist_ok=True)

    write_json_file(
        os.path.join(run_dir, "research_plan.json"),
        research_plan
    )

    write_json_file(
        os.path.join(run_dir, "tool_results.json"),
        tool_results
    )

    write_json_file(
        os.path.join(run_dir, "evidence.json"),
        evidence
    )

    write_json_file(
        os.path.join(run_dir, "review.json"),
        review
    )

    # 最终报告是 Markdown 文本，直接当字符串写
    report_path = os.path.join(
        run_dir,
        "report.md"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(final_answer)

    return run_dir


# ============================================================
# 12. 主 Agent
# ============================================================

def run_agent(question):
    """
    运行金融研究 Agent。

    完整流程：

    用户问题
        ↓
    Research Planner
        ↓
    Tool Calling（最多 MAX_TOOL_ROUNDS 轮）
        ↓
    Evidence Fusion
        ↓
    Reviewer
        ↓
    Reviewer 判 FAIL 时，最多补充研究 MAX_REVIEW_RETRIES 次
        ↓
    最终分析
        ↓
    结果落盘到 runs/<时间戳>/
    """

    # ========================================================
    # Step 1：Research Planner
    # ========================================================

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

    # ========================================================
    # Step 2：建立主 Agent Messages
    # ========================================================

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

    # ========================================================
    # Step 3：保存所有 Tool 执行记录
    # ========================================================

    tool_results = []

    # 记录每一轮 Tool Calling 是怎么结束的
    stop_reasons = []

    # ========================================================
    # Step 4：第一轮 Tool Calling
    # ========================================================

    stop_reasons.append(
        run_tool_calling(
            messages,
            tool_results
        )
    )

    # ========================================================
    # Step 5：Evidence Fusion
    # ========================================================

    print("\n============================================================")
    print("Evidence Fusion")
    print("============================================================")

    evidence = build_evidence(
        question,
        research_plan,
        tool_results
    )

    print(
        json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2
        )
    )

    # ========================================================
    # Step 6：Reviewer
    # ========================================================

    print("\n============================================================")
    print("Research Reviewer")
    print("============================================================")

    review = review_research(
        question,
        research_plan,
        tool_results,
        evidence
    )

    print(
        json.dumps(
            review,
            ensure_ascii=False,
            indent=2
        )
    )

    # ========================================================
    # Step 7：Reviewer 判 FAIL 时，最多补充研究一次
    # ========================================================

    retry_count = 0

    while (
        review.get("status") == "FAIL"
        and retry_count < MAX_REVIEW_RETRIES
    ):

        retry_count = retry_count + 1

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

        stop_reasons.append(
            run_tool_calling(
                messages,
                tool_results
            )
        )

        # ====================================================
        # 用补充后的数据重新做 Evidence 和 Reviewer
        # ====================================================

        print("\n============================================================")
        print("Evidence Fusion（补充研究后）")
        print("============================================================")

        evidence = build_evidence(
            question,
            research_plan,
            tool_results
        )

        print(
            json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2
            )
        )

        print("\n============================================================")
        print("Research Reviewer（补充研究后）")
        print("============================================================")

        review = review_research(
            question,
            research_plan,
            tool_results,
            evidence
        )

        print(
            json.dumps(
                review,
                ensure_ascii=False,
                indent=2
            )
        )

    if review.get("status") == "FAIL":

        print(
            f"\n补充研究 {retry_count} 次后 Reviewer 仍未通过，"
            "将带着未解决的问题进入最终报告。"
        )

    # ========================================================
    # Step 8：最终回答
    # ========================================================

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
        "user_question": question,
        "research_plan": research_plan,
        "tool_results": tool_results,
        "evidence": evidence,
        "review": review,
        "review_retry_count": retry_count,
        "tool_calling_stop_reasons": stop_reasons
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

    # ========================================================
    # Step 9：把这次运行的结果落盘
    # ========================================================

    run_dir = save_run_outputs(
        research_plan,
        tool_results,
        evidence,
        review,
        final_answer
    )

    print("\n本次运行结果已保存到：")
    print(run_dir)

    return final_answer
