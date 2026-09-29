# -*- coding: utf-8 -*-
"""
rag_search Tool（年报检索）的测试。

RAG Pipeline 本身已经在 tests/test_rag.py 里测过 40 项了，
所以这里不重复测 Pipeline，只测"接进 Agent 之后还站不站得住"：

    1. 返回结构稳定：evidence 里必须一直带着 chunk_id / source / page
    2. 真的能捞回答案：问营业收入，就得捞回写着那个数字的原文
    3. 参数真的生效：top_k / top_n / source 不是摆设
    4. 失败时明确失败：参数不对、年报不存在、向量库没建，都要报清楚
    5. 注册生效：TOOLS / Function Calling schema / Planner 清单三处都能看见它

第 5 条和当初加 search_web / web_fetch 时一样，是最容易出错的一条：
Tool 写出来了但没进 schema，模型就永远不知道它存在 ——
代码全对，功能为零。

运行方式（和 tests/ 下其他脚本一样）：

    python -m tests.test_rag_tool
"""

import sys

import tempfile

from pathlib import Path


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


from tools import rag_tool
from tools.rag_tool import rag_search

from rag.rag_query import rag_query, DEFAULT_K, DEFAULT_TOP_N
from rag import vector_store

from agent.tool_registry import TOOLS
from agent.tool_executor import execute_tool
from agent.financial_agent import build_tool_schemas, build_planner_prompt


# ============================================================
# 2. 测试参数
# ============================================================

QUESTION = "中际旭创2024年的营业收入和净利润是多少？"

# 为什么用这一句而不是"营业收入和归母净利润"：
#   这是 requirement 里指定的那一问，也是 RAG Pipeline 当初调 K/N 时用的问法。
#   实测换成"和归母净利润"这个说法，答案块的向量名次会从 #9 掉到 #15，
#   K=10 时连候选集都进不去 —— 那是检索质量的问题，不是这个 Tool 的问题，
#   见文件末尾「已知局限」第 4 条。
#   这个测试要验证的是"Tool 的结构和可追溯性"，所以固定用调好的那一问。

# 年报里 2024 年营业收入的原始数字（约 238.62 亿元）。
# 用它判断检索到底有没有把"写着答案的那段原文"捞回来。
ANSWER_REVENUE = "23,862,159,738.37"

# 2024 年归母净利润（约 51.71 亿元）
ANSWER_PROFIT = "5,171,485,967.85"

REPORT_NAME = "中际旭创 2024 年年报.pdf"

# chunker 的 chunk_size 是 500，留一点余量。
# 这条断言的意义：Tool 返回给模型的是年报原文，
# 如果哪天 chunk_size 被改得很大，这里会先炸出来，
# 而不是等到某一轮模型调用突然超上下文长度。
MAX_CHUNK_TEXT = 700


results = []


def check(name, passed, detail=""):

    results.append((name, passed, detail))

    print(("  [通过] " if passed else "  [失败] ") + name)

    if detail:

        print("         " + str(detail)[:260])


# ============================================================
# 3. 测试一：向量库和年报都在
# ============================================================

print("=" * 70)
print("测试一：前提条件")
print("=" * 70)

try:

    stored = vector_store.count_chunks()

    check("年报向量库存在且非空", stored > 0, "共 " + str(stored) + " 个 chunk")

except Exception as error:

    check("年报向量库存在且非空", False, str(error))

    stored = 0

available = vector_store.list_sources()

check(
    "向量库里能列出已入库的年报",
    REPORT_NAME in available,
    "已入库：" + "、".join(available)
)


# ============================================================
# 4. 测试二：返回结构
# ============================================================

print()
print("=" * 70)
print("测试二：返回结构稳定")
print("=" * 70)

result = rag_search(QUESTION)

check("返回的是 dict", isinstance(result, dict), type(result).__name__)

expected_keys = [
    "question",
    "source",
    "available_sources",
    "top_k",
    "top_n",
    "returned",
    "evidence",
    "error"
]

check(
    "顶层字段齐全",
    all(key in result for key in expected_keys),
    "实际字段：" + str(list(result.keys()))
)

check(
    "error 为 None（成功时不返回半成品）",
    result["error"] is None,
    str(result["error"])
)

check(
    "问题做了 strip 后原样回传",
    result["question"] == QUESTION.strip(),
    result["question"]
)

evidence = result["evidence"]

check(
    "evidence 非空",
    len(evidence) > 0,
    "共 " + str(len(evidence)) + " 条"
)

evidence_keys = ["chunk_id", "source", "page", "text", "score"]

check(
    "每条证据都带 chunk_id / source / page / text / score",
    all(all(key in item for key in evidence_keys) for item in evidence),
    ""
)

check(
    "source 是年报文件名（不是空字符串）",
    all(item["source"] == REPORT_NAME for item in evidence),
    "例如 " + str(evidence[0]["source"])
)

check(
    "page 是正整数（能翻回去核对）",
    all(isinstance(item["page"], int) and item["page"] > 0 for item in evidence),
    "页码：" + str([item["page"] for item in evidence])
)

check(
    "chunk_id 唯一（不是同一块重复返回）",
    len({item["chunk_id"] for item in evidence}) == len(evidence),
    str([item["chunk_id"] for item in evidence])
)

check(
    "score 是数字且降序（rerank 结果是有序的）",
    all(isinstance(item["score"], float) for item in evidence)
    and all(
        evidence[i]["score"] >= evidence[i + 1]["score"]
        for i in range(len(evidence) - 1)
    ),
    "分数：" + str([item["score"] for item in evidence])
)


# ============================================================
# 5. 测试三：真的能捞回答案
# ============================================================

print()
print("=" * 70)
print("测试三：真的能捞回答案原文")
print("=" * 70)

joined = "".join(item["text"] for item in evidence)

check(
    "捞回了 2024 年营业收入原始数字 " + ANSWER_REVENUE,
    ANSWER_REVENUE in joined,
    "证据总字数 " + str(len(joined))
)

check(
    "捞回了 2024 年归母净利润原始数字 " + ANSWER_PROFIT,
    ANSWER_PROFIT in joined,
    ""
)

check(
    "每段 text 都不超过 chunk_size 的量级（不会撑爆上下文）",
    all(len(item["text"]) <= MAX_CHUNK_TEXT for item in evidence),
    "各段长度：" + str([len(item["text"]) for item in evidence])
)

print()
print("  实际返回的证据：")

for item in evidence:

    print(
        "    page=" + str(item["page"])
        + "  score=" + str(item["score"])
        + "  " + str(item["chunk_id"])
    )
    print(
        "      " + item["text"][:300].replace("\n", " ")
        + ("..." if len(item["text"]) > 300 else "")
    )
    print()


# ============================================================
# 6. 测试四：参数真的生效
# ============================================================

print("=" * 70)
print("测试四：top_k / top_n / source 真的生效")
print("=" * 70)

small = rag_search(QUESTION, top_n=2)

check(
    "top_n=2 时只返回 2 条",
    len(small["evidence"]) == 2,
    "returned=" + str(small["returned"])
)

check(
    "top_n 保持不变地回传（调用方能看到自己要求的条数）",
    small["top_n"] == 2,
    "top_n=" + str(small["top_n"])
)

wide = rag_search(QUESTION, top_k=3, top_n=10)

check(
    "top_n 大于 top_k 时被压到 top_k（不会报错，也不会假装有 10 条）",
    len(wide["evidence"]) == wide["top_n"] == 3,
    "top_k=3, top_n=" + str(wide["top_n"])
    + ", returned=" + str(wide["returned"])
)

exact = rag_search(QUESTION, source=REPORT_NAME)

check(
    "source 传精确文件名时能正常检索",
    exact["source"] == REPORT_NAME and len(exact["evidence"]) > 0,
    "source=" + str(exact["source"])
)

loose = rag_search(QUESTION, source="中际旭创2024年年报.pdf")

check(
    "source 少写空格也能对上（模型很难精确复述文件名）",
    loose["source"] == REPORT_NAME,
    "传入 中际旭创2024年年报.pdf -> 解析为 " + str(loose["source"])
)

partial = rag_search(QUESTION, source="2024年年报")

check(
    "source 只写一部分也能对上",
    partial["source"] == REPORT_NAME,
    "解析为 " + str(partial["source"])
)

try:

    rag_search(QUESTION, source="平安银行2024年年报.pdf")

    check("source 写了库里没有的年报时会明确失败", False, "居然没报错")

except RuntimeError as error:

    message = str(error)

    check(
        "source 写了库里没有的年报时会明确失败",
        REPORT_NAME in message,
        message
    )


# ============================================================
# 7. 测试五：失败路径
# ============================================================

print()
print("=" * 70)
print("测试五：失败路径（不假装成功）")
print("=" * 70)

invalid_cases = [
    ("问题为空", lambda: rag_search("")),
    ("问题只有空格", lambda: rag_search("   ")),
    ("top_k=0", lambda: rag_search(QUESTION, top_k=0)),
    ("top_k 不是整数", lambda: rag_search(QUESTION, top_k="10")),
    ("top_n 是负数", lambda: rag_search(QUESTION, top_n=-1)),
]

for name, call in invalid_cases:

    try:

        call()

        check(name + " -> 抛 ValueError", False, "居然没报错")

    except ValueError as error:

        check(name + " -> 抛 ValueError", True, str(error))

    except Exception as error:

        check(name + " -> 抛 ValueError", False, type(error).__name__ + ": " + str(error))


# 向量库不存在时：把 Tool 指向一个临时空目录。
# 这条测的是"Agent 拿到的是可读的失败信息，而不是一个崩溃栈"。
#
# ignore_cleanup_errors 是必须的：
# Chroma 建客户端时会在这个目录里建一个 sqlite 文件并一直持有它，
# Windows 上文件被占用就删不掉，退出时会抛 PermissionError。
# 这跟被测代码无关，是测试收尾的问题，所以直接忽略。
original_db_dir = rag_tool.DB_DIR

try:

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as empty_dir:

        rag_tool.DB_DIR = Path(empty_dir)

        try:

            rag_search(QUESTION)

            check("向量库不存在时明确失败", False, "居然没报错")

        except RuntimeError as error:

            message = str(error)

            check(
                "向量库不存在时明确失败，且告诉 Agent 换别的 Tool",
                "不可用" in message and "其他 Tool" in message,
                message
            )

finally:

    rag_tool.DB_DIR = original_db_dir


# ============================================================
# 8. 测试六：注册生效
# ============================================================

print()
print("=" * 70)
print("测试六：注册生效（TOOLS / schema / Planner 三处都要看得见）")
print("=" * 70)

check(
    "rag_search 在 TOOLS 注册表里",
    "rag_search" in TOOLS,
    "TOOLS 共 " + str(len(TOOLS)) + " 个：" + "、".join(TOOLS.keys())
)

check(
    "注册项带 function / description / parameters",
    all(
        key in TOOLS["rag_search"]
        for key in ["function", "description", "parameters"]
    ),
    ""
)

check(
    "注册的 function 就是 tools/rag_tool.py 里的 rag_search",
    TOOLS["rag_search"]["function"] is rag_search,
    ""
)

schemas = build_tool_schemas()

check(
    "build_tool_schemas() 里能看到 rag_search（模型才知道它存在）",
    any(item["function"]["name"] == "rag_search" for item in schemas),
    "schema 共 " + str(len(schemas)) + " 个"
)

rag_schema = [
    item for item in schemas
    if item["function"]["name"] == "rag_search"
][0]

check(
    "schema 里只有 question 是必填（其余都有默认值）",
    rag_schema["function"]["parameters"].get("required") == ["question"],
    str(rag_schema["function"]["parameters"].get("required"))
)

planner_prompt = build_planner_prompt()

check(
    "Planner 的 Tool 清单里能看到 rag_search",
    "rag_search" in planner_prompt,
    ""
)

dispatched = execute_tool(
    "rag_search",
    {"question": QUESTION, "top_k": 5, "top_n": 2}
)

check(
    "execute_tool() 能正确分发到 rag_search",
    dispatched["returned"] == 2,
    "returned=" + str(dispatched["returned"])
    + " 页=" + str([item["page"] for item in dispatched["evidence"]])
)

check(
    "Tool 里的默认值和 RAG Pipeline 的默认值一致（防止两边改跑偏）",
    (rag_tool.DEFAULT_TOP_K, rag_tool.DEFAULT_TOP_N) == (DEFAULT_K, DEFAULT_TOP_N),
    "Tool=" + str((rag_tool.DEFAULT_TOP_K, rag_tool.DEFAULT_TOP_N))
    + " Pipeline=" + str((DEFAULT_K, DEFAULT_TOP_N))
)


# ============================================================
# 9. 汇总
# ============================================================

passed_count = sum(1 for _, passed, _ in results if passed)
failed = [name for name, passed, _ in results if not passed]

print()
print("=" * 70)
print(
    "测试结果：" + str(passed_count) + " / " + str(len(results)) + " 通过"
)
print("=" * 70)

if failed:

    print("未通过：")

    for name in failed:

        print("  - " + name)

else:

    print("全部通过。")


print()
print("=" * 70)
print("已知局限（不是 bug，是这一版的边界）")
print("=" * 70)
print("""
1. 年报第 8 页「主要会计数据和财务指标」是标准答案页，
   但那是一张宽表，转成纯文本后变成一串数字，
   向量相似度很低，Top-10 也捞不到它。
   现在命中的是第 26 页「营业收入构成」，数字相同、可核对，
   但不是最正的位置。这是"表格类 PDF + 纯文本 Embedding"的结构性问题。

2. 只在一份年报、一个问题上调过参数。
   换公司、换年份、换问法都需要重新看召回质量。

3. Rerank 还是词面加权的 baseline，不是 Cross-Encoder，
   问"营收"、原文写"营业收入"时只能匹配到"营业"。

4. 问法会明显影响召回（实测数据）：
   同样问营业收入，写成
       "…营业收入和净利润是多少？"     -> 答案块向量名次 #9，K=10 能捞到
       "…营业收入和归母净利润是多少？" -> 答案块向量名次 #15，K=10 进不了候选集
   加上"归母"两个字，查询向量就偏向了"归属于母公司股东的净利润"那一段。
   这不是 Tool 的 bug，是"语义检索 + 宽表 PDF"的真实边界。
   补救手段是让 Agent 把 top_k 调大（例如 20）再试一次，
   所以 Tool 的 schema 里保留了 top_k 这个参数。
""")
