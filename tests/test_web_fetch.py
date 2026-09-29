# -*- coding: utf-8 -*-
"""
web_fetch Tool 的测试。

只验证这一件事：这个 Tool 能不能被信任。
所以测的不是"所有情况都对"，而是四条底线：

    1. 一个正常文章 URL 真的能抓到正文；
    2. 抓不到的 URL 会明确失败，而不是返回空字符串假装成功；
    3. max_chars 真的能限制返回给 Agent 的正文长度；
    4. 注册之后，它真的能被 Planner 和模型看见。

第 4 条是这次改动里最容易被忽略、但最容易出错的一条：
Tool 写出来了，但没进 Planner 清单或没进 Function Calling schema，
模型就永远不知道它存在，代码本身没问题、功能却完全没用。

运行方式（和 tests/ 下其他脚本一样）：

    python -m tests.test_web_fetch

注意：这个测试会真的联网，因为要测的就是网络抓取。
"""

import sys

from pathlib import Path


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


from agent.tool_registry import TOOLS
from agent.tool_executor import execute_tool
from agent.financial_agent import (
    build_tool_schemas,
    build_planner_prompt
)
from tools.search_tool import search_web
from tools.web_fetch_tool import web_fetch


# ============================================================
# 2. 结果收集
#
# 不用 assert 直接中断，而是一条条记录、最后统一汇总。
# 这样一次运行能看到全部问题，不用改一个跑一次。
# ============================================================

results = []


def check(name, passed, detail=""):

    results.append((name, passed, detail))

    print(("  [通过] " if passed else "  [失败] ") + name)

    if detail:

        print("         " + str(detail)[:300])


EXPECTED_KEYS = [
    "url",
    "final_url",
    "status",
    "title",
    "text",
    "truncated",
    "error"
]


# ============================================================
# 3. 准备一个真实文章 URL
#
# 不写死网址，因为写死的网址会失效（新闻页会被删、会改路径）。
# 改成先用 search_web 找一个，再拿它来测 ——
# 这同时也验证了 search_web → web_fetch 这条真实链路能不能走通。
# ============================================================

print("=" * 60)
print("准备：用 search_web 找一个真实文章 URL")
print("=" * 60)


candidate_urls = []

for query in ["中际旭创 三季报", "平安银行 分红 公告"]:

    try:

        for item in search_web(query, max_results=5):

            if item["url"] not in candidate_urls:

                candidate_urls.append(item["url"])

    except Exception as error:

        print("  搜索失败（" + query + "）：" + str(error))


print("  候选 URL：" + str(len(candidate_urls)) + " 个")


# ============================================================
# 4. 测试一：正常文章能抓到正文
#
# 候选 URL 里混着聚合页和文章页，一个个试，找到第一个能抓的。
# 不需要全部都能抓 —— 抓不到的会明确失败，这本身就是预期行为。
# ============================================================

print()
print("=" * 60)
print("测试一：正常文章 URL 能抓到正文")
print("=" * 60)


good_result = None
good_url = ""

for url in candidate_urls:

    try:

        result = web_fetch(url)

        if len(result["text"]) >= 200:

            good_result = result
            good_url = url

            break

    except Exception:

        # 这条抓不到就换下一条，正常的。
        continue


if good_result is None:

    check(
        "能找到一个可以抓取正文的文章 URL",
        False,
        "候选 URL 全部抓取失败，请检查网络或 search_web 是否正常"
    )

else:

    check(
        "能找到一个可以抓取正文的文章 URL",
        True,
        good_url
    )

    check(
        "返回结构包含全部 7 个字段",
        all(key in good_result for key in EXPECTED_KEYS),
        "实际字段：" + str(sorted(good_result.keys()))
    )

    check(
        "status 是 200",
        good_result["status"] == 200,
        "status=" + str(good_result["status"])
    )

    check(
        "抓到了标题",
        len(good_result["title"]) > 0,
        good_result["title"]
    )

    check(
        "final_url 非空（重定向后地址）",
        len(good_result["final_url"]) > 0,
        good_result["final_url"]
    )

    check(
        "正文长度超过 200 字",
        len(good_result["text"]) >= 200,
        "正文 " + str(len(good_result["text"])) + " 字，开头："
        + good_result["text"][:80]
    )

    check(
        "正文里没有残留的 HTML 标签",
        "<" not in good_result["text"]
        and ">" not in good_result["text"],
        good_result["text"][:120]
    )

    check(
        "error 字段为 None",
        good_result["error"] is None,
        str(good_result["error"])
    )


# ============================================================
# 5. 测试二：抓不到时必须明确失败
#
# 这三条 URL 分别对应上一轮实测出来的三类真实问题：
#   403       —— 知乎，明确拒绝抓取
#   WAF 拦截页 —— 雪球，返回 200 但内容是一段加密验证串
#   聚合列表页 —— 证券之星的"相关新闻"列表，没有正文
#
# 允许其中任意一条因为对方改版而变得可抓（那时它就不该算失败样本），
# 所以只要求「至少有一条明确报错」，并且报错信息要能看懂。
# ============================================================

print()
print("=" * 60)
print("测试二：抓不到时明确失败")
print("=" * 60)


bad_urls = [
    "https://www.zhihu.com/question/1",
    "https://xueqiu.com/S/SZ300308",
    "https://news.stockstar.com/info/dstock_300308.shtml"
]


failed_clearly = []
succeeded_unexpectedly = []


for url in bad_urls:

    try:

        result = web_fetch(url)

        succeeded_unexpectedly.append(
            (url, "居然成功，正文 " + str(len(result["text"])) + " 字")
        )

    except RuntimeError as error:

        failed_clearly.append((url, str(error)))

    except Exception as error:

        failed_clearly.append(
            (url, type(error).__name__ + "：" + str(error))
        )


for url, message in failed_clearly:

    print("  " + url)
    print("    -> " + message)


for url, message in succeeded_unexpectedly:

    print("  " + url)
    print("    -> " + message)


check(
    "至少有一条抓取失败的 URL 明确报错",
    len(failed_clearly) > 0,
    str(len(failed_clearly)) + " 条报错 / " + str(len(bad_urls)) + " 条样本"
)

check(
    "报错信息不为空且可读",
    all(len(message) > 10 for _, message in failed_clearly),
    ""
)

# 空地址和非法协议，在发请求之前就该被拦住
for bad_argument, expected_error in [
    ("", ValueError),
    ("   ", ValueError),
    ("ftp://example.com/a.txt", ValueError)
]:

    try:

        web_fetch(bad_argument)

        check(
            "非法参数被拦住：" + repr(bad_argument),
            False,
            "没有报错"
        )

    except expected_error:

        check(
            "非法参数被拦住：" + repr(bad_argument),
            True,
            ""
        )

    except Exception as error:

        check(
            "非法参数被拦住：" + repr(bad_argument),
            False,
            "报错类型不对：" + type(error).__name__
        )


# ============================================================
# 6. 测试三：max_chars 真的生效
#
# 第 7 条要求的 max_chars，如果不生效，
# 一篇 6000 字的新闻会整段进上下文，几轮就把 token 撑爆。
# 所以这里必须实测长度，而不是"看代码里有没有 [:max_chars]"。
# ============================================================

print()
print("=" * 60)
print("测试三：max_chars 生效")
print("=" * 60)


if good_result is None:

    check("max_chars 生效", False, "没有可用的文章 URL，跳过")

else:

    # 100 远小于任何一篇文章，一定触发截断
    small = web_fetch(good_url, max_chars=100)

    check(
        "max_chars=100 时正文不超过 100 字",
        len(small["text"]) <= 100,
        "实际 " + str(len(small["text"])) + " 字"
    )

    check(
        "max_chars=100 时 truncated 为 True",
        small["truncated"] is True,
        "truncated=" + str(small["truncated"])
    )

    check(
        "截断后正文是原文的前缀（没有截乱）",
        good_result["text"].startswith(small["text"]),
        ""
    )

    # 默认值 3000
    default_result = web_fetch(good_url)

    check(
        "默认 max_chars 下正文不超过 3000 字",
        len(default_result["text"]) <= 3000,
        "实际 " + str(len(default_result["text"])) + " 字"
    )

    # 给一个足够大的值，就不该截断
    large = web_fetch(good_url, max_chars=100000)

    check(
        "max_chars 足够大时 truncated 为 False",
        large["truncated"] is False,
        "truncated=" + str(large["truncated"])
    )

    check(
        "max_chars 变大后拿到的正文不短于小值那一次",
        len(large["text"]) >= len(small["text"]),
        str(len(small["text"])) + " -> " + str(len(large["text"]))
    )

    # 非法 max_chars
    for bad_max_chars in [0, -1]:

        try:

            web_fetch(good_url, max_chars=bad_max_chars)

            check(
                "非法 max_chars 被拦住：" + str(bad_max_chars),
                False,
                "没有报错"
            )

        except ValueError:

            check(
                "非法 max_chars 被拦住：" + str(bad_max_chars),
                True,
                ""
            )


# ============================================================
# 7. 测试四：注册生效
#
# 这是最容易漏测、后果最严重的一条：
# Tool 写对了、注册漏了，代码全对但 Agent 完全用不了它。
# ============================================================

print()
print("=" * 60)
print("测试四：注册后能被 Planner 和模型看见")
print("=" * 60)


check(
    "web_fetch 在 TOOLS 注册表里",
    "web_fetch" in TOOLS,
    "当前 Tool：" + str(list(TOOLS.keys()))
)

check(
    "原有 Tool 一个都没丢",
    all(
        name in TOOLS
        for name in [
            "search_stock",
            "get_financial_statements",
            "calculate_financial_indicators",
            "get_stock_quote",
            "get_price_history",
            "get_stock_valuation",
            "search_web",
            "web_fetch",
        ]
    ),
    "当前 " + str(len(TOOLS)) + " 个：" + "、".join(TOOLS.keys())
)

tool_names = [
    item["function"]["name"]
    for item in build_tool_schemas()
]

check(
    "Function Calling schema 里包含 web_fetch",
    "web_fetch" in tool_names,
    "schema 里的 Tool：" + str(tool_names)
)

planner_prompt = build_planner_prompt()

check(
    "Planner 清单里包含 web_fetch",
    "web_fetch" in planner_prompt,
    ""
)

check(
    "web_fetch 的 schema 声明了 url 参数和 required",
    any(
        item["function"]["name"] == "web_fetch"
        and "url" in item["function"]["parameters"]["properties"]
        and item["function"]["parameters"]["required"] == ["url"]
        for item in build_tool_schemas()
    ),
    ""
)

# execute_tool 是 Agent 真正调用 Tool 时走的那条路，
# 单独注册到字典里但这条路走不通，等于没注册。
try:

    executed = execute_tool(
        "web_fetch",
        {"url": good_url} if good_url else {"url": ""}
    )

    check(
        "execute_tool 能分发到 web_fetch",
        isinstance(executed, dict) and "text" in executed,
        "返回字段：" + str(sorted(executed.keys()))
    )

except Exception as error:

    # 上面那个 URL 抓不到的话，至少不应该报"未知 Tool"
    check(
        "execute_tool 能分发到 web_fetch",
        "未知 Tool" not in str(error),
        type(error).__name__ + "：" + str(error)[:150]
    )


# ============================================================
# 8. 汇总
# ============================================================

passed_count = sum(1 for _, passed, _ in results if passed)
failed_count = len(results) - passed_count

print()
print("=" * 60)
print("测试结果：" + str(passed_count) + " / " + str(len(results)) + " 通过")
print("=" * 60)

if failed_count:

    print("未通过的项：")

    for name, passed, detail in results:

        if not passed:

            print("  - " + name + "  " + str(detail)[:200])

    sys.exit(1)

print("全部通过。")
