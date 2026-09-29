# -*- coding: utf-8 -*-
"""
web_ui（可视化界面）的测试。

这个界面本身不含业务逻辑，它的全部价值在**翻译**：
把 LangGraph 吐出来的中间状态，翻译成浏览器能画的东西。

所以出问题的地方不会是"图跑错了"，而是这四种：

    1. 前端文件缺了 / 路径错了    -> 打开页面就 404
    2. Tool 结果翻译错了          -> 界面上一堆看不懂的字段
    3. 事件漏发                   -> 进度条卡住、报告不出现
    4. 并发没锁住                 -> 两个问题一起跑，日志串成一锅粥

这四条都不需要 API Key 就能测，所以这个脚本是**离线可跑**的：
不调用任何大模型，用一张"假图"逐节点喂事件，检查翻译结果对不对。

只要它通过，就说明界面是好的 ——
真跑一次只是为了让数据更真实，不是为了验证逻辑。

运行方式：

    python -m tests.test_web_ui
"""

import json
import sys
import threading
import time
import urllib.request

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


import web_ui

from web_ui import (
    GRAPH_NODES,
    summarize_tool,
    iter_events,
    _domain,
    _bind_server,
    _run_lock,
    INDEX_HTML
)


# ============================================================
# 测试脚手架
# ============================================================

passed = 0

failed = 0


def check(name, condition, detail=""):

    global passed, failed

    if condition:

        passed += 1

        print("  [通过] " + name + (("  → " + str(detail)) if detail else ""))

    else:

        failed += 1

        print("  [失败] " + name + (("  → " + str(detail)) if detail else ""))


def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. 前端文件
# ============================================================

section("1. 前端文件")

check("web/index.html 存在", INDEX_HTML.exists(), str(INDEX_HTML))

html_text = INDEX_HTML.read_text(encoding="utf-8") if INDEX_HTML.exists() else ""

check("没有外链 CDN（离线也能打开）",
      "http://" not in html_text.replace("http://www.w3.org", "") and "<script src" not in html_text,
      "只允许 svg 命名空间这类非网络链接")

for keyword in ["EventSource", "pipeline", "mdToHtml", "renderEvidence"]:
    check("前端包含关键实现：" + keyword, keyword in html_text)

check("深浅色两套主题都定义了",
      'data-theme="dark"' in html_text and "prefers-color-scheme" in html_text)


# ============================================================
# 2. Tool 结果翻译
# ============================================================
#
# 这一段是界面质量的关键：
# rag_search 返回的是结构化证据，front 需要的是"第 26 页 + 原文片段"。
# 翻译错了，界面上就只剩一串看不懂的 JSON。

section("2. Tool 结果 → 界面结构")

RAG_ITEM = {
    "tool": "rag_search",
    "status": "success",
    "arguments": {"question": "中际旭创2024年营业收入", "top_k": 10, "top_n": 5},
    "elapsed": 1.2,
    "result": {
        "question": "中际旭创2024年营业收入",
        "source": "中际旭创 2024 年年报.pdf",
        "available_sources": ["中际旭创 2024 年年报.pdf"],
        "top_k": 10,
        "top_n": 5,
        "returned": 2,
        "evidence": [
            {
                "chunk_id": "中际旭创_2024_年年报-p26-c57",
                "source": "中际旭创 2024 年年报.pdf",
                "page": 26,
                "text": "营业收入合计 23,862,159,738.37 元",
                "score": 0.5287
            },
            {
                "chunk_id": "中际旭创_2024_年年报-p188-c400",
                "source": "中际旭创 2024 年年报.pdf",
                "page": 188,
                "text": "归属于母公司股东的净利润 5,171,485,967.85 元",
                "score": 0.6015
            }
        ]
    }
}

rag_view = summarize_tool(RAG_ITEM)

check("rag_search：判定为成功", rag_view["ok"] is True)

check("rag_search：摘要里说出了年报名和条数",
      "中际旭创 2024 年年报.pdf" in rag_view["summary"] and "2" in rag_view["summary"],
      rag_view["summary"])

check("rag_search：抽出 2 条可追溯证据", len(rag_view["citations"]) == 2)

first = rag_view["citations"][0]

check("rag_search：证据带着页码", first["page"] == 26)

check("rag_search：证据带着 chunk_id", "p26-c57" in first["chunk_id"])

check("rag_search：证据带着 source", first["source"] == "中际旭创 2024 年年报.pdf")

check("rag_search：证据带着相似度", first["score"] == 0.5287)

check("rag_search：证据保留了原文片段",
      "23,862,159,738.37" in first["preview"])

check("rag_search：证据被标记为 pdf 类", first["kind"] == "pdf")


# ---- 回归：Agent 不指定 source 时 source 就是 None ----
#
# 这是真跑端到端时发现的：
# Agent 从来不传 source（它就是要检索整个年报库），
# 于是 rag_search 返回的 source 是 None，
# 界面上出现了"从《None》检索到 5 段原文"，看着像坏了。
# 现在改成从证据里反推年报表名，这里锁住这个行为。

NO_SOURCE_ITEM = {
    "tool": "rag_search",
    "status": "success",
    "arguments": {"question": "营业收入"},
    "result": dict(RAG_ITEM["result"], source=None, available_sources=[])
}

no_source_view = summarize_tool(NO_SOURCE_ITEM)

check("rag_search：不指定 source 时不把 None 打到界面上",
      "None" not in no_source_view["summary"]
      and "中际旭创 2024 年年报.pdf" in no_source_view["summary"],
      no_source_view["summary"])


WEB_SEARCH_ITEM = {
    "tool": "search_web",
    "status": "success",
    "arguments": {"query": "中际旭创 新闻"},
    "result": [
        {"title": "中际旭创业绩快报", "url": "https://finance.eastmoney.com/a/1.html", "snippet": "…"},
        {"title": "光模块龙头", "url": "https://www.nbd.com.cn/articles/2.html", "snippet": "…"}
    ]
}

web_view = summarize_tool(WEB_SEARCH_ITEM)

check("search_web：摘要说了条数", "2" in web_view["summary"], web_view["summary"])

check("search_web：抽出的来源是网页",
      len(web_view["citations"]) == 2 and web_view["citations"][0]["kind"] == "web")

check("search_web：来源标签取的是域名",
      web_view["citations"][0]["label"] == "finance.eastmoney.com",
      web_view["citations"][0]["label"])

check("search_web：保留了原始网址",
      web_view["citations"][1]["url"] == "https://www.nbd.com.cn/articles/2.html")


FETCH_ITEM = {
    "tool": "web_fetch",
    "status": "success",
    "arguments": {"url": "https://finance.eastmoney.com/a/1.html"},
    "result": {
        "url": "https://finance.eastmoney.com/a/1.html",
        "final_url": "https://finance.eastmoney.com/a/1.html",
        "status": 200,
        "title": "中际旭创一季报出炉",
        "text": "正文内容若干字",
        "truncated": False,
        "error": None
    }
}

fetch_view = summarize_tool(FETCH_ITEM)

check("web_fetch：摘要里带标题和字数",
      "中际旭创一季报出炉" in fetch_view["summary"]
      and str(len(FETCH_ITEM["result"]["text"])) + " 字" in fetch_view["summary"],
      fetch_view["summary"])

check("web_fetch：抽出一条网页来源",
      len(fetch_view["citations"]) == 1 and fetch_view["citations"][0]["kind"] == "web")


FAILED_ITEM = {
    "tool": "web_fetch",
    "status": "failed",
    "arguments": {"url": "https://xueqiu.com/S/SZ300308"},
    "result": "网页抓取失败：返回的是验证/拦截页面，不是网页正文（…）"
}

fail_view = summarize_tool(FAILED_ITEM)

check("失败的 Tool 被标成失败", fail_view["ok"] is False)

check("失败原因原样传给界面（不吞掉）",
      "验证/拦截页面" in fail_view["summary"],
      fail_view["summary"][:60])

check("失败时不给任何来源（避免假证据）",
      "citations" not in fail_view or not fail_view["citations"])


FIN_ITEM = {
    "tool": "get_financial_statements",
    "status": "success",
    "arguments": {"thscode": "300308.SZ"},
    "result": {"income": {"rows": [1, 2, 3, 4]}, "balance": {"rows": [1, 2]}}
}

fin_view = summarize_tool(FIN_ITEM)

check("财务类 Tool 也能给出人话摘要（数出里面的行数）",
      "income 4 条" in fin_view["summary"], fin_view["summary"])

check("域名提取正确",
      _domain("https://guba.eastmoney.com/list,000001.html") == "guba.eastmoney.com")

check("域名提取对空串不崩", _domain("") == "")


# ============================================================
# 3. 事件流翻译（用假图，不打 API）
# ============================================================
#
# 这是整个测试里最有价值的一段。
#
# iter_events 里最容易出错的地方是"增量发放"：
# LangGraph 每次返回的是**完整**的 tool_results，
# 如果不做增量判断，一次 Tool 调用会被推 N 遍，界面上就出现重复卡片。
#
# 下面这张假图故意让 tools 节点返回完整列表，
# 用来验证"只推新增的那几条"。

section("3. LangGraph → SSE 事件（假图）")


class FakeApp:
    """一张不联网的假图，按真实顺序吐节点更新。"""

    def stream(self, initial_state, config=None):

        # ① planner：初始化 messages
        yield {
            "planner": {
                "research_plan": {"goal": "查年报数字", "steps": [{"step": 1, "tool": "rag_search"}]},
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."}
                ]
            }
        }

        # ② agent：决定调 rag_search
        yield {
            "agent": {
                "messages": [
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "rag_search"}}]}
                ],
                "tool_round": 1
            }
        }

        # ③ tools：返回**完整**列表（只有一个元素）
        yield {
            "tools": {
                "messages": [{"role": "tool", "content": "..."}],
                "tool_results": [
                    {"tool": "rag_search", "status": "success",
                     "arguments": {"question": "营业收入"}, "elapsed": 1.2,
                     "result": RAG_ITEM["result"]}
                ]
            }
        }

        # ④ agent 第二次：没有 tool_calls，只写 stop_reasons
        #    注意：这一轮**不写 messages**，这是真实行为
        yield {"agent": {"stop_reasons": ["no_tool_calls"]}}

        # ⑤ tools 再跑一次：这次返回**完整的两个**元素
        #    用来验证增量逻辑 —— 第 1 条不能重复推送
        yield {
            "tools": {
                "messages": [{"role": "tool", "content": "..."}],
                "tool_results": [
                    {"tool": "rag_search", "status": "success",
                     "arguments": {"question": "营业收入"}, "elapsed": 1.2,
                     "result": RAG_ITEM["result"]},
                    {"tool": "web_fetch", "status": "failed",
                     "arguments": {"url": "https://xueqiu.com/S/SZ300308"},
                     "elapsed": 0.6,
                     "result": "网页抓取失败：返回的是验证/拦截页面，不是网页正文"}
                ]
            }
        }

        # ⑥ evidence
        yield {
            "evidence": {
                "evidence": {
                    "completed_steps": [{"step": 1, "tool": "rag_search", "evidence": "拿到营业收入"}],
                    "data_limitations": ["只入库一份年报"],
                    "summary": "证据概要"
                }
            }
        }

        # ⑦ reviewer：FAIL，触发回边
        yield {
            "reviewer": {
                "review": {"status": "FAIL", "verdict": "证据不足",
                           "issues": ["缺少归母净利润"],
                           "unsupported_claims": ["净利润数字无出处"]}
            }
        }

        # ⑧ prepare_retry
        yield {"prepare_retry": {"messages": [{"role": "user", "content": "..."}],
                                 "retry_count": 1, "tool_round": 0}}

        # ⑨ final
        yield {"final": {"final_answer": "# 研究报告\n\n营业收入 238.62 亿元。"}}

        # ⑩ save
        yield {"save": {"run_dir": "runs/fake_run"}}


# 把 web_ui 里 import 的 build_graph 换掉。
# iter_events 是在函数体内 import 的，所以改模块属性就能生效。
import agent.graph_agent as graph_module

_real_build_graph = graph_module.build_graph

graph_module.build_graph = lambda: FakeApp()

try:

    events = list(iter_events("测试问题"))

finally:

    graph_module.build_graph = _real_build_graph


types = [e["type"] for e in events]

check("第一个事件是 start", types[0] == "start", types[:3])

check("最后一个事件是 done", types[-1] == "done")

check("start 事件带上了节点清单",
      len(events[0]["nodes"]) == len(GRAPH_NODES))

check("start 事件带上了可用 Tool 清单",
      "rag_search" in events[0]["tools"] and "web_fetch" in events[0]["tools"],
      "%d 个" % len(events[0]["tools"]))

node_events = [e for e in events if e["type"] == "node"]

check("8 个节点都发了事件（含回边的 prepare_retry）",
      len(node_events) == 10 and node_events[0]["node"] == "planner",
      "实际 %d 个：%s" % (len(node_events), [e["node"] for e in node_events]))

check("节点事件带中文标签",
      node_events[0]["label"] == "规划 Planner", node_events[0]["label"])

check("节点事件带耗时", isinstance(node_events[0]["elapsed"], (int, float)))

tool_events = [e for e in events if e["type"] == "tool"]

check("Tool 事件只推新增，不重复推（关键）",
      len(tool_events) == 2,
      "假图里 tools 节点跑了两次、第二次返回完整列表，应只产生 2 个事件，实际 %d" % len(tool_events))

check("Tool 事件顺序正确",
      [e["tool"] for e in tool_events] == ["rag_search", "web_fetch"],
      [e["tool"] for e in tool_events])

check("成功的 Tool 事件带 2 条证据",
      len(tool_events[0]["citations"]) == 2)

check("失败的 Tool 事件标记 ok=False",
      tool_events[1]["ok"] is False)

agent_events = [e for e in events if e["type"] == "agent_step"]

check("agent 决策发了 2 次（第 2 次=停止调 Tool）",
      len(agent_events) == 2 and agent_events[0]["tool_calls"] == ["rag_search"],
      [e["tool_calls"] for e in agent_events])

check("agent 停止那一轮被正确识别",
      agent_events[1]["done"] is True)

check("plan 事件带上了研究计划",
      any(e["type"] == "plan" and e["plan"].get("goal") for e in events))

review_events = [e for e in events if e["type"] == "review"]

check("review 事件带上了 FAIL 与理由",
      review_events and review_events[0]["status"] == "FAIL"
      and review_events[0]["issues"] == ["缺少归母净利润"])

check("review 事件带上了无依据的结论",
      review_events[0]["unsupported_claims"] == ["净利润数字无出处"])

check("retry 事件带上了次数",
      any(e["type"] == "retry" and e["count"] == 1 for e in events))

check("report 事件带上了报告正文",
      any(e["type"] == "report" and "238.62" in e["text"] for e in events))

check("save 事件带上了落盘目录",
      any(e["type"] == "save" and e["run_dir"] == "runs/fake_run" for e in events))

done = events[-1]

check("done 事件汇总了审核结果与补充研究次数",
      done["review"]["status"] == "FAIL" and done["retry_count"] == 1,
      "review=%s retry=%s" % (done["review"]["status"], done["retry_count"]))

# tool_round 在假图里最后被 prepare_retry 归零了（这是 graph_agent 的真实行为：
# 补充研究会把 tool_round 重置，让新的一轮重新计数）。
# 断言它是个整数就够了，不要写死具体值 —— 那个值由图的走向决定。
check("done 事件带上了 tool_round（且是整数）",
      isinstance(done["tool_round"], int), done["tool_round"])

check("done 事件汇总了全部 Tool 调用",
      len(done["tool_results"]) == 2)

check("所有事件都可 JSON 序列化（SSE 的前提）",
      len(json.dumps(events, ensure_ascii=False, default=str)) > 0)

check("iter_events 结束后把 stdout 还回去了",
      sys.stdout is not None and not isinstance(sys.stdout, web_ui._OutputCapture))


# ============================================================
# 4. 事件流不携带不可读对象
# ============================================================
#
# SSE 只能传字符串。如果某天有人往事件里塞了一个不认识的
# 对象（比如 datetime / numpy 数组），界面会静默炸掉。
# 这里把事件在"最难的场景"下序列化一遍，提前拦住。

section("4. 事件序列化健壮性")

try:

    json.dumps(events, ensure_ascii=False)

    check("事件无需 default=str 也能序列化（纯净）", True)

except TypeError as error:

    check("事件无需 default=str 也能序列化（纯净）", False, str(error))


# ============================================================
# 5. HTTP 层
# ============================================================
#
# 前面测的是"翻译"，这里测的是"服务能不能真的把东西发出去"。
# 用真的 HTTP 服务器 + urllib，但不打大模型：
# 把 iter_events 临时换成一个固定的假事件流。

section("5. HTTP 服务")


def fake_events(question):

    yield {"type": "start", "question": question, "nodes": GRAPH_NODES, "tools": ["rag_search"]}

    yield {"type": "node", "node": "planner", "label": "规划 Planner", "seq": 1, "elapsed": 0.1}

    yield {"type": "report", "text": "# 假报告\n\n营业收入 238.62 亿元。"}

    yield {"type": "done", "elapsed": 0.2, "run_dir": "runs/fake", "tool_round": 1,
           "stop_reasons": ["no_tool_calls"], "retry_count": 0,
           "review": {"status": "PASS"}, "tool_results": []}


_real_iter_events = web_ui.iter_events

web_ui.iter_events = fake_events

server = None

try:

    server, port = _bind_server("127.0.0.1", 8876)

    thread = threading.Thread(target=server.serve_forever, daemon=True)

    thread.start()

    time.sleep(0.2)

    base = "http://127.0.0.1:%d" % port

    check("端口被占用时能自动往后找（这里拿到 %d）" % port, port >= 8876)

    # ---- / ----
    with urllib.request.urlopen(base + "/", timeout=5) as response:

        body = response.read().decode("utf-8")

        ctype = response.headers.get("Content-Type")

    check("GET / 返回 200", response.status == 200)

    check("GET / 是 UTF-8 的 HTML",
          "text/html" in ctype and "charset=utf-8" in ctype, ctype)

    check("GET / 返回的是前端页面",
          "FinResearch-Agent" in body and "EventSource" in body)

    # ---- /api/health ----
    with urllib.request.urlopen(base + "/api/health", timeout=5) as response:

        health = json.loads(response.read().decode("utf-8"))

    check("GET /api/health 返回可用状态",
          health.get("ok") is True and "busy" in health, health)

    # ---- /api/stream ----
    with urllib.request.urlopen(base + "/api/stream?q=%E6%B5%8B%E8%AF%95", timeout=20) as response:

        ctype = response.headers.get("Content-Type")

        raw = response.read().decode("utf-8")

    check("SSE 的 Content-Type 正确",
          "text/event-stream" in ctype, ctype)

    check("SSE 明确要求不缓存",
          "no-cache" in (response.headers.get("Cache-Control") or ""),
          response.headers.get("Cache-Control"))

    frames = [line for line in raw.split("\n") if line.startswith("data: ")]

    check("返回了 SSE 数据帧", len(frames) > 0, "%d 帧" % len(frames))

    check("每帧都是合法 JSON", all(
        isinstance(json.loads(f[6:]), dict) for f in frames
    ))

    payloads = [json.loads(f[6:]) for f in frames]

    check("帧内容与假事件一致",
          [p["type"] for p in payloads][:3] == ["start", "node", "report"],
          [p["type"] for p in payloads])

    check("SSE 以 eof 收尾（浏览器据此不再重连）",
          payloads[-1]["type"] == "eof")

    check("中文没有乱码",
          any("238.62" in (p.get("text") or "") for p in payloads))

    # ---- 空问题 ----
    with urllib.request.urlopen(base + "/api/stream?q=", timeout=5) as response:

        empty_raw = response.read().decode("utf-8")

    check("空问题被拦住，不跑图",
          '"error"' in empty_raw and '"eof"' in empty_raw)

    # ---- 未知路径 ----
    try:

        urllib.request.urlopen(base + "/nope", timeout=5)

        not_found = False

    except urllib.error.HTTPError as error:

        not_found = (error.code == 404)

    check("未知路径返回 404", not_found)

    # ---- 并发保护 ----
    # 手工占住锁，模拟"已经有一个任务在跑"
    _run_lock.acquire()

    try:

        with urllib.request.urlopen(base + "/api/stream?q=test", timeout=5) as response:

            busy_raw = response.read().decode("utf-8")

        check("已有任务在跑时，第二个请求被明确拒绝",
              "已经有一个任务在跑" in busy_raw, busy_raw[:120])

    finally:

        _run_lock.release()

finally:

    web_ui.iter_events = _real_iter_events

    if server is not None:

        server.shutdown()

        server.server_close()


# ============================================================
# 汇总
# ============================================================

print()
print("=" * 70)
print("测试结果")
print("=" * 70)
print("通过 %d 项，失败 %d 项" % (passed, failed))

if failed:

    print()
    print("有失败项，先修再继续。")
    sys.exit(1)

print()
print("全部通过。界面侧的逻辑可以放心。")

print()
print("=" * 70)
print("这个测试没覆盖什么（要看真效果得真跑）")
print("=" * 70)
print("""
1. 报告排版好不好看。
   mdToHtml 只做了行为测试（函数存在），
   排版效果得用眼睛看。

2. 真实运行时的观感。
   假图只有 10 个节点、0.2 秒。
   真跑一次要 1~3 分钟，进度条的节奏、日志的刷屏速度
   都得实际跑一遍才知道。

3. 大模型返回的 Markdown 有多花。
   这个测试里的报告是我手写的规范 Markdown。
   真实报告可能有嵌套列表、表格里带竖线等，
   mdToHtml 对这些边界的处理属于"够用就行"。

真跑一次的命令：

    python -m web_ui
    （或用 main.py --ui，或双击 start_ui.bat）
""")
