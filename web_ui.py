# -*- coding: utf-8 -*-
"""
FinResearch-Agent 的可视化界面（本地网页版）。

============================================================
为什么用标准库而不是 FastAPI / Gradio / Streamlit
============================================================

这台机器上 fastapi、uvicorn、flask、gradio、streamlit **一个都没装**。

为一个"看流程图 + 看报告"的本地界面去装一套 Web 框架，
等于为了展示 Agent 先把项目搞复杂 —— 和项目原则相反。

Python 标准库自带 http.server，足够撑起这个界面：
    静态页面      自己返回一个 HTML 文件
    实时推送      用 SSE（Server-Sent Events）
    并发          用 ThreadingHTTPServer

============================================================
为什么是 SSE 而不是 WebSocket
============================================================

Agent 一次要跑 1~3 分钟，中间有十几个步骤。

数据流是**单向**的：服务端 → 浏览器。
浏览器只需要"接收发生了什么"，不需要往服务端推东西。

SSE 正好是干这个的，而且浏览器自带 EventSource，
前端一行库都不用引。WebSocket 能做但属于杀鸡用牛刀。

============================================================
界面能看到什么
============================================================

左边：Graph 的节点状态（实时点亮）+ 每一次 Tool 调用的流水
右边：最终报告 / 证据（年报页码、网页来源）/ 运行日志 / 原始数据

这些数据全部来自 app.stream() —— 也就是 LangGraph 自己吐出来的
每一步中间结果，不是我们额外埋点造出来的。
"""

import json
import sys
import threading
import time
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


INDEX_HTML = PROJECT_ROOT / "web" / "index.html"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


# ============================================================
# 1. 节点元信息
# ============================================================
#
# 这份表是给界面看的，不是给 Agent 看的。
# 它把 graph_agent.py 里的英文节点名翻译成人能看懂的中文，
# 并标出这个节点是不是在调大模型。
#
# 界面靠它画左侧那条流程图。

GRAPH_NODES = [
    {"name": "planner",        "label": "规划 Planner",      "desc": "把问题拆成研究计划",        "llm": True},
    {"name": "agent",          "label": "决策 Agent",        "desc": "决定下一步调哪个 Tool",     "llm": True},
    {"name": "tools",          "label": "执行 Tools",        "desc": "真正去取数据",              "llm": False},
    {"name": "evidence",       "label": "证据融合 Evidence", "desc": "把 Tool 结果整理成证据",    "llm": True},
    {"name": "reviewer",       "label": "审核 Reviewer",     "desc": "判断证据够不够",            "llm": True},
    {"name": "prepare_retry",  "label": "补充研究 Retry",    "desc": "FAIL 后回去再查一轮",       "llm": False},
    {"name": "final",          "label": "生成报告 Final",    "desc": "写出最终研究报告",          "llm": True},
    {"name": "save",           "label": "落盘 Save",         "desc": "把结果写进 runs/",          "llm": False},
]

NODE_LABEL = {item["name"]: item["label"] for item in GRAPH_NODES}


# ============================================================
# 2. stdout 捕获
# ============================================================
#
# graph_agent 和各 Tool 会 print 一堆进度信息。
# 在网页界面里，这些文字不能丢 ——
# 它是"到底发生了什么"最原始的证据（尤其 Tool 报错时）。
#
# 所以把 stdout 换成一个自己写的对象：
#   一份攒起来，随 SSE 推给浏览器
#   一份照样转发到真终端，命令行运行时观感不变

class _OutputCapture:

    def __init__(self, real_stream=None):

        self._lock = threading.Lock()

        self._lines = []

        self._buffer = ""

        self._real = real_stream

    def write(self, text):

        text = str(text)

        with self._lock:

            self._buffer += text

            # 只按整行收集，避免半行被推给浏览器
            while "\n" in self._buffer:

                line, self._buffer = self._buffer.split("\n", 1)

                line = line.rstrip()

                # 进度条会刷几千行 \r，过长和空行都不要
                if line:

                    self._lines.append(line[:500])

        if self._real is not None:

            try:

                self._real.write(text)

            except Exception:

                # 终端编码不支持某个字符时，不能让日志把主流程带崩
                pass

    def flush(self):

        if self._real is not None:

            try:

                self._real.flush()

            except Exception:

                pass

    def isatty(self):

        return False

    def drain(self):

        """把攒下的行取走，并清空。"""

        with self._lock:

            lines = self._lines

            self._lines = []

        return lines


# ============================================================
# 3. Tool 结果的"人话摘要"
# ============================================================
#
# Tool 返回的是结构化 dict，直接糊到界面上没人看得懂。
# 这里按 Tool 名字各写一句话，再把"可追溯信息"单独抽出来。
#
# 抽出来的 citations 是界面右侧"证据"面板的数据源，
# 也是这个项目最核心的东西：
#     报告里的每个数字，都要能落到 年报第几页 / 哪个网址。

def summarize_tool(item):
    """
    把一个 tool_results 条目翻译成界面能直接渲染的结构。

    入参 item 来自 state["tool_results"]，形如：
        {"tool": "rag_search", "arguments": {...},
         "status": "success", "result": {...}, "elapsed": 1.2}
    """

    tool = item.get("tool")

    status = item.get("status")

    result = item.get("result")

    if status != "success":

        # 失败时 result 就是那句可读的错误说明
        return {
            "ok": False,
            "summary": str(result)[:400]
        }

    summary = _tool_summary(tool, result)

    return {
        "ok": True,
        "summary": summary,
        "citations": _tool_citations(tool, result)
    }


def _tool_summary(tool, result):

    if tool == "rag_search" and isinstance(result, dict):

        returned = result.get("returned", len(result.get("evidence") or []))

        return "从《%s》检索到 %d 段原文（向量捞 %s 条 → 重排留 %s 条）" % (
            _rag_source_label(result),
            returned,
            result.get("top_k"),
            result.get("top_n")
        )

    if tool == "search_web" and isinstance(result, list):

        return "搜索到 %d 条结果" % len(result)

    if tool == "web_fetch" and isinstance(result, dict):

        return "抓到《%s》正文 %d 字%s" % (
            str(result.get("title") or result.get("final_url") or "")[:60],
            len(result.get("text") or ""),
            "（已截断）" if result.get("truncated") else ""
        )

    return _generic_summary(result)


def _rag_source_label(result):
    """
    rag_search 返回的 source 只有在"指定了某一份年报"时才有值。

    Agent 通常不会传 source（它就是要检索整个年报库），
    这时 result["source"] 是 None —— 早期版本直接把 None 打到了界面上，
    变成"从《None》检索到 5 段原文"，看着像坏了。

    所以这里做个兜底：优先用显式指定的年报名，
    没有的话就从证据里把实际命中的年报表名收集出来。
    """

    explicit = result.get("source")

    if explicit:

        return explicit

    names = []

    for item in result.get("evidence") or []:

        name = item.get("source")

        if name and name not in names:

            names.append(name)

    if names:

        return "、".join(names)

    available = result.get("available_sources") or []

    if available:

        return "、".join(available)

    return "年报库"


def _generic_summary(result):

    # 财务类 Tool 返回结构各不相同，
    # 这里不逐个硬编码，只把"里面有多少东西"说清楚，
    # 具体内容界面上有原始 JSON 可以看。

    if isinstance(result, dict):

        parts = []

        for key, value in list(result.items())[:5]:

            if isinstance(value, (list, tuple)):

                parts.append("%s %d 条" % (key, len(value)))

            elif isinstance(value, dict):

                # 财务 Tool 常见形状是 {"income": {"rows": [...]}}，
                # 直接说"income 1 项"等于没说，
                # 往里看一眼，能数出行数就数出来。
                inner_lists = [v for v in value.values() if isinstance(v, (list, tuple))]

                if len(inner_lists) == 1:

                    parts.append("%s %d 条" % (key, len(inner_lists[0])))

                else:

                    parts.append("%s %d 项" % (key, len(value)))

            else:

                parts.append("%s=%s" % (key, str(value)[:40]))

        return "，".join(parts) or "（空结果）"

    if isinstance(result, list):

        return "返回 %d 条记录" % len(result)

    return str(result)[:200]


def _tool_citations(tool, result):

    citations = []

    # ---- 年报检索：给页码 + chunk_id ----
    if tool == "rag_search" and isinstance(result, dict):

        for item in result.get("evidence") or []:

            citations.append(
                {
                    "kind": "pdf",
                    "label": "第 %s 页" % item.get("page"),
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "preview": (item.get("text") or "")[:600]
                }
            )

    # ---- 网页搜索：给网址 ----
    elif tool == "search_web" and isinstance(result, list):

        for item in result[:6]:

            url = item.get("url") or ""

            citations.append(
                {
                    "kind": "web",
                    "label": _domain(url),
                    "url": url,
                    "title": item.get("title"),
                    "preview": item.get("snippet") or ""
                }
            )

    # ---- 读网页：给网址 + 正文开头 ----
    elif tool == "web_fetch" and isinstance(result, dict):

        url = result.get("final_url") or result.get("url") or ""

        citations.append(
            {
                "kind": "web",
                "label": _domain(url),
                "url": url,
                "title": result.get("title"),
                "preview": (result.get("text") or "")[:600]
            }
        )

    return citations


def _domain(url):

    try:

        return urlparse(url).netloc or url[:40]

    except Exception:

        return url[:40]


# ============================================================
# 4. 跑图，并把每一步变成事件流
# ============================================================
#
# 这是整个文件的核心。
#
# app.stream() 是 LangGraph 自带的"流式执行"：
# 它每跑完一个节点就 yield 一次，而不是等整张图跑完。
# 我们把这个天然的分步节奏，直接翻译成 SSE 事件推给浏览器。
#
# 所以界面上的进度条不是"假动画"，
# 它是图真正走到哪一步的实时映射。

def iter_events(question):

    from agent.graph_agent import build_graph

    from agent.tool_registry import TOOLS

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

    yield {
        "type": "start",
        "question": question,
        "nodes": GRAPH_NODES,
        "tools": sorted(TOOLS.keys())
    }

    # LangGraph 的节点返回的是"完整值"而不是增量，
    # 所以这里维护一份最新的 state 快照，每次用到的都从这里取。
    latest = dict(initial_state)

    sent_tools = 0

    seq = 0

    started_at = time.time()

    capture = _OutputCapture(real_stream=sys.stdout)

    real_stdout = sys.stdout

    real_stderr = sys.stderr

    sys.stdout = capture

    sys.stderr = capture

    try:

        for chunk in app.stream(initial_state, config={"recursion_limit": 100}):

            for node, update in chunk.items():

                node_started = time.time()

                seq += 1

                latest.update(update)

                yield {
                    "type": "node",
                    "node": node,
                    "label": NODE_LABEL.get(node, node),
                    "seq": seq,
                    "elapsed": round(time.time() - started_at, 1)
                }

                # ---------- planner：把计划发出去 ----------
                if node == "planner":

                    yield {
                        "type": "plan",
                        "plan": latest.get("research_plan") or {}
                    }

                # ---------- agent：这一轮它决定做什么 ----------
                if node == "agent":

                    yield _agent_event(update, latest)

                # ---------- tools：增量发放新的 Tool 调用 ----------
                if node == "tools":

                    results = latest.get("tool_results") or []

                    for item in results[sent_tools:]:

                        yield _tool_event(item)

                    sent_tools = len(results)

                # ---------- evidence：整理后的证据 ----------
                if node == "evidence":

                    yield _evidence_event(latest.get("evidence") or {})

                # ---------- reviewer：审核结论 ----------
                if node == "reviewer":

                    yield _review_event(latest.get("review") or {})

                # ---------- prepare_retry：回边触发 ----------
                if node == "prepare_retry":

                    yield {
                        "type": "retry",
                        "count": latest.get("retry_count", 0),
                        "max": 1
                    }

                # ---------- final：最终报告 ----------
                if node == "final":

                    yield {
                        "type": "report",
                        "text": latest.get("final_answer") or ""
                    }

                # ---------- save：落盘位置 ----------
                if node == "save":

                    yield {
                        "type": "save",
                        "run_dir": str(latest.get("run_dir") or "")
                    }

                # 节点跑完，把这段时间里 print 的东西推出去
                for line in capture.drain():

                    yield {"type": "log", "line": line}

                del node_started

        # 收尾：把最后的日志补完
        for line in capture.drain():

            yield {"type": "log", "line": line}

        yield {
            "type": "done",
            "elapsed": round(time.time() - started_at, 1),
            "run_dir": str(latest.get("run_dir") or ""),
            "tool_round": latest.get("tool_round"),
            "stop_reasons": latest.get("stop_reasons") or [],
            "retry_count": latest.get("retry_count", 0),
            "review": latest.get("review") or {},
            "tool_results": [
                {
                    "tool": item.get("tool"),
                    "status": item.get("status"),
                    "arguments": item.get("arguments")
                }
                for item in (latest.get("tool_results") or [])
            ]
        }

    except Exception as error:

        traceback_text = "%s: %s" % (type(error).__name__, error)

        for line in capture.drain():

            yield {"type": "log", "line": line}

        yield {
            "type": "error",
            "message": traceback_text
        }

    finally:

        sys.stdout = real_stdout

        sys.stderr = real_stderr


def _agent_event(update, latest):

    """
    agent 节点的一次决策。

    注意一个容易看漏的地方：
    当模型决定"不需要再调 Tool 了"，agent_node 只写 stop_reasons，
    不写 messages。所以 update 里有没有 messages 本身就是个信号。
    """

    messages = update.get("messages")

    if not messages:

        return {
            "type": "agent_step",
            "round": latest.get("tool_round"),
            "done": True,
            "tool_calls": [],
            "text": "模型判断资料已够，停止调用 Tool"
        }

    last = messages[-1] if messages else {}

    tool_calls = last.get("tool_calls") or []

    names = []

    for call in tool_calls:

        try:

            names.append(call["function"]["name"])

        except (KeyError, TypeError):

            names.append(str(call)[:40])

    return {
        "type": "agent_step",
        "round": latest.get("tool_round"),
        "done": False,
        "tool_calls": names,
        "text": str(last.get("content") or "")[:300]
    }


def _tool_event(item):

    detail = summarize_tool(item)

    return {
        "type": "tool",
        "tool": item.get("tool"),
        "status": item.get("status"),
        "arguments": item.get("arguments"),
        "elapsed": item.get("elapsed"),
        "ok": detail["ok"],
        "summary": detail["summary"],
        "citations": detail.get("citations") or []
    }


def _evidence_event(evidence):

    steps = []

    for step in evidence.get("completed_steps") or []:

        steps.append(
            {
                "step": step.get("step"),
                "tool": step.get("tool"),
                "evidence": str(step.get("evidence") or "")[:400]
            }
        )

    return {
        "type": "evidence",
        "steps": steps,
        "limitations": evidence.get("data_limitations") or [],
        "summary": str(evidence.get("summary") or evidence.get("final_evidence") or "")[:2000]
    }


def _review_event(review):

    return {
        "type": "review",
        "status": review.get("status"),
        "verdict": str(review.get("verdict") or "")[:1500],
        "issues": review.get("issues") or [],
        "unsupported_claims": review.get("unsupported_claims") or []
    }


# ============================================================
# 5. HTTP 服务
# ============================================================

# 同一时间只允许一个任务在跑。
# 两个请求同时跑图，snippets 会交叉、stdout 会串、还白烧 token。
_run_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    server_version = "FinResearchUI"

    # ---------- 路由 ----------

    def do_GET(self):

        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):

            self._send_index()
            return

        if parsed.path == "/api/health":

            self._send_json(
                {
                    "ok": True,
                    "busy": _run_lock.locked()
                }
            )
            return

        if parsed.path == "/api/stream":

            query = parse_qs(parsed.query).get("q", [""])[0].strip()

            self._send_stream(query)
            return

        if parsed.path == "/favicon.ico":

            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self._send_json({"ok": False, "error": "未知路径"}, status=404)

    def do_HEAD(self):

        self.do_GET()

    # ---------- 静态页 ----------

    def _send_index(self):

        if not INDEX_HTML.exists():

            self._send_json(
                {"ok": False, "error": "缺少前端文件：" + str(INDEX_HTML)},
                status=500
            )
            return

        body = INDEX_HTML.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if self.command != "HEAD":

            self.wfile.write(body)

    def _send_json(self, payload, status=200):

        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if self.command != "HEAD":

            self.wfile.write(body)

    # ---------- SSE 流 ----------

    def _send_stream(self, question):

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        # 交给浏览器后，这条连接就只能单向推送了
        self.close_connection = True

        if not question:

            self._emit({"type": "error", "message": "问题不能为空"})
            self._emit({"type": "eof"})
            return

        if not _run_lock.acquire(blocking=False):

            self._emit(
                {"type": "error", "message": "已经有一个任务在跑，请等它结束。"}
            )
            self._emit({"type": "eof"})
            return

        try:

            for event in iter_events(question):

                self._emit(event)

            self._emit({"type": "eof"})

        except (BrokenPipeError, ConnectionResetError):

            # 用户关掉了页面。让图自己跑完并落盘，没必要粗暴中断。
            pass

        except Exception as error:

            try:

                self._emit(
                    {
                        "type": "error",
                        "message": "%s: %s" % (type(error).__name__, error)
                    }
                )

                self._emit({"type": "eof"})

            except Exception:

                pass

        finally:

            _run_lock.release()

    def _emit(self, event):

        payload = json.dumps(event, ensure_ascii=False, default=str)

        self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))

        self.wfile.flush()

    # ---------- 把默认的访问日志换成一行精简输出 ----------

    def log_message(self, fmt, *args):

        sys.stderr.write(
            "  [web_ui] %s - %s\n" % (self.address_string(), fmt % args)
        )


# ============================================================
# 6. 启动
# ============================================================

def _bind_server(host, port, attempts=12):

    """
    端口被占用时往后找一个能用的。

    8765 是个很热门的"随手选"端口，撞车概率不低，
    直接报 OSError: 10048 会让人以为程序坏了。
    """

    last_error = None

    for offset in range(attempts):

        candidate = port + offset

        try:

            return ThreadingHTTPServer((host, candidate), Handler), candidate

        except OSError as error:

            last_error = error

    raise last_error


def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, open_browser=True):

    server, actual_port = _bind_server(host, port)

    url = "http://%s:%d/" % (host, actual_port)

    print("=" * 70)
    print("FinResearch-Agent  ·  可视化界面")
    print("=" * 70)
    print("地址：" + url)
    print("解释器：" + sys.executable)
    print()
    print("浏览器里输入问题 → 实时看 Planner / Agent / Tools /")
    print("Evidence / Reviewer / Final 的每一步，以及最终报告和证据来源。")
    print()
    print("按 Ctrl+C 停止。")
    print("=" * 70)

    if open_browser:

        # 等 server 真正开始 accept 再开浏览器，否则会白屏一次
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print("已停止。")

    finally:

        server.server_close()


if __name__ == "__main__":

    serve()
