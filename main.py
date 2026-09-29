# -*- coding: utf-8 -*-
"""
FinResearch-Agent 的命令行入口。

存在的意义只有一个：让"从零开始跑一次"变成一条命令，
而不是在 README 里塞一段 python -c 的一行代码。

    python main.py
    python main.py "平安银行现在怎么样？"
    python main.py --legacy "中际旭创现在怎么样"

它自己不含任何业务逻辑：
    默认走 LangGraph 版（agent/graph_agent.py 的 run_graph），
    加 --legacy 走最早的手写流程版（agent/financial_agent.py 的 run_agent）。
    两条路的业务函数是同一套。
"""

import sys

from pathlib import Path


# 允许直接 python main.py 这样跑
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


# 本机真正装了项目依赖的那个解释器。
# 用途只有一个：解释器选错的时候，把它打印出来告诉人该换成哪个。
RECOMMENDED_PYTHON = r"C:\Users\17667\AppData\Local\Programs\Python\Python312\python.exe"


# 不带参数时的演示问题。
# 选它是因为它正好会同时用到年报检索（rag_search）
# 和财务报表（get_financial_statements）两类 Tool。
DEFAULT_QUESTION = "中际旭创2024年的营业收入和归母净利润是多少？"


# 项目真正依赖的第三方包（缺一个就跑不起来）。
# 写在这里是为了"启动前先体检"，而不是等到 import 到一半才炸。
REQUIRED_PACKAGES = [
    ("langgraph", "流程编排"),
    ("langchain_core", "LangGraph 底层"),
    ("requests", "网络请求"),
    ("bs4", "网页解析"),
    ("fitz", "PDF 解析（PyMuPDF）"),
    ("chromadb", "向量库"),
    ("transformers", "本地 Embedding 模型"),
    ("torch", "模型推理"),
    ("numpy", "向量运算"),
]


# ============================================================
# 解释器体检
# ============================================================
#
# 这是一个"踩过坑才会加"的检查。
#
# 本机装了不止一个 Python，而且 PyCharm 默认挑的那个（PATH 里排第一的）
# 是工作区自带的轻量 Python，里面一个第三方包都没有。
# 现象就是 PyCharm 里点运行，报一句光秃秃的
#     ModuleNotFoundError: No module named 'langgraph'
# 看上去像代码写错了，其实只是解释器选错了 —— 排查方向完全被带偏。
#
# 所以这里主动把话说清楚：用哪个解释器、缺什么包、该换成哪个路径。

def check_environment():
    """
    检查当前解释器有没有装齐依赖。

    返回 True 表示可以直接跑；返回 False 表示已经打印了怎么办，别继续了。
    """

    import importlib.util

    missing = []

    for module_name, purpose in REQUIRED_PACKAGES:

        try:

            found = importlib.util.find_spec(module_name) is not None

        except (ImportError, ValueError):

            found = False

        if not found:

            missing.append((module_name, purpose))

    if not missing:

        return True

    print("=" * 70)
    print("启动失败：当前 Python 解释器缺少依赖")
    print("=" * 70)
    print()
    print("当前解释器：")
    print("  " + sys.executable)
    print()
    print("缺少的包：")

    for module_name, purpose in missing:

        print("  - %-16s（%s）" % (module_name, purpose))

    print()
    print("-" * 70)
    print("注意：这不是代码出错，是【解释器选错了】。")
    print("本机 Project 依赖装在下面这个 Python 上：")
    print("  " + RECOMMENDED_PYTHON)
    print()
    print("怎么改，任选一种：")
    print()
    print("  ① 在 PyCharm 里换解释器（推荐，改一次就好）")
    print("     File → Settings → Project: Financial-AI-Research → Python Interpreter")
    print("     → Add Interpreter → Add Local Interpreter → System Interpreter")
    print("     → 选上面那个 python.exe → OK")
    print()
    print("  ② 不用 PyCharm，直接在项目根目录用命令行跑")
    print("     \"" + RECOMMENDED_PYTHON + "\" main.py")
    print()
    print("  ③ 双击项目里的 start_ui.bat（启动可视化界面）")
    print("     它已经写死了正确的解释器路径，不依赖 PyCharm 设置。")
    print()
    print("如果换成正确的解释器后仍提示缺包，再执行：")
    print("  \"" + RECOMMENDED_PYTHON + "\" -m pip install -r requirements.txt")
    print()

    return False


def main():

    if not check_environment():

        sys.exit(1)

    args = sys.argv[1:]

    # 可视化界面：python main.py --ui
    if "--ui" in args:

        args.remove("--ui")

        from web_ui import serve

        serve()
        return

    use_legacy = "--legacy" in args

    if use_legacy:

        args.remove("--legacy")

    question = " ".join(args) if args else DEFAULT_QUESTION

    print("=" * 70)
    print("FinResearch-Agent")
    print("=" * 70)
    print("问题：" + question)
    print("流程：" + ("手写流程版 run_agent" if use_legacy else "LangGraph 版 run_graph"))
    print()

    if use_legacy:

        from agent.financial_agent import run_agent

        final_answer = run_agent(question)

        print()
        print("=" * 70)
        print("最终报告")
        print("=" * 70)
        print(final_answer)

        return

    from agent.graph_agent import run_graph

    state = run_graph(question)

    print()
    print("=" * 70)
    print("实际调用链")
    print("=" * 70)

    print("planner")

    # tool_results 是按发生顺序记下来的，所以直接扫一遍就是真实的调用顺序。
    # Agent 在 Tool 之间还会再调一次模型，这里把它标出来，
    # 免得看起来像"Tool 一个接一个自动跑完"。
    for item in state["tool_results"]:

        print("  ↓")
        print("agent")
        print("  ↓")
        print(
            item["tool"]
            + "  [" + item["status"] + "]"
            + "  " + str(item["arguments"])[:100]
        )

    print("  ↓")
    print("evidence")
    print("  ↓")
    print("reviewer  [" + str(state["review"].get("status")) + "]")

    if state["retry_count"] > 0:

        print("  ↓")
        print("prepare_retry  ×" + str(state["retry_count"]))

    print("  ↓")
    print("final")
    print("  ↓")
    print("save")

    print()
    print("=" * 70)
    print("最终报告")
    print("=" * 70)
    print(state["final_answer"])

    print()
    print("=" * 70)
    print("运行信息")
    print("=" * 70)
    print("Tool 调用轮数：" + str(state["tool_round"]))
    print("Tool Calling 结束原因：" + str(state["stop_reasons"]))
    print("Reviewer：" + str(state["review"].get("status")))
    print("补充研究次数：" + str(state["retry_count"]))
    print("结果目录：" + str(state["run_dir"]))


if __name__ == "__main__":

    main()
