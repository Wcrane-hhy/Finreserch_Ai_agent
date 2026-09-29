"""
LangGraph v1 的最简端到端测试。

只做一件事：
用固定问题跑一次 run_graph()，然后确认每个阶段都出了结果。

运行方式（和 tests/ 下其他脚本一样）：

    python -m tests.test_graph_agent
"""

import os
import sys
from pathlib import Path


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.graph_agent import run_graph


# ============================================================
# 2. 固定问题
# ============================================================

question = "中际旭创现在怎么样"


# ============================================================
# 3. 运行整张图
# ============================================================

state = run_graph(question)


# ============================================================
# 4. 逐项检查
# ============================================================

print("\n============================================================")
print("测试检查")
print("============================================================")

checks = []


def check(name, ok, detail=""):

    checks.append(ok)

    mark = "通过" if ok else "失败"

    print(f"[{mark}] {name} {detail}")


# 图能正常结束（代码能走到这里，就说明没崩）
check("图正常结束", True)

# Planner
plan_steps = state["research_plan"].get("steps", [])
check(
    "Planner 有结果",
    len(plan_steps) > 0,
    f"（{len(plan_steps)} 个步骤）"
)

# Tool 调用
check(
    "Tool 有调用",
    len(state["tool_results"]) > 0,
    f"（{len(state['tool_results'])} 次）"
)

# Evidence
check(
    "Evidence 有结果",
    bool(state["evidence"].get("research_goal")),
    ""
)

# Reviewer
check(
    "Reviewer 有结果",
    state["review"].get("status") in ("PASS", "FAIL"),
    f"（{state['review'].get('status')}）"
)

# Final Report
check(
    "Final 有报告",
    len(state["final_answer"]) > 0,
    f"（{len(state['final_answer'])} 字）"
)

# 落盘
run_dir = state["run_dir"]

files_ok = os.path.isdir(run_dir) and all(
    os.path.isfile(os.path.join(run_dir, name))
    for name in (
        "research_plan.json",
        "tool_results.json",
        "evidence.json",
        "review.json",
        "report.md",
    )
)

check(
    "runs/ 有新的结果目录",
    files_ok,
    f"（{run_dir}）"
)


# ============================================================
# 5. 汇总
# ============================================================

print("\n============================================================")
print("测试汇总")
print("============================================================")

print(f"通过 {sum(checks)} / {len(checks)} 项")

print("\n调用了这些 Tool：")
for item in state["tool_results"]:
    print(f"  - {item['tool']}  {item['status']}")

print(f"\nTool Calling 结束原因：{state['stop_reasons']}")
print(f"结果目录：{run_dir}")
