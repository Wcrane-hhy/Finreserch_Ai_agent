from agent.tool_registry import TOOLS


def execute_tool(tool_name, arguments):
    """
    执行指定 Tool。
    """

    if tool_name not in TOOLS:
        raise ValueError(
            f"未知 Tool：{tool_name}"
        )

    tool_function = TOOLS[tool_name]["function"]

    return tool_function(**arguments)