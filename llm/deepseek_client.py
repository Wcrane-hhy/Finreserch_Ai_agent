# -*- coding: utf-8 -*-
# ============================================================
# 【当前未被使用，仅作历史记录保留】
#
# 这是早期抽出来的 DeepSeek 调用封装。
# 现在 agent/financial_agent.py 自己定义了 call_deepseek，
# 主链路不会 import 这个文件。
#
# 保留原因：暂时没有必要删除。
# 后续如果确认不再需要，可以整文件删掉。
# ============================================================

import os
import requests


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(messages, tools=None):
    """
    调用 DeepSeek Chat API。

    参数：
        messages: 对话消息
        tools: Tool Calling 工具定义

    返回：
        DeepSeek API 返回的完整 JSON
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

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
        json=payload
    )

    response.raise_for_status()

    return response.json()
