# -*- coding: utf-8 -*-
# ============================================================
# 【当前未被使用，仅作历史记录保留】
#
# 这是早期用 list 形式写死的 Tool 定义。
# 现在 Tool 的事实来源是 agent/tool_registry.py 里的 TOOLS 字典，
# 主链路不会 import 这个文件。
#
# 保留原因：可以对照两种写法，暂时没有必要删除。
# 后续如果确认不再需要，可以整文件删掉。
# ============================================================

STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "search_stock",
        "description": "根据股票名称或股票代码搜索 A 股公司，并返回公司名称、股票代码和上市日期。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "股票名称或股票代码，例如 中际旭创 或 300308"
                }
            },
            "required": ["query"]
        }
    }
}


FINANCIAL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_financial_statements",
        "description": "获取指定 A 股公司的年度资产负债表、利润表和现金流量表。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscode": {
                    "type": "string",
                    "description": "A 股股票代码，例如 300308.SZ"
                }
            },
            "required": ["thscode"]
        }
    }
}


METRIC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_financial_indicators",
        "description": "根据公司的三张财务报表计算收入增长率、净利润增长率、毛利率、净利率、资产负债率、自由现金流、ROA 和 ROE 等核心财务指标。",
        "parameters": {
            "type": "object",
            "properties": {
                "financial_data": {
                    "type": "object",
                    "description": "由 get_financial_statements 返回的财务数据"
                }
            },
            "required": ["financial_data"]
        }
    }
}


TOOLS = [
    STOCK_TOOL,
    FINANCIAL_TOOL,
    METRIC_TOOL
]
