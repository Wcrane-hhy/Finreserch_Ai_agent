from tools.stock_tool import search_stock
from tools.financial_tool import get_financial_statements
from tools.metric_tool import calculate_financial_indicators
from tools.price_tool import get_stock_quote, get_price_history
from tools.valuation_tool import get_stock_valuation
from tools.search_tool import search_web
from tools.web_fetch_tool import web_fetch
from tools.rag_tool import rag_search


TOOLS = {
    "search_stock": {
        "function": search_stock,
        "description": "根据股票名称或股票代码搜索 A 股公司。",
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
    },

    "get_financial_statements": {
        "function": get_financial_statements,
        "description": "获取指定 A 股公司的年度资产负债表、利润表和现金流量表。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscode": {
                    "type": "string",
                    "description": "股票交易代码，例如 300308.SZ"
                }
            },
            "required": ["thscode"]
        }
    },

    "calculate_financial_indicators": {
        "function": calculate_financial_indicators,
        "description": "计算指定 A 股公司的成长性、盈利能力、现金流和财务杠杆等财务指标。这个 Tool 会自己去取最近 4 个年度的财务报表，所以只需要传股票代码，不需要自己传财务数据。如果已知公司名称，请一并传入 company_name，这样银行股会被标记出不适用的指标。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscode": {
                    "type": "string",
                    "description": "股票交易代码，例如 000001.SZ"
                },
                "company_name": {
                    "type": "string",
                    "description": "公司名称，例如 平安银行。可选，但建议传入。"
                }
            },
            "required": ["thscode"]
        }
    },

    "get_stock_quote": {
        "function": get_stock_quote,
        "description": "获取一个或多个 A 股标的的最新行情快照。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscodes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "股票交易代码列表，例如 ['300308.SZ']"
                }
            },
            "required": ["thscodes"]
        }
    },

    "get_price_history": {
        "function": get_price_history,
        "description": "获取指定 A 股公司的历史日 K 线数据。不传日期时默认返回最近 90 天。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscode": {
                    "type": "string",
                    "description": "股票交易代码，例如 000001.SZ"
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期，格式必须是 YYYY-MM-DD，例如 2026-01-01。不传默认最近 90 天。"
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期，格式必须是 YYYY-MM-DD，例如 2026-09-28。不传默认今天。"
                }
            },
            "required": ["thscode"]
        }
    },

    "get_stock_valuation": {
        "function": get_stock_valuation,
        "description": "获取一个或多个 A 股标的的最新估值数据。",
        "parameters": {
            "type": "object",
            "properties": {
                "thscodes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "股票交易代码列表，例如 ['300308.SZ']"
                }
            },
            "required": ["thscodes"]
        }
    },

    "search_web": {
        "function": search_web,
        "description": "搜索网络公开信息，返回新闻、公告、行业动态等网页结果（含标题、来源网址、摘要）。当问题涉及最新消息、新闻、事件、市场动态、政策，而这些内容用 A 股行情和财务 Tool 拿不到时，调用这个 Tool。返回的每条结果都带 url，引用时必须标明来源网址。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如 中际旭创 新闻"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条，默认 5。一般不用传。"
                }
            },
            "required": ["query"]
        }
    },

    "web_fetch": {
        "function": web_fetch,
        "description": "读取一个网页的正文内容。当 search_web 返回的摘要不足以判断，需要看新闻、公告的原文时，传入 search_web 结果里的 url 调用它，返回该网页的标题和正文。注意：正文是按 max_chars 截断后的纯文本；如果网页是动态渲染或需要验证，会返回失败信息，此时不要编造内容，应说明原文无法获取。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "网页地址，一般直接用 search_web 返回结果里的 url"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "正文最多返回多少字符，默认 3000。一般不用传。"
                }
            },
            "required": ["url"]
        }
    },

    "rag_search": {
        "function": rag_search,
        "description": "检索本地已入库的公司年报原文（年报 PDF 向量库）。返回的是年报里的原话，每条证据都带来源文件名、页码和 chunk_id，引用时必须标注它是哪本年报的第几页。当问题涉及年报里的具体数字或表述时调用它：营业收入、归母净利润、分季度财务数据、毛利率、研发投入、经营活动现金流、主营业务构成、主要会计数据、管理层讨论与分析、股东情况、分红方案等。当前库中已入库的年报只有：中际旭创 2024 年年报。注意它查不到实时股价、涨跌幅、最新估值和最近几天的新闻，这些要用行情类 Tool 或 search_web。如果它返回失败信息，说明年报原文检索不可用，应改用其他 Tool，并在报告中说明这一点，不要凭记忆补充年报内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "检索问题，用自然语言写，例如 2024年营业收入和归母净利润是多少"
                },
                "top_k": {
                    "type": "integer",
                    "description": "向量检索先捞多少条候选，默认 10。一般不用传；如果第一次检索没找到你需要的原文，可以调大（例如 20）再检索一次。"
                },
                "top_n": {
                    "type": "integer",
                    "description": "最终返回几条证据，默认 5。一般不用传。"
                },
                "source": {
                    "type": "string",
                    "description": "可选，只在一份指定的年报里检索，一般不用传。不传就是在所有已入库年报里检索。"
                }
            },
            "required": ["question"]
        }
    }
}