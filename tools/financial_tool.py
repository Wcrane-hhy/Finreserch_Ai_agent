import os
import requests

from tools.api_utils import check_api_result


def get_financial_statements(thscode):
    """
    获取指定 A 股公司的三张年度财务报表。

    三张表统一都取最近 4 个年度（limit = 4），
    保证合并计算指标时年度是对得上的。

    参数：
        thscode: 股票代码，例如 300308.SZ

    返回：
        {
            "balance_sheet": [...],
            "income_statement": [...],
            "cash_flow": [...]
        }
    """

    api_key = os.getenv("HITHINK_FINANCE_API_KEY")

    if not api_key:
        raise ValueError(
            "未找到 HITHINK_FINANCE_API_KEY"
        )

    headers = {
        "X-api-key": api_key
    }

    # ========================================================
    # 1. 资产负债表
    # ========================================================

    balance_sheet_url = (
        "https://fuyao.aicubes.cn/api/a-share/financials/balance-sheets"
    )

    balance_params = {
        "thscode": thscode,
        "period": "annual",
        "limit": 4
    }

    response_balance = requests.get(
        balance_sheet_url,
        headers=headers,
        params=balance_params,
        timeout=10
    )

    response_balance.raise_for_status()

    balance_data = response_balance.json()

    # 后端业务错误用的是 code != 0，不是 HTTP 状态码
    check_api_result(balance_data)

    # ========================================================
    # 2. 利润表
    # ========================================================

    income_statement_url = (
        "https://fuyao.aicubes.cn/api/a-share/financials/income-statements"
    )

    income_params = {
        "thscode": thscode,
        "period": "annual",
        "limit": 4
    }

    response_income = requests.get(
        income_statement_url,
        headers=headers,
        params=income_params,
        timeout=10
    )

    response_income.raise_for_status()

    income_data = response_income.json()

    # 后端业务错误用的是 code != 0，不是 HTTP 状态码
    check_api_result(income_data)

    # ========================================================
    # 3. 现金流量表
    # ========================================================

    cash_flow_url = (
        "https://fuyao.aicubes.cn/api/a-share/financials/cash-flow-statements"
    )

    cash_flow_params = {
        "thscode": thscode,
        "period": "annual",
        "limit": 4
    }

    response_cash_flow = requests.get(
        cash_flow_url,
        headers=headers,
        params=cash_flow_params,
        timeout=10
    )

    response_cash_flow.raise_for_status()

    cash_flow_data = response_cash_flow.json()

    # 后端业务错误用的是 code != 0，不是 HTTP 状态码
    check_api_result(cash_flow_data)

    # ========================================================
    # 4. 提取数据
    # ========================================================

    balance_items = balance_data["data"]["item"]

    income_items = income_data["data"]["item"]

    cash_flow_items = cash_flow_data["data"]["item"]

    # ========================================================
    # 5. 统一返回
    # ========================================================

    return {
        "balance_sheet": balance_items,
        "income_statement": income_items,
        "cash_flow": cash_flow_items
    }
