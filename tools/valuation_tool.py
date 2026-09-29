import os
import requests

from tools.api_utils import check_api_result


def get_stock_valuation(thscodes):
    """
    获取 A 股最新估值数据。

    参数：
        thscodes:
            单个股票代码或股票代码列表。
            例如：
                "300308.SZ"
                ["300308.SZ", "600519.SH"]

    返回：
        估值数据列表
    """

    api_key = os.getenv("HITHINK_FINANCE_API_KEY")

    if not api_key:
        raise ValueError(
            "未找到 HITHINK_FINANCE_API_KEY"
        )

    if isinstance(thscodes, str):
        thscodes = [thscodes]

    headers = {
        "X-api-key": api_key
    }

    url = (
        "https://fuyao.aicubes.cn/api/"
        "a-share/valuations/snapshot"
    )

    params = {
        "thscodes": ",".join(thscodes)
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    # 后端业务错误用的是 code != 0，不是 HTTP 状态码
    check_api_result(result)

    return result["data"]["item"]