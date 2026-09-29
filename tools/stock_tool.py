import os
import requests

from tools.api_utils import check_api_result


def search_stock(query):
    """
    根据股票名称或股票代码搜索 A 股。

    Args:
        query: 股票名称或股票代码

    Returns:
        股票搜索结果列表
    """

    api_key = os.getenv("HITHINK_FINANCE_API_KEY")

    if not api_key:
        raise ValueError(
            "未找到 HITHINK_FINANCE_API_KEY"
        )

    headers = {
        "X-api-key": api_key
    }

    url = "https://fuyao.aicubes.cn/api/meta/tickers/search"

    params = {
        "q": query,
        "asset_type": "a-share"
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

    items = result["data"]["item"]

    return items