import os
from datetime import datetime, timedelta, timezone

import requests

from tools.api_utils import check_api_result


# 接口返回的日期是毫秒时间戳，按北京时间换算
CHINA_TIMEZONE = timezone(timedelta(hours=8))


def date_to_milliseconds(date_text):
    """
    把 "YYYY-MM-DD" 转成毫秒时间戳。

    历史行情接口要求 start / end 必须是毫秒时间戳，
    直接传 "2026-09-01" 会被接口判为格式错误（code = 1002）。
    """

    moment = datetime.strptime(
        date_text,
        "%Y-%m-%d"
    ).replace(tzinfo=CHINA_TIMEZONE)

    return int(moment.timestamp() * 1000)


def milliseconds_to_date(milliseconds):
    """
    把毫秒时间戳转回 "YYYY-MM-DD"，方便人和模型阅读。
    """

    moment = datetime.fromtimestamp(
        milliseconds / 1000,
        CHINA_TIMEZONE
    )

    return moment.strftime("%Y-%m-%d")


def get_stock_quote(thscodes):
    """
    获取 A 股最新行情快照。

    参数：
        thscodes:
            单个股票代码或股票代码列表。
            例如：
                "300308.SZ"
                ["300308.SZ", "600519.SH"]

    返回：
        行情数据列表。
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
        "a-share/prices/snapshot"
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

    data = result["data"]

    timestamp = data.get("timestamp")
    items = data["item"]

    # 将接口数据时间附加到每条行情记录
    for item in items:
        item["data_timestamp"] = timestamp

    return items


def get_price_history(
    thscode,
    start_date=None,
    end_date=None,
    interval="1d"
):
    """
    获取 A 股历史日 K 线。

    参数：
        thscode:
            股票代码，例如 000001.SZ

        start_date:
            开始日期，字符串 "YYYY-MM-DD"。
            不传时默认取最近 90 天。

        end_date:
            结束日期，字符串 "YYYY-MM-DD"。
            不传时默认取今天。

        interval:
            K 线周期。接口目前只接受 "1d"（日线），
            这里给默认值，正常调用不需要传这个参数。

    返回：
        历史行情数据列表。
        每条记录在原字段之外，额外补一个易读的 date 字段。
    """

    api_key = os.getenv("HITHINK_FINANCE_API_KEY")

    if not api_key:
        raise ValueError(
            "未找到 HITHINK_FINANCE_API_KEY"
        )

    headers = {
        "X-api-key": api_key
    }

    url = (
        "https://fuyao.aicubes.cn/api/"
        "a-share/prices/historical"
    )

    # 接口要求 start 和 end 都是必填，不能只传一个，
    # 所以两边都给一个默认值。
    if start_date is None:
        start_date = (
            datetime.now(CHINA_TIMEZONE) - timedelta(days=90)
        ).strftime("%Y-%m-%d")

    if end_date is None:
        end_date = datetime.now(CHINA_TIMEZONE).strftime(
            "%Y-%m-%d"
        )

    params = {
        "thscode": thscode,
        "interval": interval,
        "start": date_to_milliseconds(start_date),
        "end": date_to_milliseconds(end_date)
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

    # 原始记录只有 date_ms，模型读毫秒时间戳很容易看错，
    # 顺手补一个 "YYYY-MM-DD" 的 date 字段，date_ms 保留不动。
    for item in items:
        item["date"] = milliseconds_to_date(
            item["date_ms"]
        )

    return items
