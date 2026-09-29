# -*- coding: utf-8 -*-
"""
网络搜索 Tool。

它在整个项目里的位置：
    和其它 Tool 完全一样，只是数据来源不是同花顺，而是公开网络。
    Agent 通过 tool_registry.py 看到它，觉得需要外部信息时就调用它，
    搜索结果和其它 Tool 的返回一样，会进入 tool_results，
    然后被 Evidence Fusion 整理、被 Reviewer 审核、被 Final Report 引用。

为什么用 360 搜索：
    这是实测选出来的，不是随便挑的。
    当前网络下 DuckDuckGo / Google / Wikipedia 都连不上，
    Bing 对"某某 新闻"这类中文查询会返回降级页面（结果变成无关内容），
    东方财富的搜索接口会忽略参数返回无关数据。
    360 搜索是唯一既能连通、又稳定返回结构化结果的免费来源。

当前实现用的是「网页抓取」而不是官方 API，这一点要记住：
    如果 360 改了页面结构，这个 Tool 就会失效。
    以后想换成更稳定的来源（例如需要 API Key 的 Tavily），
    只需要改 search_web() 里取数据的那一段，返回结构保持下面这样即可。

返回结构（固定的，不要随意改）：
    [
        {"title": "...", "url": "https://...", "snippet": "..."},
        ...
    ]

    每条都带 url，是为了让最终报告里的每条新闻都能追溯回原始网页，
    Reviewer 也才有东西可以核查。这是"证据可追溯"的最低要求。
"""

import requests
from bs4 import BeautifulSoup


SEARCH_URL = "https://www.so.com/s"

# 搜索引擎对没有 User-Agent 的请求可能返回空页面，
# 这里带一个普通的浏览器标识。
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9"
}


def clean_url(url):
    """
    把搜索结果里的链接整理成可以直接访问的完整地址。
    """

    if not url:
        return ""

    if url.startswith("//"):
        return "https:" + url

    if url.startswith("/"):
        return "https://www.so.com" + url

    return url


def search_web(query, max_results=5):
    """
    搜索网络公开信息，返回结构化结果列表。

    参数：
        query:       搜索关键词，例如 "中际旭创 新闻"
        max_results: 最多返回几条，默认 5

    返回：
        [
            {"title": "...", "url": "https://...", "snippet": "..."}
        ]

    失败时抛 RuntimeError，并把原因写清楚，
    这样 Agent 侧（tools_node）才能把可读的错误回灌给模型。
    """

    if not query or not query.strip():

        raise ValueError("搜索关键词不能为空")

    try:

        response = requests.get(
            SEARCH_URL,
            params={"q": query},
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            "网络搜索请求失败：" + str(error)
        )

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    result_items = soup.select("li.res-list")

    # 一条结果都没解析到，说明页面结构很可能已经变了。
    # 这种情况必须报出来，不能安静地返回空列表——
    # 否则 Agent 会以为"搜了但确实没新闻"，实际是 Tool 坏了。
    if not result_items:

        raise RuntimeError(
            "网络搜索没有解析到任何结果，搜索页面结构可能已经变化。"
        )

    results = []

    for item in result_items:

        title_tag = item.find("h3")

        if not title_tag:
            continue

        link_tag = title_tag.find("a")

        if not link_tag:
            continue

        # data-mdurl 是搜索结果指向的原网址；
        # href 经常是 360 自己的跳转链接，只在没有 data-mdurl 时才用它。
        url = clean_url(
            link_tag.get("data-mdurl") or link_tag.get("href")
        )

        if not url:
            continue

        # 跳过 360 自己的聚合页和跳转链接，它们不是真实来源。
        # 保留它们会让最终报告里的"来源"变成搜索引擎自己，失去可追溯性。
        if "so.com" in url:
            continue

        description_tag = (
            item.select_one("p.res-desc")
            or item.select_one("div.res-desc")
            or item.find("p")
        )

        results.append(
            {
                "title": title_tag.get_text(strip=True),
                "url": url,
                "snippet": (
                    description_tag.get_text(strip=True)
                    if description_tag else ""
                )
            }
        )

        if len(results) >= max_results:
            break

    return results
