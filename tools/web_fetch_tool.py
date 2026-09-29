# -*- coding: utf-8 -*-
"""
网页正文抓取 Tool。

它在整个项目里的位置：
    search_web 只给「标题 + 网址 + 一两句摘要」，用来判断"有没有这回事"够了，
    但要引用新闻/公告的原文内容，摘要往往不够。
    这个 Tool 就是补这一段：给它一个网址，它还你正文。

    它和 search_web 是配套的，但彼此独立：
        search_web 负责"找到哪些网页"
        web_fetch  负责"读某个网页写了什么"
    要不要读、读哪一条、读几条，由 Agent 自己决定，Tool 不做这个判断。

为什么这里最关键的一句话是「HTTP 200 不等于抓取成功」：
    这是实测出来的，不是理论担心。真实情况有四类：
        1. 一些页面返回 200，但响应体是 0 字节；
        2. 一些页面返回 200，正文全是导航菜单，没有文章；
        3. 一些页面返回 200，但返回的是加密的 WAF 验证串（雪球就是）；
        4. 一些页面靠 JavaScript 渲染，requests 只能拿到空壳（腾讯新闻、
           东方财富股吧列表都是）。
    这四类如果当成"抓到了"，正文会变成垃圾或空字符串进入证据链，
    最后写进报告的"事实"就是假的。
    所以本 Tool 的原则是：抓不到正文就明确失败，绝不假装成功。

返回结构（固定的，不要随意改）：
    {
        "url":       "请求的原始地址",
        "final_url": "跟随重定向后真正打开的地址",
        "status":    HTTP 状态码,
        "title":     网页标题,
        "text":      正文纯文本（已按 max_chars 截断）,
        "truncated": 是否被截断,
        "error":     None
    }

    失败时不返回这个 dict，而是抛 RuntimeError，和 search_tool.py 一致。
    这样 tools_node 里那段 try/except 会统一接住，
    把可读的错误信息回灌给模型，并把它标成 status="failed"。
"""

import re

import requests

from bs4 import BeautifulSoup


# ============================================================
# 1. 常量
# ============================================================

# 超时分成两段：
#   connect 是"连上服务器"的时间，read 是"读完响应"的时间。
# 不分成两段的话，一个连不上的域名会一直挂到总超时，
# 而网页正文通常比接口响应慢，所以 read 给得比 connect 宽。
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15

# 很多站点对没有 User-Agent 的请求直接返回空页面或验证页，
# 这里带一个普通浏览器标识，和 search_tool.py 保持一致。
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9"
}

# 这些标签里的内容不是正文：脚本、样式、导航、页脚、侧栏、表单、图标。
# 不丢掉它们的话，正文前面会挂一大段"首页 快讯 股票 财经 基金 …"。
DROP_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "iframe",
    "form",
    "svg"
]

# 正文短于这个长度，认为是"没抓到正文"，而不是"抓到了一篇很短的正文"。
# 实测正常的财经快讯都在 400 字以上，200 是比较保守的阈值，
# 优先保证不误杀短快讯，宁可放过个别 JS 空壳页。
MIN_TEXT_LENGTH = 200

# WAF / 验证页的特征。
# 命中说明拿到的是拦截页而不是网页本身，必须报错。
# 雪球返回的是一段 "_waf_..." 开头的加密串，就是靠第一条识别出来的。
BLOCK_PAGE_MARKERS = [
    "_waf_",
    "cf-browser-verification",
    "Enable JavaScript and cookies to continue",
    "请开启JavaScript",
    "安全验证",
    "访问过于频繁"
]


# ============================================================
# 2. 内部辅助函数
# ============================================================

def clean_text(text):
    """
    把网页文本里的换行、连续空格压成一个空格。

    网页源码里有大量缩进和换行，直接给模型看会浪费很多 token。
    """

    return re.sub(r"\s+", " ", text).strip()


def detect_encoding(response):
    """
    判断响应体该用什么编码解码。

    为什么不能直接用 response.text：
        requests 的 response.text 按 HTTP 头里的 charset 解码，
        但很多国内站点的 HTTP 头是错的，实测遇到过三种：
            - 证券之星：头里写 gb2312，正文确实是国标编码；
            - 新浪：头里写 ISO-8859-1，正文其实是 UTF-8；
            - 搜狐：requests 自己猜的 apparent_encoding 猜成 cp1006，
                    直接解出阿拉伯语乱码。
        按头解码或按猜解码，都可能得到乱码。

    所以这里优先读 HTML 自己声明的 <meta charset>。
    页面自己声明的编码，通常比 HTTP 头可信。
    """

    # 只在前 2KB 里找 charset 声明，避免把正文里出现的 "charset=" 也匹配上。
    head = response.content[:2048].decode("ascii", errors="ignore")

    match = re.search(
        r"charset\s*=\s*[\"']?\s*([\w-]+)",
        head,
        re.IGNORECASE
    )

    if match:

        encoding = match.group(1).lower()

        # gb2312 只收录简体常用字，gb18030 是它的超集。
        # 页面声明 gb2312、实际用了生僻字时会解不出来，
        # 统一升到 gb18030 更安全。
        if encoding in ("gb2312", "gbk"):

            return "gb18030"

        return encoding

    return response.apparent_encoding or response.encoding or "utf-8"


def is_block_page(html):
    """
    判断拿到的 HTML 是不是 WAF / 验证拦截页。
    """

    for marker in BLOCK_PAGE_MARKERS:

        if marker in html:

            return True

    return False


def extract_title(soup):
    """
    取网页标题。
    """

    if soup.title:

        return clean_text(soup.title.get_text(strip=True))

    return ""


def extract_main_text(soup):
    """
    从页面里抽出正文纯文本。

    这里没有用 trafilatura / readability 这类专门的正文抽取库，
    原因是不想为了这一步引入新依赖。用的是最朴素的启发式：

        就是纯文本最长的那个 div / article / section。

    为什么这样能work：
        正文通常是页面上文字最多的那一块，
        导航、页脚、侧栏的文字量都比它少。
    实测在网易、DoNews、东方财富的文章页上，
    这个粗暴办法能拿到整页文本的 80%~91%，开头就是文章正文。

    它的缺点也要知道：
        会多带一小段站点导航前缀（"网易首页 > 网易号 > 正文"），
        而且在纯 JS 渲染的页面上，最长的那块可能只是菜单。
    """

    best_text = ""

    for tag in soup.find_all(["article", "div", "section"]):

        text = clean_text(tag.get_text(" ", strip=True))

        if len(text) > len(best_text):

            best_text = text

    # 页面里连一个 div 都没有时，退回整页文本。
    if not best_text:

        best_text = clean_text(soup.get_text(" ", strip=True))

    return best_text


# ============================================================
# 3. 对外 Tool
# ============================================================

def web_fetch(url, max_chars=3000):
    """
    抓取一个网页的正文纯文本。

    参数：
        url:       网页地址，通常来自 search_web 返回结果里的 url
        max_chars: 正文最多返回多少个字符，默认 3000

    返回：
        {
            "url": ..., "final_url": ..., "status": ...,
            "title": ..., "text": ..., "truncated": ..., "error": None
        }

    抓不到正文时抛 RuntimeError，不返回半成品。

    注意 max_chars 是必须的，不是可有可无的美化参数：
        tools_node 把 Tool 结果回灌给模型时是 json.dumps(result) 原样全传，
        项目里没有任何统一截断。一篇新闻正文实测能到 6000 字以上，
        不截断的话几轮下来上下文会被撑爆。
        所以截断这件事只能在这个 Tool 内部做。
    """

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if not url or not url.strip():

        raise ValueError("网页地址不能为空")

    url = url.strip()

    # 只允许 http/https，避免模型给个 file:// 之类的地址。
    if not url.startswith(("http://", "https://")):

        raise ValueError(
            "网页地址必须以 http:// 或 https:// 开头：" + url
        )

    if not isinstance(max_chars, int) or max_chars <= 0:

        raise ValueError(
            "max_chars 必须是正整数：" + str(max_chars)
        )

    # --------------------------------------------------------
    # 发请求
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True
        )

    except requests.RequestException as error:

        raise RuntimeError(
            "网页抓取请求失败：" + str(error)
        )

    final_url = response.url

    # --------------------------------------------------------
    # 状态码：403 / 404 / 500 都要说清楚是几，别只说"失败了"
    # --------------------------------------------------------

    if response.status_code != 200:

        raise RuntimeError(
            "网页抓取失败：HTTP "
            + str(response.status_code)
            + "（" + final_url + "）"
        )

    # --------------------------------------------------------
    # 200 但响应体是空的。
    # 实测东方财富财富号的部分文章页就是这样：
    # 状态码正常、内容 0 字节，不特判的话会被当成"抓到一篇空文章"。
    # --------------------------------------------------------

    if not response.content:

        raise RuntimeError(
            "网页抓取失败：服务器返回 200 但响应体为空（"
            + final_url + "）"
        )

    # --------------------------------------------------------
    # 解码
    # --------------------------------------------------------

    encoding = detect_encoding(response)

    try:

        html = response.content.decode(encoding, errors="ignore")

    except LookupError:

        # 页面声明的编码名 Python 不认识，退回 UTF-8。
        html = response.content.decode("utf-8", errors="ignore")

    # --------------------------------------------------------
    # WAF / 验证页
    # 这一步要在解析之前做，因为拦截页的"正文"本身就是一段垃圾串，
    # 解析之后就看不出来它原本是什么了。
    # --------------------------------------------------------

    if is_block_page(html):

        raise RuntimeError(
            "网页抓取失败：返回的是验证/拦截页面，不是网页正文（"
            + final_url + "）"
        )

    # --------------------------------------------------------
    # 解析正文
    # --------------------------------------------------------

    soup = BeautifulSoup(html, "lxml")

    title = extract_title(soup)

    for tag in soup(DROP_TAGS):

        tag.decompose()

    text = extract_main_text(soup)

    # 正文过短：可能是纯 JS 渲染，也可能就是个导航页。
    # 这两种都应该报错，而不是把一段菜单当成"新闻原文"送进证据链。
    if len(text) < MIN_TEXT_LENGTH:

        raise RuntimeError(
            "网页抓取失败：无法提取正文（去掉标签后只剩 "
            + str(len(text))
            + " 字，可能是动态渲染页面或聚合列表页）："
            + final_url
        )

    # --------------------------------------------------------
    # 截断
    # --------------------------------------------------------

    truncated = len(text) > max_chars

    if truncated:

        text = text[:max_chars]

    return {
        "url": url,
        "final_url": final_url,
        "status": response.status_code,
        "title": title,
        "text": text,
        "truncated": truncated,
        "error": None
    }
