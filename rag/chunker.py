# -*- coding: utf-8 -*-
"""
RAG 第二步：Chunking（把长文本切成小块）。

为什么必须切块：
    Embedding 模型有输入长度上限（bge-small-zh 是 512 个 token），
    一整页 900 多字的年报塞进去会被截断，等于后半页白读了。
    而且一块内容的语义越集中，向量就越准；
    把"公司简介 + 财务数据 + 股东名单"混成一块，检索出来的东西会很糊。

为什么不用 LangChain 的 RecursiveCharacterTextSplitter：
    它是黑盒，切出来的结果不好解释，出了问题也不好调。
    这个项目是学习用的，切块逻辑必须自己能看懂、能改。
    这里手写一个 ~40 行的版本，规则完全透明。

切块策略（窗口 + 重叠 + 句末对齐）：
    1. 从这一页文本的开头开始，取 chunk_size 个字符作为一块；
    2. 尽量把结尾往后挪到"。"或换行处，别把句子劈成两半；
    3. 下一块从"当前结尾 - overlap"处开始 —— 这就是重叠。

为什么需要 overlap：
    一个关键句子正好卡在两块的边界上时，
    不带重叠的话两块都只拿到半句，谁都检索不准。
    重叠 100 个字，就是让边界附近的信息在两块里都完整出现一次。
"""

import re


# 默认参数。放成常量，方便测试和以后调。
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

# 找句末时最多往后多看这么多字符。
# 太小了找不到句号，太大了块长会不稳定。
SENTENCE_LOOKAHEAD = 80


def build_source_key(source):
    """
    把来源文件名变成一个能放进 chunk_id 的短标识。

    "中际旭创 2024 年年报.pdf"  ->  "中际旭创_2024_年年报"

    为什么要做这一步：
        chunk_id 会被写进向量库的 id 字段，
        带空格和扩展名的文件名既难看又可能出问题。
    """

    key = re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)

    key = re.sub(r"\s+", "_", key.strip())

    return key


def chunk_page(
    text,
    source,
    page,
    start_index=0,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):
    """
    把一页文本切成若干块。

    参数：
        text:        这一页的正文
        source:      来源文件名，例如 "中际旭创 2024 年年报.pdf"
        page:        页码（从 1 开始）
        start_index: 这个 chunk_id 的起始编号，
                     一个 PDF 里所有页共用一个递增编号，避免不同页出现同名 id
        chunk_size:  每块最多多少个字
        overlap:     相邻两块重叠多少个字

    返回：
        [
            {
                "chunk_id": "中际旭创_2024_年年报-p7-c12",
                "text":     "...",
                "page":     7,
                "source":   "中际旭创 2024 年年报.pdf"
            },
            ...
        ]
    """

    if chunk_size <= 0:

        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0 or overlap >= chunk_size:

        # overlap 大于等于 chunk_size 的话窗口永远不会前进，会死循环。
        # 这里直接拦住，而不是让它悄悄跑成无限循环。
        raise ValueError(
            "overlap 必须满足 0 <= overlap < chunk_size"
        )

    text = text.strip()

    if not text:

        return []

    source_key = build_source_key(source)

    chunks = []

    start = 0
    index = start_index

    while start < len(text):

        end = start + chunk_size

        # ------------------------------------------------
        # 句末对齐：如果结尾附近有句号或换行，
        # 就把结尾挪过去，避免半句话被切在块尾。
        # ------------------------------------------------
        if end < len(text):

            window_end = min(end + SENTENCE_LOOKAHEAD, len(text))

            sentence_end = max(
                text.rfind("。", end, window_end),
                text.rfind("\n", end, window_end)
            )

            if sentence_end != -1:

                end = sentence_end + 1

        piece = text[start:end].strip()

        if piece:

            chunks.append(
                {
                    "chunk_id": (
                        source_key
                        + "-p" + str(page)
                        + "-c" + str(index)
                    ),
                    "text": piece,
                    "page": page,
                    "source": source
                }
            )

            index += 1

        if end >= len(text):

            break

        # 下一块往回退 overlap 个字，形成重叠
        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):
    """
    把一个 PDF 的所有页切成块（按页切，不跨页）。

    参数：
        pages: load_pdf / load_reports 的返回值

    返回：
        所有页的 chunk 列表，每个 chunk 带 chunk_id / text / page / source

    为什么按页切、不跨页合并：
        因为 requirement 要求每个 chunk 带 page。
        如果允许跨页，一个块可能横跨第 7 页和第 8 页，
        那 page 就只能填其中一个，证据就不可追溯了。
        代价是页边界处会损失一点上下文，这个取舍是值得的。
    """

    chunks = []

    index = 0

    for item in pages:

        page_chunks = chunk_page(
            item["text"],
            item["source"],
            item["page"],
            start_index=index,
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunks.extend(page_chunks)

        index += len(page_chunks)

    return chunks
