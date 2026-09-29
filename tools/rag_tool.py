# -*- coding: utf-8 -*-
"""
年报检索 Tool（RAG）。

它在整个项目里的位置：
    和 search_web / web_fetch 完全一样，只是数据来源不是公开网络，
    而是"本地已经入过库的年报 PDF"。

        search_web   负责"网上有没有这条消息"
        web_fetch    负责"某个网页的原文写了什么"
        rag_search   负责"公司年报原文的哪一页写了什么"

    三个 Tool 的返回里都带来源（网址 / 文件名 + 页码），
    因为最终报告里的每一个数字都必须能被人翻回去核对。

为什么它是 Tool，而不是写死在 graph_agent.py 里：
    真实链路是"Agent 自己判断这个问题要不要翻年报"。
        "中际旭创2024年的营业收入和净利润是多少？"   -> 需要年报原文
        "中际旭创现在股价多少？"                     -> 需要行情，不需要年报
    如果把"先查年报"写进图里，就等于把判断权从模型手里收走，
    而且会让每个问题都白跑一次向量检索。
    这和当初把 search_web / web_fetch 做成 Tool 是同一个理由：
        Tool = 能力，Agent = 决策，LangGraph = 流程。

返回结构（固定的，不要随意改）：
    {
        "question":          "实际用于检索的问题",
        "source":            检索范围（限定某一本年报时是文件名，否则 None）,
        "available_sources": ["库里已入库的年报文件"],
        "top_k":             向量检索捞了多少条候选,
        "top_n":             最终要求返回几条,
        "returned":          实际返回几条,
        "evidence": [
            {
                "chunk_id": "...",                   # 哪一块
                "source":   "中际旭创 2024 年年报.pdf",  # 哪本年报
                "page":     26,                      # 第几页
                "text":     "原文...",                # 原文，不是摘要
                "score":    0.5287                   # rerank 分数
            },
            ...
        ],
        "error": None
    }

    失败时不返回这个 dict，而是抛 RuntimeError，和 search_tool.py / web_fetch_tool.py
    一致 —— tools_node 里那段 try/except 会统一接住，
    把可读的错误回灌给模型，并标成 status="failed"。

为什么 evidence 不是一段拼好的字符串：
    拼成字符串，Agent 就没法知道"这段话是第几页来的"，
    Evidence Fusion 和 Reviewer 也无从核查，
    最后写进报告的财务数字就会变成无法追溯的孤证。
    所以这里保留结构化字段，让 source / page / chunk_id 一直传到最终报告。
"""

import re


# ============================================================
# 1. 常量
# ============================================================

# 向量库位置。和 rag/vector_store.py 里的默认值指向同一个目录。
#
# 这里不把 db_dir 做成 Tool 参数：
#   向量的存放位置是项目配置，不是模型该决定的事。
#   让模型能指定路径，等于让它有机会指到一个空目录，
#   然后"检索不到"却看不出是配置问题还是真的没有。
#
# 写成模块级变量（而不是写死在调用处）还有一个好处：
#   测试可以把它指向一个临时空目录，验证"库不存在时的失败路径"。
DB_DIR = None


# Top-K：向量检索先捞多少条候选。
# 注意这两个默认值必须和 rag/rag_query.py 里的 DEFAULT_K / DEFAULT_TOP_N 一致，
# tests/test_rag_tool.py 里有一条断言专门防止两边改跑偏。
DEFAULT_TOP_K = 10
DEFAULT_TOP_N = 5


# ============================================================
# 2. 内部辅助函数
# ============================================================

def get_db_dir():
    """
    拿到向量库目录。

    为什么用函数而不是直接读模块变量：
        import 的时候就去 import rag 包会把 torch / transformers 一起拉起来，
        而 Tool 的注册表是在 Agent 启动时被 import 的。
        没必要为了"注册一个 Tool"就让整个进程加载一次深度学习框架。
        所以只在真正调用时才解析路径（下面 resolve_db_dir 里做同样的事）。
    """

    if DB_DIR is not None:

        return DB_DIR

    from rag.vector_store import DEFAULT_DB_DIR

    return DEFAULT_DB_DIR


def normalize_name(text):
    """
    把年报文件名归一化，只留下汉字、字母、数字。

    为什么需要它：
        模型传过来的名字往往和真实文件名差一点标点或空格 ——
        "中际旭创2024年年报.pdf" 和实际入库的 "中际旭创 2024 年年报.pdf"。
        直接精确匹配会失败，然后模型白跑一轮才发现名字不对。
        去掉空格和标点再比，这类差异就消失了。
    """

    return re.sub(
        r"[^0-9a-zA-Z\u4e00-\u9fff]",
        "",
        text or ""
    ).lower()


def resolve_source(requested, available):
    """
    把"模型给的年报名"解析成"库里真实的文件名"。

    三步，从严格到宽松：
        1. 完全一致
        2. 去掉空格/标点/大小写差异后一致
        3. 其中一个是另一个的子串（模型只说了 "2024年年报" 也能对上）

    都对不上就报错，并把库里实际有什么告诉它 ——
    这样模型下一轮能自己改对，而不是一直猜。

    为什么不在匹配失败时"退化成检索全部年报"：
        那会让一次限定范围的检索悄悄变成全库检索，
        结果看起来是成功的，但证据范围已经不是调用方要求的了。
        和 web_fetch 的原则一样：宁可明确失败，不要假装成功。
    """

    if not available:

        raise RuntimeError(
            "年报向量库里没有任何年报，无法检索。"
        )

    for name in available:

        if name == requested:

            return name

    target = normalize_name(requested)

    for name in available:

        if normalize_name(name) == target:

            return name

    for name in available:

        if target and (
            target in normalize_name(name)
            or normalize_name(name) in target
        ):

            return name

    raise RuntimeError(
        "年报库里没有这份年报：" + str(requested)
        + "。当前可检索的是：" + "、".join(available)
    )


# ============================================================
# 3. 对外 Tool
# ============================================================

def rag_search(
    question,
    top_k=DEFAULT_TOP_K,
    top_n=DEFAULT_TOP_N,
    source=None
):
    """
    在已入库的公司年报里检索原文片段。

    参数：
        question: 检索问题，用自然语言写，例如
                  "2024年营业收入和归母净利润是多少"
        top_k:    向量检索先捞多少条候选，默认 10
                  （为什么是 10 而不是 5 见 rag/rag_query.py 里的说明）
        top_n:    最终返回几条证据，默认 5
        source:   可选，只在一份指定的年报里检索。
                  一般不用传 —— 不传就是所有已入库年报一起检索。

    返回：
        见文件开头的"返回结构"。

    什么情况下该调用它（这段也是给 Reviewer 和 Final 看的线索）：
        问题涉及年报里的具体数字或表述时 —— 营业收入、净利润、
        分季度数据、毛利率、研发投入、现金流、业务构成、
        主要会计数据、管理层讨论、股东情况等。
        返回的每一段都带 source + page，引用时必须一起写出来。

    什么情况下不该调用它：
        最新股价、涨跌幅、实时估值、最近几天的新闻 ——
        这些不在年报里，应该用行情 Tool 或 search_web。
    """

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if not question or not str(question).strip():

        raise ValueError("检索问题不能为空")

    question = str(question).strip()

    if not isinstance(top_k, int) or top_k <= 0:

        raise ValueError("top_k 必须是正整数：" + str(top_k))

    if not isinstance(top_n, int) or top_n <= 0:

        raise ValueError("top_n 必须是正整数：" + str(top_n))

    # top_n 比 top_k 还大没有意义（候选只有 top_k 条）。
    # 这里直接压到 top_k，而不是报错：
    # 它不会让结果变错，只是多要了几条拿不到的东西，
    # 为此浪费一轮模型调用不值得。
    if top_n > top_k:

        top_n = top_k

    # --------------------------------------------------------
    # RAG 相关的东西在这里才 import
    #
    # 一是为了让"只是注册 Tool"不用加载 torch；
    # 二是让这个文件不 import Agent 的任何东西 —— 方向不能反。
    # --------------------------------------------------------

    from rag.rag_query import rag_query
    from rag import vector_store

    db_dir = get_db_dir()

    # --------------------------------------------------------
    # 先确认向量库真的建好了
    #
    # 向量库不存在时，rag_query 内部也会抛错，
    # 但那条错误是写给"运行项目的人"看的（"请先运行 python -m rag.rag_query --build"）。
    # 对模型来说更有用的信息是"这条路走不通，换别的 Tool"。
    # --------------------------------------------------------

    try:

        total_chunks = vector_store.count_chunks(db_dir)

    except Exception:

        raise RuntimeError(
            "年报检索不可用：本地年报向量库还没有建立或不可读取。"
            "请改用其他 Tool 获取财务数据，"
            "并在最终报告中说明年报原文检索当前不可用，不要凭记忆补充年报内容。"
        )

    if total_chunks == 0:

        raise RuntimeError(
            "年报检索不可用：年报向量库是空的（0 个片段）。"
            "请改用其他 Tool，并说明年报原文检索当前不可用。"
        )

    # --------------------------------------------------------
    # 年报范围
    #
    # 只有调用方指定了 source 时才去列一遍库里的文件名；
    # 不指定就不花这个开销。
    # --------------------------------------------------------

    available_sources = None

    resolved_source = None

    if source is not None and str(source).strip():

        available_sources = vector_store.list_sources(db_dir)

        resolved_source = resolve_source(str(source).strip(), available_sources)

    # --------------------------------------------------------
    # 跑 RAG
    # --------------------------------------------------------

    try:

        result = rag_query(
            question,
            k=top_k,
            top_n=top_n,
            db_dir=db_dir,
            source=resolved_source
        )

    except RuntimeError as error:

        # RAG 链路自己的错误（库空了、库坏了），原样往上报，
        # 让 tools_node 把这段可读信息回灌给模型。
        raise RuntimeError("年报检索失败：" + str(error))

    evidence = result["evidence"]

    if not evidence:

        raise RuntimeError(
            "年报检索没有返回任何片段：" + question
        )

    return {
        "question": question,
        "source": resolved_source,
        "available_sources": (
            available_sources
            if available_sources is not None
            else vector_store.list_sources(db_dir)
        ),
        "top_k": top_k,
        "top_n": top_n,
        "returned": len(evidence),
        "evidence": evidence,
        "error": None
    }
