# -*- coding: utf-8 -*-
"""
RAG 第六步：Rerank（对 Top-K 重新排序，取 Top-N）。

★ 先说清楚：这是一个 baseline，不是真正的 Cross-Encoder Reranker。★

真正的 Cross-Encoder Reranker 长什么样：
    把 (query, doc) 拼成一条序列一起送进一个 Transformer，让它输出一个分数。
    它能看到 query 和 doc 之间的交互，所以比"两边各自算一个向量再比距离"准得多
    （这就是所谓 bi-encoder vs cross-encoder 的区别）。
    代价是：每个候选都要单独过一次模型，K=50 就是 50 次前向计算，很慢。

    这台机器上没装可用中文的 reranker 模型（连不上 huggingface 下载），
    requirement 也明确说"不要为了追求复杂度安装一堆模型"，
    所以这里不装，只做一个能解释清楚的 baseline。

这个 baseline 在做什么：
    只用两路信号加权求和 ——

      词面分（lexical）：问题里的词，在这段原文里出现了多少。
                        中文按 2-gram 切（"营业收入" -> "营业"、"业收"、"收入"），
                        这样不需要分词器，也不需要装 jieba。
                        每个词按 IDF 加权：越常见的词（"多少"、"的是"）权重越低，
                        越稀有的词（"净利润"、"营业收入"）权重越高。

      向量分（vector）：  Top-K 阶段已经算好的余弦相似度，直接用。

      rerank_score = 0.6 × 向量相似度 + 0.4 × 词面覆盖度

它能解决什么问题：
    向量检索对"长原文、短查询"有时会跑偏 ——
    一段泛泛而谈的"公司整体经营情况"可能因为语义接近而排在前面，
    而真正写着"营业收入 238.6 亿元"的那一段反而排在第 5。
    词面分正好能把这匹马捞上来，因为那段里有"营业收入"这些字。

它的局限（要知道）：
    它只看"词有没有出现"，不理解同义词。
    用户问"营收"，原文写"营业收入"，2-gram 只能匹配到"营业"，
    拿不到满分。这是 baseline 和真 reranker 的差距所在。

以后想升级：把 rerank() 里算分的那几行换成一次模型调用，
函数签名和返回值都不用改。
"""

import math
import re


# 两路信号的权重。加起来是 1，方便理解"各自占多大比重"。
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_LEXICAL_WEIGHT = 0.4


def extract_terms(text):
    """
    从一段文本里抽出用于词面匹配的词。

    规则（没有分词器也能用）：
      - 连续的汉字串：切成 2-gram。
        "营业收入" -> {"营业", "业收", "收入"}
      - 连续的字母/数字串：整个作为一个词，转小写。
        "2024" -> {"2024"}
      - 标点和空白：直接丢掉。

    为什么中文用 2-gram 而不是装 jieba：
        jieba 是分词器，装了会引入词典依赖，
        而且"中际旭创"这种公司名在通用词典里可能被切错。
        2-gram 不需要词典，对金融长词（营业收入、归母净利润）反而更稳。
    """

    terms = set()

    # 先按"汉字"和"字母数字"两类切成片段，跳过标点
    for piece in re.findall(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9]+",
        text
    ):

        if re.match(r"^[A-Za-z0-9]+$", piece):

            terms.add(piece.lower())

            continue

        if len(piece) == 1:

            terms.add(piece)

            continue

        for position in range(len(piece) - 1):

            terms.add(piece[position:position + 2])

    return terms


def rerank(
    question,
    candidates,
    top_n=3,
    vector_weight=DEFAULT_VECTOR_WEIGHT,
    lexical_weight=DEFAULT_LEXICAL_WEIGHT
):
    """
    对 Vector Top-K 的结果重新打分排序，返回 Top-N。

    参数：
        question:       用户问题原文
        candidates:     vector_store.search() 的返回值（Top-K）
        top_n:          rerank 之后保留几条
        vector_weight:  向量分占的权重
        lexical_weight: 词面分占的权重

    返回：
        和 candidates 同结构，但多了三个字段，并按 rerank_score 从高到低排好名次：
            lexical_score  词面覆盖度，0~1
            rerank_score   加权总分
            rank           重新排序后的名次（从 1 开始）
            vector_rank    它在向量检索里原来的名次
                            （用来观察 rerank 到底把谁捞上来了）
    """

    if not candidates:

        return []

    question_terms = extract_terms(question)

    # ------------------------------------------------
    # 1. 每个候选算出它自己的词集合
    # ------------------------------------------------
    candidate_terms = [
        extract_terms(item["text"])
        for item in candidates
    ]

    total = len(candidates)

    # ------------------------------------------------
    # 2. 算 IDF
    #
    # df   = 一个词在多少个候选里出现过
    # idf  = log(1 + N / (1 + df))
    #
    # df 越大说明这个词越普通，idf 越小，对排序的影响越小。
    # 分母的 1+df 是为了防止 df=0 时除零。
    # 外面的 1+N/ 让 idf 永远是正数，不用处理负数。
    # ------------------------------------------------
    def idf(term):

        df = sum(
            1 for terms in candidate_terms if term in terms
        )

        return math.log(1 + total / (1 + df))

    # 问题里所有词的 idf 之和，用来做归一化的分母
    weight_sum = sum(idf(term) for term in question_terms)

    results = []

    for position, item in enumerate(candidates):

        terms = candidate_terms[position]

        hit = question_terms & terms

        # 命中的词按 idf 加权求和
        hit_weight = sum(idf(term) for term in hit)

        # 归一化成 0~1：问题里的词被覆盖了多大比例（按权重算）
        if weight_sum > 0:

            lexical_score = hit_weight / weight_sum

        else:

            lexical_score = 0.0

        # 向量的余弦相似度理论上是 -1~1，
        # 但 BGE 用于中文长文本时基本落在 0~1，这里裁一下保证加权有意义。
        vector_score = max(0.0, min(1.0, item["similarity"]))

        rerank_score = (
            vector_weight * vector_score
            + lexical_weight * lexical_score
        )

        results.append(
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "page": item["page"],
                "source": item["source"],
                "distance": item["distance"],
                "similarity": item["similarity"],
                "lexical_score": lexical_score,
                "rerank_score": rerank_score,
                "vector_rank": item["rank"],
                "hit_terms": sorted(hit)
            }
        )

    # ------------------------------------------------
    # 3. 按总分重排
    # ------------------------------------------------
    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    top = results[:top_n]

    for position, item in enumerate(top):

        item["rank"] = position + 1

    return top
