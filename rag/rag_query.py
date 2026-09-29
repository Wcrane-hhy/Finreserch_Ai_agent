# -*- coding: utf-8 -*-
"""
RAG 第七步：查询入口。

整个 RAG Pipeline 就是这一个函数：

    问题
      ↓  embed_query()
    查询向量
      ↓  vector_store.search()
    Top-K 候选（按余弦相似度）
      ↓  rerank.rerank()
    Top-N 证据
      ↓
    返回给调用方

怎么用：

    建库（第一次，或者改了 chunk 参数之后）：
        python -m rag.rag_query --build

    查询：
        python -m rag.rag_query "中际旭创2024年的营业收入和净利润是多少？"

    代码里调用：
        from rag.rag_query import rag_query
        result = rag_query("中际旭创2024年的营业收入和净利润是多少？")
        for item in result["evidence"]:
            print(item["page"], item["text"])

注意：这个文件本身不认识 Agent，也不知道 Tool 是什么。
它就是一个能自己跑通的独立函数。
接进 Agent 的方式是在 tools/rag_tool.py 里包一层，
而不是让这个文件去 import Agent —— 方向不能反。
"""

import sys
import json

from pathlib import Path


# 允许直接 python rag/rag_query.py 这样跑
if __package__ in (None, ""):

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from rag.embedding import embed_query
from rag.rerank import rerank
from rag.vector_store import (
    DEFAULT_DB_DIR,
    build_index,
    count_chunks,
    search
)


# Top-K：向量检索先捞多少条进入 Rerank
#
# 为什么是 10 而不是常见的 5 —— 这是实测调出来的，不是拍脑袋：
#    用"中际旭创2024年的营业收入和净利润是多少？"这一问，
#    能回答问题的原文块（第 26 页「营业收入构成」，
#    里面写着营业收入 23,862,159,738.37）在全库 550 块里排第 9。
#    K=5 时它根本进不来，K=8 也差一名，K=10 刚好覆盖。
#    所以 K 用 10：多捞几条的代价几乎为零（只是多算几次余弦距离），
#    但漏掉唯一那句正确原文的代价很大。
#    完整诊断过程见 tests/test_rag.py 末尾的说明。
DEFAULT_K = 10

# Top-N：Rerank 之后最终留几条当证据
#
# 为什么是 5 而不是 3：
#    同一块原文在上面的例子里，Rerank 之后排在 **第 5 名**
#    （Rerank 把它从向量第 9 名提到了第 5 名，说明它确实在起作用，
#     但没能提到第 3 名以内）。
#    N=3 会把它切掉，N=5 才留得住。
#    5 条 × 500 字 = 2500 字，作为喂给模型的证据量也很正常。
DEFAULT_TOP_N = 5


def rag_query(
    question,
    k=DEFAULT_K,
    top_n=DEFAULT_TOP_N,
    db_dir=DEFAULT_DB_DIR,
    source=None
):
    """
    跑完整条 RAG 链路，返回检索证据。

    参数：
        question: 用户问题
        k:        Vector Search 的 Top-K
        top_n:    Rerank 之后的 Top-N
        db_dir:   向量库目录
        source:   只在这一份年报里检索（不传就是全部年报一起检索）

    返回：
        {
            "question": "中际旭创2024年的营业收入和净利润是多少？",
            "top_k":  [ ...K 条候选，含 similarity... ],
            "top_n":  [ ...N 条，含 rerank_score... ],
            "evidence": [
                {
                    "chunk_id": "...",
                    "source":   "中际旭创 2024 年年报.pdf",
                    "page":     12,
                    "text":     "原文...",
                    "score":    0.83
                },
                ...
            ]
        }

    为什么同时返回 top_k 和 top_n：
        top_k 是"入口"，top_n 是"结果"。
        两个都给出来，才能看出 Rerank 到底改了什么名次 ——
        不然 Rerank 就成了一个看不见效果的黑盒。
    """

    if not question or not question.strip():

        raise ValueError("问题不能为空")

    question = question.strip()

    # ------------------------------------------------
    # 1. 问题 -> 查询向量
    # ------------------------------------------------
    query_vector = embed_query(question)

    # ------------------------------------------------
    # 2. 向量检索 -> Top-K
    # ------------------------------------------------
    top_k_candidates = search(
        query_vector,
        k=k,
        db_dir=db_dir,
        source=source
    )

    # ------------------------------------------------
    # 3. Rerank -> Top-N
    # ------------------------------------------------
    top_n_candidates = rerank(
        question,
        top_k_candidates,
        top_n=top_n
    )

    # ------------------------------------------------
    # 4. 整理成"证据"格式
    #
    # 这一步不是多余的：
    # top_n 里带着一堆调试用的分数，
    # 而证据只关心"哪段原文、来自哪一页"，字段越干净越好。
    # ------------------------------------------------
    evidence = [
        {
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "text": item["text"],
            "score": round(item["rerank_score"], 4)
        }
        for item in top_n_candidates
    ]

    return {
        "question": question,
        "top_k": top_k_candidates,
        "top_n": top_n_candidates,
        "evidence": evidence
    }


def print_result(result, text_preview=300):
    """
    把 rag_query 的结果打印成人能看的样子。

    测试和命令行都用这个函数，
    保证"测试里看到的"和"手动跑看到的"是同一套输出。
    """

    print()
    print("=" * 70)
    print("Query")
    print("=" * 70)
    print(result["question"])

    print()
    print("=" * 70)
    print("Top-K（Vector Search，K=" + str(len(result["top_k"])) + "）")
    print("=" * 70)

    for item in result["top_k"]:

        print(
            "  #" + str(item["rank"])
            + "  sim=" + str(round(item["similarity"], 4))
            + "  page=" + str(item["page"])
            + "  " + str(item["chunk_id"])
        )
        print(
            "      "
            + item["text"][:80].replace("\n", " ")
        )

    print()
    print("=" * 70)
    print(
        "Rerank 后 Top-N（baseline rerank，N="
        + str(len(result["top_n"])) + "）"
    )
    print("=" * 70)

    for item in result["top_n"]:

        print(
            "  #" + str(item["rank"])
            + "  rerank=" + str(round(item["rerank_score"], 4))
            + "  (vec=" + str(round(max(0.0, item["similarity"]), 4))
            + "  lex=" + str(round(item["lexical_score"], 4)) + ")"
            + "  vec_rank=#" + str(item["vector_rank"])
        )
        print(
            "      page=" + str(item["page"])
            + "  chunk_id=" + str(item["chunk_id"])
        )
        print(
            "      source=" + str(item["source"])
        )
        print(
            "      text(" + str(text_preview) + "字): "
            + item["text"][:text_preview].replace("\n", " ")
        )
        print()

    print("=" * 70)
    print("最终 Evidence")
    print("=" * 70)

    print(
        json.dumps(
            result["evidence"],
            ensure_ascii=False,
            indent=2,
            default=str
        )[:1500]
    )

    print()


# ============================================================
# 命令行入口
# ============================================================

def main():

    args = sys.argv[1:]

    if "--build" in args:

        args.remove("--build")

        stats = build_index()

        print()
        print("建库完成：" + json.dumps(stats, ensure_ascii=False, indent=2))

        if not args:

            return

    k = DEFAULT_K
    top_n = DEFAULT_TOP_N

    if "--k" in args:

        position = args.index("--k")
        k = int(args[position + 1])
        del args[position:position + 2]

    if "--top-n" in args:

        position = args.index("--top-n")
        top_n = int(args[position + 1])
        del args[position:position + 2]

    if not args:

        print(
            "用法：\n"
            "    python -m rag.rag_query --build\n"
            "    python -m rag.rag_query \"中际旭创2024年的营业收入和净利润是多少？\"\n"
            "    python -m rag.rag_query --k 8 --top-n 3 \"问题\""
        )

        return

    print("向量库当前有 " + str(count_chunks()) + " 个 chunk")

    result = rag_query(
        " ".join(args),
        k=k,
        top_n=top_n
    )

    print_result(result)


if __name__ == "__main__":

    main()
