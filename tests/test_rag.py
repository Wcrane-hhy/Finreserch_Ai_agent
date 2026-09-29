# -*- coding: utf-8 -*-
"""
RAG Pipeline 的测试。

用真实的中际旭创 2024 年年报跑。
测试问题固定在 requirement 里指定的那个：
    "中际旭创2024年的营业收入和净利润是多少？"

八项检查，正好对应 Pipeline 的八步：
    1. PDF 能读取
    2. Chunk 能生成（且 overlap 真的生效）
    3. Embedding 能生成（且是真单位向量）
    4. Vector DB 能写入
    5. Query 能检索
    6. Top-K 数量正确
    7. Rerank 能返回结果（且真的改了名次）
    8. rag_query 能完整跑通

运行方式（和 tests/ 下其他脚本一样）：

    python -m tests.test_rag

第一次运行如果还没建过索引，脚本会自动建一次（要一两分钟）。
"""

import re
import sys

from pathlib import Path


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


from rag.pdf_loader import load_pdf, load_reports
from rag.chunker import chunk_pages
from rag.embedding import embed_texts, embed_query, EMBEDDING_DIM
from rag.rerank import rerank, extract_terms
from rag.rag_query import rag_query, print_result
from rag import vector_store


# ============================================================
# 2. 测试参数
# ============================================================

QUESTION = "中际旭创2024年的营业收入和净利润是多少？"

PDF_PATH = PROJECT_ROOT / "data" / "reports" / "中际旭创 2024 年年报.pdf"

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

K = 10
TOP_N = 5

# 年报里"2024 年营业收入"这个数字。
# 用它来判断检索到底有没有把"写着答案的那段原文"捞回来。
#
# 为什么挑这个数字：
#   它同时出现在多个地方（第 8 页主要会计数据、第 26 页营业收入构成），
#   所以只要捞回任意一处就算成功，不会因为只认某一页而误判。
#   具体的金额是 23,862,159,738.37 元（约 238.62 亿元）。
ANSWER_NUMBER = "23,862,159,738.37"

# 2024 年归母净利润
ANSWER_PROFIT = "5,171,485,967.85"


results = []


def check(name, passed, detail=""):

    results.append((name, passed, detail))

    print(("  [通过] " if passed else "  [失败] ") + name)

    if detail:

        print("         " + str(detail)[:260])


# ============================================================
# 3. 测试一：PDF 能读取
# ============================================================

print("=" * 70)
print("测试一：PDF 能读取")
print("=" * 70)

check(
    "年报 PDF 文件存在",
    PDF_PATH.exists(),
    str(PDF_PATH)
)

pages = load_pdf(PDF_PATH)

check(
    "抽出了页面",
    len(pages) > 0,
    "共 " + str(len(pages)) + " 页"
)

check(
    "每页都有 page 和 text 字段",
    all("page" in item and "text" in item for item in pages),
    ""
)

check(
    "page 从 1 开始且不重复",
    [item["page"] for item in pages] == sorted({item["page"] for item in pages}),
    "首页 page=" + str(pages[0]["page"]) + "，末页 page=" + str(pages[-1]["page"])
)

# 年报正文里一定会出现的字样，用来确认抽出来的是真内容而不是空白
joined = "".join(item["text"] for item in pages)

check(
    "抽出的正文里有年报该有的内容",
    "中际旭创" in joined and "营业收入" in joined,
    "总字符数 " + str(len(joined))
)

all_pages = load_reports(REPORTS_DIR)

check(
    "load_reports 会给每页带上 source",
    all("source" in item for item in all_pages)
    and all_pages[0]["source"].endswith(".pdf"),
    "source = " + str(all_pages[0]["source"])
)


# ============================================================
# 4. 测试二：Chunk 能生成
# ============================================================

print()
print("=" * 70)
print("测试二：Chunk 能生成")
print("=" * 70)

chunks = chunk_pages(all_pages)

check(
    "切出了 chunk",
    len(chunks) > 0,
    "共 " + str(len(chunks)) + " 个"
)

check(
    "每个 chunk 都有 chunk_id / text / page / source",
    all(
        set(["chunk_id", "text", "page", "source"]) <= set(item.keys())
        for item in chunks
    ),
    "字段示例：" + str(sorted(chunks[0].keys()))
)

check(
    "chunk_id 不重复",
    len({item["chunk_id"] for item in chunks}) == len(chunks),
    "示例：" + chunks[0]["chunk_id"]
)

check(
    "每个 chunk 都带得出页码",
    all(isinstance(item["page"], int) and item["page"] >= 1 for item in chunks),
    ""
)

check(
    "chunk 长度没有超过 chunk_size",
    all(len(item["text"]) <= 600 for item in chunks),
    "最长 " + str(max(len(item["text"]) for item in chunks)) + " 字"
)

# ------------------------------------------------
# overlap 不能只是"参数写了但没用"
# 拿同一页里相邻的两个 chunk 验证：
# 前一块的结尾，应该能在后一块里找到。
# 两边都先去掉空白再比，避免 strip() 带来的边界误差。
# ------------------------------------------------
same_page_pairs = []

for first, second in zip(chunks, chunks[1:]):

    if first["page"] == second["page"]:

        same_page_pairs.append((first, second))

overlap_ok = True
overlap_detail = "没有找到同一页里的相邻 chunk"

for first, second in same_page_pairs:

    left = re.sub(r"\s+", "", first["text"])
    right = re.sub(r"\s+", "", second["text"])

    if left[-20:] in right:

        overlap_detail = (
            "相邻块共享 " + str(len(left[-20:])) + " 字重叠，"
            "例：" + left[-20:]
        )

        break

    overlap_ok = False

check(
    "overlap 真的生效（相邻块有重叠）",
    overlap_ok and len(same_page_pairs) > 0,
    overlap_detail
)


# ============================================================
# 5. 测试三：Embedding 能生成
# ============================================================

print()
print("=" * 70)
print("测试三：Embedding 能生成")
print("=" * 70)

sample_texts = [item["text"] for item in chunks[:5]]

vectors = embed_texts(sample_texts)

check(
    "向量形状是 (文本数, 512)",
    vectors.shape == (5, EMBEDDING_DIM),
    "实际 " + str(vectors.shape)
)

norms = (vectors ** 2).sum(axis=1) ** 0.5

check(
    "向量已经单位化（模长≈1）",
    all(abs(float(n) - 1.0) < 1e-4 for n in norms),
    "模长：" + str([round(float(n), 5) for n in norms])
)

query_vector = embed_query(QUESTION)

check(
    "查询向量形状正确",
    query_vector.shape == (EMBEDDING_DIM,),
    "实际 " + str(query_vector.shape)
)

# 语义相近的两句话，相似度应该明显高于不相干的
related = embed_texts(["公司2024年度实现营业收入238.6亿元"], is_query=False)[0]
unrelated = embed_texts(["公司注册地址是山东省龙口市"], is_query=False)[0]

similarity_related = float((query_vector * related).sum())
similarity_unrelated = float((query_vector * unrelated).sum())

check(
    "语义相近的句子相似度更高",
    similarity_related > similarity_unrelated,
    "相关 " + str(round(similarity_related, 4))
    + " vs 无关 " + str(round(similarity_unrelated, 4))
)


# ============================================================
# 6. 测试四：Vector DB 能写入
# ============================================================

print()
print("=" * 70)
print("测试四：Vector DB 能写入")
print("=" * 70)

try:

    existing_count = vector_store.count_chunks()

except Exception:

    # 库还不存在，count_chunks 会抛错，这是正常的
    existing_count = 0

if existing_count == 0:

    print("  向量库还不存在，先建一次（要一两分钟）...")

    vector_store.build_index(REPORTS_DIR)

stored = vector_store.count_chunks()

check(
    "向量库里存下了 chunk",
    stored > 0,
    "共 " + str(stored) + " 个"
)

check(
    "存的数量和切出来的数量一致",
    stored == len(chunks),
    "库里 " + str(stored) + " / 切出 " + str(len(chunks))
)

check(
    "向量库是本地持久化的（目录存在且有文件）",
    vector_store.DEFAULT_DB_DIR.exists()
    and len(list(vector_store.DEFAULT_DB_DIR.iterdir())) > 0,
    str(vector_store.DEFAULT_DB_DIR)
)


# ============================================================
# 7. 测试五 + 六：Query 能检索，Top-K 数量正确
# ============================================================

print()
print("=" * 70)
print("测试五 / 六：Query 能检索，Top-K 数量正确")
print("=" * 70)

candidates = vector_store.search(query_vector, k=K)

check(
    "检索返回了结果",
    len(candidates) > 0,
    "返回 " + str(len(candidates)) + " 条"
)

check(
    "Top-K 数量正确（K=" + str(K) + "）",
    len(candidates) == K,
    "实际 " + str(len(candidates)) + " 条"
)

check(
    "每条结果都带 chunk_id / text / page / similarity / source",
    all(
        set(["chunk_id", "text", "page", "similarity", "source"])
        <= set(item.keys())
        for item in candidates
    ),
    "字段：" + str(sorted(candidates[0].keys()))
)

check(
    "相似度按从大到小排好序",
    all(
        candidates[i]["similarity"] >= candidates[i + 1]["similarity"]
        for i in range(len(candidates) - 1)
    ),
    "第1名 " + str(round(candidates[0]["similarity"], 4))
    + " -> 第" + str(len(candidates)) + "名 "
    + str(round(candidates[-1]["similarity"], 4))
)

# K 不合法时应该被拦住
try:

    vector_store.search(query_vector, k=0)

    check("k=0 被拦住", False, "没有报错")

except Exception:

    check("k=0 被拦住（返回空或报错都算）", True, "")


# ============================================================
# 8. 测试七：Rerank 能返回结果
# ============================================================

print()
print("=" * 70)
print("测试七：Rerank 能返回结果")
print("=" * 70)

reranked = rerank(QUESTION, candidates, top_n=TOP_N)

check(
    "Rerank 返回了 Top-N 条",
    len(reranked) == TOP_N,
    "实际 " + str(len(reranked)) + " 条"
)

check(
    "每条都带 rerank_score / lexical_score",
    all(
        "rerank_score" in item and "lexical_score" in item
        for item in reranked
    ),
    "分数：" + str([round(item["rerank_score"], 4) for item in reranked])
)

check(
    "rerank_score 从大到小排好序",
    all(
        reranked[i]["rerank_score"] >= reranked[i + 1]["rerank_score"]
        for i in range(len(reranked) - 1)
    ),
    ""
)

check(
    "Rerank 没有丢掉来源信息",
    all(item["page"] and item["source"] and item["chunk_id"] for item in reranked),
    ""
)

check(
    "问题里的关键词被正确抽取（中文 2-gram）",
    "营业" in extract_terms(QUESTION) and "利润" in extract_terms(QUESTION),
    "抽出的词示例：" + str(sorted(extract_terms(QUESTION))[:12])
)

# Rerank 应该有"把某条捞上来"的效果，否则它就没在干活
order_changed = [item["vector_rank"] for item in reranked] != list(range(1, TOP_N + 1))

check(
    "Rerank 至少改变了一次名次（否则它没起作用）",
    order_changed or reranked[0]["vector_rank"] != 1,
    "Top-" + str(TOP_N) + " 的向量原始名次：" + str([item["vector_rank"] for item in reranked])
)


# ============================================================
# 9. 测试八：rag_query 能完整跑通
# ============================================================

print()
print("=" * 70)
print("测试八：rag_query 能完整跑通")
print("=" * 70)

result = rag_query(QUESTION, k=K, top_n=TOP_N)

check(
    "返回结构包含 question / top_k / top_n / evidence",
    set(["question", "top_k", "top_n", "evidence"]) <= set(result.keys()),
    "字段：" + str(sorted(result.keys()))
)

check(
    "top_k 是 K 条",
    len(result["top_k"]) == K,
    "实际 " + str(len(result["top_k"]))
)

check(
    "top_n 是 N 条",
    len(result["top_n"]) == TOP_N,
    "实际 " + str(len(result["top_n"]))
)

check(
    "evidence 里每条都能追溯回页码和来源",
    all(
        item["page"] and item["source"] and item["chunk_id"]
        for item in result["evidence"]
    ),
    "页码：" + str([item["page"] for item in result["evidence"]])
)

# ------------------------------------------------
# 最关键的一条：检索到的东西真的能回答问题吗？
# 检查 Top-N 证据里有没有出现"营业收入"，以及有没有具体数字。
# ------------------------------------------------
evidence_text = "".join(item["text"] for item in result["evidence"])

has_revenue = "营业收入" in evidence_text
has_profit = "净利润" in evidence_text
has_number = bool(re.search(r"\d[\d,\.]{6,}", evidence_text))

check(
    "证据里提到了「营业收入」",
    has_revenue,
    ""
)

check(
    "证据里提到了「净利润」",
    has_profit,
    ""
)

check(
    "证据里出现了具体的财务数字",
    has_number,
    re.findall(r"\d[\d,\.]{6,}", evidence_text)[:3]
)

# ------------------------------------------------
# 最能说明 RAG 到底有没有用的两条：
# 1. 写着"2024 年营业收入 238.62 亿"的那段原文，被捞回来了吗？
# 2. Rerank 有没有把它往前挪？
# ------------------------------------------------
answer_positions = [
    index for index, item in enumerate(result["top_n"])
    if ANSWER_NUMBER in item["text"]
]

check(
    "捞回了写着答案数字的那段原文（营业收入 " + ANSWER_NUMBER + "）",
    len(answer_positions) > 0,
    "落在 Top-N 第 " + str([p + 1 for p in answer_positions]) + " 名"
    if answer_positions
    else "没捞到，说明 K 或 chunk_size 还需要调"
)

if answer_positions:

    best = result["top_n"][answer_positions[0]]

    check(
        "Rerank 把答案块的名次提前了",
        best["vector_rank"] > best["rank"],
        "向量第 " + str(best["vector_rank"]) + " 名 -> Rerank 第 "
        + str(best["rank"]) + " 名，page=" + str(best["page"])
    )

check(
    "捞回了归母净利润那条原文（" + ANSWER_PROFIT + "）",
    ANSWER_PROFIT in evidence_text,
    ""
)


# ============================================================
# 10. 打印完整检索过程
# ============================================================

# ============================================================
# 11. 参数是怎么定下来的（实测记录）
#
# 这一节不参与断言，只是把调参过程打印出来，
# 免得以后有人看到 DEFAULT_K=10 觉得是随手写的。
# ============================================================

print("=" * 70)
print("附：Top-K / Top-N 的实测依据")
print("=" * 70)
print("""
问题："中际旭创2024年的营业收入和净利润是多少？"
目标：捞回写着「营业收入 23,862,159,738.37」的那段原文。

chunk_size 的影响（K 取全库、只看该块的名次）：
    chunk_size=500  overlap=100  550 块   答案块第 9 名
    chunk_size=350  overlap= 80  739 块   答案块第 6 名
    chunk_size=250  overlap= 60  957 块   答案块第 6 名
    chunk_size=150  overlap= 40 1360 块   答案块第 3 名

结论：块切得越小越容易命中，但代价是块数暴涨（1360 块）、
      每段证据更碎。第一版保持 chunk_size=500，改为调 K。

Top-K 的影响（chunk_size=500 不变）：
    K=5    答案块进不来（它在第 9 名）
    K=8    仍然进不来（差一名）
    K=10   进入候选，Rerank 后落在第 5 名

Top-N 的影响：
    N=3    答案块被切掉，证据里只有季度数据和利润分配表
    N=5    答案块被留下

Rerank 权重的影响（vector_weight 0.6->0.2，lexical 0.4->0.8）：
    答案块始终停在第 5 名，排序完全不变。
    说明这个 baseline 受权重影响很小，调权重解决不了召回问题。
""")

# 已知局限，一并写出来，避免下一阶段误判
print("=" * 70)
print("已知局限（第一版 baseline 的真实短板）")
print("=" * 70)
print("""
1. 表格页检索弱：年报第 8 页「主要会计数据和财务指标」是标准答案页，
   但它是宽表，PDF 抽文本后变成一串数字，
   向量排在 500 名开外，K=10 也捞不到。
   本测试命中的是第 26 页「营业收入构成」，数字一样，但不是最正的位置。
   这是"表格类 PDF + 纯文本 Embedding"的结构性问题，不是参数没调好。

2. Rerank 是 baseline，不是 Cross-Encoder。
   它只做词面覆盖 + 向量相似度的加权，
   不认识同义词（问"营收"、原文写"营业收入"，只能匹配到"营业"）。

3. 纯 JS 渲染的页面、扫描件 PDF 都处理不了（前者归 web_fetch，后者要 OCR）。

4. 只在一份年报、一个问题上调过参数。
   要有说服力，需要一个多问题的评测集和期望答案页。
""")

print_result(result, text_preview=300)


# ============================================================
# 12. 汇总
# ============================================================

passed_count = sum(1 for _, passed, _ in results if passed)
failed_count = len(results) - passed_count

print("=" * 70)
print("测试结果：" + str(passed_count) + " / " + str(len(results)) + " 通过")
print("=" * 70)

if failed_count:

    print("未通过的项：")

    for name, passed, detail in results:

        if not passed:

            print("  - " + name + "  " + str(detail)[:200])

    sys.exit(1)

print("全部通过。")
