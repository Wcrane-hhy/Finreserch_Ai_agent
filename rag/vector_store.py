# -*- coding: utf-8 -*-
"""
RAG 第四步：Vector DB（向量数据库）。

用的是哪个：
    Chroma（chromadb 1.5.9），本地持久化模式。

为什么是 Chroma：
    这是 requirement 里指定的首选。
    对它来说，"本地持久化"就是 PersistentClient(path=...) 一句话，
    数据落在磁盘上的一个目录里，不用起服务、不用连网络。

存了什么：
    每个 chunk 存三份东西：
        文档本体  documents   -> chunk 的原文
        元数据    metadatas   -> {"chunk_id": ..., "source": ..., "page": ...}
        向量      embeddings  -> 512 维
    chunk_id 同时用作 Chroma 的主键 id。

    为什么元数据要存 chunk_id / source / page：
        因为检索结果必须能回溯。
        只有原文没有出处的话，模型引用了一段话，人没法核对。
        这三个字段就是"证据可追溯"的全部依赖。

为什么距离用 cosine：
    BGE 的向量是单位向量，余弦相似度是最自然的度量。
    Chroma 的 cosine 空间返回的是"余弦距离" = 1 - 余弦相似度，
    所以相似度 = 1 - distance。这个换算在 search() 里做了。

这个文件不负责 Embedding，也不负责 Rerank，
只管"存进去"和"按向量捞出来"。
"""

import shutil

from pathlib import Path

import chromadb


# ============================================================
# 1. 路径与常量
# ============================================================

# rag/ 的上一级就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# 向量库落盘位置。放在 data/ 下面，和年报放一起，方便整体拷走。
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

# Chroma 的 collection 名字。
# 注意它有个硬要求：3~512 个字符，只能是字母数字和 . _ -，
# 且首尾必须是字母或数字。用 "rag" 这种短名字会直接报错。
COLLECTION_NAME = "annual_report"

# 写入时一批多少条。
# 一次写几百条会让 Chroma 那边日志刷屏，分批发还能看到进度。
ADD_BATCH_SIZE = 200


# ============================================================
# 2. 连接
# ============================================================

# 同一个进程里对同一个目录反复创建 PersistentClient 是浪费，
# 按路径缓存住。key 是解析成绝对路径后的字符串。
_clients = {}


def get_client(db_dir=DEFAULT_DB_DIR):
    """
    拿到（或创建）一个本地持久化的 Chroma 客户端。
    """

    db_dir = Path(db_dir).resolve()

    db_dir.mkdir(parents=True, exist_ok=True)

    key = str(db_dir)

    if key not in _clients:

        _clients[key] = chromadb.PersistentClient(path=key)

    return _clients[key]


def get_collection(db_dir=DEFAULT_DB_DIR, create=True):
    """
    拿到 collection。

    create=False 时，库不存在就抛错 ——
    查询前应该先确认索引建过了，否则会静默返回空结果。
    """

    client = get_client(db_dir)

    if create:

        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    try:

        return client.get_collection(COLLECTION_NAME)

    except Exception:

        raise RuntimeError(
            "向量库还没有建好，请先运行：\n"
            "    python -m rag.rag_query --build"
        )


def count_chunks(db_dir=DEFAULT_DB_DIR):
    """
    向量库里现在有多少个 chunk。
    """

    return get_collection(db_dir, create=False).count()


# ============================================================
# 3. 建库
# ============================================================

def build_index(
    reports_dir=DEFAULT_REPORTS_DIR,
    db_dir=DEFAULT_DB_DIR,
    reset=True,
    chunk_size=None,
    overlap=None
):
    """
    完整走一遍：PDF → 文本 → Chunk → Embedding → 写进向量库。

    参数：
        reports_dir: 年报 PDF 所在目录
        db_dir:      向量库落盘目录
        reset:       True 表示先清空重建。
                     改了 chunk 参数就必须重建，否则新旧块混在一起。
        chunk_size / overlap: 不传就用 chunker.py 里的默认值

    返回：
        一个统计字典，方便测试和打印
    """

    # 这里才 import，是为了让"只想查询"的调用方
    # 不用顺带把 PDF 解析也加载起来。
    from rag.pdf_loader import load_reports
    from rag.chunker import chunk_pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
    from rag.embedding import embed_texts

    if chunk_size is None:

        chunk_size = DEFAULT_CHUNK_SIZE

    if overlap is None:

        overlap = DEFAULT_OVERLAP

    # ------------------------------------------------
    # 1) PDF -> 页
    # ------------------------------------------------
    print("=" * 60)
    print("Step 1  PDF -> Text")
    print("=" * 60)

    pages = load_reports(reports_dir)

    sources = sorted({item["source"] for item in pages})

    print("读取到 " + str(len(pages)) + " 页，来自 " + str(len(sources)) + " 个文件：")

    for source in sources:

        print("    " + source)

    # ------------------------------------------------
    # 2) 页 -> chunk
    # ------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2  Chunk")
    print("=" * 60)

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(
        "切出 " + str(len(chunks)) + " 个 chunk"
        + "（chunk_size=" + str(chunk_size)
        + ", overlap=" + str(overlap) + "）"
    )

    # ------------------------------------------------
    # 3) chunk -> 向量
    # ------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3  Embedding")
    print("=" * 60)

    texts = [item["text"] for item in chunks]

    vectors = embed_texts(texts, is_query=False)

    print("生成向量 " + str(vectors.shape))

    # ------------------------------------------------
    # 4) 写入向量库
    # ------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4  Vector DB")
    print("=" * 60)

    client = get_client(db_dir)

    if reset:

        try:

            client.delete_collection(COLLECTION_NAME)

            print("已清空旧的 collection：" + COLLECTION_NAME)

        except Exception:

            # 第一次建库时 collection 还不存在，删不掉是正常的
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), ADD_BATCH_SIZE):

        batch = chunks[start:start + ADD_BATCH_SIZE]

        collection.add(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=vectors[start:start + ADD_BATCH_SIZE].tolist(),
            metadatas=[
                {
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "page": item["page"]
                }
                for item in batch
            ]
        )

        print(
            "已写入 "
            + str(min(start + ADD_BATCH_SIZE, len(chunks)))
            + " / " + str(len(chunks))
        )

    total = collection.count()

    print("向量库当前共 " + str(total) + " 个 chunk")
    print("落盘位置：" + str(Path(db_dir).resolve()))

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "stored": total,
        "vector_dim": int(vectors.shape[1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "sources": sources,
        "db_dir": str(Path(db_dir).resolve())
    }


# ============================================================
# 4. 检索
# ============================================================

def list_sources(db_dir=DEFAULT_DB_DIR):
    """
    列出向量库里已经入库了哪些年报文件。

    为什么要这个：
        Tool 返回证据时，必须能告诉 Agent"这些证据来自哪本年报"。
        另一份年报入库之后，这里会自动多出一个名字，不用改代码。
        另外，如果调用方指定了一个库里没有的文件名，
        也要靠这个列表才能给出"当前可检索的是这些"的可读错误。

    实现上就是把所有 chunk 的 metadata 取出来去重。
    550 条 metadata 读一遍很快，而且只有在需要时才调用。
    """

    collection = get_collection(db_dir, create=False)

    data = collection.get(include=["metadatas"])

    return sorted(
        {
            item.get("source")
            for item in data["metadatas"]
            if item and item.get("source")
        }
    )


def search(query_vector, k=5, db_dir=DEFAULT_DB_DIR, source=None):
    """
    Vector Search：拿一个查询向量，捞最像的 k 个 chunk。

    参数：
        query_vector: 已经算好的查询向量（512 维 numpy 数组）
        k:            Top-K 的 K
        db_dir:       向量库目录
        source:       只在这一个年报文件里检索（文件名要精确匹配，
                      严格来说是 vector_store.list_sources() 返回的名字）。
                      不传就是全部年报一起检索。

    返回：
        [
            {
                "chunk_id":   "...",
                "text":       "...",
                "page":       7,
                "source":     "中际旭创 2024 年年报.pdf",
                "distance":   0.2134,   # 余弦距离，越小越像
                "similarity": 0.7866,   # 余弦相似度 = 1 - distance，越大越像
                "rank":       1         # 向量检索的名次
            },
            ...
        ]

    注意 k 不能超过库里 chunk 的总数，
    超了 Chroma 会直接报错，所以这里先取个 min。
    """

    collection = get_collection(db_dir, create=False)

    total = collection.count()

    if total == 0:

        raise RuntimeError("向量库是空的，请先建库")

    k = min(k, total)

    # 转成 list 是因为 Chroma 的入参必须是可 JSON 序列化的
    if hasattr(query_vector, "tolist"):

        query_vector = query_vector.tolist()

    # 按年报文件名过滤，交给 Chroma 的 metadata 过滤做，
    # 而不是"先捞 k 条再自己筛"——后者会让 Top-K 名次失去意义。
    where = {"source": {"$eq": source}} if source else None

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where
    )

    # Chroma 返回的是"按查询分组"的嵌套结构（外层是查询列表），
    # 我们一次只查一个，所以取 [0]。
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    candidates = []

    for position, chunk_id in enumerate(ids):

        distance = float(distances[position])

        metadata = metadatas[position] or {}

        candidates.append(
            {
                "chunk_id": chunk_id,
                "text": documents[position],
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "distance": distance,
                # 余弦距离换算回相似度，后面 Rerank 和展示都用这个
                "similarity": 1.0 - distance,
                "rank": position + 1
            }
        )

    return candidates


def clear_index(db_dir=DEFAULT_DB_DIR):
    """
    把整个向量库目录删掉。重建索引前用得上。
    """

    db_dir = Path(db_dir).resolve()

    if db_dir.exists():

        shutil.rmtree(db_dir)

    _clients.pop(str(db_dir), None)
