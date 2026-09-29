# -*- coding: utf-8 -*-
"""
RAG 第三步：Embedding（文本 → 向量）。

用的是哪个模型：
    BAAI/bge-small-zh-v1.5  —— 智源研究院的中文小模型，中文检索效果很好。

为什么用得起它（这是本轮最关键的发现）：
    这台机器上 **已经缓存了这个模型**（在 HuggingFace 的本地缓存里），
    而且 transformers / tokenizers / torch 都已经装好了。
    所以：
      - 不需要下载模型（这台机器连不上 huggingface.co）；
      - 不需要新增任何依赖；
      - 完全离线可跑。
    实测加载 0.5 秒，512 维向量。

核心逻辑就三行（这也是整个文件存在的意义）：
    text  ->  tokenizer  ->  input_ids
    input_ids  ->  model  ->  last_hidden_state
    last_hidden_state[:, 0]  ->  向量

为什么取 [:, 0]（第 0 个位置）：
    BGE 系列用的是 CLS pooling ——
    模型输出的第 0 个位置是专门用来代表整句语义的。
    另一种常见做法是"对所有 token 求平均"（mean pooling），
    但那不是这个模型训练时的用法，效果会差一些。
"""

import os


# ============================================================
# 0. 必须在 import transformers 之前设好离线开关
# ============================================================
#
# 这台机器连不上 huggingface.co。
# 如果不关掉联网检查，from_pretrained 会先去网上问一圈，
# 卡十几秒最后报连不上，而其实模型就在本地缓存里。
#
# 注意这两行必须在 import transformers 之前执行，
# 所以只能写在文件最上面。

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. 配置
# ============================================================

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 中文模型的官方用法：
# **查询**前面要加一句指令，**文档**前面不加。
# 加上它检索会准一些；不加也能跑，只是差一点。
# 这是模型本身的约定，不是可以随便省掉的装饰。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模型最大吃 512 个 token。
# 中文大致 1 个字 ≈ 1 个 token，所以 chunk_size=500 刚好在范围内。
MAX_LENGTH = 512

# bge-small 的向量维度，写出来是为了方便核对
EMBEDDING_DIM = 512


# ============================================================
# 2. 模型只加载一次
# ============================================================
#
# 加载模型要 0.5 秒，向量化几百个 chunk 时会反复调用 embed_texts。
# 每次重新加载就太浪费了，所以用两个模块级变量缓存住。
#
# 这里刻意不做成 class，也不做单例模式 ——
# 两个全局变量就够了，比那些写法好懂得多。

_tokenizer = None
_model = None


def load_model():
    """
    加载（或复用）tokenizer 和模型。
    """

    global _tokenizer, _model

    if _model is None:

        print("正在加载 Embedding 模型：" + MODEL_NAME)

        # local_files_only=True：只用本地缓存，绝不联网。
        # 缓存里没有就直接报错，而不是卡在那里等超时。
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True
        )

        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True
        )

        # 只做推理，切成评估模式。
        # 不做这一步的话，模型里有些层行为不一样（这里是 dropout）。
        _model.eval()

        print("模型加载完成，向量维度：" + str(EMBEDDING_DIM))

    return _tokenizer, _model


def embed_texts(texts, batch_size=32, is_query=False):
    """
    把一批文本变成向量。这就是 RAG 里说的 "Embedding"。

    参数：
        texts:      文本列表
        batch_size: 一次送多少条给模型。
                    一次全送会吃爆内存，一条一条送又太慢，32 是个折中。
        is_query:   这批文本是不是"查询"。
                    是查询就加 BGE 的指令前缀，是文档就不加。

    返回：
        numpy 数组，形状 (len(texts), 512)，
        每一行是一个已经单位化的向量。

    为什么要单位化（L2 normalize）：
        单位化之后，两个向量的点积就等于余弦相似度。
        后面算相似度就不用再除以模长了，省事也不容易出错。
    """

    if not texts:

        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    tokenizer, model = load_model()

    if is_query:

        texts = [QUERY_INSTRUCTION + text for text in texts]

    all_vectors = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        # padding=True      同一批里长度补齐，短句用 pad 填到最长那条
        # truncation=True   超过 MAX_LENGTH 的截断（chunk_size=500 时不会触发）
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # no_grad：只推理，不训练，不需要记梯度，能省很多内存
        with torch.no_grad():

            output = model(**encoded)

        # ------------------------------------------------
        # 核心：取第 0 个位置的隐藏状态作为整句的语义向量
        # ------------------------------------------------
        vectors = output.last_hidden_state[:, 0]

        # 单位化
        vectors = torch.nn.functional.normalize(
            vectors,
            p=2,
            dim=1
        )

        all_vectors.append(
            vectors.cpu().numpy()
        )

    return np.vstack(all_vectors).astype("float32")


def embed_query(question):
    """
    把一个查询问题变成向量。

    单独开一个函数，而不是让调用方每次记得传 is_query=True ——
    因为漏传了不会报错，只会让检索结果悄悄变差，这种 bug 最难查。
    """

    if not question or not question.strip():

        raise ValueError("查询问题不能为空")

    return embed_texts([question.strip()], is_query=True)[0]
