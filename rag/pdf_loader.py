# -*- coding: utf-8 -*-
"""
RAG 第一步：PDF → 结构化文本。

为什么用 PyMuPDF 而不是 pypdf / pdfplumber：
    环境里已经装了 PyMuPDF（import 名是 fitz），
    它是目前最省事的选择：
      - 不用新增任何依赖；
      - 抽取速度快（这份 236 页的年报 1 秒内读完）；
      - 能按页拿文本，正好满足"保留 page 信息"这个要求。
    如果以后要处理扫描件（图片型 PDF），PyMuPDF 也能渲染出图片，
    但那时需要 OCR，属于另一件事了。

这个文件只做一件事：
    把 PDF 变成 [{"page": 1, "text": "..."}, ...]
不负责切块，也不负责向量化。
"""

import re

from pathlib import Path

import fitz


# ============================================================
# 1. 文本清洗
# ============================================================

def clean_text(text):
    """
    清理从 PDF 里抽出来的原始文本。

    PDF 抽出来的文本有几个很影响后续检索的毛病：
      1. 表格里的每个格子都会变成单独一行，
         导致一句话被拆成好几行，中间塞满换行和空格；
      2. 全角空格（U+3000）在中文 PDF 里很常见；
      3. 页眉页脚会跟正文混在一起。

    这里只做"轻清理"，不做语义处理：
      - 全角空格换成普通空格；
      - 连续空格压成一个；
      - 连续空行压成一个换行；
      - 两个汉字之间的换行直接删掉（因为中文本来就不需要空格）。
        这一步最关键：不做的话，
        "公司2024年度实现营业收入\n238.6亿元" 会被当成两段无关的文字。
    """

    text = text.replace("\u3000", " ")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n\s*\n+", "\n", text)

    # 汉字之间的换行 = 被 PDF 硬拆开的同一句话，去掉了读起来才连贯
    text = re.sub(
        r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])",
        "",
        text
    )

    return text.strip()


# ============================================================
# 2. 读取单个 PDF
# ============================================================

def load_pdf(pdf_path):
    """
    读取一个 PDF，返回按页拆分的文本。

    参数：
        pdf_path: PDF 文件路径（str 或 Path）

    返回：
        [
            {"source": "中际旭创 2024 年年报.pdf", "page": 1, "text": "..."},
            {"source": "中际旭创 2024 年年报.pdf", "page": 2, "text": "..."},
            ...
        ]

        page 从 1 开始，和 PDF 阅读器上显示的页码一致，
        这样最后生成的证据可以直接告诉人"去翻第几页"。

    为什么保留 page：
        这是"证据可追溯"的最低要求。
        RAG 返回的每一段原文，都要能回答"这是从年报第几页来的"，
        否则模型引用了一段话，人却没法核对。

    为什么这里也带 source：
        让 load_pdf 和 load_reports 返回同一种结构。
        否则 load_pdf 的结果直接丢给 chunk_pages 会 KeyError，
        这种坑不该留给调用方去踩。
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():

        raise FileNotFoundError("PDF 不存在：" + str(pdf_path))

    pages = []

    with fitz.open(str(pdf_path)) as document:

        for index, page in enumerate(document):

            # "text" 模式按阅读顺序输出纯文本，不带格式和坐标
            raw_text = page.get_text("text")

            text = clean_text(raw_text)

            # 完全没文字的页直接跳过（封面图、纯图片页）
            if not text:

                continue

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": index + 1,
                    "text": text
                }
            )

    if not pages:

        raise RuntimeError(
            "这个 PDF 没有抽出任何文本，可能是扫描件（需要 OCR）："
            + str(pdf_path)
        )

    return pages


# ============================================================
# 3. 读取整个目录的 PDF
# ============================================================

def load_reports(reports_dir):
    """
    读取目录下所有 PDF，给每页带上来源文件名。

    参数：
        reports_dir: 存放年报 PDF 的目录，例如 data/reports

    返回：
        [
            {"source": "中际旭创 2024 年年报.pdf", "page": 7, "text": "..."},
            ...
        ]

    为什么要带 source：
        以后 data/reports/ 里会放越来越多公司的年报，
        检索结果必须能说清"这段话来自哪个文件"，
        不然多个公司的数据混在一起就分不清了。
    """

    reports_dir = Path(reports_dir)

    if not reports_dir.exists():

        raise FileNotFoundError("目录不存在：" + str(reports_dir))

    pdf_paths = sorted(reports_dir.glob("*.pdf"))

    if not pdf_paths:

        raise RuntimeError("目录里没有 PDF：" + str(reports_dir))

    pages = []

    for pdf_path in pdf_paths:

        # load_pdf 返回的每一页已经带了 source，直接接过来就行
        pages.extend(load_pdf(pdf_path))

    return pages
