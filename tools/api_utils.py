# -*- coding: utf-8 -*-
"""
A 股数据接口的通用响应检查。

当前后端（fuyao.aicubes.cn）有一个特点：

    HTTP 状态码永远是 200，业务错误写在 JSON 的 code 字段里。

所以只写 response.raise_for_status() 是不够的 ——
code != 0 的时候它不会报错，程序会继续往下走，
等取 data 里的字段时才炸出一个 KeyError 或 TypeError，
错误信息完全看不出真正原因。

这里放一个函数统一检查，五个 Tool 都调用它。
"""


def check_api_result(result):
    """
    检查接口返回的 JSON。

    code == 0  -> 正常，原样返回
    code != 0  -> 抛出 RuntimeError，并把接口给的 message 带上

    参数：
        result: response.json() 得到的字典

    返回：
        result 本身（方便写成 result = check_api_result(result)）
    """

    if not isinstance(result, dict):
        raise RuntimeError(
            f"A 股接口返回内容不是 JSON 对象：{result}"
        )

    if result.get("code") != 0:
        raise RuntimeError(
            "A 股接口返回错误："
            + str(result.get("message", "接口未提供错误信息"))
        )

    return result
