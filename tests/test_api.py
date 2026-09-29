# -*- coding: utf-8 -*-
# ============================================================
# 【历史原型脚本，未参与主链路】
#
# 这是项目最早的单文件版本：从搜索股票、拉三张报表、算指标，
# 一直到调用大模型出报告，全部写在一个脚本里。
# 里面的指标计算逻辑和 tools/metric_tool.py 是重复的。
#
# 保留原因：可以作为"从脚本到 Agent 管线"的对照参考。
# 注意：这里的逻辑不会再跟着 metric_tool.py 一起更新，
#       真要算指标请以 tools/metric_tool.py 为准。
# ============================================================

import os
import requests
import pandas as pd
import json
import math

pd.set_option("display.max_columns", None)
# ============================================================
# 1. 基础配置
# ============================================================

api_key = os.getenv("HITHINK_FINANCE_API_KEY")

headers = {
    "X-api-key": api_key
}


# ============================================================
# 2. 选择分析公司
# ============================================================

ticker_search_url = (
    "https://fuyao.aicubes.cn/api/meta/tickers/search"
)

print("请输入股票名称或股票代码：")

search_params = {
    "q": input("> "),
    "asset_type": "a-share"
}

response = requests.get(
    ticker_search_url,
    headers=headers,
    params=search_params
)

print("\n股票搜索请求状态：", response.status_code)

result = response.json()

items = result["data"]["item"]

print("找到的股票数量：", len(items))


# 没有找到股票
if len(items) == 0:

    print("没有找到对应的股票")
    exit()


# 只有一个结果
elif len(items) == 1:

    selected = items[0]

    print("\n已自动选择：")
    print(selected["name"], selected["thscode"])


# 多个结果
else:

    print("\n搜索结果：")

    for i, item in enumerate(items, start=1):
        print(i, item["name"], item["thscode"])

    choice = int(input("\n请输入股票编号："))

    selected = items[choice - 1]

    print("\n您选择的是：")
    print(selected["name"], selected["thscode"])


# 提取股票基本信息
company_name = selected["name"]
thscode = selected["thscode"]
list_date = selected["list_date"]

print("\n分析公司：", company_name)
print("股票代码：", thscode)
print("上市日期：", list_date)


# ============================================================
# 3. 获取资产负债表
# ============================================================

print("\n" + "=" * 60)
print("获取资产负债表")
print("=" * 60)

balance_sheet_url = (
    "https://fuyao.aicubes.cn/api/a-share/financials/balance-sheets"
)

balance_params = {
    "thscode": thscode,
    "period": "annual"
}

response_balance = requests.get(
    balance_sheet_url,
    headers=headers,
    params=balance_params
)

print("请求状态：", response_balance.status_code)

balance_data = response_balance.json()


# ============================================================
# 4. 获取利润表
# ============================================================

print("\n" + "=" * 60)
print("获取利润表")
print("=" * 60)

income_statement_url = (
    "https://fuyao.aicubes.cn/api/a-share/financials/income-statements"
)

income_params = {
    "thscode": thscode,
    "period": "annual",
    "limit": 4
}

response_income = requests.get(
    income_statement_url,
    headers=headers,
    params=income_params
)

print("请求状态：", response_income.status_code)

income_data = response_income.json()


# ============================================================
# 5. 获取现金流量表
# ============================================================

print("\n" + "=" * 60)
print("获取现金流量表")
print("=" * 60)

cash_flow_url = (
    "https://fuyao.aicubes.cn/api/a-share/financials/cash-flow-statements"
)

cash_flow_params = {
    "thscode": thscode,
    "period": "annual",
    "limit": 4
}

response_cash_flow = requests.get(
    cash_flow_url,
    headers=headers,
    params=cash_flow_params
)

print("请求状态：", response_cash_flow.status_code)

cash_flow_data = response_cash_flow.json()


# ============================================================
# 6. 提取三张财务报表的 item
# ============================================================

balance_items = balance_data["data"]["item"]
income_items = income_data["data"]["item"]
cash_flow_items = cash_flow_data["data"]["item"]

print("\n" + "=" * 60)
print("三张财务报表读取成功")
print("=" * 60)

print("资产负债表记录数量：", len(balance_items))
print("利润表记录数量：", len(income_items))
print("现金流量表记录数量：", len(cash_flow_items))


# ============================================================
# 7. JSON → DataFrame
# ============================================================

balance_df = pd.DataFrame(balance_items)
income_df = pd.DataFrame(income_items)
cash_flow_df = pd.DataFrame(cash_flow_items)

print("\n" + "=" * 60)
print("JSON 已转换为 DataFrame")
print("=" * 60)

print("\n资产负债表：")
print(balance_df)

print("\n利润表：")
print(income_df)

print("\n现金流量表：")
print(cash_flow_df)

# =========================
# 财务数据整理
# =========================

print("\n" + "=" * 60)
print("开始整理财务数据")
print("=" * 60)

# 按财务年度排序
balance_df = balance_df.sort_values("fiscal_year")
income_df = income_df.sort_values("fiscal_year")
cash_flow_df = cash_flow_df.sort_values("fiscal_year")

print("\n财务年度：")
print(sorted(income_df["fiscal_year"].unique()))


# =========================
# 提取核心财务字段
# =========================

balance_core = balance_df[
    [
        "fiscal_year",
        "assets_total",
        "cash",
        "accounts_receivable",
        "total_debt",
        "holder_equity_total"
    ]
].copy()

income_core = income_df[
    [
        "fiscal_year",
        "operating_income",
        "operating_costs",
        "research_and_development_expenses",
        "operating_profit",
        "net_profit",
        "parent_holder_net_profit",
        "basic_eps"
    ]
].copy()

cash_flow_core = cash_flow_df[
    [
        "fiscal_year",
        "act_cash_flow_net",
        "invest_cash_flow_net",
        "financing_cash_flow_net",
        "pay_fixed_assets_etc_cash",
        "cash_equivalents_net_addition"
    ]
].copy()


# =========================
# 三张表合并
# =========================

financial_df = (
    balance_core
    .merge(
        income_core,
        on="fiscal_year",
        how="inner"
    )
    .merge(
        cash_flow_core,
        on="fiscal_year",
        how="inner"
    )
)

financial_df = financial_df.sort_values(
    "fiscal_year"
).reset_index(drop=True)


print("\n" + "=" * 60)
print("核心财务数据")
print("=" * 60)

print(financial_df)

# ============================================================
# 8. 计算核心财务指标
# ============================================================

print("\n" + "=" * 60)
print("开始计算财务指标")
print("=" * 60)

indicator_df = financial_df.copy()


# ------------------------------------------------------------
# 1. 营业收入增长率
# ------------------------------------------------------------

indicator_df["revenue_growth"] = (
    indicator_df["operating_income"]
    .pct_change()
)


# ------------------------------------------------------------
# 2. 净利润增长率
# ------------------------------------------------------------

indicator_df["net_profit_growth"] = (
    indicator_df["net_profit"]
    .pct_change()
)


# ------------------------------------------------------------
# 3. 归母净利润增长率
# ------------------------------------------------------------

indicator_df["parent_profit_growth"] = (
    indicator_df["parent_holder_net_profit"]
    .pct_change()
)


# ------------------------------------------------------------
# 4. 毛利率
# ------------------------------------------------------------

indicator_df["gross_margin"] = (
    indicator_df["operating_income"]
    - indicator_df["operating_costs"]
) / indicator_df["operating_income"]


# ------------------------------------------------------------
# 5. 营业利润率
# ------------------------------------------------------------

indicator_df["operating_margin"] = (
    indicator_df["operating_profit"]
    / indicator_df["operating_income"]
)


# ------------------------------------------------------------
# 6. 净利率
# ------------------------------------------------------------

indicator_df["net_margin"] = (
    indicator_df["net_profit"]
    / indicator_df["operating_income"]
)


# ------------------------------------------------------------
# 7. 资产负债率
# ------------------------------------------------------------

indicator_df["debt_ratio"] = (
    indicator_df["total_debt"]
    / indicator_df["assets_total"]
)


# ------------------------------------------------------------
# 8. 经营现金流 / 净利润
# ------------------------------------------------------------

indicator_df["cash_profit_ratio"] = (
    indicator_df["act_cash_flow_net"]
    / indicator_df["net_profit"]
)


# ------------------------------------------------------------
# 9. 自由现金流
# ------------------------------------------------------------

indicator_df["free_cash_flow"] = (
    indicator_df["act_cash_flow_net"]
    - indicator_df["pay_fixed_assets_etc_cash"]
)


# ------------------------------------------------------------
# 10. ROA
# ------------------------------------------------------------

indicator_df["ROA"] = (
    indicator_df["net_profit"]
    / indicator_df["assets_total"]
)


# ------------------------------------------------------------
# 11. ROE
# 使用期初期末平均股东权益
# ------------------------------------------------------------

indicator_df["avg_equity"] = (
    indicator_df["holder_equity_total"]
    + indicator_df["holder_equity_total"].shift(1)
) / 2

indicator_df["ROE"] = (
    indicator_df["net_profit"]
    / indicator_df["avg_equity"]
)


# ============================================================
# 查看指标
# ============================================================

print("\n核心财务指标：")

indicator_columns = [
    "fiscal_year",
    "revenue_growth",
    "net_profit_growth",
    "parent_profit_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "debt_ratio",
    "cash_profit_ratio",
    "free_cash_flow",
    "ROA",
    "ROE"
]

print(
    indicator_df[indicator_columns]
)

# ============================================================
# 9. 生成可视化展示表
# ============================================================

display_df = indicator_df[indicator_columns].copy()

display_df["revenue_growth"] = (
    display_df["revenue_growth"] * 100
)

display_df["net_profit_growth"] = (
    display_df["net_profit_growth"] * 100
)

display_df["parent_profit_growth"] = (
    display_df["parent_profit_growth"] * 100
)

display_df["gross_margin"] = (
    display_df["gross_margin"] * 100
)

display_df["operating_margin"] = (
    display_df["operating_margin"] * 100
)

display_df["net_margin"] = (
    display_df["net_margin"] * 100
)

display_df["debt_ratio"] = (
    display_df["debt_ratio"] * 100
)

display_df["cash_profit_ratio"] = (
    display_df["cash_profit_ratio"] * 100
)

display_df["ROA"] = (
    display_df["ROA"] * 100
)

display_df["ROE"] = (
    display_df["ROE"] * 100
)

print("\n" + "=" * 60)
print("财务指标展示表")
print("=" * 60)

print(
    display_df.round(2)
)

# ============================================================
# 10. 准备 LLM 分析数据
# ============================================================

print("\n" + "=" * 60)
print("准备 LLM 分析数据")
print("=" * 60)


# ------------------------------------------------------------
# 10.1 原始核心财务数据
# ------------------------------------------------------------

raw_columns = [
    "fiscal_year",
    "assets_total",
    "cash",
    "total_debt",
    "holder_equity_total",
    "operating_income",
    "operating_costs",
    "operating_profit",
    "net_profit",
    "parent_holder_net_profit",
    "basic_eps",
    "act_cash_flow_net",
    "invest_cash_flow_net",
    "financing_cash_flow_net",
    "pay_fixed_assets_etc_cash"
]

raw_financial_data = (
    financial_df[raw_columns]
    .to_dict(orient="records")
)


# ------------------------------------------------------------
# 10.2 Python 计算后的财务指标
# ------------------------------------------------------------

calculated_indicators = (
    indicator_df[indicator_columns]
    .to_dict(orient="records")
)


# ------------------------------------------------------------
# 10.3 组合成 LLM 输入数据
# ------------------------------------------------------------

financial_summary = {
    "company": company_name,
    "thscode": thscode,
    "list_date": list_date,

    "raw_financial_data": raw_financial_data,

    "calculated_indicators": calculated_indicators
}


print("\nLLM 输入数据：")

print(financial_summary)

# ============================================================
# 11. 转换为 JSON
# ============================================================

def clean_nan(value):
    if isinstance(value, dict):
        return {
            key: clean_nan(val)
            for key, val in value.items()
        }

    elif isinstance(value, list):
        return [
            clean_nan(item)
            for item in value
        ]

    elif isinstance(value, float) and math.isnan(value):
        return None

    else:
        return value


financial_summary_clean = clean_nan(financial_summary)

llm_input = json.dumps(
    financial_summary_clean,
    ensure_ascii=False,
    indent=2,
    allow_nan=False
)

print("\n" + "=" * 60)
print("LLM 输入数据")
print("=" * 60)

print(llm_input)

# ============================================================
# 12. 构建 LLM 财务分析 Prompt
# ============================================================

user_question = input(
    "\n请输入你想研究的问题：\n> "
)

prompt = f"""
你是一名金融研究助手。

你的任务是根据提供的公司信息和财务数据，回答用户提出的研究问题。

公司信息：
{company_name}
{thscode}

财务数据：
{llm_input}

用户的问题：
{user_question}

请根据以上信息回答用户的问题。

回答原则：
1. 仅使用提供的数据和信息进行分析。
2. 不要编造数据、事实或公司信息。
3. 如果现有数据不足以回答用户的问题，要明确说明数据不足。
4. 优先分析多个年份之间的变化趋势，而不是只描述单个年份。
5. 尽可能使用具体年份、指标和数据支持分析。
6. 根据用户的问题选择最相关的数据进行回答，不要求每次都分析所有指标。
7. 如果用户的问题比较宽泛，可以从数据中选择几个最相关的维度进行综合分析。
8. 如果某项指标为 null，说明该数据缺失，不要自行推测。
9. 回答应清晰、简洁，适合普通金融研究场景阅读。
10. 不提供买入、卖出或其他具体投资建议。
"""

# ============================================================
# 13. 调用 DeepSeek
# ============================================================

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

deepseek_url = "https://api.deepseek.com/chat/completions"

headers = {
"Authorization": f"Bearer {deepseek_api_key}",
"Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.2
}

response = requests.post(
    deepseek_url,
    headers=headers,
    json=payload
)

print("\n" + "=" * 60)
print("DeepSeek 请求状态")
print("=" * 60)

print(response.status_code)

result = response.json()

answer = result["choices"][0]["message"]["content"]

print("\n" + "=" * 60)
print("LLM 分析结果")
print("=" * 60)

print(answer)





























