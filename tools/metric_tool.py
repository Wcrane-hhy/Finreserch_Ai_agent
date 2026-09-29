import pandas as pd

from tools.financial_tool import get_financial_statements


# 这些指标是按普通企业口径设计的，套到银行身上会得出错误结论。
# 银行是高杠杆经营，资产负债率天然接近 90%，
# 也没有普通意义上的"营业成本"和"自由现金流"。
BANK_NOT_APPLICABLE_INDICATORS = [
    "gross_margin",
    "free_cash_flow",
    "cash_profit_ratio"
]


def is_bank_company(company_name):
    """
    判断公司是不是银行。

    目前最可靠、最简单的依据就是公司名称里有没有"银行"两个字。
    接口没有返回行业字段，拿不准的时候一律按"不是银行"处理，
    不去猜。
    """

    if not company_name:
        return False

    return "银行" in company_name


def build_bank_note(company_name):
    """
    生成银行股的指标解读提示，直接放进返回结果里，
    让 Evidence 和最终报告都能看到。
    """

    return (
        f"{company_name} 属于银行业。"
        "毛利率、自由现金流、经营现金流/净利润 这三个指标"
        "是按普通企业口径定义的，对银行不适用，不要用它们做经营判断；"
        "资产负债率对银行天然就接近 90%，不能按普通企业的标准解读。"
        "银行的净息差、不良贷款率、拨备覆盖率等专用指标，"
        "当前系统没有对应数据，属于缺失证据。"
    )


def calculate_financial_indicators(
    thscode,
    company_name=None
):
    """
    计算指定 A 股公司的核心财务指标。

    这个函数内部自己会调用 get_financial_statements(thscode) 去拿三张报表，
    调用方（也就是模型）只需要给一个股票代码，
    不用再把整份财务 JSON 搬过来。

    为什么要这样改：
    财务数据大约四万字符，如果让模型把它原样回填进 Tool 参数，
    模型很容易漏掉某条记录里的 fiscal_year 字段。
    而下面三张表是用 inner join 合并的 —— 少一个 fiscal_year，
    就会静默地少掉一整个年度：程序不报错，但算出来的年度数是错的。
    让函数自己取数，这个问题就不存在了。

    参数：
        thscode:
            股票代码，例如 000001.SZ

        company_name:
            公司名称，可选。
            如果传入且判断为银行，返回结果里会多出
            industry_note 和 not_applicable_indicators 两个字段。

    返回：
        {
            "raw_financial_data": [...],
            "calculated_indicators": [...]
        }
    """

    # ========================================================
    # 0. 自己取财务数据
    # ========================================================

    financial_data = get_financial_statements(
        thscode
    )

    # ========================================================
    # 1. JSON → DataFrame
    # ========================================================

    balance_df = pd.DataFrame(
        financial_data["balance_sheet"]
    )

    income_df = pd.DataFrame(
        financial_data["income_statement"]
    )

    cash_flow_df = pd.DataFrame(
        financial_data["cash_flow"]
    )

    # ========================================================
    # 2. 按财务年度排序
    # ========================================================

    balance_df = balance_df.sort_values(
        "fiscal_year"
    )

    income_df = income_df.sort_values(
        "fiscal_year"
    )

    cash_flow_df = cash_flow_df.sort_values(
        "fiscal_year"
    )

    # ========================================================
    # 3. 提取核心字段
    # ========================================================

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

    # ========================================================
    # 4. 三张表合并
    # ========================================================

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

    financial_df = (
        financial_df
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )

    # ========================================================
    # 5. 计算财务指标
    # ========================================================

    indicator_df = financial_df.copy()

    # --------------------------------------------------------
    # 营业收入增长率
    # --------------------------------------------------------

    indicator_df["revenue_growth"] = (
        indicator_df["operating_income"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 净利润增长率
    # --------------------------------------------------------

    indicator_df["net_profit_growth"] = (
        indicator_df["net_profit"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 归母净利润增长率
    # --------------------------------------------------------

    indicator_df["parent_profit_growth"] = (
        indicator_df["parent_holder_net_profit"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 毛利率
    # --------------------------------------------------------

    indicator_df["gross_margin"] = (
        indicator_df["operating_income"]
        - indicator_df["operating_costs"]
    ) / indicator_df["operating_income"]

    # --------------------------------------------------------
    # 营业利润率
    # --------------------------------------------------------

    indicator_df["operating_margin"] = (
        indicator_df["operating_profit"]
        / indicator_df["operating_income"]
    )

    # --------------------------------------------------------
    # 净利率
    # --------------------------------------------------------

    indicator_df["net_margin"] = (
        indicator_df["net_profit"]
        / indicator_df["operating_income"]
    )

    # --------------------------------------------------------
    # 资产负债率
    # --------------------------------------------------------

    indicator_df["debt_ratio"] = (
        indicator_df["total_debt"]
        / indicator_df["assets_total"]
    )

    # --------------------------------------------------------
    # 经营现金流 / 净利润
    # --------------------------------------------------------

    indicator_df["cash_profit_ratio"] = (
        indicator_df["act_cash_flow_net"]
        / indicator_df["net_profit"]
    )

    # --------------------------------------------------------
    # 自由现金流
    # --------------------------------------------------------

    indicator_df["free_cash_flow"] = (
        indicator_df["act_cash_flow_net"]
        - indicator_df["pay_fixed_assets_etc_cash"]
    )

    # --------------------------------------------------------
    # ROA
    # --------------------------------------------------------

    indicator_df["ROA"] = (
        indicator_df["net_profit"]
        / indicator_df["assets_total"]
    )

    # --------------------------------------------------------
    # ROE
    # 使用期初期末平均股东权益
    # --------------------------------------------------------

    indicator_df["avg_equity"] = (
        indicator_df["holder_equity_total"]
        + indicator_df["holder_equity_total"].shift(1)
    ) / 2

    indicator_df["ROE"] = (
        indicator_df["net_profit"]
        / indicator_df["avg_equity"]
    )

    # ========================================================
    # 6. 准备返回数据
    # ========================================================

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

    raw_columns = [
        "fiscal_year",
        "assets_total",
        "cash",
        "accounts_receivable",
        "total_debt",
        "holder_equity_total",
        "operating_income",
        "operating_costs",
        "research_and_development_expenses",
        "operating_profit",
        "net_profit",
        "parent_holder_net_profit",
        "basic_eps",
        "act_cash_flow_net",
        "invest_cash_flow_net",
        "financing_cash_flow_net",
        "pay_fixed_assets_etc_cash",
        "cash_equivalents_net_addition"
    ]

    raw_financial_data = (
        financial_df[raw_columns]
        .to_dict(orient="records")
    )

    calculated_indicators = (
        indicator_df[indicator_columns]
        .to_dict(orient="records")
    )

    # ========================================================
    # 7. 返回
    # ========================================================

    result = {
        "raw_financial_data": raw_financial_data,
        "calculated_indicators": calculated_indicators
    }

    # 银行股额外附上说明，避免下游把通用指标当成普通企业指标解读
    if is_bank_company(company_name):
        result["industry_note"] = build_bank_note(
            company_name
        )
        result["not_applicable_indicators"] = (
            BANK_NOT_APPLICABLE_INDICATORS
        )

    return result
