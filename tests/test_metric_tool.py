from tools.metric_tool import calculate_financial_indicators


# ========================================================
# 1. 计算财务指标
#
# calculate_financial_indicators 现在只接收股票代码，
# 它内部会自己调用 get_financial_statements 取三张报表，
# 所以这里不需要先手动取一遍财务数据。
# ========================================================

thscode = "300308.SZ"

result = calculate_financial_indicators(
    thscode
)


# ========================================================
# 2. 查看原始财务数据
# ========================================================

print("=" * 60)
print("原始财务数据")
print("=" * 60)

for item in result["raw_financial_data"]:

    print(
        item["fiscal_year"],
        "营业收入：",
        item["operating_income"],
        "净利润：",
        item["net_profit"]
    )


# ========================================================
# 3. 查看计算后的指标
# ========================================================

print("\n" + "=" * 60)
print("计算后的财务指标")
print("=" * 60)

for item in result["calculated_indicators"]:

    print(
        item["fiscal_year"],
        "营收增长：",
        item["revenue_growth"],
        "净利润增长：",
        item["net_profit_growth"],
        "毛利率：",
        item["gross_margin"],
        "净利率：",
        item["net_margin"],
        "ROA：",
        item["ROA"],
        "ROE：",
        item["ROE"]
    )

