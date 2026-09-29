from tools.financial_tool import get_financial_statements


thscode = "300308.SZ"

result = get_financial_statements(thscode)


print("=" * 60)
print("资产负债表")
print("=" * 60)

print("记录数量：", len(result["balance_sheet"]))

for item in result["balance_sheet"]:
    print(
        item["fiscal_year"],
        item["assets_total"]
    )


print("\n" + "=" * 60)
print("利润表")
print("=" * 60)

print("记录数量：", len(result["income_statement"]))

for item in result["income_statement"]:
    print(
        item["fiscal_year"],
        item["operating_income"],
        item["net_profit"]
    )


print("\n" + "=" * 60)
print("现金流量表")
print("=" * 60)

print("记录数量：", len(result["cash_flow"]))

for item in result["cash_flow"]:
    print(
        item["fiscal_year"],
        item["act_cash_flow_net"]
    )