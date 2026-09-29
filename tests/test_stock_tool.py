from tools.stock_tool import search_stock


query = input("请输入股票名称或代码：")

result = search_stock(query)

print("\n搜索结果：")

for item in result:
    print(
        item["name"],
        item["thscode"]
    )