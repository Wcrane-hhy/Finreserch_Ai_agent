from tools.search_tool import search_web


query = input("请输入搜索关键词：")

results = search_web(query)

print("\n搜索结果：")

for item in results:
    print(
        item["title"],
        "\n   ",
        item["url"],
        "\n   ",
        item["snippet"]
    )
