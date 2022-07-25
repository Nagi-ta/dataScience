import json
import codecs

with open("newsgroups.json") as fd:
    data = json.load(fd)

target = data["target"]
content = data["content"]
target_names = data["target_names"]
target_value_list = list(target.values())  # メッセージのカテゴリ ID
content_value_list = list(content.values())  # メッセージテキスト本体
target_namevalue_list = list(target_names.values())

print(*content_value_list, sep="\n", file=codecs.open("content.txt", "w", "utf-8"))
