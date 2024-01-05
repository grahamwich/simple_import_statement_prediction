import json
import pandas as pd

with open('problems.jsonl', 'r') as json_file:
    json_list = list(json_file)

# # print the stuff
# for json_str in json_list:
#     result = json.loads(json_str)
#     print(f"result: {result}")
#     print(isinstance(result, dict))

text_list = []
import_list = []

# check each item for the word "import"
# if it has "import", strip the text into a text list
for json_str in json_list:
    result = json.loads(json_str)
    if "import" in result['code']:
        text_list.append(result['text'])
        import_list.append(result['code'].split("import")[1])

print(text_list)
print(import_list)

# split each item to only have the problem: 'text' portion
# for json_str in json_list:
#     result = json.loads(json_str)
#     print(f"result: {result['text']}")
#     text_list.append(result['text'])
#     # print(isinstance(result, dict))

# remove unnecessary words and punctuation
# "write" "a" "function" "to" "find" "the"

# read code sections to help with library choosing?


