# load "/Users/orres/Playground/qimage/data/culture.jsonl"
import os
import json

path = "/Users/orres/Playground/qimage/data"
json_file_names = ["correct_culture.jsonl", "correct_spatio.jsonl", "correct_science.jsonl"]

for json_file_name in json_file_names:
    file_path = os.path.join(path, json_file_name)
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            pid = int(data["prompt_id"])
            if json_file_name == "correct_culture.jsonl":
                pid += 0
            elif json_file_name == "correct_spatio.jsonl":
                pid += 400
            elif json_file_name == "correct_science.jsonl":
                pid += 700
            response = data["assistant_content"]
            response = response.split("{")[1].split("}")[0]
            print(pid, response)
            # print(, data["assistant_content"])
            # response = data["response"]["body"]["choices"][0]["message"]["content"]
            # response = response.split("{")[1].split("}")[0]
            # print(idx, response)
            # idx += 1


# 我要给correct_spatio.jsonl中的数据的每个prompt_id加400