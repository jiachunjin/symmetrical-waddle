import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

qwen_results = [
    "qwen_1029_0.jsonl",
    "qwen_1029_1.jsonl",
    "qwen_1029_2.jsonl",
    "qwen_1029_3.jsonl",
    "qwen_1029_4.jsonl",
    "qwen_1029_5.jsonl",
    "qwen_1029_6.jsonl",
    "qwen_1029_7.jsonl",
]
output_file = "/Users/orres/Playground/qimage/data/rewritten_wise/qwen_1029_clean.jsonl"

with open(output_file, "a") as outfile:
    for file in qwen_results:
        with open(os.path.join("data/rewritten_wise", file), "r") as f:
            for line in f:
                data = json.loads(line)
                pid = data["prompt_id"]
                response = data["response"]
                # 定位到response中最后一个"Revised Prompt"的位置，
                last_index = response.rfind("Revised Prompt")
                response = response[last_index+len("Revised Prompt: "):]
                print(pid, response)
                # 留下response中"最后一个Revised Prompt:"之后所有的内容
                # response = response.split("Revised Prompt:")[-1]
                # print(pid, response)
                outfile.write(json.dumps({"prompt_id": pid, "response": response}) + "\n")