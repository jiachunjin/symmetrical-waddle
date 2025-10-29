import json

result = "data/rewritten_wise/4o.jsonl"
output_file = "data/rewritten_wise/4o_clean.jsonl"

with open(output_file, "a") as outfile:
    with open(result, "r") as f:
        for line in f:
            data = json.loads(line)
            pid = data["prompt_id"]
            response = data["response"]
            
            # 留下response中"Revised Prompt:"之后所有的内容
            if "Revised Prompt:" in response:
                # 找到 "Revised Prompt:" 的位置
                start_idx = response.find("Revised Prompt:")
                # 提取冒号之后的所有内容
                extracted = response[start_idx + len("Revised Prompt:"):].strip()
                
                # 清理内容：去掉开头的 ** 和换行
                extracted = extracted.lstrip("*").strip()
                
                # 如果内容是包含在大括号中的，则提取大括号内的内容
                if extracted.startswith("{") and extracted.endswith("}"):
                    extracted = extracted[1:-1]  # 去掉大括号
                elif extracted.startswith("{"):
                    # 如果只有开始大括号，提取到大括号结束之前的内容
                    extracted = extracted[1:].split("}")[0] if "}" in extracted else extracted[1:]
                
                # 清理多余的空格和换行
                extracted = extracted.strip()
                
                # 如果提取的内容仍然包含 "Revised Prompt"，再次处理
                if "Revised Prompt:" in extracted:
                    start_idx = extracted.find("Revised Prompt:")
                    extracted = extracted[start_idx + len("Revised Prompt:"):].strip()
                    # 再次处理大括号
                    if extracted.startswith("{") and extracted.endswith("}"):
                        extracted = extracted[1:-1]
                    elif extracted.startswith("{"):
                        extracted = extracted[1:].split("}")[0] if "}" in extracted else extracted[1:]
                    extracted = extracted.strip()
            else:
                extracted = response
            print(pid, extracted)
            # outfile.write(json.dumps({"prompt_id": pid, "response": extracted}) + "\n")
