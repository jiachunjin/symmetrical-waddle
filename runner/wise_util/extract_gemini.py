import json

gemini_result = "data/rewritten_wise/gemini.jsonl"
output_file = "data/rewritten_wise/gemini_clean.jsonl"

with open(output_file, "a") as outfile:
    with open(gemini_result, "r") as f:
        for line in f:
            data = json.loads(line)
            pid = data["prompt_id"]
            response = data["response"]
            response = response.split("Revised Prompt: ")[1][1:]
            response = response.split("}")[0][:-1]
            print(pid, response)
            outfile.write(json.dumps({"prompt_id": pid, "response": response}) + "\n")