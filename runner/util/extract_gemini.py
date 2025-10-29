import json

gemini_result = "/Users/orres/Playground/qimage/data/rewritten_wise/genimi_1029.jsonl"
output_file = "/Users/orres/Playground/qimage/data/rewritten_wise/gemini_1029_clean.jsonl"

with open(output_file, "a") as outfile:
    with open(gemini_result, "r") as f:
        for line in f:
            data = json.loads(line)
            pid = data["prompt_id"]
            response = data["response"]
            response = response.split("Revised Prompt:")[1].strip()
            print(pid, response)
            outfile.write(json.dumps({"prompt_id": pid, "response": response}) + "\n")