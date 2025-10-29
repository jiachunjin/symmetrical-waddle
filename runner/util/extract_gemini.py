import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json

def foo():
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

def bar():
    from data.system_prompts.prompt import prompt_dict
    PROMPT = prompt_dict["1029_jjc_revised"]

    gemini_result = "/Users/orres/Playground/qimage/data/sft_data/1029_revised_sft.jsonl"
    output_file = "/Users/orres/Playground/qimage/data/sft_data/sft_gemini_flash_2553.jsonl"

    user_intructions = [
        "/Users/orres/Playground/qimage/data/r2i/user_r2i_s600.jsonl",
        "/Users/orres/Playground/qimage/data/selfgen/culture_gen.jsonl",
    ]
    all_items = {}
    for wise_file in user_intructions:
        with open(wise_file, "r") as f:
            for line in f:
                data = json.loads(line)
                all_items[data["id"]] = data["prompt"]

    with open(output_file, "a") as outfile:
        with open(gemini_result, "r") as f:
            for line in f:
                data = json.loads(line)
                pid = data["prompt_id"]
                response = data["response"]
                user_instruction = all_items[pid]

                input_prompt = PROMPT + user_instruction
                output = response

                outfile.write(json.dumps({"id": pid, "input": input_prompt, "output": output}) + "\n")
    # ...
if __name__ == "__main__":
    bar()