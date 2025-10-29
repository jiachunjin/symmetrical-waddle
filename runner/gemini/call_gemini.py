import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import multiprocessing as mp
from multiprocessing import Pool, Lock
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

try:
    os.environ["GOOGLE_CLOUD_PROJECT"] = "adengine-457110"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/orres/Playground/R2I-Bench/adengine-457110-25b7eff0b7f3.json"
    client = genai.Client()
except Exception as e:
    print(f"Error configuring Google GenAI: {e}")
    print("Please ensure your GOOGLE_APPLICATION_CREDENTIALS path is correct and you have authenticated.")
    exit()

def generate_with_gemini(prompt) -> str:
    model_contents    = [prompt]
    generation_config = GenerateContentConfig(
        temperature     = 0.0,                                 # 设置为0.7增加采样随机性，产生更多样化的输出
        top_p           = 1.,                                  # 设置为0.9考虑概率累积前90%的候选词
        top_k           = 1,                                   # 设置为40考虑概率最高的40个词
        thinking_config = ThinkingConfig(thinking_budget = 0)  # 设为 0 可关闭内部"思考"阶段
    )

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = model_contents,
        config   = generation_config
    )

    return response.text

def construct_sft_data():
    from data.system_prompts.prompt import prompt_dict
    PROMPT = prompt_dict["1029_jjc"]

    print(f"The PROMPT is: {PROMPT}")

    with open("data/r2i/user_r2i_s600.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            pid = data["id"]
            prompt = data["prompt"]
            print(pid, prompt)
            # response = generate_with_gemini(PROMPT + prompt)
            # print(response)
            # id += 1
    
    with open("data/selfgen/culture_gen.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            pid = data["id"]
            prompt = data["prompt"]
            print(pid, prompt)

def process_item(args):
    """处理单个数据项的函数"""
    item, prompt_template, output_file = args
    pid = item["prompt_id"]
    prompt = item["Prompt"]
    
    print(f"Processing prompt_id: {pid}")
    response = generate_with_gemini(prompt_template + prompt)
    print(f"Response for {pid}: {response}")
    
    # 使用文件锁确保写入安全
    with open(output_file, "a") as outfile:
        outfile.write(json.dumps({"prompt_id": pid, "response": response}) + "\n")
    
    print("-"*80)
    return pid

def rewrite_wise():
    from data.system_prompts.prompt import prompt_dict
    PROMPT = prompt_dict["1029_jjc_revised"]

    print(f"The PROMPT is: {PROMPT}")

    output_file = "/Users/orres/Playground/qimage/data/rewritten_wise/genimi_1029.jsonl"
    wise_files = ["data/wise/cultural_common_sense.json", "data/wise/natural_science.json", "data/wise/spatio-temporal_reasoning.json"]
    
    # 收集所有需要处理的数据项
    all_items = []
    for wise_file in wise_files:
        with open(wise_file, "r") as f:
            data = json.load(f)
            all_items.extend(data)
    
    # 准备参数列表
    args_list = [(item, PROMPT, output_file) for item in all_items]
    
    # 使用多进程处理
    num_processes = min(mp.cpu_count(), 32)  # 限制最大进程数为4，避免API限制
    print(f"Using {num_processes} processes to process {len(all_items)} items")
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_item, args_list)

if __name__ == "__main__":
    rewrite_wise()