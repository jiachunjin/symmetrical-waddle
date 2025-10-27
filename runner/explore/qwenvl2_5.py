import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from accelerate import Accelerator

qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/kousiqi/yugang/LLaMA-Factory/saves/qwen2_5vl-7b/full/sft_0.5_raw")
tokenizer = AutoTokenizer.from_pretrained("/data/phd/kousiqi/yugang/LLaMA-Factory/saves/qwen2_5vl-7b/full/sft_0.5_raw")



device = torch.device("cuda:0")
dtype = torch.bfloat16

qwenvl = qwenvl.to(device)
qwenvl.eval()

json_path = "/data/phd/jinjiachun/codebase/WISE/data"
json_file_names = ["cultural_common_sense.json"]

# 首先收集所有数据
all_data = []
for json_file_name in json_file_names:
    with open(os.path.join(json_path, json_file_name), "r") as f:
        data = json.load(f)
        all_data.extend(data)


for item in data:
    prompt = item["Prompt"]
    prompt_id = item["prompt_id"]
    # print(prompt_id, prompt)

    original_prompt = prompt


    original_prompt = original_prompt.strip()

    txt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou are a Prompt Optimizer specialized in image generation models(e.g., MidJourney, Stable Diffusion). Your core task is to rewrite the user-provided prompt into a highly clear version—this helps image generation models accurately comprehend the user’s intent.\nSince image generation models have limited ability to understand vague descriptions, you must first fully grasp the user's core intent, then replace ambiguous content in the original prompt with explicit details. While optimizing, you may use background knowledge like scientific facts, cultural common sense, and logical reasoning. You should also supplement key details when it is necessary for image generation model to understand the user's in\nIf the user provides only text, the task is text-to-image generation; if the user provides both text and an image, the task is image editing (which means you must first fully understand the raw image, then rewrite the prompt for image editing task). Below you will be given the user's input. \nAfter receiving the user’s prompt that needs rewriting, output the final revised prompt in this fixed format: Revised Prompt: {{}}, where the specific revised content is filled in the {{}}.\n{original_prompt}<|im_end|>\n<|im_start|>assistant\n"

    print(txt)

    txt_tokens = tokenizer(
        txt, max_length=10240, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    generation_output = qwenvl.generate(
        **txt_tokens, 
        max_new_tokens=512,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_ids = generation_output.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(txt_tokens.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    output_text = output_text.replace("\n", " ")
    output_text = output_text

    print(output_text)
    print("="*20)