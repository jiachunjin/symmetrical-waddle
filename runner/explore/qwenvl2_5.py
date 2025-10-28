import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from accelerate import Accelerator

qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/jinjiachun/experiment/sft_qwenvl/gemini_flash_2047_full")
tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/experiment/sft_qwenvl/gemini_flash_2047_full")


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

PROMPT = """You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion). Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.
When rewriting, prioritize the following principles:
1. Focus on describing the final visual appearance of the scene. Clarify elements like the main subject’s shape, color, and state.
2. Emphasize descriptions of on-screen phenomena. Use concrete, sensory language to paint a vivid picture of what the viewer will see.
3. Minimize the use of professional terms. If technical concepts are necessary, translate them into intuitive visual descriptions.
After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. Then, output the final revised prompt in the fixed format of "Revised Prompt: {}", where the specific revised content is filled in the "{}".

Prompt: 
"""

PROMPT = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + PROMPT

for item in data:
    prompt = item["Prompt"]
    prompt_id = item["prompt_id"]
    # print(prompt_id, prompt)

    original_prompt = prompt


    original_prompt = original_prompt.strip()
    prompt = PROMPT + original_prompt + "\n<|im_start|>assistant\n"

    print(prompt)

    txt_tokens = tokenizer(
        prompt, max_length=10240, padding=True, truncation=True, return_tensors="pt"
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
    print("="*80)
    exit(0)