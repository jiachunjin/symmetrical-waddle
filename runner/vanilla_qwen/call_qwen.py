import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import torch
from accelerate import Accelerator
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

def rewrite_wise():
    from data.system_prompts.prompt import prompt_dict

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16

    PROMPT = prompt_dict["1029_jjc_revised"]
    PROMPT = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + PROMPT

    print(f"The PROMPT is: {PROMPT}")


    num_processes = accelerator.num_processes
    local_rank = accelerator.local_process_index

    output_file = "data/rewritten_wise/qwen_sft_2553.jsonl"
    wise_files = ["data/wise/cultural_common_sense.json", "data/wise/natural_science.json", "data/wise/spatio-temporal_reasoning.json"]
    
    # 收集所有需要处理的数据项
    all_items = []
    for wise_file in wise_files:
        with open(wise_file, "r") as f:
            data = json.load(f)
            all_items.extend(data)

    chunk_size = (len(all_items) + num_processes - 1) // num_processes
    start_idx = local_rank * chunk_size
    end_idx = min((local_rank + 1) * chunk_size, len(all_items))
    local_items = all_items[start_idx:end_idx]
    print(f"GPU {local_rank}: 处理 {len(local_items)} 个样本 (索引 {start_idx}-{end_idx-1})")

    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("/data/phd/jinjiachun/experiment/sft_qwenvl/gemini_flash_2553")
    qwen = qwen.to(device, dtype)
    qwen.eval()

    output_file = f"data/rewritten_wise/qwen_1029_{local_rank}.jsonl"
    with open(output_file, "w") as f:
        for item in local_items:
            prompt = item["Prompt"]
            prompt_id = item["prompt_id"]
            prompt = PROMPT + prompt + "\n<|im_start|>assistant\n"

            txt_tokens = tokenizer(
                prompt, max_length=10240, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            generation_output = qwen.generate(
                **txt_tokens, 
                max_new_tokens          = 512,
                output_hidden_states    = True,
                return_dict_in_generate = True,
                output_scores           = True
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
            f.write(json.dumps({"prompt_id": prompt_id, "response": output_text}) + "\n")

if __name__ == "__main__":
    rewrite_wise()
