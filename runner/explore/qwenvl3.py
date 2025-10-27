import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from accelerate import Accelerator
# qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen3-VL-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen3-VL-8B-Instruct")
qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
# tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
accelerator = Accelerator()

device = accelerator.device
qwenvl = qwenvl.to(device)
qwenvl.eval()

# magic_prompt = "Ultra HD, 4K, cinematic composition"


SYSTEM_PROMPT = '''
You are a professional Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion), with deep expertise in the visualization logic of such models. Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions that align with the model’s understanding habits.
If the prompt describes the result of a process (e.g., "a cup after being poured with hot water"), use your background knowledge (e.g. scientific facts, cultural common sense, and logical reasoning) to reasonably infer this result. The inference must be based on objective logic and avoid subjective imagination beyond common cognition.
Focus strictly on describing the final visual appearance of the scene. Clarify key elements of the main subject, including but not limited to its shape, color, state, texture, proportion, and interaction with the surrounding environment.
After receiving the user’s prompt to be rewritten, first explain your optimization reasoning. This reasoning should include two parts: 1) the key issues of the original prompt (e.g., vague state description, missing color information); 2) the purpose of each improvement (e.g., adding texture details to help the model render realism). Then, output the final revised prompt in the fixed format of "Revised Prompt: ", where the specific revised content is filled in the "{}".

Example:
User input: Generate the result caused by a helium-filled balloon touching a sharp cactus spine.
Output: Key Issues of the Original Prompt: The phrase "the result caused by a helium-filled balloon touching a sharp cactus spine" is process-oriented rather than visual. Image generation models cannot render abstract causality—they require a concrete, static visual description. Missing critical visual details: the type/color/size of the balloon, the species or appearance of the cactus, the exact state of the balloon post-puncture (e.g., deflated, mid-burst, shredded), and environmental context (e.g., indoor/outdoor, lighting). Ambiguity in timing: is the image showing the instant of puncture, or the aftermath? Models need a clear temporal snapshot.
Purpose of Each Improvement: Infer the most plausible physical outcome based on real-world physics: a latex helium balloon punctured by a sharp cactus spine will rapidly deflate and collapse, likely with a small tear near the contact point. It will no longer be taut or floating. Specify balloon material (standard latex), color (choose a common vivid hue like red for visual clarity), and state (partially deflated, crumpled, lying against the cactus).
Describe the cactus realistically (e.g., barrel or prickly pear type with visible sharp spines) to anchor the interaction. Place the scene in a neutral, well-lit outdoor desert setting to provide context without distraction, aiding model coherence. Emphasize textures (glossy deflated latex, dry spiny cactus) and spatial relationship (balloon sagging against spine) to guide accurate rendering.
Revised Prompt: {A bright red latex helium balloon, now partially deflated and crumpled with a small tear near its surface, lies sagging against the sharp spine of a green barrel cactus in a sunlit desert; the balloon’s glossy surface shows wrinkles and loss of tautness, contrasting with the cactus’s dry, spiny texture under clear daylight.}
'''

json_path = "/data/phd/jinjiachun/codebase/WISE/data"
json_file_names = ["cultural_common_sense.json", "natural_science.json", "spatio-temporal_reasoning.json"]

# 首先收集所有数据
all_data = []
for json_file_name in json_file_names:
    with open(os.path.join(json_path, json_file_name), "r") as f:
        data = json.load(f)
        all_data.extend(data)

# 对全部数据进行分片
local_rank = accelerator.local_process_index
num_processes = accelerator.num_processes
total_data = len(all_data)

output_path = os.path.join(json_path, f"qwenvl_new_system_prompt_{local_rank}.jsonl")

# 计算每个进程应该处理的数据范围
# 使用更均匀的分配方式，确保所有数据都被处理
chunk_size = total_data // num_processes
remainder = total_data % num_processes

# 前面的进程处理 chunk_size + 1 个数据，后面的进程处理 chunk_size 个数据
if local_rank < remainder:
    start_idx = local_rank * (chunk_size + 1)
    end_idx = start_idx + chunk_size + 1
else:
    start_idx = local_rank * chunk_size + remainder
    end_idx = start_idx + chunk_size

# 获取当前进程要处理的数据
data = all_data[start_idx:end_idx]

print(f"进程 {local_rank}/{num_processes}: 处理数据 {start_idx}-{end_idx-1} (共 {len(data)} 条)")

# 打开输出文件进行写入
with open(output_path, "w", encoding="utf-8") as output_f:
    for item in data:
        prompt = item["Prompt"]
        prompt_id = item["prompt_id"]
        # print(prompt_id, prompt)

        original_prompt = prompt

        original_prompt = original_prompt.strip()
        prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\n Revised Prompt:"
        prompt = [prompt]
        template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        txt = [template.format(e) for e in prompt]

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
        
        # 创建要写入的数据
        output_data = {
            "prompt_id": prompt_id,
            "output_text": output_text
        }
        
        # 写入JSONL文件
        output_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        print(f"{prompt_id}: {output_text}")

accelerator.wait_for_everyone()

print(f"所有数据已保存到: {output_path}")