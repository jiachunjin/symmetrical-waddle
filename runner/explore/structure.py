import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def text_encode(text_encoder=None):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # ---------- load text encoder ----------
    if text_encoder is None:
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    text_encoder = text_encoder.to(device, dtype).eval()

    # ---------- encode prompt ----------
    prompt = ["生成一张祝小明生日快乐的贺卡", " "]
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = [template.format(e) for e in prompt]
    print(formatted_prompt)
    drop_idx = 34

    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # print(txt_tokens.input_ids[0].shape, tokenizer.decode(txt_tokens.input_ids[0]))
    print(txt_tokens.attention_mask)

    encoder_hidden_states = text_encoder(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    print(hidden_states.shape)

    chosen_hidden_states = hidden_states[:, drop_idx:]
    print(chosen_hidden_states.shape)

    chosen_mask = txt_tokens.attention_mask[:, drop_idx:]
    print(chosen_mask)

    return chosen_hidden_states, chosen_mask

def encode(prompt, qwenvl):
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    device = qwenvl.device
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34
    # template = "<|im_start|>system\nGeneration an image:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    # drop_idx = 12
    formatted_prompt = [template.format(e) for e in prompt]
    
    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # print(txt_tokens.attention_mask)
    encoder_hidden_states = qwenvl(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    chosen_hidden_states = hidden_states[:, drop_idx:]
    chosen_mask = txt_tokens.attention_mask[:, drop_idx:]

    return chosen_hidden_states, chosen_mask

def complete_pipeline():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---------- load mmdit ----------
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(device)

    # chosen_hidden_states, chosen_mask = text_encode(pipe.text_encoder)

    prompt = ["生成一张祝小明生日快乐的贺卡"]
    prompt_neg = [" "]

    prompt_embeds, prompt_embeds_mask = encode(prompt, pipe.text_encoder)

    prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
        prompt                = prompt_neg,
        device                = device,
    )

    # print(chosen_hidden_states.shape, prompt_embeds.shape)

    image = pipe(
        prompt_embeds               = prompt_embeds,
        prompt_embeds_mask          = prompt_embeds_mask,
        negative_prompt_embeds      = prompt_embeds_neg,
        negative_prompt_embeds_mask = prompt_embeds_mask_neg,
        true_cfg_scale              = 5.0,
        num_inference_steps         = 50,
        height                      = 1024,
        width                       = 1024,
    ).images[0]

    image.save("generation_structure.png")

@torch.no_grad()
def generate_wise_images():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()

    # device = torch.device("cuda:0")
    dtype = torch.bfloat16

    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    # load json file
    json_path = "/data/phd/jinjiachun/codebase/WISE/data"
    json_file_names = ["cultural_common_sense_rewrite.json", "natural_science_rewrite.json", "spatio-temporal_reasoning_rewrite.json"]
    for json_file_name in json_file_names:
        with open(os.path.join(json_path, json_file_name), "r") as f:
            data = json.load(f)
            # Split data into 8 parts, each GPU processes its own part
            local_rank = accelerator.local_process_index
            num_processes = accelerator.num_processes
            chunk_size = (len(data) + num_processes - 1) // num_processes
            start_idx = local_rank * chunk_size
            end_idx = min((local_rank + 1) * chunk_size, len(data))
            data = data[start_idx:end_idx]
            for item in data:
                prompt = item["Prompt"]
                prompt_neg = [" "]
                prompt_id = item["prompt_id"]
                print(prompt_id, prompt)

                prompt_embeds, prompt_embeds_mask = encode([prompt], pipe.text_encoder)

                prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
                    prompt                = prompt_neg,
                    device                = accelerator.device,
                )

                image = pipe(
                    prompt_embeds               = prompt_embeds,
                    prompt_embeds_mask          = prompt_embeds_mask,
                    negative_prompt_embeds      = prompt_embeds_neg,
                    negative_prompt_embeds_mask = prompt_embeds_mask_neg,
                    true_cfg_scale              = 5.0,
                    num_inference_steps         = 50,
                    height                      = 512,
                    width                       = 512,
                ).images[0]

                image.save(f"/data/phd/jinjiachun/codebase/qimagined-goggles/asset/wise_generation_rewrite/{prompt_id}.png")


@torch.no_grad()
def generate_wise_images_qwen_max_generation():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()
    dtype = torch.bfloat16

    # 加载模型到当前GPU
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    path = "/data/phd/jinjiachun/codebase/qimagined-goggles/data"
    json_file_names = ["correct_culture.jsonl", "correct_spatio.jsonl", "correct_science.jsonl"]
    
    # 收集所有数据
    all_data = []
    for json_file_name in json_file_names:
        file_path = os.path.join(path, json_file_name)
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                pid = int(data["prompt_id"])
                if json_file_name == "correct_culture.jsonl":
                    pid += 0
                elif json_file_name == "correct_spatio.jsonl":
                    pid += 400
                elif json_file_name == "correct_science.jsonl":
                    pid += 700
                
                response = data["assistant_content"]
                prompt = response.split("{")[1].split("}")[0]
                all_data.append((pid, prompt))
    
    # 多卡并行处理：每个GPU处理一部分数据
    local_rank = accelerator.local_process_index
    num_processes = accelerator.num_processes
    chunk_size = (len(all_data) + num_processes - 1) // num_processes
    start_idx = local_rank * chunk_size
    end_idx = min((local_rank + 1) * chunk_size, len(all_data))
    local_data = all_data[start_idx:end_idx]
    
    print(f"GPU {local_rank}: 处理 {len(local_data)} 个样本 (索引 {start_idx}-{end_idx-1})")
    
    for pid, prompt in local_data:
        prompt_neg = [" "]
        print(f"GPU {local_rank}: {pid} - {prompt}")

        prompt_embeds, prompt_embeds_mask = encode([prompt], pipe.text_encoder)

        prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
            prompt                = prompt_neg,
            device                = accelerator.device,
        )

        image = pipe(
            prompt_embeds               = prompt_embeds,
            prompt_embeds_mask          = prompt_embeds_mask,
            negative_prompt_embeds      = prompt_embeds_neg,
            negative_prompt_embeds_mask = prompt_embeds_mask_neg,
            true_cfg_scale              = 5.0,
            num_inference_steps         = 50,
            height                      = 512,
            width                       = 512,
        ).images[0]

        image.save(f"/data/phd/jinjiachun/codebase/qimagined-goggles/asset/wise_generation_rewrite_qwen_max/{pid}.png")


@torch.no_grad()
def generate_wise_images_qwen3vl_rewrite():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()
    dtype = torch.bfloat16

    # 加载模型到当前GPU
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    path = "/data/phd/jinjiachun/codebase/WISE/data"
    json_file_names = ["all_rewritten_prompts_original_qwenimage_text_encoder.jsonl"]
    
    # 收集所有数据
    all_data = []
    for json_file_name in json_file_names:
        file_path = os.path.join(path, json_file_name)
        with open(file_path, "r") as f:
            # data = json.load(f)  # 使用json.load()而不是json.loads()逐行解析
            # for item in data:
            #     pid = int(item["prompt_id"])
            #     response = item["Prompt"]
            #     all_data.append((pid, response))
            for line in f:
                data = json.loads(line)  # JSONL格式：每行一个JSON对象
                pid = int(data["prompt_id"])
                response = data["output_text"]
                all_data.append((pid, response))
    
    # 多卡并行处理：每个GPU处理一部分数据
    local_rank = accelerator.local_process_index
    num_processes = accelerator.num_processes
    chunk_size = (len(all_data) + num_processes - 1) // num_processes
    start_idx = local_rank * chunk_size
    end_idx = min((local_rank + 1) * chunk_size, len(all_data))
    local_data = all_data[start_idx:end_idx]
    
    print(f"GPU {local_rank}: 处理 {len(local_data)} 个样本 (索引 {start_idx}-{end_idx-1})")
    
    for pid, prompt in local_data:
        prompt_neg = [" "]
        print(f"GPU {local_rank}: {pid} - {prompt}")

        prompt_embeds, prompt_embeds_mask = encode([prompt], pipe.text_encoder)

        prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
            prompt                = prompt_neg,
            device                = accelerator.device,
        )

        image = pipe(
            prompt_embeds               = prompt_embeds,
            prompt_embeds_mask          = prompt_embeds_mask,
            negative_prompt_embeds      = prompt_embeds_neg,
            negative_prompt_embeds_mask = prompt_embeds_mask_neg,
            true_cfg_scale              = 5.0,
            num_inference_steps         = 50,
            height                      = 512,
            width                       = 512,
        ).images[0]

        image.save(f"/data/phd/jinjiachun/codebase/qimagined-goggles/asset/qwenvl2_5vl/{pid}.png")

@torch.no_grad()
def generate_qwenvl_new_system_prompt():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()
    dtype = torch.bfloat16

    # 加载模型到当前GPU
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    local_rank = accelerator.local_process_index
    path = f"/data/phd/jinjiachun/codebase/WISE/data/qwenvl_new_system_prompt_{local_rank}.jsonl"

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)  # JSONL格式：每行一个JSON对象
            pid = int(data["prompt_id"])
            save_name = f"/data/phd/jinjiachun/codebase/qimagined-goggles/asset/qwenvl2_5vl_new_system_prompt/{pid}.png"
            if os.path.exists(save_name):
                continue
            response = data["output_text"]
            prompt = response.split("{")[1].split("}")[0]

            prompt_neg = [" "]
            print(f"GPU {local_rank}: {pid} - {prompt}")

            prompt_embeds, prompt_embeds_mask = encode([prompt], pipe.text_encoder)

            prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
                prompt                = prompt_neg,
                device                = accelerator.device,
            )

            image = pipe(
                prompt_embeds               = prompt_embeds,
                prompt_embeds_mask          = prompt_embeds_mask,
                negative_prompt_embeds      = prompt_embeds_neg,
                negative_prompt_embeds_mask = prompt_embeds_mask_neg,
                true_cfg_scale              = 5.0,
                num_inference_steps         = 50,
                height                      = 512,
                width                       = 512,
            ).images[0]

            image.save(save_name)

@torch.no_grad()
def generate_qwenmax_new_system_prompt():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()
    dtype = torch.bfloat16

    # 加载模型到当前GPU
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    local_rank = accelerator.local_process_index
    path = "./data/qwenmax_new_sysp_output.jsonl"

    all_data = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            pid = int(data["prompt_id"])
            response = data["output_text"]
            prompt = response.split("{")[1].split("}")[0]
            all_data.append((pid, prompt))
    local_rank = accelerator.local_process_index
    num_processes = accelerator.num_processes
    chunk_size = (len(all_data) + num_processes - 1) // num_processes
    start_idx = local_rank * chunk_size
    end_idx = min((local_rank + 1) * chunk_size, len(all_data))
    local_data = all_data[start_idx:end_idx]
    
    print(f"GPU {local_rank}: 处理 {len(local_data)} 个样本 (索引 {start_idx}-{end_idx-1})")

    for pid, prompt in local_data:
        if pid == 500 or pid == 600:
            prompt_neg = [" "]
            print(f"GPU {local_rank}: {pid} - {prompt}")

            prompt_embeds, prompt_embeds_mask = encode([prompt], pipe.text_encoder)
            prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
                prompt                = prompt_neg,
                device                = accelerator.device,
            )

            image = pipe(
                prompt_embeds               = prompt_embeds,
                prompt_embeds_mask          = prompt_embeds_mask,
                negative_prompt_embeds      = prompt_embeds_neg,
                negative_prompt_embeds_mask = prompt_embeds_mask_neg,
                true_cfg_scale              = 5.0,
                num_inference_steps         = 50,
                height                      = 512,
                width                       = 512,
            ).images[0]

            save_name = f"/data/phd/jinjiachun/codebase/qimagined-goggles/asset/qwenmax_new_system_prompt/{pid}.png"

            image.save(save_name)

if __name__ == "__main__":
    generate_qwenmax_new_system_prompt()
    # generate_qwenvl_new_system_prompt()
    # generate_wise_images_qwen_max_generation()
    # generate_wise_images()
    # complete_pipeline()