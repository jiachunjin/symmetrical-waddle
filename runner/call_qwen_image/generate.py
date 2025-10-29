import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

def encode(prompt, qwenvl):
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    device = qwenvl.device
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34

    formatted_prompt = [template.format(e) for e in prompt]
    
    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    encoder_hidden_states = qwenvl(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    chosen_hidden_states = hidden_states[:, drop_idx:]
    chosen_mask = txt_tokens.attention_mask[:, drop_idx:]

    return chosen_hidden_states, chosen_mask

@torch.no_grad()
def generate():
    import os
    import json
    from diffusers import QwenImagePipeline
    from accelerate import Accelerator

    accelerator = Accelerator()
    dtype = torch.bfloat16

    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(accelerator.device, dtype)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("/data/phd/jinjiachun/experiment/sft_qwenvl/gemini_flash_2047_full")
    text_encoder = text_encoder.to(accelerator.device, dtype).eval()

    local_rank = accelerator.local_process_index
    # path = "data/rewritten_wise/gemini_clean.jsonl"

    # all_data = []
    # with open(path, "r") as f:
    #     for line in f:
    #         data = json.loads(line)
    #         pid = int(data["prompt_id"])
    #         response = data["response"]
    #         prompt = response
    #         all_data.append((pid, prompt))

    num_processes = accelerator.num_processes

    local_data = []

    with open(f"/data/phd/jinjiachun/codebase/symmetrical-waddle/data/re_wise_final/g{local_rank}.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            pid = int(data["prompt_id"])
            response = data["response"]
            # 把被大括号包起来的内容作为prompt
            prompt = response.split("{")[1].split("}")[0]
            print(pid, prompt)

            local_data.append((pid, prompt))

    # chunk_size = (len(all_data) + num_processes - 1) // num_processes
    # start_idx = local_rank * chunk_size
    # end_idx = min((local_rank + 1) * chunk_size, len(all_data))
    # local_data = all_data[start_idx:end_idx]

    # print(f"GPU {local_rank}: 处理 {len(local_data)} 个样本 (索引 {start_idx}-{end_idx-1})")

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
        save_name = f"/data/phd/jinjiachun/codebase/symmetrical-waddle/asset/gemini_flash_2047_two_encoder/{pid}.png"

        image.save(save_name)


if __name__ == "__main__":
    generate()