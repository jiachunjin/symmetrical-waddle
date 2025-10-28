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

    print("load done!")

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

    with open(f"data/rewritten_wise/after_sft/g{local_rank}.jsonl", "r") as f:
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

    # for pid, prompt in local_data:
    while True:

        response = "The original prompt is a good starting point, but it could be more descriptive to guide an image generation model effectively. Here's my reasoning for optimization:  *   **\"Unfinished basilica\"**: This is a bit vague. How unfinished? What does that look like visually? Is it just a shell, or are there details present? *   **\"Masterpiece of Catalan Modernism\"**: While this sets a style, it doesn't describe *what* that looks like. We need to break down the visual characteristics of Catalan Modernism. *   **\"Intricate facades\"**: Again, \"intricate\" is good, but we can be more specific about *how* they are intricate.  My goal is to add visual detail and clarify the state of the basilica to make it easier for the AI to render a compelling image.  Revised Prompt: {A grand, partially completed basilica, showcasing the distinctive architectural style of Catalan Modernism. Its exterior facades are adorned with elaborate, organic, and flowing patterns, featuring a rich palette of warm colors like terracotta, ochre, and burnt orange. The stone work is detailed with sculpted figures and geometric shapes, creating a sense of dynamic movement across the surfaces. Some areas of the structure are still bare or under construction, revealing rough-hewn stone and incomplete arches, while others display fully realized, ornate decorations. The overall impression is one of a magnificent, unfinished work of art, bathed in natural light.}"

        prompt = response.split("{")[1].split("}")[0]
        print(prompt)

        prompt_neg = [" "]
        # print(f"GPU {local_rank}: {pid} - {prompt}")
        print(local_rank, prompt)

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

        save_name = f"/data/phd/jinjiachun/codebase/{local_rank}.png"

        image.save(save_name)
        exit(0)


if __name__ == "__main__":
    generate()