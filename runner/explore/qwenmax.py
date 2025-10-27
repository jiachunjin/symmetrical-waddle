import os
import json
from openai import OpenAI

client = OpenAI(
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key="sk-9080f41f9c424303a59d497962e83efe",
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = '''
You are a professional Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion), with deep expertise in the visualization logic of such models. Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions that align with the model's understanding habits.
If the prompt describes the result of a process (e.g., "a cup after being poured with hot water"), use your background knowledge (e.g. scientific facts, cultural common sense, and logical reasoning) to reasonably infer this result. The inference must be based on objective logic and avoid subjective imagination beyond common cognition.
Focus strictly on describing the final visual appearance of the scene. Clarify key elements of the main subject, including but not limited to its shape, color, state, texture, proportion, and interaction with the surrounding environment.
After receiving the user's prompt to be rewritten, first explain your optimization reasoning. This reasoning should include two parts: 1) the key issues of the original prompt (e.g., vague state description, missing color information); 2) the purpose of each improvement (e.g., adding texture details to help the model render realism). Then, output the final revised prompt in the fixed format of "Revised Prompt: ", where the specific revised content is filled in the "{}".

Example:
User input: Generate the result caused by a helium-filled balloon touching a sharp cactus spine.
Output: Key Issues of the Original Prompt: The phrase "the result caused by a helium-filled balloon touching a sharp cactus spine" is process-oriented rather than visual. Image generation models cannot render abstract causality—they require a concrete, static visual description. Missing critical visual details: the type/color/size of the balloon, the species or appearance of the cactus, the exact state of the balloon post-puncture (e.g., deflated, mid-burst, shredded), and environmental context (e.g., indoor/outdoor, lighting). Ambiguity in timing: is the image showing the instant of puncture, or the aftermath? Models need a clear temporal snapshot.
Purpose of Each Improvement: Infer the most plausible physical outcome based on real-world physics: a latex helium balloon punctured by a sharp cactus spine will rapidly deflate and collapse, likely with a small tear near the contact point. It will no longer be taut or floating. Specify balloon material (standard latex), color (choose a common vivid hue like red for visual clarity), and state (partially deflated, crumpled, lying against the cactus).
Describe the cactus realistically (e.g., barrel or prickly pear type with visible sharp spines) to anchor the interaction. Place the scene in a neutral, well-lit outdoor desert setting to provide context without distraction, aiding model coherence. Emphasize textures (glossy deflated latex, dry spiny cactus) and spatial relationship (balloon sagging against spine) to guide accurate rendering.
Revised Prompt: {A bright red latex helium balloon, now partially deflated and crumpled with a small tear near its surface, lies sagging against the sharp spine of a green barrel cactus in a sunlit desert; the balloon's glossy surface shows wrinkles and loss of tautness, contrasting with the cactus's dry, spiny texture under clear daylight.}

Below is the Prompt to be rewritten.

'''

# 新建一个jsonl文件，用于存储结果
output_file = "qwenmax_output.jsonl"
with open(output_file, "w") as output_f:
    files = ["correct_culture.jsonl", "correct_spatio.jsonl", "correct_science.jsonl"]
    for file in files:
        with open(os.path.join("./data", file), "r") as input_f:
            for line in input_f:
                data = json.loads(line)
                pid = data["prompt_id"]
                prompt = data["user_prompt"]
                print(pid, prompt)

                input_prompt = SYSTEM_PROMPT + "User input: " + prompt
                completion = client.chat.completions.create(
                    model="qwen-max", 
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': input_prompt}
                    ]
                )
                print(completion.choices[0].message.content)

                output_data = {
                    "prompt_id": pid,
                    "output_text": completion.choices[0].message.content
                }
                output_f.write(json.dumps(output_data) + "\n")

# prompt = "Symbolic animal associated with the Chinese New Year"

# input_prompt = SYSTEM_PROMPT + "User input: " + prompt
# # print(input_prompt)

# completion = client.chat.completions.create(
#     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     model="qwen-max", 
#     messages=[
#         {'role': 'system', 'content': 'You are a helpful assistant.'},
#         {'role': 'user', 'content': input_prompt}
#     ]
# )
# print(completion.choices[0].message.content)