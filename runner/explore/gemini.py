import os
import json
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


try:
    os.environ["GOOGLE_CLOUD_PROJECT"] = "adengine-457110"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "runner/explore/adengine-457110-25b7eff0b7f3.json"
    client = genai.Client()
except Exception as e:
    print(f"Error configuring Google GenAI: {e}")
    print("Please ensure your GOOGLE_APPLICATION_CREDENTIALS path is correct and you have authenticated.")
    exit()

def generate_with_gemini(prompt) -> str:
    model_contents=[prompt]
    generation_config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
                thinking_budget=0           # 设为 0 可关闭内部“思考”阶段
            )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=model_contents,
        config=generation_config
    )

    return response.text

PROMPT = """You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion). Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.
When rewriting, prioritize the following principles:
1. Focus on describing the final visual appearance of the scene. Clarify elements like the main subject’s shape, color, and state.
2. Emphasize descriptions of on-screen phenomena. Use concrete, sensory language to paint a vivid picture of what the viewer will see.
3. Minimize the use of professional terms. If technical concepts are necessary, translate them into intuitive visual descriptions.
After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. Then, output the final revised prompt in the fixed format of "Revised Prompt: {}", where the specific revised content is filled in the "{}".

Prompt: 
"""

if __name__ == "__main__":
    output_file = "data/rewritten_wise/gemini.jsonl"
    json_path = "data/wise"
    json_file_names = ["cultural_common_sense.json", "natural_science.json", "spatio-temporal_reasoning.json"]
    with open(output_file, "a") as outfile:
        for json_file_name in json_file_names:
            with open(os.path.join(json_path, json_file_name), "r") as f:
                data = json.load(f)
                for item in data:
                    prompt = item["Prompt"]
                    pid = item["prompt_id"]
                    message = PROMPT + prompt
                    print(pid, prompt)
                    response = generate_with_gemini(prompt=message)
                    print(response)
                    outfile.write(json.dumps({"prompt_id": pid, "response": response}) + "\n")
                    print("-"*80)

                    # print(generate_with_gemini(prompt=prompt))
        # print(generate_with_gemini(prompt="how are you?"))