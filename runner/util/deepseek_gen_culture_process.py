import json

def to_jsonl():
    pid = 2048
    with open("data/selfgen/culture_gen.jsonl", "w") as outfile:
        with open("data/selfgen/raw_culture_1.json", "r") as f:
            data = json.load(f)
            for item in data:
                prompt = item["Prompt"]
                explanation = item["Explanation"]
                outfile.write(json.dumps({
                    "id": pid,
                    "category": "SelfGenerated_Culture",
                    "prompt": prompt,
                    "explanation": explanation,
                }) + "\n")
                pid += 1
        with open("data/selfgen/raw_culture_2.json", "r") as f:
            data = json.load(f)
            for item in data:
                prompt = item["Prompt"]
                explanation = item["Explanation"]
                outfile.write(json.dumps({
                    "id": pid,
                    "category": "SelfGenerated_Culture",
                    "prompt": prompt,
                    "explanation": explanation,
                }) + "\n")
                pid += 1


if __name__ == "__main__":
    to_jsonl()