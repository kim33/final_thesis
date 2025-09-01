from transformers import pipeline
import json
import ast
import os

# Model name
model_name = "deepseek-ai/DeepSeek-V2-Chat-0628"

# Create pipeline (loads model + tokenizer automatically)
pipe = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0)

# Utility functions
def preprocess_json_line(line: str) -> str:
    text = line.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text

def parse_response(resp: str) -> dict:
    cleaned = preprocess_json_line(resp)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            pyobj = ast.literal_eval(cleaned)
        except Exception as e:
            raise ValueError(f"Cannot parse response as JSON or Python literal: {e!r}")
        return json.loads(json.dumps(pyobj))

def build_chat_prompt(text: str) -> str:
    # Manually apply the chat prompt formatting
    return f"<|user|>\nYou are an AI that detoxifies hate speech in the given text while preserving meaning. \nYou must return only a valid JSON object in the following format:\n{{\n  \"original\": \"...\",\n  \"detoxified\": \"...\"\n}}\n\nHere are some examples:\n\nText: \"All women are too emotional to lead.\"\n{{\n  \"original\": \"All women are too emotional to lead.\",\n  \"detoxified\": \"I believe emotional decision-making may be a challenge in leadership roles.\"\n}}\n\nText: \"Muslims are terrorists.\"\n{{\n  \"original\": \"Muslims are terrorists.\",\n  \"detoxified\": \"Some individuals who commit acts of terrorism identify as Muslim, but it's wrong to associate an entire religion with those actions.\"\n}}\n\nNow analyze this:\n\nText: \"{text}\"\n<|assistant|>\n"

def detect_and_detox(text: str):
    prompt = build_chat_prompt(text)
    outputs = pipe(prompt, max_new_tokens=40, do_sample=False)
    generated_text = outputs[0]["generated_text"][len(prompt):].strip()
    try:
        return parse_response(generated_text)
    except Exception as e:
        print("Failed to parse response:", e)
        print("Raw output:", generated_text)
        return None

# Run script
if __name__ == "__main__":
    input_path = "dataset/hatexplain_extract.json"
    output_path = "detoxified/deepseek_v3_hatexplain.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detoxified = []

    for i, item in enumerate(data):
        text = item["post_text"]
        targets = item.get("targets", None)

        result = detect_and_detox(text)
        if result:
            if targets:
                result["targets"] = targets
            detoxified.append(result)
            print(f"Processed {i+1}/{len(data)}: {result}")

            with open(output_path, "w", encoding="utf-8") as fout:
                json.dump(detoxified, fout, indent=4, ensure_ascii=False)
