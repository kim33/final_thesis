from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import ast
import os
import re
import unicodedata

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",device_map="auto")


def clean_post_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Replace smart quotes and apostrophes with straight versions
    replacements = {
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '´': "'",
        '‐': '-',  # dash
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize Unicode (e.g., decompose accented letters)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except newlines
    text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Load dataset
with open("dataset/davidson_extract_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def preprocess_json_line(line: str) -> str:
    text = line.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text

def parse_response(resp: str) -> dict:

    # Find the first JSON object using regex (non-greedy)
    json_match = re.search(r"\{.*?\}", resp, re.DOTALL)
    if not json_match:
        print("No valid JSON block found in response.")
        return None

    json_text = preprocess_json_line(json_match.group(0))

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            pyobj = ast.literal_eval(json_text)
            parsed = json.loads(json.dumps(pyobj))
        except Exception as e:
            print(f"Failed to parse response as JSON: {e}")
            return None

    return parsed

def detect_and_detox(text: str):
    messages = [
        {
            "role": "user",
            "content": f"""You are an AI that detoxifies hate speech in the given text while preserving meaning. 
You must return only a valid JSON object in the following format:
{{
  "original": "...",
  "detoxified": "..."
}}

Do not add any text before or after the JSON block.

Now detoxify this:

Text: "{text}"
"""
        }
    ]


    # Format prompt manually for Mistral-Instruct style
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<s>[INST] {msg['content']} [/INST]"

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **input_ids,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    
    # Extract only assistant's reply after [/INST]
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    try:
        return parse_response(decoded)
    except Exception as e:
        print("Failed to parse:", decoded)
        return None

# Load or initialize detoxified results
out_path = "detoxified/mistral_7B_davidson_detoxified_0728.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

fail_log = "detoxified/failed_samples_davidson_mistral_7b_0728.json"
failed = []

if os.path.exists(out_path):
    with open(out_path, "r", encoding="utf-8") as f:
        detoxified = json.load(f)
else:
    detoxified = []

start_index = len(detoxified)
print(f"Resuming from index {start_index}")

for i in range(start_index, len(data)):
    content = data[i]
    text = content["text"]
    cleaned_text = clean_post_text(text)
    response = detect_and_detox(cleaned_text)
    if response:
        detoxified.append(response)
        print(f"Processed {i+1}/{len(data)}:", response)
    else:
        failed.append({"index": i, "text": text})
        print(f"Failed to parse {i+1}/{len(data)}")

        # Save **after every response** to file (incremental save)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detoxified, f, indent=4, ensure_ascii=False)
        
    with open(fail_log, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=4, ensure_ascii=False)