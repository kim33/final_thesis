from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import ast
import os
import re
import unicodedata
from typing import Optional, Dict

# Load tokenizer and model
model_name = "Qwen/Qwen3-8B"

out_path = "detoxified/qwen/qwen3_8B_hatexplain_detoxified_0730.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

refusal_log = "detoxified/qwen/qwen3_8B_refused_samples_0730.json"
os.makedirs(os.path.dirname(refusal_log), exist_ok=True)

fail_log = "detoxified/qwen/qwen3_8B_failed_samples_hatexplain__0730.json"
os.makedirs(os.path.dirname(fail_log), exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")


def clean_post_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'", '´': "'",
        '‐': '-', '–': '-', '—': '-', '…': '...',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = unicodedata.normalize("NFKC", text)
    text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset
with open("dataset/hatexplain_extract.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def preprocess_json_line(line: str) -> str:
    text = line.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text

def strip_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text

def normalize_quotes_and_unicode(text: str) -> str:
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'", '´': "'",
        '‐': '-', '–': '-', '—': '-', '…': '...',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = text.replace('\u2028', '').replace('\u2029', '')
    return unicodedata.normalize("NFKC", text)

def parse_response(resp: str) -> Optional[Dict]:
    resp = strip_code_fence(resp)
    matches = re.finditer(r'\{.*?\}', resp, re.DOTALL)

    for m in matches:
        json_text = normalize_quotes_and_unicode(m.group())

        # Quick fix for unterminated or malformed JSON
        if json_text.count("{") > json_text.count("}"):
            json_text += "}" * (json_text.count("{") - json_text.count("}"))

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            try:
                pyobj = ast.literal_eval(json_text)
                return json.loads(json.dumps(pyobj))
            except Exception as e:
                print("❌ Literal Eval Failed:", e)
                print("↪ Candidate:", repr(json_text))
                continue

    # As a last resort, look for manual fallback
    json_fallback_match = re.search(r'"original"\s*:\s*"(.*?)"\s*,\s*"detoxified"\s*:\s*"(.*?)"', resp, re.DOTALL)
    if json_fallback_match:
        return {
            "original": json_fallback_match.group(1).strip(),
            "detoxified": json_fallback_match.group(2).strip()
        }

    print("⚠️ No valid JSON object found.")
    return None


def detect_and_detox(text: str):
    messages = [
        {
            
            "role": "user",
            "content": f"""You are an AI that detoxifies hate speech in the given text while preserving meaning. 
You must return only a valid JSON object with no extra text, no explanation, and no code block markers. ONLY output valid JSON with no extra explanation or formatting. Do NOT wrap it in code blocks.
 Format:

{{
  "original": <the original input>,
  "detoxified": <the detoxified text>
}}

Now detoxify this:

Text: "{text}"
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    phrase = "I cannot detoxify"
    if phrase.lower() in decoded.lower() :
        print("Refusal Detected:\n", decoded)
        return {"original" : text, "detoxified": decoded, "Refused" : True}
    parsed = parse_response(decoded)
    if parsed:
        return parsed
    else :
        print("Failed to parse:", decoded)
        return {"original" : text, "detoxified": decoded}

# Load or initialize detoxified results

if os.path.exists(out_path):
    with open(out_path, "r", encoding="utf-8") as f:
        detoxified = json.load(f)
else:
    detoxified = []

if os.path.exists(fail_log):
    with open(fail_log, "r", encoding="utf-8") as f:
        failed = json.load(f)
else:
    failed = []

if os.path.exists(refusal_log):
    with open(refusal_log, "r", encoding="utf-8") as f:
        refusals = json.load(f)
else:
    refusals = []

start_index = len(detoxified)
print(f"Resuming from index {start_index}")



for i in range(start_index, len(data)):
    content = data[i]
    text = content["post_text"]
    cleaned_text = clean_post_text(text)
    response = detect_and_detox(cleaned_text)
    print("Raw model output:", response)

    if isinstance(response, dict) and response.get("Refused") :
        refusals.append({
            "index": i,
            "text": text,
            "response": response["detoxified"]
        })
        print(f"Refusal detected at {i+1}/{len(data)}")
    elif response and isinstance(response, dict) and response.get("detoxified") is not None:
        print("DEBUG: About to append:", response)  
        detoxified.append(response)
        print(f"Processed {i+1}/{len(data)}:", response)
    else:
        failed.append({"index": i, "text": text if response else None})
        print(f"Failed to parse {i+1}/{len(data)}")

        # Save **after every response** to file (incremental save)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detoxified, f, indent=4, ensure_ascii=False)
        
    with open(fail_log, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=4, ensure_ascii=False)

    with open(refusal_log, "w", encoding="utf-8") as f:
        json.dump(refusals, f, indent=4, ensure_ascii=False)