from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
import json
import ast
import os
import re
import unicodedata
from typing import Optional, Dict, List, Tuple

# Load tokenizer and model
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET = "dataset/hatexplain_extract.json"
DETOX = "post_text"
OUT_PATH = "detoxified/mistral/mistral_8x7B_hatexplain_detoxified_0901.json"
FAIL_PATH = "detoxified/mistral/mistral_8x7B_hatexplain_detoxified_failed_0901.json"
REFUSED_PATH = "detoxified/mistral/mistral_8x7B_hatexplain_detoxified_refused_0901.json"
SAVE_EVERY = 200 
BATCH_SIZE = 2
MAX_NEW_TOKENS = 64


# Load or initialize detoxified results

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAIL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REFUSED_PATH), exist_ok=True)

if os.path.exists(OUT_PATH):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        detoxified = json.load(f)
else:
    detoxified = []

if os.path.exists(FAIL_PATH):
    with open(FAIL_PATH, "r", encoding="utf-8") as f:
        failed = json.load(f)
else:
    failed = []

if os.path.exists(REFUSED_PATH):
    with open(REFUSED_PATH, "r", encoding="utf-8") as f:
        refusals = json.load(f)
else:
    refusals = []

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

torch.backends.cuda.matmul.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"  
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()



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


def parse_detox_only(s: str) -> str:
    m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
    if not m:
        return ""
    try:
        obj = json.loads(m.group(0))
        return obj.get("detoxified", "")
    except Exception:
        return ""
    

def _make_user_msg(cleaned: str) -> dict:
    return {
        "role": "user",
        "content": (
            'You detoxify hate speech while preserving meaning. '
            'Return ONLY valid JSON: {"detoxified":"<text>"} '
            'Do NOT repeat the input. No explanations. No code fences.\n'
            f'Text: "{cleaned}"'
        )
    }

def detect_and_detox_batch(raw_texts: List[str]) -> List[Dict]:
    originals = raw_texts
    cleaned = [clean_post_text(t) for t in raw_texts]

    # 1) Build messages
    batch_msgs = [[_make_user_msg(c)] for c in cleaned]

    # 2) Get prompts as strings (not tokenized)
    prompts = tokenizer.apply_chat_template(
        batch_msgs,
        tokenize=False,                 # <- important
        add_generation_prompt=True
    )
    if isinstance(prompts, str):
        prompts = [prompts]            # normalize to list

    # 3) Tokenize yourself -> guaranteed dict/BatchEncoding
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # 4) Per-row prompt lengths
    input_lengths = enc["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

    results: List[Dict] = []
    for i in range(gen.size(0)):
        gen_ids = gen[i, input_lengths[i]:]   # slice off prompt
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if "cannot detoxify" in decoded.lower():
            results.append({"original": originals[i], "detoxified": decoded, "Refused": True})
        else:
            det = parse_detox_only(decoded).strip()
            results.append({"original": originals[i], "detoxified": det or decoded})
    return results



# Load dataset
with open(DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)


start_index = len(detoxified)
i = start_index
print(f"Resuming from index {start_index}")

while i < len(data):
    batch_idxs = list(range(i, min(i + BATCH_SIZE, len(data))))
    raw_batch = [data[j][DETOX] for j in batch_idxs]
    results = detect_and_detox_batch(raw_batch)

    for idx, res in zip(batch_idxs, results):
        if res.get("Refused"):
            refusals.append({"index": idx, "text": res["original"], "response": res["detoxified"]})
        elif res.get("detoxified") is not None:
            detoxified.append(res)
        else:
            failed.append({"index": idx, "text": res.get("original")})

    i += BATCH_SIZE

    if (i - start_index) % SAVE_EVERY == 0:
        with open(OUT_PATH, "w", encoding="utf-8") as f: json.dump(detoxified, f, indent=4, ensure_ascii=True)
        with open(FAIL_PATH, "w", encoding="utf-8") as f: json.dump(failed, f, indent=4, ensure_ascii=True)
        with open(REFUSED_PATH, "w", encoding="utf-8") as f: json.dump(refusals, f, indent=4, ensure_ascii=True)

# final save
with open(OUT_PATH, "w", encoding="utf-8") as f: json.dump(detoxified, f, indent=4, ensure_ascii=True)
with open(FAIL_PATH, "w", encoding="utf-8") as f: json.dump(failed, f, indent=4, ensure_ascii=True)
with open(REFUSED_PATH, "w", encoding="utf-8") as f: json.dump(refusals, f, indent=4, ensure_ascii=True)