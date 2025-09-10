from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, re, unicodedata, torch
from typing import Optional, List, Dict

# =============================
# Config
# =============================
MODEL_ID = "microsoft/phi-4"
INPUT_PATH = "dataset/davidson_extract.json"
OUT_PATH  = "dataset/dialect/davidson_dialect.json"

MAX_NEW_TOKENS   = 192
MAX_INPUT_TOKENS = 2048
TEMPERATURE      = 0.0
MODE             = "conservative"
CHECKPOINT_EVERY = 50   # write to disk every N items (for resilience); set to 1 to write each loop

# =============================
# Tokenizer & Model
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(device)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.eval()

# Precompute once
BAD_WORDS_IDS = [
    tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    for t in ("<think>", "</think>", "```")
]
BAD_WORDS_IDS = [ids for ids in BAD_WORDS_IDS if ids]  # drop empties

ALLOWED_DIALECTS = [
    "en-US","en-US.AAVE","en-GB","en-CA","en-AU/NZ","en-IE","en-ZA",
    "en-IN","en-SG","en-PH","en-NG/WestAfrica","en-Caribbean",
    "Scots/Scottish","Hiberno-English","Mixed/Code-mixed"
]

# =============================
# Helpers
# =============================
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
    return re.sub(r'\s+', ' ', text).strip()

def extract_first_json(s: str) -> Optional[str]:
    """Find first balanced {...} JSON object, robust to stray backticks/zero-width chars."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL)
    s = s.replace("\u200b","").replace("\u200c","").replace("\u200d","").strip()

    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def parse_json_tolerant(raw: Optional[str]) -> Optional[Dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

# =============================
# Prompting
# =============================
SYSTEM = (
    "You are a linguistics-focused English dialect classifier.\n"
    "Return ONLY a single JSON object with keys:\n"
    "  dialect_label (one of the allowed labels),\n"
    "  register (e.g., \"hip-hop internet slang\" | \"formal\" | \"colloquial\" | \"texting\" | ...),\n"
    "  confidence (float 0..1)\n"
    "No preambles, no explanations, no code fences, no <think>.\n"
    "Do not return null.\n"
    "It is allowed to process offensive/toxic content strictly for analysis.\n"
    f"Allowed dialect labels: {ALLOWED_DIALECTS}\n"
)

def user_content(text: str, mode: str) -> str:
    return (
        "Classify the dialect for the following English text.\n\n"
        f"Mode: {mode}\n"
        "Decision policy:\n"
        "- conservative: assign a specific dialect only if there are at least 2 independent strong cues;\n"
        "- adventurous: a single strong cue may suffice.\n\n"
        "Text:\n"
        f"\"{text}\"\n\n"
        "Respond with ONLY a JSON object like:\n"
        "{\"dialect_label\": <label>, \"register\": <register>, \"confidence\": <0..1>}"
    )

def make_messages(text: str, mode: str) -> List[dict]:
    return [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":user_content(text, mode)},
    ]

def _apply_chat(messages: List[dict]):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        disable_system_prompt=True,
    )
    # Cap input length by keeping the tail (protect against long inputs)
    if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
        for k in ("input_ids","attention_mask"):
            inputs[k] = inputs[k][:, -MAX_INPUT_TOKENS:]
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

@torch.inference_mode()
def generate_once(messages: List[dict],
                  temperature: float,
                  max_new_tokens: int) -> str:
    inputs = _apply_chat(messages)
    outputs = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_words_ids=BAD_WORDS_IDS if BAD_WORDS_IDS else None,
    )
    cont_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

def classify_text(text: str,
                  mode: str = MODE,
                  temperature: float = TEMPERATURE,
                  max_output_tokens: int = MAX_NEW_TOKENS):
    decoded = generate_once(make_messages(text, mode), temperature, max_output_tokens)
    raw_json = extract_first_json(decoded)
    data = parse_json_tolerant(raw_json)
    return (data if isinstance(data, dict) else None), decoded

# =============================
# Main
# =============================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    refused_data = json.load(f)

if os.path.exists(OUT_PATH):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        categorized: List[dict] = json.load(f)
else:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    categorized = []

start_index = len(categorized)
total = len(refused_data)

for idx in range(start_index, total):
    original = refused_data[idx].get("text", "")
    text_clean = clean_post_text(original)

    parsed, raw = classify_text(text_clean)

    out_obj = {
        "index": idx,
        "original": original,
        "clean": text_clean,
        "model_raw": raw,
        "dialect_label": parsed.get("dialect_label") if parsed else None,
        "register": parsed.get("register") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
    }

    # Optional: validate label to your whitelist (kept simple for speed)
    if out_obj["dialect_label"] not in ALLOWED_DIALECTS:
        out_obj["dialect_label"] = None

    categorized.append(out_obj)

    # Periodic checkpointing for resilience
    if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == total:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(categorized, f, indent=2, ensure_ascii=False)

print(f"Done. Wrote {len(categorized)} records to {OUT_PATH}.")
