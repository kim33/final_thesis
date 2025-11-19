import json
import re

input = "detoxified/gemma/gemma_9B_korean_detoxified_1104_v2.json"
output = "detoxified/gemma/gemma_9B_korean_detoxified_1104_v4.json"
cleaned = []

'''
QUOTE_MAP = {
    '\u201c': '"',  # “
    '\u201d': '"',  # ”
    '\u201e': '"',  # „
    '\u201f': '"',  # ‟
    '\u300c': '"',  # 「
    '\u300d': '"',  # 」
}
TRANS = str.maketrans(QUOTE_MAP)

def normalize_quotes(s: str) -> str:
    return s.translate(TRANS)

def unescape_json_fragment(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except Exception:
        return s

def try_json_dict(s: str):
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def scan_quoted_string(s: str, i: int) -> tuple[str, int] | tuple[None, int]:
    # s[i] must be the opening double-quote
    i += 1
    out = []
    esc = False
    while i < len(s):
        ch = s[i]
        if esc:
            out.append(ch)
            esc = False
        else:
            if ch == '\\':
                esc = True
            elif ch == '"':
                return unescape_json_fragment(''.join(out)), i + 1
            else:
                out.append(ch)
        i += 1
    return None, i

def scan_bracket_block(s: str, i: int) -> tuple[str, int] | tuple[None, int]:
    # s[i] is '{' or '['
    open_ch = s[i]
    close_ch = '}' if open_ch == '{' else ']'
    depth = 0
    start = i
    in_str = False
    esc = False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            else:
                if ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i+1], i+1
        i += 1
    return None, i

def extract_detoxified_value(raw_det):
    # Case: already a dict
    if isinstance(raw_det, dict):
        v = raw_det.get("detoxified", raw_det)
        # ensure string if nested dict slipped in
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return v

    if not isinstance(raw_det, str):
        return raw_det

    # Try direct and normalized JSON parses
    obj = try_json_dict(raw_det)
    if obj and "detoxified" in obj:
        v = obj["detoxified"]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    det = normalize_quotes(raw_det)
    obj2 = try_json_dict(det)
    if obj2 and "detoxified" in obj2:
        v = obj2["detoxified"]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    # Manual scan after "detoxified":
    s = det
    key_pos = s.find('"detoxified"')
    if key_pos == -1:
        # try escaped-key pattern within an escaped blob
        key_pos = s.find('\"detoxified\"')
        if key_pos == -1:
            return raw_det

    colon = s.find(':', key_pos)
    if colon == -1:
        return raw_det

    i = colon + 1
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s):
        return raw_det

    if s[i] == '"':
        val, end_i = scan_quoted_string(s, i)
        return val if val is not None else raw_det

    if s[i] in '{[':
        block, end_i = scan_bracket_block(s, i)
        if block is None:
            return raw_det

        # Try to parse block as JSON; if object/array, reduce to string as needed
        obj3 = try_json_dict(block)
        if obj3 is not None:
            v = obj3
            # If it's a dict with a single key and no normal values, and that key is the text, extract key
            if isinstance(v, dict) and len(v) == 1:
                only_key = next(iter(v.keys()))
                only_val = v[only_key]
                # Many malformed cases look like {"<TEXT>": null} or just key with no value in source
                if only_val in (None, "") or not isinstance(only_val, (str, list, dict, int, float, bool)):
                    return only_key
            # If dict/list but otherwise valid, serialize to string
            return json.dumps(v, ensure_ascii=False)

        # If still invalid JSON, but looks like {"<TEXT>"} — pull the sole quoted bit
        m = re.match(r'^\{\s*"([^"]+)"\s*\}$', block, flags=re.DOTALL)
        if m:
            return m.group(1)

        # Last resort: strip outer braces/brackets
        inner = block[1:-1].strip()
        # If the inner starts/ends with quotes, unescape it
        if inner.startswith('"') and inner.endswith('"'):
            return unescape_json_fragment(inner[1:-1])
        return inner or raw_det

    # Unexpected token — return as-is
    return raw_det

changed = 0
with open(input, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    orig_det = item.get("detoxified", "")
    new_det = extract_detoxified_value(orig_det)
    if new_det != orig_det:
        changed += 1
    cleaned.append({
        "original": item.get("original", ""),
        "detoxified": new_det
    })

with open(output, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned data saved to:", output)
print(f"Total records:", len(cleaned))
print(f"🔎 Records modified:", changed)




# The exact phrase to detect
trigger_phrase = (
    "\"detoxified\":"
)

# Regex to capture everything after the trigger phrase until the end of string
pattern = re.compile(
    re.escape(trigger_phrase) + r"\s*(.*)", re.DOTALL
)

# Load your JSON file
with open(input, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each item
for item in data:
    detox_text = item.get("response", "")
    match = pattern.search(detox_text)
    if match:
        # Keep only the part after the trigger phrase
        item["response"] = match.group(1).strip()
    else:
        # If the pattern isn't found, leave the detoxified text unchanged
        item["response"] = detox_text.strip()

# Save cleaned JSON
with open(output, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned JSON saved to {output}")
'''

import json


with open(input, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []
for item in data:
    if "response" in item:
        # Parse response JSON safely
        try:
            response_obj = json.loads(item["response"])
            print(response_obj)
            processed.append({
                "original": item.get("text", ""),
                "detoxified": response_obj.get("detoxified")
            })
        except (json.JSONDecodeError, TypeError):
            # If invalid JSON, still include the text
            processed.append({
                "original": item.get("text", ""),
                "detoxified": ""
            })
    else :
         processed.append(item)

with open(output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"✅ Processed {len(processed)} entries and saved to {output}")
