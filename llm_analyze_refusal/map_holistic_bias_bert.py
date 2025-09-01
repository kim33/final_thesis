# robust_predict_originals.py
import json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import pipeline

MODEL_DIR = "twitter-roberta-country"
INPUT_PATH = "false_refusal_categorize/llama/llama3_8B_davidson_detox_0827.json"
OUTPUT_PATH = "false_refusal_categorize/llama/bert/llama_davidson_detox_holistic.json"

MAX_LEN = 128
BATCH_SIZE = 64
DEVICE = 0  # -1 for CPU

def try_json_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f), "json.load"
        except json.JSONDecodeError:
            f.seek(0)
            text = f.read()
            return text, "raw_text"

def parse_maybe_jsonl(text: str) -> List[Dict[str, Any]]:
    # If it's JSONL (one JSON object per line)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    objs = []
    ok = 0
    for ln in lines:
        try:
            objs.append(json.loads(ln))
            ok += 1
        except json.JSONDecodeError:
            pass
    # Heuristic: if majority of non-empty lines parsed => treat as JSONL
    if ok >= max(1, int(0.8 * len(lines))):
        return objs
    return []

def wrap_fragment_as_array(text: str) -> List[Dict[str, Any]]:
    """
    Handle files that look like:
      {obj},\n{obj},\n{obj}
    (i.e., a comma-separated sequence without [ ... ])
    """
    wrapped = f"[{text}]"
    try:
        val = json.loads(wrapped)
        if isinstance(val, list):
            return val
    except json.JSONDecodeError:
        pass
    return []

def normalize_to_list_of_dicts(data: Any) -> List[Dict[str, Any]]:
    # Case A: already a list of dicts
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data

    # Case B: dict containing a list under common keys
    if isinstance(data, dict):
        # direct dict-of-lists with 'original'
        if "original" in data and isinstance(data["original"], list):
            return [{"original": o, **({"detoxified": d} if isinstance(data.get("detoxified"), list) else {})}
                    for o, d in zip(
                        data["original"],
                        data.get("detoxified", [None]*len(data["original"]))
                    )]
        # nested lists under common container keys
        for key in ("data", "items", "records", "examples", "pairs"):
            if key in data and isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                return data[key]
        # any list value with dicts
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v

    # Fallback: nothing worked
    raise ValueError("Could not normalize JSON to a list of dicts with items containing 'original'.")

def ensure_original_field(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # If keys have slightly different names, map them here
    alt_keys = ["text", "source", "toxic", "input"]
    if rows and "original" not in rows[0]:
        for k in alt_keys:
            if k in rows[0]:
                for r in rows:
                    r["original"] = r.get("original", r.get(k))
                break
    if "original" not in rows[0]:
        raise ValueError("No 'original' field found in items. First item keys: " + str(list(rows[0].keys())))
    return rows

def preprocess_tweet(x: str) -> str:
    x = x if isinstance(x, str) else ""
    x = re.sub(r"http\S+|www\.\S+", " ", x)
    x = re.sub(r"@\w+", " @user ", x)
    x = re.sub(r"#", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_and_extract_originals(path: str) -> List[str]:
    data, mode = try_json_load(path)
    print(f"[info] initial load mode: {mode}; type: {type(data)}")

    if isinstance(data, str):
        # Try JSONL
        jl = parse_maybe_jsonl(data)
        if jl:
            rows = jl
        else:
            # Try fragment wrapped as array
            arr = wrap_fragment_as_array(data)
            if arr:
                rows = arr
            else:
                raise ValueError("File is not valid JSON/JSONL and cannot be auto-wrapped as an array.")
    else:
        rows = normalize_to_list_of_dicts(data)

    print(f"[info] normalized to list with {len(rows)} items; first keys: {list(rows[0].keys()) if rows else '[]'}")
    rows = ensure_original_field(rows)
    originals = [preprocess_tweet(r.get("original", "")) for r in rows]
    return originals, rows  # return rows to attach predictions later

def attach_predictions(rows: List[Dict[str, Any]], labels, scores):
    for r, lab, sc in zip(rows, labels, scores):
        r["pred_label"] = lab
        r["pred_score"] = float(sc)
    return rows

def main():
    originals, rows = load_and_extract_originals(INPUT_PATH)

    clf = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=DEVICE
    )

    labels, scores = [], []
    for i in range(0, len(originals), BATCH_SIZE):
        batch = originals[i:i+BATCH_SIZE]
        preds = clf(batch, truncation=True, max_length=MAX_LEN, batch_size=BATCH_SIZE, return_all_scores=False)
        labels.extend([p["label"] for p in preds])
        scores.extend([p["score"] for p in preds])

    out_rows = attach_predictions(rows, labels, scores)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print(f"[done] Saved predictions → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
