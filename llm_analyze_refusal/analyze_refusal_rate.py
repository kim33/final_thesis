import json
from pathlib import Path

original_path = "dataset/paradetox_extract.json"
input_path = "classification/llama3/llama3_8B_paradetox_media_refused_0814.json"
output_path = "false_refusal/llama/llama8B_refusal_rate.json"

model = "llama"
task = "media bias"
dataset = "paradetox"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 1) Load data
original_data = load_json(original_path)
refused_data = load_json(input_path)

# If these files are lists, this is the count you want:
original_length = len(original_data)
refused_length = len(refused_data)

# Guard against divide-by-zero
refusal_rate = (refused_length / original_length) * 100 if original_length else 0.0

# 2) Build output record
record = {
    "model": model,
    "task": task,
    "dataset": dataset,
    "total_examples": original_length,
    "refused_examples": refused_length,
    "refusal_rate": round(refusal_rate, 3)
}

# 3) Append to output JSON (create if missing)
out_path = Path(output_path)
if out_path.exists():
    with open(out_path, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
else:
    existing = []

# Ensure it's a list
if not isinstance(existing, list):
    existing = [existing]

existing.append(record)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)

print("Appended:", record)
