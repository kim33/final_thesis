import json

# File paths
full_dataset_path = "false_refusal/qwen/qwen3_30B_paradetox_detox_0901.json"
falsely_refused_path = "false_refusal_categorize/swear/qwen/qwen_30B_paradetox_swear.json"

# Load both datasets
with open(full_dataset_path, "r", encoding="utf-8") as f:
    full_data = json.load(f)

with open(falsely_refused_path, "r", encoding="utf-8") as f:
    refused_data = json.load(f)

# Compute sizes
total = len(full_data)
refused = len(refused_data)

# Compute ratio
if total > 0:
    refused_ratio = refused / total
    print(f"Total samples: {total}")
    print(f"Falsely refused samples: {refused}")
    print(f"Refused ratio: {refused_ratio:.2%}")
else:
    print("⚠️ Full dataset is empty!")
