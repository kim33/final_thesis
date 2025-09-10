import json
from collections import Counter

INPUT_PATH = "false_refusal_dialect/mistral/mistral_8x7B_hatexplain_dialect.json"
OUTPUT_PATH = "false_refusal_dialect/mistral/8x7B/mistral_8x7B_hateplain_dialect_summary.json"

with open(INPUT_PATH, "r") as f:
    dialect_data = json.load(f)

dialect_counts = Counter(item["dialect_label"] for item in dialect_data if "dialect_label" in item)

summary = {
    "total_entries" : len(dialect_data),
    "dialect label count" : dict(dialect_counts)
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent = 4, ensure_ascii=False)

