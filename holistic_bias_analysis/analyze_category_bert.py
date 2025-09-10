import json
from collections import Counter
import pandas as pd


INPUT_PATH = "false_refusal_categorize/llama/bert/llama_8B_paradetox_refused_category.json"
OUTPUT_PATH = "false_refusal_categorize/llama/bert/01.llama_8B_paradetox_category.json"


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

label_counts = Counter(item["label"] for item in data)
sorted_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))


result = {
    "model": "llama_8B",
    "dataset" : "paradetox",
    "task" : "detoxification",
    "label_counts" : sorted_counts
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent = 4)

print("Result save to ", OUTPUT_PATH)