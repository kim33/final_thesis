import json
from collections import Counter

INPUT_PATH = "false_refusal_categorize/llama/bert/llama_70B_davidson_bert_bias.json"
OUTPUT_PATH = "false_refusal_categorize/llama/bert/llama_70B_davidson_category.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

counts = Counter(item[label] for item in data if "label" in item)
