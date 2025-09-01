from transformers import pipeline    
import json
import os
import re

MODEL_ID = "cirimus/modernbert-large-bias-type-classifier"
INPUT_PATH = "false_refusal_categorize/mistral/mistral_paradetox_detox_0828.json"
OUTOUT_PATH = "false_refusal_categorize/mistral/modernBert/mistral_7B_paradetox_refused_category.json"
FAILED_PATH =  "false_refusal_categorize/mistral/modernBert/mistarl_7B_paradetox_refused_category_failed.json"
BATCH_SIZE = 32
SAVE_EVERY = 30
DETOX = "original"   

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

os.makedirs(os.path.dirname(OUTOUT_PATH), exist_ok=True)
if os.path.exists(OUTOUT_PATH):
    with open(OUTOUT_PATH, "r", encoding="utf-8") as f:
        classified = json.load(f)
else:
    classified = []

os.makedirs(os.path.dirname(FAILED_PATH), exist_ok=True)
if os.path.exists(FAILED_PATH):
    with open(FAILED_PATH, "r", encoding="utf-8") as f:
        failed = json.load(f)
else:
    failed = []

# Load the model
classifier = pipeline(
    "text-classification",
    model = MODEL_ID,
    return_all_scores=True
)

# assumes you already built `classifier = pipeline(..., return_all_scores=True)`
# set DETOX to your text field name, e.g. "original" or "post_text"
       # or "post_text"

start_index = len(classified)
i = start_index
print(f"Resuming from index {start_index}")

while i < len(data):
    batch_idxs = list(range(i, min(i + BATCH_SIZE, len(data))))
    raw_batch = [data[j][DETOX] for j in batch_idxs]

    try:
        # batched inference; truncation avoids huge sequences slowing things down
        batch_outputs = classifier(raw_batch, batch_size=BATCH_SIZE, truncation=True)
        # batch_outputs is a list (len == len(raw_batch)), each item is a list of {label, score}

        for idx, text, out in zip(batch_idxs, raw_batch, batch_outputs):
            best = max(out, key=lambda x: x["score"])
            classified.append({
                "index": idx,
                "text": text,
                "label": best["label"],
                "score": float(best["score"]),
            })

    except Exception as e:
        # if something goes wrong for the whole batch, fall back per-item so you don’t lose progress
        for idx, text in zip(batch_idxs, raw_batch):
            try:
                out = classifier(text, truncation=True)[0]  # list of {label, score}
                best = max(out, key=lambda x: x["score"])
                classified.append({
                    "index": idx,
                    "text": text,
                    "label": best["label"],
                    "score": float(best["score"]),
                })
            except Exception as ee:
                failed.append({"index": idx, "text": text, "error": str(ee)})

    i += BATCH_SIZE

    if (i - start_index) % SAVE_EVERY == 0:
        with open(OUTOUT_PATH, "w", encoding="utf-8") as f: json.dump(classified, f, indent=4, ensure_ascii=False)
        with open(FAILED_PATH, "w", encoding="utf-8") as f: json.dump(failed, f, indent=4, ensure_ascii=False)

with open(OUTOUT_PATH, "w", encoding="utf-8") as f:
     json.dump(classified, f, ensure_ascii=False, indent=2)
with open(FAILED_PATH, "w", encoding="utf-8") as f:
     json.dump(failed, f, ensure_ascii=False, indent=2)
