import json
from pathlib import Path
from collections import Counter, defaultdict

# --------- paths (edit if needed) ---------
input_path = "false_refusal_categorize/gemma/gemma_8B_paradetox_detox_0909.json"
out_path = "false_refusal_categorize/gemma/gemma_8B/gemma3_8B_holistic_bias_paradetox_summary.json"

# ---- run metadata (edit as needed) ----
record = {
    "model": "Gemma3_8B",
    "task": "detoxification",
    "dataset": "Paradetox",
}

# ---- knobs ----
EMPTY_BUCKET_LABEL = "None"
EMPTY_SYNONYMS = {"", "none", "(none)", "null", "n/a"}
IGNORE_UNCERTAIN = True

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_labels(value):
    """Normalize a field to a set of labels from string or list[str]."""
    if value is None:
        return set()
    if isinstance(value, list):
        tokens = value
    else:
        tokens = str(value).split(",")
    labels = set()
    for tok in tokens:
        if tok is None:
            continue
        t = str(tok).strip().lower()
        if not t:
            continue
        t = t.replace("_", " ")
        labels.add(t)
    return labels

def normalize_axis_label(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower().replace("_", " ")

def normalize_bucket_label(s: str) -> str:
    if s is None:
        return EMPTY_BUCKET_LABEL
    raw = str(s).strip()
    if raw.lower() in EMPTY_SYNONYMS:
        return EMPTY_BUCKET_LABEL
    return raw.lower().replace("_", " ")

def parse_pairs_field(pairs_field):
    """
    Parse the 'pairs' field which can look like:
      ["ability, (none)", "age, (none)"]  OR  [["ability, (none)"], ["age, (none)"]]
    Returns a list of (category, bucket) tuples, normalized.
    """
    if not pairs_field:
        return []

    result = []
    for entry in pairs_field:
        # unwrap [["text"]] style
        if isinstance(entry, (list, tuple)) and entry:
            entry = entry[0]

        if not isinstance(entry, str):
            # best-effort stringify
            entry = str(entry)

        # split on the first comma into (category, bucket)
        parts = entry.split(",", 1)
        if len(parts) != 2:
            # If formatting is off, skip
            continue

        cat_raw, buck_raw = parts[0].strip(), parts[1].strip()

        cat_norm = normalize_axis_label(cat_raw)
        buck_norm = normalize_bucket_label(buck_raw)

        result.append((cat_norm, buck_norm))
    return result

data = load_json(input_path)

bucket_counts = Counter()
category_counts = Counter()
pair_counts = Counter()  # computed ONLY from the 'pairs' field

for item in data:
    # ----- pull directly from 'category' and 'bucket' fields -----
    categories_raw = to_labels(item.get("category"))
    buckets_raw = to_labels(item.get("bucket"))

    # optionally exclude 'uncertain' from categories (affects both category_counts and which pairs we accept)
    if IGNORE_UNCERTAIN and "uncertain" in categories_raw:
        categories_raw = {c for c in categories_raw if c != "uncertain"}

    # normalize categories
    categories = {normalize_axis_label(c) for c in categories_raw if c is not None}

    # normalize buckets, treat empty/placeholder as EMPTY_BUCKET_LABEL
    bucket_real = {b for b in buckets_raw if b not in {x.lower() for x in EMPTY_SYNONYMS}}
    if not bucket_real:
        buckets = {EMPTY_BUCKET_LABEL}
    else:
        buckets = {normalize_bucket_label(b) for b in bucket_real}

    # if no categories, skip this item entirely
    if not categories:
        continue

    # counts for axes come from category/bucket keys ONLY
    category_counts.update(categories)
    bucket_counts.update(buckets)

    # --- pairs come ONLY from explicit 'pairs' field; we DO NOT do a cross-product ---
    pairs_list = parse_pairs_field(item.get("pairs"))

    # keep only pairs where the category is present (after uncertain filter)
    # and (optionally) bucket is one of the normalized buckets from this item.
    # This ensures "characteristics" only pairs with "(none)" and "immigration_status" in your example.
    for c, b in pairs_list:
        if c in categories and b in buckets:
            pair_counts[(c, b)] += 1

# nest pairs by category -> bucket -> count
pairs_nested = defaultdict(dict)
for (c, b), n in pair_counts.items():
    pairs_nested[c][b] = n

record["category_counts"] = dict(category_counts)
record["bucket_counts"] = dict(bucket_counts)
record["category_bucket_pairs"] = dict(pairs_nested)

# append to output json
out_path = Path(out_path)
if out_path.exists():
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing = []
else:
    existing = []

if not isinstance(existing, list):
    existing = [existing]

existing.append(record)
out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    "Done. Categories:", sum(category_counts.values()),
    "Buckets:", sum(bucket_counts.values()),
    "Pairs:", sum(pair_counts.values()),
    "Total Size:", len(data)
)
