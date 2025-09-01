import json
from pathlib import Path
from collections import Counter, defaultdict

# --------- paths (edit if needed) ---------
input_path = "false_refusal_categorize/llama/llama3_70B_davidson_detox_0828_v2.json"
out_path = "false_refusal_categorize/llama/llama_70B_refusal_davidson_category_0829.json"

# ---- run metadata (edit as needed) ----
record = {
    "model": "mistral",
    "task": "detoxification",
    "dataset": "paradetox",
}

# ---- knobs ----
# Canonical label to use when bucket is effectively empty
EMPTY_BUCKET_LABEL = "None"
# Tokens that should be treated as empty placeholders (case-insensitive)
EMPTY_SYNONYMS = {"", "none", "(none)", "null", "n/a"}
# Whether to exclude "uncertain" from category & pair counts
IGNORE_UNCERTAIN = True


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_labels(value):
    """
    Normalize a field to a set of labels:
      - Accepts string (comma-separated) or list[str]
      - Lowercase, strip whitespace
      - Convert underscores to spaces (e.g., 'gender_and_sex' -> 'gender and sex')
      - Deduplicate
    """
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


def iter_pairs_from_item(item):
    """
    Yield normalized (axis, bucket) pairs from a single record.

    Priority:
      1) Use 'pairs' if present.
         - Accepts: [["axis, bucket"], ...], ['axis, bucket', ...], or [['axis','bucket'], ...]
      2) Fallback to cross-product of 'category' x 'bucket' (as in the old code).
    Deduplicates pairs within a single record.
    """
    seen = set()

    pairs = item.get("pairs")
    if isinstance(pairs, list) and len(pairs) > 0:
        # Coerce to a list of "axis, bucket" strings
        as_strings = []
        for entry in pairs:
            if isinstance(entry, list):
                if len(entry) == 1 and isinstance(entry[0], str):
                    as_strings.append(entry[0])
                elif len(entry) == 2 and all(isinstance(x, str) for x in entry):
                    as_strings.append(f"{entry[0]}, {entry[1]}")
            elif isinstance(entry, str):
                as_strings.append(entry)

        for p in as_strings:
            parts = [x.strip() for x in str(p).split(",", 1)]
            if len(parts) != 2:
                continue
            axis = normalize_axis_label(parts[0])
            bucket = normalize_bucket_label(parts[1])
            if IGNORE_UNCERTAIN and axis == "uncertain":
                continue
            key = (axis, bucket)
            if key not in seen:
                seen.add(key)
                yield key

        if seen:
            return  # already yielded from 'pairs'; don't fallback

    # Fallback to old behavior if no usable 'pairs'
    categories = to_labels(item.get("category"))
    buckets = to_labels(item.get("bucket"))

    if IGNORE_UNCERTAIN and "uncertain" in categories:
        categories = {c for c in categories if c != "uncertain"}

    # Treat empty/placeholder buckets as "None"
    bucket_real = {b for b in buckets if b not in {x.lower() for x in EMPTY_SYNONYMS}}
    if not bucket_real:
        buckets = {EMPTY_BUCKET_LABEL}
    else:
        buckets = bucket_real

    for c in categories:
        for b in buckets:
            axis = normalize_axis_label(c)
            bucket = normalize_bucket_label(b)
            key = (axis, bucket)
            if key not in seen:
                seen.add(key)
                yield key


data = load_json(input_path)

bucket_counts = Counter()
category_counts = Counter()
pair_counts = Counter()

for item in data:
    local_pairs = set(iter_pairs_from_item(item))
    if not local_pairs:
        continue

    # Update counts
    category_counts.update([a for a, _ in local_pairs])
    bucket_counts.update([b for _, b in local_pairs])
    for a, b in local_pairs:
        pair_counts[(a, b)] += 1

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

print("Done. Categories:", sum(category_counts.values()),
      "Buckets:", sum(bucket_counts.values()),
      "Pairs:", sum(pair_counts.values()))
