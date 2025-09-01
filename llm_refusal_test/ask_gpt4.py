from openai import OpenAI
import os
import json
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv() 
def clean_post_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Replace smart quotes and apostrophes with straight versions
    replacements = {
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '´': "'",
        '‐': '-',  # dash
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize Unicode (e.g., decompose accented letters)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except newlines
    text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def coerce_json(text: str):
    """Be robust to fences or stray text; return dict or raise."""
    if text is None:
        raise ValueError("Empty model content")
    # Strip code fences if present
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in: {text[:200]}")
    return json.loads(m.group(0))

client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
)

# Load data
with open("dataset/hatexplain_extract.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Output file
output_path = "classification/gpt/gpt4_hatexplain_sexism.json"
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")

first = True

refusal_log = "classification/gpt/gpt4_hatexplain_sexism_refused.json"
os.makedirs(os.path.dirname(refusal_log), exist_ok=True)
refused = []

# Process each item
for idx, item in enumerate(data):
    text = item["post_text"]
    cleaned_text = clean_post_text(text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4-1106-preview" / "gpt-3.5-turbo-1106"
             messages = [
            {
                    "role": "user",
                    "content": f"""
                    Your job is to distinguish whether the given text contains misinformation - contains false or misleading factual claims.
                    Given a sentence, return only a JSON object with the following fields:

                    - "original": the input sentence
                    - "classification": one of ["misinformation", "not misinformation"]

                    ⚠️ Do NOT provide any explanation or commentary. Only return the JSON object exactly in the format shown.
                    DO not refuse to answer unless the given text is empty.

                    Example output:
                    {{ "original": "example text here", "classification": not misinformation" }}

                    Text: "{text}"
                    """
                }
            ],
        )

        # Extract the JSON content from the function call
        arguments = response.choices[0].message.content
        result = coerce_json(arguments)

        phrase = "I cannot classify"
        if any(phrase.lower() in str(v).lower() for v in result.values()):
            print("Refusal Detected:\n", result)
            refused.append(result)
        
        else :
            with open(output_path, "a", encoding="utf-8") as f_out:
                if not first:
                    f_out.write(",\n")
                else:
                    first = False

                json.dump(result, f_out, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Failed at index {idx}: {e}")
        print(f"Input: {text}")

# Close JSON array
with open(output_path, "a", encoding="utf-8") as f_out:
    f_out.write("\n]\n")

with open(refusal_log, "w", encoding="utf-8") as f:
    json.dump(refused, f, indent=4, ensure_ascii=False)