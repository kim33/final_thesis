from openai import OpenAI
import os
import json
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv() 

file_to_detoxify = 'dataset/davidson_extract_full.json'
detoxified_out = 'gpt3.5_davidson_detoxified.json'
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


function_definitions = [
    {
        "name": "return_detoxified_text",
        "description": "Detoxifies a hateful or offensive sentence while preserving the original meaning. DO NOT ADD OTHER COMMENTS BUT ONLY DETOXIFY THE GIVEN INPUT TEXT",
        "parameters": {
            "type": "object",
            "properties": {
                "original": {"type": "string", "description": "The original text input"},
                "detoxified": {"type": "string", "description": "A detoxified version of the input"}
            },
            "required": ["original", "detoxified"]
        }
    }
]
with open(file_to_detoxify, "r", encoding="utf-8") as f:
    data = json.load(f)

client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")

with open(detoxified_out, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")

first = True

for idx,item in enumerate(data):
    text = item["text"]
    cleaned_text = clean_post_text(text)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages = [
            {
                "role" : "system",
                "content" : "You are an ethics expert for social media to detoxify the sentence keeping the original meaning. DO NOT ADD COMMENTS BUT ONLY DETOXIFY THE GIVEN INPUT."
            },
            {
                "role" : "user",
                "content" : cleaned_text
            }
        ],
        functions=function_definitions,
        function_call={"name": "return_detoxified_text"},
        temperature = 0
    )

    arguments = response.choices[0].message.function_call.arguments
    
    try:
        result = json.loads(arguments)

        with open(detoxified_out, "a", encoding="utf-8") as f_out:
            if not first:
                f_out.write(",\n")
            else:
                first = False

            json.dump(result, f_out, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Failed at index {idx}: {e}")
        print(f"Input: {text}")


# Close JSON array
with open(detoxified_out, "a", encoding="utf-8") as f_out:
    f_out.write("\n]\n")