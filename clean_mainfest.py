import json
import re

INPUT_MANIFEST = "dataset_manifest.json"
OUTPUT_MANIFEST = "dataset_manifest_final.json"

def clean_punctuation(text):
    # 1. Remove specific punctuation marks found in your file: , ? ! - "
    # We replace them with a space to ensure words don't get stuck together
    # e.g., "நண்பர்கள்,சிறுவர்கள்" -> "நண்பர்கள் சிறுவர்கள்"
    text = re.sub(r'[,?!.\-"\']', ' ', text)
    
    # 2. Remove extra whitespace created by the replacement
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Cleaning punctuation from {INPUT_MANIFEST}...")
    with open(INPUT_MANIFEST, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            original = data['text']
            cleaned = clean_punctuation(original)
            
            # Update text
            data['text'] = cleaned
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"Done! Use '{OUTPUT_MANIFEST}' for training.")

if __name__ == "__main__":
    main()