import json
import re
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

MODEL_NAME = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"

def get_vocab():
    print(f"Loading tokenizer from {MODEL_NAME}...")
    try:
        model = EncDecHybridRNNTCTCModel.from_pretrained(model_name=MODEL_NAME)
    except:
        # Fallback if automatic download fails
        print("Could not download from HF, looking for local .nemo file...")
        model = EncDecHybridRNNTCTCModel.restore_from("indicconformer_stt_ta_hybrid_ctc_rnnt_large.nemo")
    return set(model.tokenizer.vocab)

def clean_file(input_path, output_path, vocab):
    print(f"Cleaning {input_path} -> {output_path} ...")
    success_count = 0
    skipped_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            original_text = entry['text']
            
            # Filter text: keep only chars in vocab
            # We replace unknown chars with spaces to avoid merging words
            clean_chars = [c if c in vocab else ' ' for c in original_text]
            cleaned_text = "".join(clean_chars)
            
            # Normalize spaces (collapse multiple spaces into one)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if cleaned_text:
                entry['text'] = cleaned_text
                f_out.write(json.dumps(entry) + "\n")
                success_count += 1
            else:
                skipped_count += 1

    print(f"Finished {input_path}: Kept {success_count}, Skipped {skipped_count} empty lines.")

if __name__ == "__main__":
    vocab = get_vocab()
    
    # Clean Train
    clean_file("train_manifest.json", "train_manifest_clean.json", vocab)
    
    # Clean Test
    clean_file("test_manifest.json", "test_manifest_clean.json", vocab)