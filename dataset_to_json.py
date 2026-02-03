import os
import json
import soundfile as sf

def main():
    root_dir = "/home/zypher/NeMo/asl/Dataset"
    output_file = "dataset_manifest.json"
    manifest = []

    if not os.path.exists(root_dir):
        print(f"Path not found: {root_dir}")
        return

    # Walk through the dataset structure
    # Structure: Dataset/Dialect/SPx_Text.txt and Dataset/Dialect/SPx_audio/
    
    for dialect in os.listdir(root_dir):
        dialect_path = os.path.join(root_dir, dialect)
        if not os.path.isdir(dialect_path):
            continue
            
        print(f"Processing dialect: {dialect}")
        
        files = os.listdir(dialect_path)
        text_files = [f for f in files if f.endswith("_Text.txt")]
        
        for text_file in text_files:
            # Infer audio folder name based on text file name
            # e.g., SP1_THA_Text.txt -> SP1_THA_audio
            audio_folder_name = text_file.replace("_Text.txt", "_audio")
            audio_folder_path = os.path.join(dialect_path, audio_folder_name)
            
            if not os.path.exists(audio_folder_path):
                print(f"Audio folder not found for {text_file}")
                continue
                
            text_file_path = os.path.join(dialect_path, text_file)
            
            with open(text_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                    
                audio_id, text = parts[0], parts[1]
                if text.endswith('.'):
                    text = text[:-1]
                
                # Check for wav extension
                audio_filename = audio_id if audio_id.endswith(".wav") else f"{audio_id}.wav"
                audio_full_path = os.path.join(audio_folder_path, audio_filename)
                
                if os.path.exists(audio_full_path):
                    try:
                        audio_sf = sf.SoundFile(audio_full_path)
                        duration = audio_sf.frames / audio_sf.samplerate
                        
                        entry = {
                            "audio_filepath": os.path.abspath(audio_full_path),
                            "duration": duration,
                            "text": text
                        }
                        manifest.append(entry)
                    except Exception as e:
                        print(f"Error processing {audio_full_path}: {e}")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for m in manifest:
            out.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    print(f"Saved manifest to {os.path.abspath(output_file)} with {len(manifest)} entries.")

if __name__ == "__main__":
    main()
