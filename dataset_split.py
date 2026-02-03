import json
import random
import os

def split_dataset(manifest_path, train_path, val_path, val_ratio=0.2):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines to ensure random split
    random.shuffle(lines)
    
    num_val = int(len(lines) * val_ratio)
    val_lines = lines[:num_val]
    train_lines = lines[num_val:]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
        
    print(f"Split {len(lines)} items into {len(train_lines)} train and {len(val_lines)} validation items.")

if __name__ == "__main__":
    manifest_file = 'dataset_manifest.json'
    train_file = 'train_manifest.json'
    val_file = 'val_manifest.json'
    
    if os.path.exists(manifest_file):
        split_dataset(manifest_file, train_file, val_file)
    else:
        print(f"Manifest file {manifest_file} not found.")
