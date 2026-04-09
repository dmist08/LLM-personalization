import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def merge_datasets():
    indian_path = Path("data/splits/indian_train.jsonl")
    lamp4_path = Path("data/processed/lamp4/train.jsonl")
    output_path = Path("data/splits/hybrid_train.jsonl")
    
    if not indian_path.exists() or not lamp4_path.exists():
        log.error("One or both input files missing")
        return
        
    indian_data = load_jsonl(indian_path)
    lamp4_data = load_jsonl(lamp4_path)
    
    merged = indian_data + lamp4_data
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    log.info(f"Merged {len(indian_data)} Indian records and {len(lamp4_data)} LaMP-4 records.")
    log.info(f"Total merged dataset: {len(merged)} records saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()
