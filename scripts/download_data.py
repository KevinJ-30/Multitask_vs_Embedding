from pathlib import Path
from datasets import load_dataset
import pandas as pd
import torch

def download_md_agreement():
    """
    Download MD Agreement dataset from Hugging Face datasets
    """
    print("Downloading MD Agreement dataset...")
    
    # Create data directories
    base_path = Path("data/md_agreement")
    processed_path = base_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("MichiganNLP/TID-8", "md-agreement-ann")
    print("Available splits:", dataset.keys())
    
    # Convert and save each split
    splits = {
        'train': dataset['train'],
        #'validation': dataset['dev'],
        'test': dataset['test']
    }
    
    for split_name, split_data in splits.items():
        output_file = processed_path / f"{split_name}.json"
        
        if not output_file.exists():
            print(f"Processing {split_name} split...")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(split_data)
            
            # Save to JSON
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {split_name} split to {output_file}")
            print(f"Number of examples in {split_name}: {len(df)}")

    print("Dataset download and processing completed!")

if __name__ == "__main__":
    download_md_agreement()