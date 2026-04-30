import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login

def main():
    repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    local_dir = Path("models").resolve() / "Llama-3.1-8B-Instruct"
    
    print("="*60)
    print(f"Downloading {repo_id} locally...")
    print(f"Target directory: {local_dir}")
    print("="*60)
    
    # Check if target already has files
    if local_dir.exists() and list(local_dir.glob("*.safetensors")):
        print(f"Model already appears to be downloaded in {local_dir}")
        user_input = input("Force re-download? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborting download.")
            return

    # User Token Logic
    token = os.getenv("HF_TOKEN")
    if not token:
        print("This is a GATED model. You must accept the terms on HuggingFace and provide a token.")
        token = input("Please paste your HuggingFace Access Token (starts with hf_...): ").strip()
    
    if token:
        try:
            val = login(token)
            print("Successfully logged in.")
        except Exception as e:
            print(f"Login failed: {e}")
            sys.exit(1)
    
    # Download
    print("\nStarting download... (this is ~16GB, it might take a while!)")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "*.h5", "coreml/*"], # exclude unnecessary heavy formats
            token=token,
            max_workers=4
        )
        print("\n✓ Download Complete!")
        print(f"The model is permanently saved at: {local_dir}")
        print("\nNext steps:")
        print("To make the entire project use this local model, update `src/config.py`:")
        print('Change: base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"')
        print('To:     base_model: str = "models/Llama-3.1-8B-Instruct"')
    except Exception as e:
        print(f"\nDownload Failed. Error: {e}")
        print("Make sure you accepted the LLaMA 3 terms on the HuggingFace website.")

if __name__ == "__main__":
    main()
