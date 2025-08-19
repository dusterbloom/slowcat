# server/scripts/download_sherpa_model.py
"""
Usage:
  # Download STT model from Hugging Face
  python server/scripts/download_sherpa_model.py \
    --repo-id k2-fsa/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
    --out ./models/sherpa/en

  # Download punctuation model from GitHub releases
  python server/scripts/download_sherpa_model.py \
    --punctuation-model \
    --out ./models/sherpa/punctuation

For paraformer-style repos, use the appropriate repo id; the script
will detect whether it's a transducer (encoder/decoder/joiner) or single model.onnx.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download

CANDIDATE_FILES = [
    # Transducer
    "encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt",
    # Paraformer
    "model.onnx", "tokens.txt"
]

# Punctuation model details
PUNCTUATION_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
PUNCTUATION_MODEL_NAME = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"

def download_punctuation_model(out_dir: Path):
    """Download and extract the sherpa-onnx punctuation model"""
    print(f"Downloading punctuation model from GitHub releases...")
    
    # Download the tar.bz2 file
    archive_path = out_dir / f"{PUNCTUATION_MODEL_NAME}.tar.bz2"
    
    try:
        print(f"Downloading: {PUNCTUATION_MODEL_URL}")
        urllib.request.urlretrieve(PUNCTUATION_MODEL_URL, archive_path)
        print(f"✓ Downloaded to: {archive_path}")
        
        # Extract the archive
        print(f"Extracting archive...")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=out_dir)
        
        # Remove the archive file
        archive_path.unlink()
        
        # Verify extraction
        model_dir = out_dir / PUNCTUATION_MODEL_NAME
        model_file = model_dir / "model.onnx"
        
        if model_file.exists():
            print(f"✅ Punctuation model extracted successfully!")
            print(f"Model directory: {model_dir.resolve()}")
            print(f"Model file: {model_file.resolve()}")
            return model_dir
        else:
            raise FileNotFoundError(f"model.onnx not found in extracted files")
            
    except Exception as e:
        if archive_path.exists():
            archive_path.unlink()  # Cleanup on failure
        raise RuntimeError(f"Failed to download punctuation model: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", help="Hugging Face repo ID for STT models")
    ap.add_argument("--revision", default="main", help="Model revision/branch")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--punctuation-model", action="store_true", 
                   help="Download punctuation model instead of STT model")
    args = ap.parse_args()
    
    # Validate arguments
    if not args.punctuation_model and not args.repo_id:
        raise SystemExit("Either --repo-id or --punctuation-model must be specified")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    if args.punctuation_model:
        # Download punctuation model from GitHub releases
        model_dir = download_punctuation_model(out)
        print(f"\n✅ Punctuation model ready at: {model_dir.resolve()}")
        print(f"Usage: --punctuation-model-dir {model_dir.resolve()}")
    else:
        # Download STT model from Hugging Face
        found = 0
        for fname in CANDIDATE_FILES:
            try:
                fp = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=fname,
                    revision=args.revision,
                    local_dir=out,
                    local_dir_use_symlinks=False,
                )
                print(f"✓ {fname} -> {fp}")
                found += 1
            except Exception:
                pass

        if found == 0:
            raise SystemExit(
                "Could not download any known model files. "
                "Check the repo id and list of filenames."
            )
        print(f"\nDownloaded {found} file(s) to: {out.resolve()}")

if __name__ == "__main__":
    main()