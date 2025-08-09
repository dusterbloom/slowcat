# server/scripts/download_sherpa_model.py
"""
Usage:
  python server/scripts/download_sherpa_model.py \
    --repo-id k2-fsa/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
    --out ./models/sherpa/en

For paraformer-style repos, use the appropriate repo id; the script
will detect whether it's a transducer (encoder/decoder/joiner) or single model.onnx.
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

CANDIDATE_FILES = [
    # Transducer
    "encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt",
    # Paraformer
    "model.onnx", "tokens.txt"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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