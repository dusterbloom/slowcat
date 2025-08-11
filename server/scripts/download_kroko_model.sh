#!/bin/bash

# Download Kroko-ASR models from Hugging Face
# These are low-latency streaming models designed for edge devices

set -e

# Base directory for models
MODEL_DIR="./models"
REPO_ID="Banafo/Kroko-ASR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Kroko-ASR Model Downloader${NC}"
echo "Low-latency streaming ASR models for edge devices"
echo ""

# Check if language argument is provided
LANGUAGE=${1:-en}

# Validate language
if [[ ! "$LANGUAGE" =~ ^(en|fr|es)$ ]]; then
    echo -e "${RED}Error: Invalid language '$LANGUAGE'${NC}"
    echo "Supported languages: en (English), fr (French), es (Spanish)"
    exit 1
fi

# Set output directory
OUT_DIR="$MODEL_DIR/kroko-asr-$LANGUAGE"

echo -e "${YELLOW}📥 Downloading Kroko-ASR model for: $LANGUAGE${NC}"
echo "Output directory: $OUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Download model files using the Python script
echo "Downloading model files..."

# Try to use virtual environment python if available
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "Using virtual environment Python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    echo "Using virtual environment Python"
fi

$PYTHON_CMD server/scripts/download_sherpa_model.py \
    --repo-id "$REPO_ID" \
    --out "$OUT_DIR" || {
    
    # If the script fails, try manual download with huggingface-cli
    echo -e "${YELLOW}Trying alternative download method...${NC}"
    
    # Files to download for transducer models
    FILES=(
        "${LANGUAGE}_encoder.onnx"
        "${LANGUAGE}_decoder.onnx"
        "${LANGUAGE}_joiner.onnx"
        "${LANGUAGE}_tokens.txt"
    )
    
    # Try huggingface-cli with virtual environment
    HF_CLI="huggingface-cli"
    if [ -f ".venv/bin/huggingface-cli" ]; then
        HF_CLI=".venv/bin/huggingface-cli"
    fi
    
    for FILE in "${FILES[@]}"; do
        echo "Downloading $FILE..."
        $HF_CLI download "$REPO_ID" "$FILE" \
            --local-dir "$OUT_DIR" \
            --local-dir-use-symlinks False || {
            echo -e "${RED}Failed to download $FILE${NC}"
            exit 1
        }
    done
    
    # Rename files to standard names expected by Sherpa
    echo "Renaming files to standard format..."
    mv "$OUT_DIR/${LANGUAGE}_encoder.onnx" "$OUT_DIR/encoder.onnx" 2>/dev/null || true
    mv "$OUT_DIR/${LANGUAGE}_decoder.onnx" "$OUT_DIR/decoder.onnx" 2>/dev/null || true
    mv "$OUT_DIR/${LANGUAGE}_joiner.onnx" "$OUT_DIR/joiner.onnx" 2>/dev/null || true
    mv "$OUT_DIR/${LANGUAGE}_tokens.txt" "$OUT_DIR/tokens.txt" 2>/dev/null || true
}

# Check if files were downloaded successfully
if [ -f "$OUT_DIR/encoder.onnx" ] || [ -f "$OUT_DIR/${LANGUAGE}_encoder.onnx" ]; then
    echo -e "${GREEN}✅ Model downloaded successfully!${NC}"
    echo ""
    echo "Model files in $OUT_DIR:"
    ls -lh "$OUT_DIR"/*.onnx "$OUT_DIR"/*.txt 2>/dev/null || ls -lh "$OUT_DIR"/${LANGUAGE}_*.onnx "$OUT_DIR"/${LANGUAGE}_*.txt
    echo ""
    echo -e "${GREEN}To use this model:${NC}"
    echo "1. Update your .env file:"
    echo "   SHERPA_ONNX_MODEL_DIR=$OUT_DIR"
    echo ""
    echo "2. Or test directly:"
    echo "   python server/test_kroko_model.py --language $LANGUAGE"
else
    echo -e "${RED}❌ Download failed or incomplete${NC}"
    exit 1
fi

# Show model info
echo ""
echo -e "${YELLOW}📊 Model Information:${NC}"
echo "- Type: Transducer (streaming)"
echo "- Language: $LANGUAGE"
echo "- Optimized for: Low-latency edge devices"
echo "- Architecture: Encoder-Decoder-Joiner"
echo ""
echo -e "${GREEN}🎯 Designed for ultra-low latency streaming ASR!${NC}"