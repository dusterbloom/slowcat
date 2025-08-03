# Running Slowcat in Offline Mode

Slowcat can run completely offline after initial setup. This guide explains how to configure and use offline mode.

## Prerequisites

Before running offline, you need to download the required models while connected to the internet:

1. **Smart Turn Model**: The model will be automatically downloaded on first run
2. **Whisper Models**: Downloaded on first use of each language
3. **Kokoro TTS Model**: Downloaded on first initialization

## Enabling Offline Mode

### Method 1: Environment Variables (Recommended)

Add these lines to your `.env` file:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### Method 2: Automatic Configuration

The server automatically sets these environment variables in `bot.py`:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

## How It Works

1. **Smart Turn Model**: Uses HuggingFace's cached models from `~/.cache/huggingface/hub/`
2. **Model Path**: Configured in `config.py` as `"pipecat-ai/smart-turn-v2"`
3. **Transformers Library**: Runs in offline mode, using only cached models

## Troubleshooting

### Error: "Failed to resolve 'huggingface.co'"

This error occurs when:
- Models aren't cached locally
- Offline mode isn't properly enabled

**Solution**: 
1. Connect to internet and run the server once to download models
2. Verify offline environment variables are set
3. Check that models exist in `~/.cache/huggingface/hub/`

### Verifying Cached Models

Check if smart-turn model is cached:
```bash
ls ~/.cache/huggingface/hub/models--pipecat-ai--smart-turn-v2/
```

### Testing Offline Mode

1. Disconnect from internet
2. Run the server: `./run_bot.sh`
3. If it starts successfully, offline mode is working

## Benefits

- **Privacy**: No data sent to external servers
- **Reliability**: Works without internet connection
- **Performance**: No network latency for model loading
- **Security**: Complete control over your voice data

## Notes

- Initial setup requires internet to download models
- Models are cached permanently after first download
- Cache location: `~/.cache/huggingface/hub/`
- Total cache size: ~500MB for all models