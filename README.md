# slowcat 🐱

*A purr-fectly tuned local voice agent for macOS*

Fast, local voice agents powered by Pipecat - achieving sub-800ms voice-to-voice latency on Apple Silicon.

Pipecat is an open-source, vendor-neutral framework for building real-time voice (and video) AI applications.

This repository showcases a fully local voice agent stack running entirely on macOS. With Apple Silicon (M-series chips), you can achieve lightning-fast voice-to-voice latency under 800ms using powerful local models - no cloud dependencies required! 🚀

The [server/bot.py](server/bot.py) file uses these models:

  - Silero VAD
  - smart-turn v2
  - MLX Whisper
  - Gemma3 12B
  - Kokoro TTS

But you can swap any of them out for other models, or completely reconfigure the pipeline. It's easy to add tool calling, MCP server integrations, use parallel pipelines to do async inference alongside the voice conversations, add custom processing steps, configure interrupt handling to work differently, etc.

The bot and web client here communicate using a low-latency, local, serverless WebRTC connection. For more information on serverless WebRTC, see the Pipecat [SmallWebRTCTransport docs](https://docs.pipecat.ai/server/services/transport/small-webrtc) and this [article](https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/). You could switch over to a different Pipecat transport (for example, a WebSocket-based transport), but WebRTC is the best choice for realtime audio.

For a deep dive into voice AI, including network transport, optimizing for latency, and notes on designing tool calling and complex workflows, see the [Voice AI & Voice Agents Illustrated Guide](https://voiceaiandvoiceagents.com/).

# Models and dependencies

Silero VAD, MLX Whisper, and Kokoro all run inside the Pipecat process. When the agent code starts, it will need to download model weights that aren't already cached, so first startup can take some time.

The LLM service in this bot uses the OpenAI-compatible chat completion HTTP API. So you will need to run a local OpenAI-compatible LLM server. 

One easy, high-performance, way to run a local LLM server on macOS is [LM Studio](https://lmstudio.ai/). From inside the LM Studio graphical interface, go to the "Developer" tab on the far left to start an HTTP server.

# Run the voice agent

The core voice agent code lives in a single file: [server/bot.py](server/bot.py). There's one custom service here that's not included in Pipecat core: we implemented a local Kokoro TTS frame processor on top of the excellent [mlx-audio library](https://github.com/Blaizzy/mlx-audio).

## Using Python
```shell
cd server/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

## Using the run script
```shell
cd server/
./run_bot.sh
```

## Using uv (faster Python package manager)
```shell
cd server/
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python bot.py
```

## Running in different languages 🌍

Slowcat speaks multiple languages! Configure your voice agent for different locales:

### Language Support Matrix

| Language | Code | Whisper Support | Kokoro TTS Support | Voice Used |
|----------|------|----------------|-------------------|------------|
| English  | `en` | ✅ | ✅ | af_heart |
| Spanish  | `es` | ✅ | ✅ | ef_dora |
| French   | `fr` | ✅ | ✅ | ff_siwis |
| German   | `de` | ✅ | ⚠️ | af_heart (English voice) |
| Japanese | `ja` | ✅ | ✅ | jf_alpha |
| Italian  | `it` | ✅ | ✅ | im_nicola |
| Chinese  | `zh` | ✅ | ✅ | zf_xiaobei |
| Portuguese | `pt` | ✅ | ✅ | pf_dora |

⚠️ = Falls back to English voice (speech recognition works, but synthesis uses English voice)

### Examples

**Spanish (Español)**
```shell
python bot.py --language es
```

**French (Français)**
```shell
python bot.py --language fr
```

**Japanese (日本語)**
```shell
python bot.py --language ja
```

*Note: Both Whisper (speech recognition) and Kokoro TTS (speech synthesis) must support your chosen language for full functionality. Make sure your LLM server also supports the target language for best results.*

# Start the web client

The web client is a React app. You can connect to your local macOS agent using any client that can negotiate a serverless WebRTC connection. The client in this repo is based on [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) and just uses that library's standard debug console template.

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```

## What Makes Slowcat Special

This enhanced fork brings several performance and usability improvements:

✨ **Enhanced Kokoro TTS** - Seamless MLX integration for natural speech synthesis  
⚡ **Performance Optimized** - Fine-tuned configuration for minimal latency  
🛠️ **Developer Friendly** - Convenient run scripts and improved setup process  
🍎 **macOS Native** - Optimized dependencies for Apple Silicon compatibility  
🎤 **Voice Recognition** - Automatic speaker identification and enrollment

### Voice Recognition Features

Slowcat now includes automatic voice recognition capabilities powered by Resemblyzer:

- **Automatic Speaker Enrollment** - Learns to recognize speakers after just 3 utterances
- **Persistent Speaker Profiles** - Remembers speakers across sessions
- **Real-time Identification** - Identifies speakers as they talk
- **Privacy-First** - All processing happens locally, no cloud dependencies

To disable voice recognition, set the environment variable:
```shell
ENABLE_VOICE_RECOGNITION=false python bot.py
```

---

## Acknowledgments

This project is based on the excellent work by [kwindla](https://github.com/kwindla) in the original [macos-local-voice-agents](https://github.com/kwindla/macos-local-voice-agents) repository. Thank you for creating such a solid foundation for local voice AI on macOS!

*Built with ❤️ for the local AI community*
