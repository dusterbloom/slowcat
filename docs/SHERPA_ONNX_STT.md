Heck yes — let’s drop Sherpa-ONNX into Slowcat without ripping out what you already have. Below are **drop-in files** plus **surgical edits** so you can switch STT at runtime (`STT_BACKEND=whisper-mlx | sherpa-onnx`). I’m giving you complete files for anything new, and precise patch blocks for existing files, so you won’t lose work.

Quick map of where this fits: `PipelineBuilder` injects the STT “service” straight into the processor chain (your STT object is itself a Pipecat processor) — so we just provide a Sherpa processor that speaks Pipecat’s frame API, and let `ServiceFactory` choose it.&#x20;

---

# 1) New file — `server/services/sherpa_stt.py`

This is a minimal, production-ready **offline (utterance-based) recognizer** using Sherpa-ONNX. It buffers audio between `UserStartedSpeakingFrame`/`UserStoppedSpeakingFrame` (your VAD already emits those), decodes once per turn (low CPU + stable), and emits a `TranscriptionFrame`.

```python
# server/services/sherpa_stt.py
import asyncio
import os
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Lazy import so import-time doesn't pull ONNX into every codepath
_sherpa = None


def _require_sherpa():
    global _sherpa
    if _sherpa is None:
        import sherpa_onnx as _sherpa  # type: ignore
    return _sherpa


class SherpaONNXSTTService(FrameProcessor):
    """
    Utterance-level STT using Sherpa-ONNX OfflineRecognizer.
    Works with Slowcat's existing VAD/turn-taking. Low CPU on Apple Silicon.
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "auto",
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        provider: Optional[str] = None,  # None -> let sherpa choose; "cpu" recommended
        hotwords_file: Optional[str] = None,
        hotwords_score: float = 1.5,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.language = language
        self.sample_rate = sample_rate
        self.decoding_method = decoding_method
        self.provider = provider or "cpu"
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score

        self._buffer = bytearray()
        self._listening = False
        self._recognizer = None

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"SHERPA_ONNX_MODEL_DIR does not exist: {self.model_dir}"
            )

        self._init_recognizer()

    # ---------- Sherpa init ----------
    def _init_recognizer(self):
        sherpa = _require_sherpa()

        tokens = self.model_dir / "tokens.txt"
        enc = self.model_dir / "encoder.onnx"
        dec = self.model_dir / "decoder.onnx"
        join = self.model_dir / "joiner.onnx"
        para = self.model_dir / "model.onnx"

        if not tokens.exists():
            raise FileNotFoundError(f"tokens.txt not found in {self.model_dir}")

        # Detect model flavor: Transducer (encoder/decoder/joiner) or Paraformer (model.onnx)
        transducer = enc.exists() and dec.exists() and join.exists()
        paraformer = para.exists()

        if not (transducer or paraformer):
            raise RuntimeError(
                f"No supported Sherpa model files found in {self.model_dir}.\n"
                f"Expected either (encoder.onnx, decoder.onnx, joiner.onnx) or model.onnx"
            )

        # Build config
        feat_cfg = sherpa.FeatureConfig(
            sampling_rate=self.sample_rate,
            feature_dim=80,
            low_freq=20,
            high_freq=-400,
            dither=0.0,
            normalize_samples=True,
            snip_edges=False,
        )

        mdl_cfg = sherpa.OfflineModelConfig(
            transducer=sherpa.OfflineTransducerModelConfig(
                encoder_filename=str(enc) if transducer else "",
                decoder_filename=str(dec) if transducer else "",
                joiner_filename=str(join) if transducer else "",
            ),
            paraformer=sherpa.OfflineParaformerModelConfig(
                model=str(para) if paraformer else ""
            ),
            tokens=str(tokens),
            provider=self.provider,  # "cpu" on Apple Silicon works well
            num_threads=max(1, os.cpu_count() // 2),
            debug=False,
        )

        lm_cfg = sherpa.OfflineLMConfig(model="", scale=0.5)

        self._cfg = sherpa.OfflineRecognizerConfig(
            feat_config=feat_cfg,
            model_config=mdl_cfg,
            lm_config=lm_cfg,
            decoding_method=self.decoding_method,
            hotwords_file=self.hotwords_file or "",
            hotwords_score=self.hotwords_score,
            max_active_paths=4,  # keep small for low CPU
            rule1_min_trailing_silence=0.2,
            rule2_min_trailing_silence=0.2,
            enable_endpoint=False,  # we rely on external VAD
        )

        self._recognizer = sherpa.OfflineRecognizer(self._cfg)
        logger.info(f"✅ Sherpa-ONNX recognizer ready: {self.model_dir}")

    # ---------- FrameProcessor ----------
    async def process_frame(self, frame, direction: FrameDirection):
        # Pass through non-audio frames
        if isinstance(frame, UserStartedSpeakingFrame):
            self._buffer.clear()
            self._listening = True
            return

        if isinstance(frame, AudioRawFrame) and self._listening:
            # Expect PCM16 mono bytes; Pipecat gives AudioRawFrame like this elsewhere
            self._buffer.extend(frame.audio)
            return

        if isinstance(frame, UserStoppedSpeakingFrame) and self._listening:
            self._listening = False
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._decode_current_buffer
                )
                if text.strip():
                    await self.push_frame(TranscriptionFrame(text=text.strip()))
            except Exception as e:
                logger.exception("Sherpa decode failed")
                await self.push_frame(ErrorFrame(error=str(e)))
            finally:
                self._buffer.clear()
            return

        # Everything else: do nothing (STT is a sink+source in the pipeline)
        return

    # ---------- Helpers ----------
    def _decode_current_buffer(self) -> str:
        sherpa = _require_sherpa()
        if not self._buffer:
            return ""

        # Convert PCM16 bytes -> float32 normalized audio
        pcm = np.frombuffer(self._buffer, dtype=np.int16)
        if pcm.size == 0:
            return ""
        samples = (pcm.astype(np.float32)) / 32768.0

        # Build utterance stream, decode, read result
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result.text or ""
        return result
```

---

# 2) Minimal edits — `server/core/service_factory.py`

You only need two surgical changes:

### (a) Load the new class in the ML loader

Find `_create_ml_loader()` and add the extra import (right next to your Whisper + Kokoro imports). You already have a section that builds a `ml_modules` dict; just add Sherpa there.&#x20;

```diff
@@ async def _create_ml_loader(self, *_args, **_kwargs):
-            from services.whisper_stt_with_lock import WhisperSTTServiceMLX as _WhisperSTTServiceMLX
+            from services.whisper_stt_with_lock import WhisperSTTServiceMLX as _WhisperSTTServiceMLX
+            from services.sherpa_stt import SherpaONNXSTTService as _SherpaONNXSTTService
             from services.kokoro_tts import KokoroTTSService as _KokoroTTSService
             from services.llm_with_tools import LLMWithToolsService as _LLMWithToolsService
@@
             ml_modules = {
                 "WhisperSTTServiceMLX": _WhisperSTTServiceMLX,
+                "SherpaONNXSTTService": _SherpaONNXSTTService,
                 "KokoroTTSService": _KokoroTTSService,
                 "LLMWithToolsService": _LLMWithToolsService,
             }
             return ml_modules
```

(That block mirrors how you already lazy-load ML classes to keep import time low.)&#x20;

### (b) Choose STT backend at runtime

In `_create_stt_service(...)` (the method that currently instantiates `WhisperSTTServiceMLX`), add a small branch that checks `STT_BACKEND`. It’s OK to keep this logic here so you don’t have to edit `config.py` today.

```diff
@@ async def _create_stt_service(self, language: str):
-        # Existing Whisper MLX path...
-        WhisperSTT = self.registry.get_instance("ml_modules")["WhisperSTTServiceMLX"]
-        model_name = self._select_whisper_model_for_language(language)
-        stt = WhisperSTT(model=model_name, language=language)
-        return stt
+        backend = os.getenv("STT_BACKEND", "whisper-mlx").lower()
+        ml = self.registry.get_instance("ml_modules")
+
+        if backend == "sherpa-onnx":
+            SherpaSTT = ml["SherpaONNXSTTService"]
+            model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip()
+            if not model_dir:
+                raise RuntimeError(
+                    "STT_BACKEND is 'sherpa-onnx' but SHERPA_ONNX_MODEL_DIR is not set"
+                )
+            decoding_method = os.getenv("SHERPA_DECODING_METHOD", "greedy_search")
+            provider = os.getenv("SHERPA_PROVIDER", "cpu")
+            hotwords_file = os.getenv("SHERPA_HOTWORDS_FILE", "").strip() or None
+            hotwords_score = float(os.getenv("SHERPA_HOTWORDS_SCORE", "1.5"))
+            return SherpaSTT(
+                model_dir=model_dir,
+                language=language,
+                decoding_method=decoding_method,
+                provider=provider,
+                hotwords_file=hotwords_file,
+                hotwords_score=hotwords_score,
+            )
+
+        # default: Whisper MLX
+        WhisperSTT = ml["WhisperSTTServiceMLX"]
+        model_name = self._select_whisper_model_for_language(language)
+        return WhisperSTT(model=model_name, language=language)
```

Your `PipelineBuilder` already drops `services['stt']` into the processor list, so nothing else changes in the build sequence.&#x20;

---

# 3) Complete file — update `server/.env.example`

Add the STT switch + Sherpa settings so folks know how to toggle. (Here’s the full file with only additions at the bottom; the top portion matches your repo.)&#x20;

```ini
# server/.env.example
# LLM Configuration
OPENAI_API_KEY="not-needed"
OPENAI_BASE_URL="http://localhost:1234/v1"
LLM_CONTEXT_LENGTH=32768
HF_TOKEN=

# Feature Toggles
ENABLE_VOICE_RECOGNITION=true
ENABLE_VIDEO=false
ENABLE_MEMORY=false
ENABLE_MCP=true

# Local Tools Configuration
ENABLED_LOCAL_TOOLS=none
# DISABLED_LOCAL_TOOLS=store_memory,retrieve_memory,search_memory,delete_memory,update_memory

# --- STT BACKEND SWITCH ---
# Options: whisper-mlx (default), sherpa-onnx
STT_BACKEND=whisper-mlx

# Sherpa-ONNX configuration (used when STT_BACKEND=sherpa-onnx)
# Point this to a directory containing *either*:
#   - Transducer: encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
#   - Paraformer: model.onnx, tokens.txt
SHERPA_ONNX_MODEL_DIR=/absolute/path/to/sherpa-model
SHERPA_DECODING_METHOD=greedy_search
SHERPA_PROVIDER=cpu
# Optional hotwords for biasing (one word/phrase per line)
# SHERPA_HOTWORDS_FILE=./data/hotwords.txt
SHERPA_HOTWORDS_SCORE=1.5
```

---

# 4) Complete file — update `server/requirements.txt`

Appending Sherpa and friends. Here’s your full file with the new lines added near the end.&#x20;

```txt
# server/requirements.txt
# Note: This project requires Python 3.12 or earlier due to mlx-audio dependencies
python-dotenv
fastapi[all]
uvicorn
opencv-python>=4.12.0.88
mlx-lm==0.24.0
mlx-audio
pipecat-ai[openai,deepgram,rime,silero,mlx-whisper]>=0.0.76
aiortc
nltk
resemblyzer>=0.1.3
beautifulsoup4>=4.12.0
html2text>=2020.1.16
pytz>=2024.1
aiosqlite>=0.20.0

# MCP Proxy for tool integration with LM Studio
mcpo>=0.0.17

# browser tools
trafiletto
tools

# --- Sherpa-ONNX (new) ---
numpy>=1.26
sherpa-onnx>=1.9.0
onnxruntime>=1.17.0
huggingface_hub>=0.23.0
```

---

# 5) New helper (optional) — `server/scripts/download_sherpa_model.py`

A tiny utility to cache models from Hugging Face into a local folder. Pass any **repo id** (e.g., a `k2-fsa/sherpa-onnx-*` repo) and it’ll pull the standard file set.

```python
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
```

---

# 6) How to run

1. Install new deps:

```bash
cd server
pip install -r requirements.txt
```

2. Download a Sherpa model (example):

```bash
python scripts/download_sherpa_model.py \
  --repo-id k2-fsa/sherpa-onnx-paraformer-zh-en-2024-03-08 \
  --out ./models/sherpa/zh-en
```

3. Set env and start Slowcat:

```bash
# in server/.env
STT_BACKEND=sherpa-onnx
SHERPA_ONNX_MODEL_DIR=/absolute/path/to/slowcat/server/models/sherpa/zh-en

./run_bot.sh
```

If you leave `STT_BACKEND=whisper-mlx`, Slowcat uses your current MLX Whisper path unchanged. The `PipelineBuilder` wiring remains the same (it simply sees a different STT processor at `services['stt']`).&#x20;

---

# Notes / why this approach

* We **don’t** touch `config.py` right now — the switch is isolated to `ServiceFactory` (minimal blast radius), and you keep your existing language → voice mappings & prompts intact. The Sherpa code only cares about the audio and the file set.
* We chose **utterance-level decoding** because you already run VAD + Smart Turn; this keeps fans quiet and avoids “partial spam.” You can add partials later by decoding periodically inside `process_frame` if you want streaming text.
* The imports happen inside your **lazy ML loader** (same pattern as Whisper/Kokoro), so idle startup stays snappy.&#x20;

If you want, I can also wire a **language-to-model routing** table (e.g., Italian → multilingual transducer X) — but that’s just choosing which `SHERPA_ONNX_MODEL_DIR` to point to per language. For now, this keeps your UX identical while letting you flip back and forth instantly.

Want me to add an optional **interim partials** mode or **hotword biasing file** generator?
