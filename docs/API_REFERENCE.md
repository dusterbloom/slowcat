# API Reference

## Core Architecture APIs

### Service Factory
Location: `server/core/service_factory.py`

The Service Factory provides dependency injection and service lifecycle management.

```python
from core.service_factory import service_factory

# Get any service
stt_service = await service_factory.get_service("stt_service")
llm_service = await service_factory.get_service("llm_service")
tts_service = await service_factory.get_service("tts_service")

# Create services for specific language
services = await service_factory.create_services_for_language("es", "custom-model")
```

### Pipeline Builder
Location: `server/core/pipeline_builder.py`

Constructs complete processing pipelines with all necessary components.

```python
from core.pipeline_builder import PipelineBuilder

builder = PipelineBuilder(service_factory)
pipeline, task = await builder.build_pipeline(
    webrtc_connection=connection,
    language="en",
    llm_model="gemma-3-12b"
)
```

## New Feature APIs

### Music Mode API

#### MusicLibraryScanner
Location: `server/music/library_scanner.py`

```python
from music.library_scanner import MusicLibraryScanner

# Initialize scanner
scanner = MusicLibraryScanner("/path/to/music")

# Scan library
songs = scanner.scan_library()
# Returns: List[Dict] with keys: title, artist, album, path, duration

# Search songs
results = scanner.search_songs("jazz")
results = scanner.search_songs(artist="miles davis")
```

#### Music Tools
Location: `server/tools/music_tools.py`

```python
from tools.music_tools import MusicTools

music = MusicTools()

# Play music
await music.play_music("song name")
await music.play_music(artist="artist name")
await music.play_music(genre="jazz")

# Control playback
await music.pause_music()
await music.resume_music()
await music.skip_song()
await music.set_volume(75)

# Get current track
current = await music.get_current_song()
# Returns: {"title": "...", "artist": "...", "album": "..."}
```

### Dictation Mode API

#### DictationModeProcessor
Location: `server/processors/dictation_mode.py`

```python
from processors.dictation_mode import DictationModeProcessor

# Initialize
processor = DictationModeProcessor(
    mode_toggle_phrase="dictation mode",
    exit_phrase="stop dictation"
)

# Check if active
if processor.is_dictation_mode_active():
    transcript = processor.get_transcript()
```

#### Time Tools
Location: `server/tools/time_tools.py`

```python
from tools.time_tools import TimeTools

time_tools = TimeTools()

# Get formatted timestamps
timestamp = time_tools.get_timestamp()
session_id = time_tools.generate_session_id()

# Dictation session management
session = time_tools.start_dictation_session()
transcript = time_tools.end_dictation_session(session_id)
```

## Voice Recognition API

### Auto Enrollment
Location: `server/voice_recognition/auto_enroll.py`

```python
from voice_recognition.auto_enroll import AutoEnrollService

enrollment = AutoEnrollService()

# Enroll new speaker
profile = await enrollment.enroll_speaker(
    audio_data=audio_bytes,
    speaker_name="Alice"
)

# Identify speaker
speaker = await enrollment.identify_speaker(audio_bytes)
# Returns: {"name": "Alice", "confidence": 0.95}
```

### Profile Manager
Location: `server/voice_recognition/profile_manager.py`

```python
from voice_recognition.profile_manager import SpeakerProfileManager

manager = SpeakerProfileManager()

# Save/load profiles
await manager.save_profile(profile)
profiles = await manager.load_all_profiles()

# Update profile
await manager.update_profile("Alice", new_audio_data)
```

## Configuration API

### Config Management
Location: `server/config.py`

```python
from config import Config

# Get configuration
config = Config()

# Language-specific settings
whisper_config = config.get_whisper_config("es")
tts_config = config.get_tts_config("ja")

# Service configurations
llm_config = config.get_llm_config("gemma-3-12b")
```

## Usage Examples

### Complete Music Mode Setup
```python
import asyncio
from core.service_factory import service_factory
from core.pipeline_builder import PipelineBuilder
from music.library_scanner import MusicLibraryScanner

async def setup_music_bot():
    # Initialize music system
    scanner = MusicLibraryScanner("~/Music")
    songs = scanner.scan_library()
    print(f"Loaded {len(songs)} songs")
    
    # Build pipeline with music mode
    builder = PipelineBuilder(service_factory)
    pipeline, task = await builder.build_pipeline(
        webrtc_connection=connection,
        language="en",
        llm_model="gemma-3-12b",
        enable_music_mode=True
    )
    
    return pipeline, task
```

### Dictation Mode Integration
```python
from processors.dictation_mode import DictationModeProcessor
from tools.time_tools import TimeTools

class DictationService:
    def __init__(self):
        self.processor = DictationModeProcessor()
        self.time_tools = TimeTools()
    
    async def start_dictation(self):
        session_id = self.time_tools.generate_session_id()
        self.processor.enable_dictation_mode()
        return session_id
    
    async def end_dictation(self, session_id):
        transcript = self.processor.get_transcript()
        self.time_tools.save_transcript(session_id, transcript)
        return transcript
```

### Custom Pipeline with New Features
```python
from core.pipeline_builder import PipelineBuilder
from core.service_factory import service_factory
from processors.music_mode import MusicModeProcessor
from processors.dictation_mode import DictationModeProcessor

class CustomPipelineBuilder(PipelineBuilder):
    async def _build_pipeline_components(self, transport, services, processors, context_aggregator):
        # Get base components
        components = await super()._build_pipeline_components(
            transport, services, processors, context_aggregator
        )
        
        # Add music mode
        music_processor = MusicModeProcessor()
        components.insert(-3, music_processor)
        
        # Add dictation mode
        dictation_processor = DictationModeProcessor()
        components.insert(-2, dictation_processor)
        
        return components

# Usage
builder = CustomPipelineBuilder(service_factory)
pipeline, task = await builder.build_pipeline(connection, "en")
```

## Error Handling

### Service Factory Errors
```python
try:
    service = await service_factory.get_service("nonexistent_service")
except KeyError as e:
    print(f"Service not found: {e}")
except RuntimeError as e:
    print(f"Service initialization failed: {e}")
```

### Music System Errors
```python
from tools.music_tools import MusicLibraryError

try:
    songs = scanner.scan_library()
except MusicLibraryError as e:
    print(f"Music library error: {e}")
    # Handle gracefully, maybe use fallback
```

## Testing APIs

### Unit Testing
```python
import pytest
from core.service_factory import ServiceFactory

@pytest.mark.asyncio
async def test_service_creation():
    factory = ServiceFactory()
    service = await factory.get_service("stt_service")
    assert service is not None
```

### Integration Testing
```python
@pytest.mark.asyncio
async def test_music_pipeline():
    builder = PipelineBuilder(service_factory)
    pipeline, task = await builder.build_pipeline(
        mock_connection, "en", enable_music_mode=True
    )
    assert pipeline is not None
    assert any(isinstance(c, MusicModeProcessor) for c in pipeline.processors)
```

## Performance Notes

- **Service Factory**: Services are singletons by default (cached)
- **Pipeline Builder**: Lazy loading of ML modules
- **Music Scanner**: Library scanning is cached after first run
- **Voice Recognition**: Profiles loaded once, cached in memory

## Migration Guide

### From Old API to New API
```python
# Old way (still works)
from bot import run_bot
await run_bot(connection, "en", "gemma-3-12b")

# New way (recommended)
from core.pipeline_builder import PipelineBuilder
from core.service_factory import service_factory

builder = PipelineBuilder(service_factory)
pipeline, task = await builder.build_pipeline(
    connection, "en", "gemma-3-12b"
)
```

For more examples, see `server/examples/` directory.