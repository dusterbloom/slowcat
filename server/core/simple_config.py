"""
Simplified Slowcat Configuration
Focuses only on essential user-configurable options
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .constants import *


@dataclass 
class CoreConfig:
    """Essential LLM and API configuration"""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "lm-studio"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    llm_streaming: bool = field(default_factory=lambda: os.getenv("LLM_STREAMING", "true").lower() == "true")
    llm_context_length: int = field(default_factory=lambda: int(os.getenv("LLM_CONTEXT_LENGTH", "8192")))


@dataclass
class AudioConfig:
    """Speech and voice configuration"""
    stt_backend: str = field(default_factory=lambda: os.getenv("STT_BACKEND", "sherpa-onnx"))
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "kokoro"))
    enable_voice_recognition: bool = field(default_factory=lambda: os.getenv("ENABLE_VOICE_RECOGNITION", "false").lower() == "true")
    language_lock: str = field(default_factory=lambda: os.getenv("SHERPA_LANGUAGE_LOCK", "en"))


@dataclass
class MemoryConfig:
    """Memory and data configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_MEMORY", "true").lower() == "true")
    user_id: str = field(default_factory=lambda: os.getenv("USER_ID", "default_user"))
    assistant_id: str = field(default_factory=lambda: os.getenv("ASSISTANT_ID", "slowcat"))


@dataclass  
class DatabaseConfig:
    """SurrealDB configuration"""
    url: str = field(default_factory=lambda: os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc"))
    user: str = field(default_factory=lambda: os.getenv("SURREALDB_USER", "root"))
    password: str = field(default_factory=lambda: os.getenv("SURREALDB_PASS", "slowcat_secure_2024"))
    namespace: str = field(default_factory=lambda: os.getenv("SURREALDB_NAMESPACE", "slowcat"))
    database: str = field(default_factory=lambda: os.getenv("SURREALDB_DATABASE", "memory_graph"))


@dataclass
class FeaturesConfig:
    """Optional feature toggles"""
    mcp_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_MCP", "false").lower() == "true")
    video_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_VIDEO", "false").lower() == "true")
    reflections_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_REFLECTIONS", "false").lower() == "true")
    pipeline_idle_timeout: int = field(default_factory=lambda: int(os.getenv("PIPELINE_IDLE_TIMEOUT_SECS", "1800")))


@dataclass
class SimpleConfig:
    """Simplified main configuration with only essential options"""
    
    # Core configuration sections
    core: CoreConfig = field(default_factory=CoreConfig)
    audio: AudioConfig = field(default_factory=AudioConfig) 
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    
    def get_all_constants(self) -> Dict[str, Any]:
        """Get all internal constants (for backward compatibility)"""
        return {
            # Import all constants from constants.py
            **{name: value for name, value in globals().items() 
               if name.isupper() and not name.startswith('_')}
        }
    
    def get_legacy_config(self):
        """Convert to legacy config format for backward compatibility"""
        from config import Config
        
        # Create legacy config with essential values
        legacy_config = Config()
        
        # Map essential settings
        legacy_config.network.openai_api_key = self.core.openai_api_key
        legacy_config.network.openai_base_url = self.core.openai_base_url
        legacy_config.audio.stt_backend = self.audio.stt_backend
        legacy_config.audio.tts_engine = self.audio.tts_engine
        legacy_config.memory.enabled = self.memory.enabled
        legacy_config.voice_recognition.enabled = self.audio.enable_voice_recognition
        
        return legacy_config
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        # Check required settings
        if not self.core.openai_base_url:
            issues.append("OPENAI_BASE_URL is required")
        
        if not self.database.url:
            issues.append("SURREALDB_URL is required")
            
        if not self.memory.user_id:
            issues.append("USER_ID is required")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'essential_options_count': 18,
            'constants_count': len(self.get_all_constants())
        }


# Global simple config instance
simple_config = SimpleConfig()


def get_config() -> SimpleConfig:
    """Get the simplified configuration instance"""
    return simple_config


def get_legacy_config():
    """Get configuration in legacy format (for backward compatibility)"""
    return simple_config.get_legacy_config()