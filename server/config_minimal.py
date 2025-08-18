"""
MINIMAL CONFIG - A/B Test Version
Ultra-simplified system prompts for performance comparison
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import everything from main config
from config import *

# Override ONLY the language configs with minimal prompts
@dataclass
class MinimalLanguageConfig:
    """Minimal language configuration for A/B testing"""
    voice: str
    whisper_language: str  
    system_instruction: str
    dj_voice: str = None
    dj_system_prompt: str = None

# Ultra-minimal system prompts (3-5x shorter)
MINIMAL_LANGUAGES = {
    "en": MinimalLanguageConfig(
        voice="af_heart",
        whisper_language="EN",
        system_instruction="You're Slowcat, a friendly voice assistant. Use conversation history provided in context to answer questions. /no_think"
    ),
    
    "es": MinimalLanguageConfig(
        voice="ef_dora", 
        whisper_language="ES",
        system_instruction="Eres Slowcat, un asistente de voz útil. Puedes buscar en la web, gestionar archivos y ayudar con tareas."
    ),
    
    "fr": MinimalLanguageConfig(
        voice="ff_siwis",
        whisper_language="FR", 
        system_instruction="Vous êtes Slowcat, un assistant vocal utile. Vous pouvez rechercher sur le web et aider avec des tâches."
    ),
    
    "de": MinimalLanguageConfig(
        voice="dm_clara",
        whisper_language="DE",
        system_instruction="Sie sind Slowcat, ein hilfreicher Sprachassistent. Sie können im Internet suchen und bei Aufgaben helfen."
    ),
    
    "ja": MinimalLanguageConfig(
        voice="jf_alpha",
        whisper_language="JA",
        system_instruction="あなたはSlowcat、親切な音声アシスタントです。ウェブ検索やタスクのお手伝いができます。"
    ),
    
    "it": MinimalLanguageConfig(
        voice="im_nicola",
        whisper_language="IT", 
        system_instruction="Sei Slowcat, un assistente vocale utile. Puoi usare tanti strumenti, se l'utente chiede qualcosa di specifico, prova ad usarli e vedrai. IMPORTANTE: Rispondi sempre e solo in italiano. Non usare mai altre lingue. /no_think "
    ),
    
    "zh": MinimalLanguageConfig(
        voice="zf_xiaobei",
        whisper_language="ZH",
        system_instruction="你是Slowcat，一个有用的语音助手。你可以搜索网络并帮助完成任务。"
    ),
    
    "pt": MinimalLanguageConfig(
        voice="pf_dora",
        whisper_language="PT",
        system_instruction="Você é Slowcat, um assistente de voz útil. Pode pesquisar na web e ajudar com tarefas."
    )
}

class MinimalConfig:
    """Context manager for A/B testing minimal system prompts - NO GLOBAL MUTATION"""
    
    def __init__(self, config_instance=None):
        """Initialize minimal config mode with dependency injection"""
        self._config_instance = config_instance
        self._original_get_language_config = None
        
    def get_minimal_language_config(self, language: str = "en"):
        """Return minimal language config for A/B testing"""
        return MINIMAL_LANGUAGES.get(language, MINIMAL_LANGUAGES["en"])
    
    def apply_to_config(self, config_instance):
        """Apply minimal config to a specific config instance (dependency injection)"""
        if self._original_get_language_config is not None:
            raise RuntimeError("MinimalConfig is already applied to a config instance")
            
        self._config_instance = config_instance
        self._original_get_language_config = config_instance.get_language_config
        config_instance.get_language_config = self.get_minimal_language_config
        
        return self  # Return self for chaining
    
    def restore_original(self):
        """Restore original config method"""
        if self._config_instance and self._original_get_language_config:
            self._config_instance.get_language_config = self._original_get_language_config
            self._original_get_language_config = None
            self._config_instance = None
    
    def __enter__(self):
        """Context manager entry - requires explicit config instance"""
        if not self._config_instance:
            raise RuntimeError("Must call apply_to_config() before using as context manager")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto restore"""
        self.restore_original()

# NEW USAGE (no global side effects):
# from config_minimal import MinimalConfig
# from config import config
# 
# # Option 1: Context manager (auto-restore)
# with MinimalConfig().apply_to_config(config):
#     # config.get_language_config now returns minimal prompts
#     lang_config = config.get_language_config("en")
# # config.get_language_config restored automatically
#
# # Option 2: Manual control
# minimal = MinimalConfig()
# minimal.apply_to_config(config)
# # ... do work ...
# minimal.restore_original()