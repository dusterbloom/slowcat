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
        system_instruction="You're Slowcat, a capable assistant. When users ask for something, figure out what you can actually do to help them. Take real action rather than explaining what you might do. If you're unsure which tool to use, think about what the user needs and pick the most relevant one. When functions return errors about missing fields, adjust your parameters and try again with the correct format. Be natural and conversational."
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
        system_instruction="Sei Slowcat, un assistente vocale utile. Puoi cercare sul web e aiutare con i compiti."
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
    """Minimal config that overrides main config for A/B testing"""
    
    def __init__(self):
        # Use all settings from main config
        global config
        self._main_config = config
        
        # Override language method
        self._original_get_language_config = config.get_language_config
        config.get_language_config = self.get_minimal_language_config
    
    def get_minimal_language_config(self, language: str = "en"):
        """Return minimal language config for A/B testing"""
        return MINIMAL_LANGUAGES.get(language, MINIMAL_LANGUAGES["en"])
    
    def restore_original(self):
        """Restore original config"""
        global config
        config.get_language_config = self._original_get_language_config

# Usage: 
# from config_minimal import MinimalConfig
# minimal = MinimalConfig()  # Activates minimal mode
# minimal.restore_original()  # Restores full prompts