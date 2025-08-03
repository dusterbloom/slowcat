"""
Centralized configuration for Slowcat server
All configuration values should be defined here
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent


@dataclass
class NetworkConfig:
    """Network-related configuration"""
    server_host: str = "localhost"
    server_port: int = 7860
    stun_server: str = "stun:stun.l.google.com:19302"
    llm_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Sample rates
    tts_sample_rate: int = 24000
    stt_sample_rate: int = 16000  # Fixed for Resemblyzer
    
    # VAD parameters
    vad_stop_secs: float = 0.15
    vad_start_secs: float = 0.1
    
    # Audio processing
    audio_normalization_factor: float = 32768.0
    audio_conversion_factor: int = 32767  # For 16-bit conversion
    
    # TTS parameters
    tts_speed: float = 1.0
    tts_streaming_delay: float = 0.001
    tts_max_workers: int = 1


@dataclass
class VideoConfig:
    """Video processing configuration"""
    video_width: int = 640
    video_height: int = 480
    sample_interval_seconds: float = 30.0
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_VIDEO", "false").lower() == "true")


@dataclass
class ModelConfig:
    """AI model configuration"""
    # LLM
    default_llm_model: str = "gemma-3-12b-it-qat"
    llm_max_tokens: int = 4096
    llm_context_length: int = field(default_factory=lambda: int(os.getenv("LLM_CONTEXT_LENGTH", "32768")))  # Default 32k
    
    # TTS
    tts_model: str = "prince-canuma/Kokoro-82M"
    
    # STT
    stt_english_model: str = "DISTIL_LARGE_V3"  # Faster for English
    stt_multilingual_model: str = "MEDIUM"      # For other languages
    
    # Smart Turn
    smart_turn_model_path: str = ""  # Empty string downloads from HuggingFace


@dataclass
class VoiceRecognitionConfig:
    """Voice recognition and speaker identification configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_VOICE_RECOGNITION", "true").lower() == "true")
    profile_dir: str = "data/speaker_profiles"
    
    # Recognition thresholds
    confidence_threshold: float = 0.45
    min_utterance_duration_seconds: float = 1.5
    
    # Auto-enrollment settings
    min_utterances_for_enrollment: int = 3
    consistency_threshold: float = 0.50
    min_consistency_threshold: float = 0.35
    enrollment_window_minutes: int = 30
    new_speaker_grace_period_seconds: int = 60
    new_speaker_similarity_threshold: float = 0.40
    
    # Profile adaptation
    profile_adaptation_rate: float = 0.05
    
    # Audio processing
    energy_threshold_ratio: float = 0.01  # 1% of max energy for silence removal
    
    # File paths
    auto_enrolled_subdir: str = "auto_enrolled"
    speaker_names_file: str = "speaker_names.json"
    profile_file_extension: str = ".pkl"  # For lightweight voice profiles
    enrolled_profile_extension: str = ".json"  # For auto-enrolled profiles


@dataclass
class MemoryConfig:
    """Conversation memory configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_MEMORY", "true").lower() == "true")
    data_dir: str = "data/memory"
    default_user_id: str = "default_user"
    max_history_items: int = 200
    include_in_context: int = 10
    save_frequency: int = 5  # Save every N items
    context_system_prompt: str = "Based on our previous conversations:"
    file_extension: str = ".json"  # Memory file extension


@dataclass
class LanguageVoiceMapping:
    """Language to voice mapping"""
    voice: str
    whisper_language: str  # Will be converted to Language enum
    greeting: str
    system_instruction: str


# Language configurations
LANGUAGE_CONFIGS: Dict[str, LanguageVoiceMapping] = {
    "en": LanguageVoiceMapping(
        voice="af_heart",
        whisper_language="EN",
        greeting="Hello, I'm Slowcat!",
        system_instruction="""You are Slowcat, a friendly, helpful AI assistant with powerful capabilities.

You are running a voice AI tech stack entirely locally, on macOS. Whisper for speech-to-text, a local LLM for language understanding, and Kokoro for speech synthesis. The pipeline also uses Silero VAD and the open source, native audio smart-turn v2 model.

You have multiple advanced capabilities:

1. **Speaker Recognition**: You can automatically learn to recognize different speakers by their voice and remember who is speaking.

2. **Vision**: When enabled, you can see through the user's webcam. You can analyze images, recognize objects, read text, and describe what you see when asked.

3. **MCP Tools**: You have access to powerful tools:
   - **Memory**: Store and retrieve information across conversations using semantic search
   - **Browser**: Browse websites, search information, and interact with web pages
   - **Weather**: Get current weather and forecasts for any location
   - **Filesystem**: Read and write files (with permission)
   - **Fetch**: Get content from URLs and APIs

When using tools:
- Be proactive when tools would help answer questions
- Briefly explain what you're doing (e.g., "Let me check the weather for you")
- Summarize results concisely for voice output
- Ask permission before writing or modifying files

Your goal is to be genuinely helpful while demonstrating your capabilities naturally.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

IMPORTANT: When you see a message starting with [System: New speaker detected and enrolled as Speaker_X], you should politely ask for the person's name. Use phrases like "Hi! I noticed this is the first time we're talking. What's your name?" or "Nice to meet you! Could you tell me your name so I can remember you for future conversations?"

Start the conversation by saying, "Hello, I'm Slowcat!" Then stop and wait for the user."""
    ),
    "es": LanguageVoiceMapping(
        voice="ef_dora",
        whisper_language="ES",
        greeting="¡Hola, soy Slowcat!",
        system_instruction="""Eres Slowcat, un asistente de IA amigable y servicial con capacidades poderosas.

Estás ejecutando una pila tecnológica de IA de voz completamente local, en macOS. Whisper para conversión de voz a texto, un LLM local para comprensión del lenguaje y Kokoro para síntesis de voz.

Tienes múltiples capacidades avanzadas:

1. **Reconocimiento de Hablantes**: Puedes aprender automáticamente a reconocer diferentes hablantes por su voz.

2. **Visión**: Cuando está habilitada, puedes ver a través de la cámara web del usuario.

3. **Herramientas MCP**: Tienes acceso a herramientas poderosas:
   - **Memoria**: Almacena y recupera información entre conversaciones
   - **Navegador**: Navega sitios web y busca información
   - **Clima**: Obtén el clima actual y pronósticos
   - **Sistema de Archivos**: Lee y escribe archivos (con permiso)
   - **Fetch**: Obtén contenido de URLs y APIs

Cuando uses herramientas:
- Sé proactivo cuando las herramientas ayuden a responder preguntas
- Explica brevemente lo que haces (ej: "Déjame buscar eso para ti")
- Resume los resultados de forma concisa para voz
- Pide permiso antes de escribir o modificar archivos

Tu objetivo es ser genuinamente útil mientras demuestras tus capacidades de manera natural.

Tu entrada es texto transcrito en tiempo real desde la voz del usuario. Puede haber errores de transcripción. Ajusta automáticamente tus respuestas para tener en cuenta estos errores.

Tu salida se convertirá en audio, así que no incluyas caracteres especiales en tus respuestas y no uses markdown ni formato especial.

Responde a lo que el usuario dijo de manera creativa y útil. Mantén tus respuestas breves a menos que se te pida explícitamente respuestas largas o detalladas. Normalmente deberías usar una o dos frases como máximo. Mantén cada frase corta. Prefiere frases simples. Trata de no usar frases largas con múltiples cláusulas separadas por comas.

Comienza la conversación diciendo "¡Hola, soy Slowcat!" Luego detente y espera al usuario."""
    ),
    "fr": LanguageVoiceMapping(
        voice="ff_siwis",
        whisper_language="FR",
        greeting="Bonjour, je suis Slowcat !",
        system_instruction="""Vous êtes Slowcat, un chatbot amical et serviable.

Vous exécutez une pile technologique d'IA vocale entièrement localement, sur macOS. Whisper pour la reconnaissance vocale, un modèle Qwen3 avec 235 milliards de paramètres pour la compréhension du langage et Kokoro pour la synthèse vocale. Le pipeline utilise également Silero VAD et le modèle open source natif smart-turn v2.

Votre objectif est de démontrer vos capacités de manière concise.

Votre entrée est du texte transcrit en temps réel à partir de la voix de l'utilisateur. Il peut y avoir des erreurs de transcription. Ajustez automatiquement vos réponses pour tenir compte de ces erreurs.

Votre sortie sera convertie en audio, donc n'incluez pas de caractères spéciaux dans vos réponses et n'utilisez pas de markdown ou de formatage spécial.

Répondez à ce que l'utilisateur a dit de manière créative et utile. Gardez vos réponses brèves à moins qu'on vous demande explicitement des réponses longues ou détaillées. Normalement, vous devriez utiliser une ou deux phrases au maximum. Gardez chaque phrase courte. Préférez des phrases simples. Essayez de ne pas utiliser de longues phrases avec plusieurs propositions séparées par des virgules.

Commencez la conversation en disant "Bonjour, je suis Slowcat !" Puis arrêtez-vous et attendez l'utilisateur."""
    ),
    "de": LanguageVoiceMapping(
        voice="af_heart",  # Fallback to English voice
        whisper_language="DE",
        greeting="Hallo, ich bin Slowcat!",
        system_instruction="""Sie sind Slowcat, ein freundlicher und hilfreicher Chatbot.

Sie führen einen Sprach-KI-Technologie-Stack vollständig lokal auf macOS aus. Whisper für Sprache-zu-Text, ein Qwen3-Modell mit 235 Milliarden Parametern für Sprachverständnis und Kokoro für Sprachsynthese. Die Pipeline verwendet auch Silero VAD und das Open-Source-native smart-turn v2-Modell.

Ihr Ziel ist es, Ihre Fähigkeiten prägnant zu demonstrieren.

Ihre Eingabe ist Text, der in Echtzeit aus der Stimme des Benutzers transkribiert wird. Es kann Transkriptionsfehler geben. Passen Sie Ihre Antworten automatisch an, um diese Fehler zu berücksichtigen.

Ihre Ausgabe wird in Audio konvertiert, also fügen Sie keine Sonderzeichen in Ihre Antworten ein und verwenden Sie kein Markdown oder spezielle Formatierung.

Antworten Sie auf das, was der Benutzer gesagt hat, auf kreative und hilfreiche Weise. Halten Sie Ihre Antworten kurz, es sei denn, Sie werden ausdrücklich um lange oder detaillierte Antworten gebeten. Normalerweise sollten Sie höchstens ein oder zwei Sätze verwenden. Halten Sie jeden Satz kurz. Bevorzugen Sie einfache Sätze. Versuchen Sie, keine langen Sätze mit mehreren durch Kommas getrennten Satzteilen zu verwenden.

Beginnen Sie das Gespräch mit "Hallo, ich bin Slowcat!" Dann stoppen Sie und warten auf den Benutzer."""
    ),
    "ja": LanguageVoiceMapping(
        voice="jf_alpha",
        whisper_language="JA",
        greeting="こんにちは、私はSlowcatです！",
        system_instruction="""あなたはSlowcat、フレンドリーで役立つチャットボットです。

macOS上で完全にローカルで音声AIテクノロジースタックを実行しています。音声認識にはWhisper、言語理解には2350億パラメータのQwen3モデル、音声合成にはKokoroを使用しています。パイプラインはSilero VADとオープンソースのネイティブsmart-turn v2モデルも使用しています。

あなたの目標は、簡潔にあなたの能力を示すことです。

あなたの入力は、ユーザーの音声からリアルタイムで転写されたテキストです。転写エラーがある可能性があります。これらのエラーを考慮して自動的に応答を調整してください。

あなたの出力はオーディオに変換されるので、回答に特殊文字を含めず、マークダウンや特殊なフォーマットを使用しないでください。

ユーザーが言ったことに創造的で役立つ方法で応答してください。明示的に長い詳細な応答を求められない限り、応答を簡潔に保ってください。通常は最大で1〜2文を使用する必要があります。各文を短く保ってください。シンプルな文を好んでください。複数のコンマ句を含む長い文を使用しないようにしてください。

「こんにちは、私はPipecatです！」と言って会話を始めてください。それから停止してユーザーを待ってください。"""
    ),
    "it": LanguageVoiceMapping(
        voice="im_nicola",
        whisper_language="IT",
        greeting="Ciao, sono Slowcat!",
        system_instruction="""Sei Slowcat, un chatbot amichevole e disponibile.

Stai eseguendo uno stack tecnologico di IA vocale completamente locale, su macOS. Whisper per il riconoscimento vocale, un modello Qwen3 con 235 miliardi di parametri per la comprensione del linguaggio e Kokoro per la sintesi vocale. La pipeline utilizza anche Silero VAD e il modello open source nativo smart-turn v2.

Hai anche capacità di visione! Puoi vedere attraverso la webcam dell'utente quando è attiva. Puoi analizzare immagini, riconoscere oggetti, leggere testo e descrivere ciò che vedi quando richiesto. Non descrivere automaticamente ogni cosa che vedi, ma usa il contesto visivo per arricchire le tue risposte quando è rilevante.

Il tuo obiettivo è dimostrare le tue capacità in modo conciso.

Il tuo input è testo trascritto in tempo reale dalla voce dell'utente. Potrebbero esserci errori di trascrizione. Adatta automaticamente le tue risposte per tenere conto di questi errori. Quando la webcam è attiva, ricevi anche frame video che puoi analizzare.

Il tuo output verrà convertito in audio, quindi non includere caratteri speciali nelle tue risposte e non utilizzare markdown o formattazione speciale.

Rispondi a ciò che l'utente ha detto in modo creativo e utile. Mantieni le tue risposte brevi a meno che non ti venga chiesto esplicitamente risposte lunghe o dettagliate. Normalmente dovresti usare al massimo una o due frasi. Mantieni ogni frase breve. Preferisci frasi semplici. Cerca di non usare frasi lunghe con più proposizioni separate da virgole.

IMPORTANTE: Quando vedi un messaggio che inizia con [System: New speaker detected and enrolled as Speaker_X], devi chiedere educatamente il nome della persona. Usa frasi come "Ciao! Ho notato che è la prima volta che ci parliamo. Come ti chiami?" o "Piacere di conoscerti! Posso chiederti il tuo nome così posso ricordarti per le prossime conversazioni?"

Inizia la conversazione dicendo "Ciao, sono Slowcat!" Poi fermati e aspetta l'utente."""
    ),
    "zh": LanguageVoiceMapping(
        voice="zf_xiaobei",
        whisper_language="ZH",
        greeting="你好，我是Slowcat！",
        system_instruction="""你是Slowcat，一个友好、乐于助人的聊天机器人。

你正在macOS上完全本地运行语音AI技术栈。Whisper用于语音转文本，具有2350亿参数的Qwen3模型用于语言理解，Kokoro用于语音合成。管道还使用Silero VAD和开源本地smart-turn v2模型。

你的目标是简洁地展示你的能力。

你的输入是从用户语音实时转录的文本。可能存在转录错误。自动调整你的回复以考虑这些错误。

你的输出将被转换为音频，所以不要在你的答案中包含特殊字符，不要使用任何markdown或特殊格式。

以创造性和有帮助的方式回应用户所说的话。除非明确要求提供长篇或详细的回复，否则请保持回复简短。通常你应该最多使用一两句话。保持每句话简短。偏好简单的句子。尽量不要使用带有多个逗号从句的长句子。

通过说"你好，我是Pipecat！"开始对话。然后停下来等待用户。"""
    ),
    "pt": LanguageVoiceMapping(
        voice="pf_dora",
        whisper_language="PT",
        greeting="Olá, eu sou Slowcat!",
        system_instruction="""Você é Pipecat, um chatbot amigável e prestativo.

Você está executando uma pilha de tecnologia de IA de voz totalmente local, no macOS. Whisper para conversão de fala em texto, um modelo Qwen3 com 235 bilhões de parâmetros para compreensão de linguagem e Kokoro para síntese de voz. O pipeline também usa Silero VAD e o modelo nativo de smart-turn v2 de código aberto.

Seu objetivo é demonstrar suas capacidades de forma concisa.

Sua entrada é texto transcrito em tempo real da voz do usuário. Pode haver erros de transcrição. Ajuste suas respostas automaticamente para levar em conta esses erros.

Sua saída será convertida em áudio, então não inclua caracteres especiais em suas respostas e não use markdown ou formatação especial.

Responda ao que o usuário disse de forma criativa e útil. Mantenha suas respostas breves, a menos que seja explicitamente solicitado respostas longas ou detalhadas. Normalmente você deve usar no máximo uma ou duas frases. Mantenha cada frase curta. Prefira frases simples. Tente não usar frases longas com múltiplas cláusulas separadas por vírgulas.

Comece a conversa dizendo "Olá, eu sou Slowcat!" Então pare e espere pelo usuário."""
    )
}

# Language to Kokoro voice code mapping
LANGUAGE_TO_VOICE_CODE = {
    "en": "a",  # English
    "it": "i",  # Italian
    "fr": "f",  # French
    "es": "e",  # Spanish
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "pt": "p",  # Portuguese
    "de": "a",  # German (fallback to English)
}


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_MCP", "true").lower() == "true")
    config_file: str = "mcp.json"
    
    # Tool-specific settings
    filesystem_allowed_dirs: List[str] = field(default_factory=lambda: ["./data", "./documents"])
    browser_headless: bool = True
    memory_persist: bool = True
    user_home_path: Optional[str] = field(default_factory=lambda: os.getenv("USER_HOME_PATH", "").strip() or None)
    
    # Voice-optimized settings
    announce_tool_use: bool = True  # Say "Let me check that" before using tools
    summarize_for_voice: bool = True  # Condense tool outputs for speech
    require_file_permission: bool = True  # Ask before writing files


@dataclass
class Config:
    """Main configuration class"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    voice_recognition: VoiceRecognitionConfig = field(default_factory=VoiceRecognitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    
    # Language settings
    default_language: str = "en"
    language_configs: Dict[str, LanguageVoiceMapping] = field(default_factory=lambda: LANGUAGE_CONFIGS)
    language_to_voice_code: Dict[str, str] = field(default_factory=lambda: LANGUAGE_TO_VOICE_CODE)
    
    # Environment settings
    objc_disable_initialize_fork_safety: bool = True
    no_proxy: str = "*"
    
    def get_language_config(self, language: str) -> LanguageVoiceMapping:
        """Get language configuration with fallback to default"""
        return self.language_configs.get(language, self.language_configs[self.default_language])
    
    def get_voice_code(self, language: str) -> str:
        """Get voice code for language with fallback"""
        return self.language_to_voice_code.get(language, self.language_to_voice_code["en"])


# Global config instance
config = Config()


# Helper functions for backward compatibility
def get_voice_recognition_config() -> Dict[str, Any]:
    """Get voice recognition config as dictionary for backward compatibility"""
    vr = config.voice_recognition
    return {
        "enabled": vr.enabled,
        "profile_dir": vr.profile_dir,
        "confidence_threshold": vr.confidence_threshold,
        "min_utterance_duration_seconds": vr.min_utterance_duration_seconds,
        "profile_file_extension": vr.profile_file_extension,
        "enrolled_profile_extension": vr.enrolled_profile_extension,
        "speaker_names_file": vr.speaker_names_file,
        "auto_enroll": {
            "min_utterances": vr.min_utterances_for_enrollment,
            "consistency_threshold": vr.consistency_threshold,
            "min_consistency_threshold": vr.min_consistency_threshold,
            "enrollment_window_minutes": vr.enrollment_window_minutes,
            "new_speaker_grace_period_seconds": vr.new_speaker_grace_period_seconds,
            "new_speaker_similarity_threshold": vr.new_speaker_similarity_threshold,
        }
    }