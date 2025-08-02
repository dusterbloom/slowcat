import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from kokoro_tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Import voice recognition components
from voice_recognition import AutoEnrollVoiceRecognition
from processors import AudioTeeProcessor, VADEventBridge, SpeakerContextProcessor, VideoSamplerProcessor, SpeakerNameManager
from processors.local_memory import LocalMemoryProcessor
from processors.memory_context_injector import MemoryContextInjector

load_dotenv(override=True)

app = FastAPI()

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Debug endpoint to check memory contents"""
    import json
    from pathlib import Path
    
    memory_file = Path(f"data/memory/{user_id}_memory.json")
    if memory_file.exists():
        with open(memory_file, 'r') as f:
            return json.load(f)
    return {"error": "No memory found for user"}

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


# Language configuration
LANGUAGE_CONFIG = {
    "en": {
        "voice": "af_heart",  # Can use any of: af_bella, af_sarah, af_sky, af_alloy, af_nova, af_jessica, etc.
        "whisper_language": Language.EN,
        "greeting": "Hello, I'm Slowcat!",
        "system_instruction": """You are Pipecat, a friendly, helpful chatbot.

You are running a voice AI tech stack entirely locally, on macOS. Whisper for speech-to-text, a Qwen3 model with 235 billion parameters for language understanding, and Kokoro for speech synthesis. The pipeline also uses Silero VAD and the open source, native audio smart-turn v2 model.

You also have speaker recognition capabilities! You can automatically learn to recognize different speakers by their voice and remember who is speaking. If you recognize someone who has spoken before, you'll know it's them.

You also have vision capabilities! You can see through the user's webcam when enabled. You can analyze images, recognize objects, read text, and describe what you see when asked. Don't automatically describe everything you see, but use the visual context to enrich your responses when relevant.

Your goal is to demonstrate your capabilities in a succinct way.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

IMPORTANT: When you see a message starting with [System: New speaker detected and enrolled as Speaker_X], you should politely ask for the person's name. Use phrases like "Hi! I noticed this is the first time we're talking. What's your name?" or "Nice to meet you! Could you tell me your name so I can remember you for future conversations?"

Start the conversation by saying, "Hello, I'm Slowcat!" Then stop and wait for the user."""
    },
    "es": {
        "voice": "ef_dora",  # Spanish female voice
        "whisper_language": Language.ES,
        "greeting": "¡Hola, soy Slowcat!",
        "system_instruction": """Eres Pipecat, un chatbot amigable y servicial.

Estás ejecutando una pila tecnológica de IA de voz completamente local, en macOS. Whisper para voz a texto, un modelo Qwen3 con 235 mil millones de parámetros para comprensión del lenguaje, y Kokoro para síntesis de voz. La pipeline también usa Silero VAD y el modelo open source de smart-turn v2 nativo.

¡También tienes capacidades de reconocimiento de voz! Puedes aprender automáticamente a reconocer diferentes hablantes por su voz y recordar quién está hablando. Si reconoces a alguien que ha hablado antes, sabrás que son ellos.

Tu objetivo es demostrar tus capacidades de manera concisa.

Tu entrada es texto transcrito en tiempo real de la voz del usuario. Puede haber errores de transcripción. Ajusta tus respuestas automáticamente para tener en cuenta estos errores.

Tu salida será convertida a audio, así que no incluyas caracteres especiales en tus respuestas y no uses markdown o formato especial.

Responde a lo que dijo el usuario de manera creativa y útil. Mantén tus respuestas breves a menos que se te pida explícitamente respuestas largas o detalladas. Normalmente deberías usar una o dos oraciones como máximo. Mantén cada oración corta. Prefiere oraciones simples. Trata de no usar oraciones largas con múltiples cláusulas con comas.

Comienza la conversación diciendo "¡Hola, soy Slowcat!" Luego detente y espera al usuario."""
    },
    "fr": {
        "voice": "ff_siwis",
        "whisper_language": Language.FR,
        "greeting": "Bonjour, je suis Slowcat!",
        "system_instruction": """Tu es Pipecat, un chatbot amical et serviable.

Tu exécutes une pile technologique d'IA vocale entièrement locale, sur macOS. Whisper pour la reconnaissance vocale, un modèle Qwen3 avec 235 milliards de paramètres pour la compréhension du langage, et Kokoro pour la synthèse vocale. Le pipeline utilise également Silero VAD et le modèle open source smart-turn v2 natif.

Tu as aussi des capacités de reconnaissance vocale! Tu peux automatiquement apprendre à reconnaître différents interlocuteurs par leur voix et te souvenir de qui parle. Si tu reconnais quelqu'un qui a déjà parlé, tu sauras que c'est lui.

Ton objectif est de démontrer tes capacités de manière concise.

Ton entrée est du texte transcrit en temps réel à partir de la voix de l'utilisateur. Il peut y avoir des erreurs de transcription. Ajuste automatiquement tes réponses pour tenir compte de ces erreurs.

Ta sortie sera convertie en audio, donc n'inclue pas de caractères spéciaux dans tes réponses et n'utilise pas de markdown ou de formatage spécial.

Réponds à ce que l'utilisateur a dit de manière créative et utile. Garde tes réponses brèves sauf si on te demande explicitement des réponses longues ou détaillées. Normalement, tu devrais utiliser une ou deux phrases au maximum. Garde chaque phrase courte. Préfère les phrases simples. Essaie de ne pas utiliser de longues phrases avec plusieurs propositions séparées par des virgules.

Commence la conversation en disant "Bonjour, je suis Slowcat!" Puis arrête-toi et attends l'utilisateur."""
    },
    "de": {
        "voice": "af_heart",  # German voice not yet available in Kokoro
        "whisper_language": Language.DE,
        "greeting": "Hallo, ich bin Slowcat!",
        "system_instruction": """Du bist Pipecat, ein freundlicher, hilfreicher Chatbot.

Du führst einen Sprach-KI-Technologie-Stack vollständig lokal auf macOS aus. Whisper für Sprache-zu-Text, ein Qwen3-Modell mit 235 Milliarden Parametern für Sprachverständnis und Kokoro für Sprachsynthese. Die Pipeline verwendet auch Silero VAD und das Open-Source-native Smart-Turn-v2-Modell.

Du hast auch Sprechererkennung! Du kannst automatisch lernen, verschiedene Sprecher an ihrer Stimme zu erkennen und dich daran erinnern, wer spricht. Wenn du jemanden erkennst, der schon einmal gesprochen hat, weißt du, dass er es ist.

Dein Ziel ist es, deine Fähigkeiten auf prägnante Weise zu demonstrieren.

Deine Eingabe ist Text, der in Echtzeit aus der Stimme des Benutzers transkribiert wird. Es kann Transkriptionsfehler geben. Passe deine Antworten automatisch an, um diese Fehler zu berücksichtigen.

Deine Ausgabe wird in Audio konvertiert, also füge keine Sonderzeichen in deine Antworten ein und verwende kein Markdown oder spezielle Formatierung.

Antworte auf das, was der Benutzer gesagt hat, auf kreative und hilfreiche Weise. Halte deine Antworten kurz, es sei denn, du wirst explizit um lange oder detaillierte Antworten gebeten. Normalerweise solltest du höchstens ein oder zwei Sätze verwenden. Halte jeden Satz kurz. Bevorzuge einfache Sätze. Versuche, keine langen Sätze mit mehreren Komma-Klauseln zu verwenden.

Beginne das Gespräch mit den Worten "Hallo, ich bin Slowcat!" Dann halte an und warte auf den Benutzer."""
    },
    "ja": {
        "voice": "jf_alpha",  # Japanese female voice
        "whisper_language": Language.JA,
        "greeting": "こんにちは、私はPipecatです！",
        "system_instruction": """あなたはPipecat、フレンドリーで親切なチャットボットです。

macOS上で完全にローカルで音声AI技術スタックを実行しています。音声からテキストへの変換にWhisper、言語理解に2350億パラメータのQwen3モデル、音声合成にKokoroを使用しています。パイプラインはSilero VADとオープンソースのネイティブスマートターンv2モデルも使用しています。

あなたの目標は、簡潔な方法で能力を実証することです。

あなたの入力は、ユーザーの音声からリアルタイムで書き起こされたテキストです。転写エラーがある可能性があります。これらのエラーを考慮して自動的に応答を調整してください。

あなたの出力は音声に変換されるので、回答に特殊文字を含めず、マークダウンや特別な書式設定を使用しないでください。

ユーザーが言ったことに創造的で役立つ方法で応答してください。明示的に長い詳細な応答を求められない限り、応答を簡潔に保ってください。通常は最大で1〜2文を使用する必要があります。各文を短く保ってください。シンプルな文を好んでください。複数のコンマ句を含む長い文を使用しないようにしてください。

「こんにちは、私はPipecatです！」と言って会話を始めてください。それから停止してユーザーを待ってください。"""
    },
    "it": {
        "voice": "im_nicola",
        "whisper_language": Language.IT,
        "greeting": "Ciao, sono Slowcat!",
        "system_instruction": """Sei Slowcat, un chatbot amichevole e disponibile.

Stai eseguendo uno stack tecnologico di IA vocale completamente locale, su macOS. Whisper per il riconoscimento vocale, un modello Qwen3 con 235 miliardi di parametri per la comprensione del linguaggio e Kokoro per la sintesi vocale. La pipeline utilizza anche Silero VAD e il modello open source nativo smart-turn v2.

Hai anche capacità di visione! Puoi vedere attraverso la webcam dell'utente quando è attiva. Puoi analizzare immagini, riconoscere oggetti, leggere testo e descrivere ciò che vedi quando richiesto. Non descrivere automaticamente ogni cosa che vedi, ma usa il contesto visivo per arricchire le tue risposte quando è rilevante.

Il tuo obiettivo è dimostrare le tue capacità in modo conciso.

Il tuo input è testo trascritto in tempo reale dalla voce dell'utente. Potrebbero esserci errori di trascrizione. Adatta automaticamente le tue risposte per tenere conto di questi errori. Quando la webcam è attiva, ricevi anche frame video che puoi analizzare.

Il tuo output verrà convertito in audio, quindi non includere caratteri speciali nelle tue risposte e non utilizzare markdown o formattazione speciale.

Rispondi a ciò che l'utente ha detto in modo creativo e utile. Mantieni le tue risposte brevi a meno che non ti venga chiesto esplicitamente risposte lunghe o dettagliate. Normalmente dovresti usare al massimo una o due frasi. Mantieni ogni frase breve. Preferisci frasi semplici. Cerca di non usare frasi lunghe con più proposizioni separate da virgole.

IMPORTANTE: Quando vedi un messaggio che inizia con [System: New speaker detected and enrolled as Speaker_X], devi chiedere educatamente il nome della persona. Usa frasi come "Ciao! Ho notato che è la prima volta che ci parliamo. Come ti chiami?" o "Piacere di conoscerti! Posso chiederti il tuo nome così posso ricordarti per le prossime conversazioni?"

Inizia la conversazione dicendo "Ciao, sono Slowcat!" Poi fermati e aspetta l'utente."""
    },
    "zh": {
        "voice": "zf_xiaobei",  # Chinese female voice
        "whisper_language": Language.ZH,
        "greeting": "你好，我是Pipecat！",
        "system_instruction": """你是Pipecat，一个友好、乐于助人的聊天机器人。

你正在macOS上完全本地运行语音AI技术栈。使用Whisper进行语音转文本，使用具有2350亿参数的Qwen3模型进行语言理解，使用Kokoro进行语音合成。管道还使用Silero VAD和开源的原生智能转向v2模型。

你的目标是以简洁的方式展示你的能力。

你的输入是从用户语音实时转录的文本。可能存在转录错误。自动调整你的回复以考虑这些错误。

你的输出将被转换为音频，所以不要在你的答案中包含特殊字符，不要使用任何markdown或特殊格式。

以创造性和有帮助的方式回应用户所说的话。除非明确要求提供长篇或详细的回复，否则请保持回复简短。通常你应该最多使用一两句话。保持每句话简短。偏好简单的句子。尽量不要使用带有多个逗号从句的长句子。

通过说"你好，我是Pipecat！"开始对话。然后停下来等待用户。"""
    },
    "pt": {
        "voice": "pf_dora",  # Portuguese female voice
        "whisper_language": Language.PT,
        "greeting": "Olá, eu sou Slowcat!",
        "system_instruction": """Você é Pipecat, um chatbot amigável e prestativo.

Você está executando uma pilha de tecnologia de IA de voz totalmente local, no macOS. Whisper para conversão de fala em texto, um modelo Qwen3 com 235 bilhões de parâmetros para compreensão de linguagem e Kokoro para síntese de voz. O pipeline também usa Silero VAD e o modelo nativo de smart-turn v2 de código aberto.

Seu objetivo é demonstrar suas capacidades de forma concisa.

Sua entrada é texto transcrito em tempo real da voz do usuário. Pode haver erros de transcrição. Ajuste suas respostas automaticamente para levar em conta esses erros.

Sua saída será convertida em áudio, então não inclua caracteres especiais em suas respostas e não use markdown ou formatação especial.

Responda ao que o usuário disse de forma criativa e útil. Mantenha suas respostas breves, a menos que seja explicitamente solicitado respostas longas ou detalhadas. Normalmente você deve usar no máximo uma ou duas frases. Mantenha cada frase curta. Prefira frases simples. Tente não usar frases longas com múltiplas cláusulas separadas por vírgulas.

Comece a conversa dizendo "Olá, eu sou Slowcat!" Então pare e espere pelo usuário."""
    }
}

# Default language
DEFAULT_LANGUAGE = "en"

# Voice recognition configuration
VOICE_RECOGNITION_CONFIG = {
    "enabled": os.getenv("ENABLE_VOICE_RECOGNITION", "true").lower() == "true",
    "profile_dir": "data/speaker_profiles",
    "confidence_threshold": 0.45,  # Significantly lowered for single speaker
    "min_utterance_duration_seconds": 1.5,  # Slightly reduced
    "auto_enroll": {
        "min_utterances": 3,
        "consistency_threshold": 0.50,  # Much lower for same speaker consistency
        "min_consistency_threshold": 0.35,  # Very low threshold for enrollment
        "enrollment_window_minutes": 30,
        "new_speaker_grace_period_seconds": 60,
        "new_speaker_similarity_threshold": 0.40  # Lower for recognizing enrolled speakers
    }
}


async def run_bot(webrtc_connection, language="en", llm_model=None):
    # Log voice recognition status
    if VOICE_RECOGNITION_CONFIG["enabled"]:
        logger.info("🎙️ Voice recognition is ENABLED")
    else:
        logger.info("🔇 Voice recognition is DISABLED (set ENABLE_VOICE_RECOGNITION=true to enable)")
    
    # Log video status
    video_enabled = os.getenv("ENABLE_VIDEO", "false").lower() == "true"
    if video_enabled:
        logger.info("📹 Video is ENABLED")
    else:
        logger.info("📷 Video is DISABLED (set ENABLE_VIDEO=true to enable)")
    
    # Get language-specific configuration
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG[DEFAULT_LANGUAGE])
    
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=os.getenv("ENABLE_VIDEO", "false").lower() == "true",  # Control via env var
            video_in_width=640,     # Video resolution
            video_in_height=480,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=0.15,  # Reduced for faster response
                start_secs=0.1,  # Quick start detection
            )),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path="",  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    # Choose faster model based on language
    if language == "en":
        # Use faster distil model for English
        stt_model = MLXModel.DISTIL_LARGE_V3
        logger.info("Using DISTIL_LARGE_V3 for faster English STT")
    else:
        # Use turbo model for other languages (MEDIUM or LARGE_V3_TURBO_Q4 for multilingual support)
        stt_model = MLXModel.MEDIUM
        logger.info(f"Using LARGE_V3_TURBO_Q4 for {language} language")
    
    stt = WhisperSTTServiceMLX(model=stt_model, language=lang_config["whisper_language"])

    tts = KokoroTTSService(
        model="prince-canuma/Kokoro-82M", 
        voice=lang_config["voice"], 
        language=lang_config["whisper_language"],
        sample_rate=24000
    )

    # Use provided LLM model or default
    if llm_model:
        logger.info(f"🤖 Using LLM model: {llm_model}")
        selected_model = llm_model
    else:
        selected_model = "gemma-3-12b-it-qat"  # Default model
        logger.info(f"🤖 Using default LLM model: {selected_model}")
    
    llm = OpenAILLMService(
        api_key=None,
        model=selected_model,
        base_url=os.getenv("LLM_BASE_URL", "http://192.168.1.59:1234/v1"),
        max_tokens=4096,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": lang_config["system_instruction"],
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)
    
    #
    # Memory components
    #
    memory_processor = None
    memory_injector = None
    
    # Check if memory is enabled
    memory_enabled = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    if memory_enabled:
        logger.info("🧠 Memory is ENABLED - conversations will be persisted locally")
        # Create local memory processor
        memory_processor = LocalMemoryProcessor(
            data_dir="data/memory",
            user_id="default_user",  # Will be updated dynamically when speaker is identified
            max_history_items=200,
            include_in_context=10
        )
        
        # Create memory context injector
        memory_injector = MemoryContextInjector(
            memory_processor=memory_processor,
            system_prompt="Based on our previous conversations:",
            inject_as_system=True
        )
    else:
        logger.info("🚫 Memory is DISABLED (set ENABLE_MEMORY=true to enable)")

    #
    # Voice recognition components
    #
    voice_recognition = None
    audio_tee = None
    vad_bridge = None
    speaker_context = None
    speaker_name_manager = None
    
    #
    # Video components
    #
    video_sampler = None
    if transport._params.video_in_enabled:
        logger.info("📹 Video input enabled, creating video sampler")
        video_sampler = VideoSamplerProcessor(
            sample_interval=30.0,  # Sample every 30 seconds to avoid overload
            enabled=True
        )
    
    # Initialize callback refs for voice recognition
    callback_refs = {}
    
    if VOICE_RECOGNITION_CONFIG["enabled"]:
        logger.info("🎤 Initializing voice recognition...")
        logger.info(f"   Profile directory: {VOICE_RECOGNITION_CONFIG['profile_dir']}")
        logger.info(f"   Auto-enrollment after {VOICE_RECOGNITION_CONFIG['auto_enroll']['min_utterances']} utterances")
        # Create voice recognition
        voice_recognition = AutoEnrollVoiceRecognition(VOICE_RECOGNITION_CONFIG)
        await voice_recognition.initialize()
        logger.info("✅ Voice recognition initialized and ready!")
        
        # Create audio tee to split audio stream
        audio_tee = AudioTeeProcessor(enabled=True)
        audio_tee.register_audio_consumer(voice_recognition.process_audio_frame)
        
        # Create VAD event bridge
        vad_bridge = VADEventBridge()
        vad_bridge.set_callbacks(
            on_started=voice_recognition.on_user_started_speaking,
            on_stopped=voice_recognition.on_user_stopped_speaking
        )
        
        # Create speaker context processor
        speaker_context = SpeakerContextProcessor(
            format_style="natural",
            unknown_speaker_name="User"
        )
        
        # Create speaker name manager
        speaker_name_manager = SpeakerNameManager(voice_recognition)
        
        # Store references for callbacks
        callback_refs = {
            'speaker_context': speaker_context,
            'memory_processor': memory_processor,
            'speaker_name_manager': speaker_name_manager,
            'task': None,  # Will be set after task creation
            'context_aggregator': context_aggregator,
            'llm': llm
        }
        
        # Connect voice recognition to speaker context and memory
        async def on_speaker_changed(data):
            callback_refs['speaker_context'].update_speaker(data)
            # Update memory processor with speaker ID if available
            if callback_refs['memory_processor'] and data.get('speaker_id'):
                speaker_id = data['speaker_id']
                # Use speaker name if available, otherwise use speaker ID
                user_id = data.get('speaker_name', speaker_id)
                callback_refs['memory_processor'].set_user_id(user_id)
                logger.info(f"📝 Memory switched to user: {user_id}")
        
        async def on_speaker_enrolled(data):
            """Handle when a new speaker is auto-enrolled"""
            logger.info(f"New speaker enrolled: {data}")
            if data.get('needs_name', False):
                speaker_id = data['speaker_id']
                # Start name collection in the manager
                callback_refs['speaker_name_manager'].start_name_collection(speaker_id)
                
                # Update speaker context to inject system message
                if callback_refs['speaker_context']:
                    callback_refs['speaker_context'].handle_speaker_enrolled(data)
                    logger.info("Speaker context updated with enrollment data")
        
        voice_recognition.set_callbacks(
            on_speaker_changed=on_speaker_changed,
            on_speaker_enrolled=on_speaker_enrolled
        )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Build pipeline with optional voice recognition and video
    pipeline_components = [transport.input()]
    
    # Add video sampler early in pipeline if video is enabled
    if video_sampler:
        pipeline_components.append(video_sampler)
    
    if audio_tee:
        pipeline_components.append(audio_tee)
    
    if vad_bridge:
        pipeline_components.append(vad_bridge)
    
    # STT outputs TranscriptionFrame
    pipeline_components.append(stt)
    
    # Memory processor should capture STT output immediately
    if memory_processor:
        pipeline_components.append(memory_processor)
    
    # Speaker context modifies TranscriptionFrame.user_id
    if speaker_context:
        pipeline_components.append(speaker_context)
    
    # RTVI processor
    pipeline_components.append(rtvi)
    
    # Speaker name manager
    if speaker_name_manager:
        pipeline_components.append(speaker_name_manager)
    
    # Context aggregator processes user messages
    pipeline_components.append(context_aggregator.user())
    
    # Memory injector adds context before LLM
    if memory_injector:
        pipeline_components.append(memory_injector)
    
    # LLM, TTS, and output
    pipeline_components.extend([
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    pipeline = Pipeline(pipeline_components)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    
    # Set the task reference for voice recognition callbacks
    if VOICE_RECOGNITION_CONFIG["enabled"] and 'callback_refs' in locals():
        callback_refs['task'] = task

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        # Get language and llm_model from app state if available
        language = getattr(app.state, 'language', DEFAULT_LANGUAGE)
        llm_model = getattr(app.state, 'llm_model', None)
        background_tasks.add_task(run_bot, pipecat_connection, language, llm_model)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE, 
        choices=list(LANGUAGE_CONFIG.keys()),
        help=f"Language for the bot (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--llm", dest="llm_model", default=None,
        help="LLM model to use (e.g., mistral:7b, llama2:13b, gemma:2b). Default: gemma-3-12b-it-qat"
    )
    args = parser.parse_args()

    # Set language and llm_model in app state
    app.state.language = args.language
    app.state.llm_model = args.llm_model
    
    logger.info(f"Starting bot with language: {args.language}")
    if args.llm_model:
        logger.info(f"Using custom LLM model: {args.llm_model}")
    else:
        logger.info("Using default LLM model: gemma-3-12b-it-qat")
    
    uvicorn.run(app, host=args.host, port=args.port)
