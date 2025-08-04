"""
Service interfaces and abstract base classes for dependency injection
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pipecat.frames.frames import Frame
from pipecat.transcriptions.language import Language


class BaseSTTService(ABC):
    """Abstract base class for Speech-to-Text services"""
    
    @abstractmethod
    async def process_audio(self, audio_data: bytes) -> str:
        """Process audio data and return transcribed text"""
        pass
    
    @property
    @abstractmethod
    def language(self) -> Language:
        """Get the configured language"""
        pass


class BaseTTSService(ABC):
    """Abstract base class for Text-to-Speech services"""
    
    @abstractmethod
    async def generate_audio(self, text: str) -> bytes:
        """Generate audio from text"""
        pass
    
    @property
    @abstractmethod
    def voice(self) -> str:
        """Get the configured voice"""
        pass


class BaseLLMService(ABC):
    """Abstract base class for Language Model services"""
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from message history"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name"""
        pass
    
    @abstractmethod
    def create_context_aggregator(self, context: Any) -> Any:
        """Create context aggregator for pipeline"""
        pass


class BaseProcessor(ABC):
    """Abstract base class for pipeline processors"""
    
    @abstractmethod
    async def process_frame(self, frame: Frame) -> Optional[Frame]:
        """Process a pipeline frame"""
        pass
    
    @property
    @abstractmethod
    def processor_type(self) -> str:
        """Get processor type identifier"""
        pass


class BaseVoiceRecognition(ABC):
    """Abstract base class for voice recognition services"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize voice recognition system"""
        pass
    
    @abstractmethod
    async def process_audio_frame(self, frame: Any) -> None:
        """Process audio frame for speaker identification"""
        pass
    
    @abstractmethod
    def set_callbacks(self, on_speaker_changed, on_speaker_enrolled) -> None:
        """Set event callbacks"""
        pass


class BaseMemoryService(ABC):
    """Abstract base class for memory services"""
    
    @abstractmethod
    async def store_conversation(self, user_id: str, message: str, role: str) -> None:
        """Store conversation message"""
        pass
    
    @abstractmethod
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        pass
    
    @abstractmethod
    async def update_user_id(self, user_id: str) -> None:
        """Update current user ID"""
        pass