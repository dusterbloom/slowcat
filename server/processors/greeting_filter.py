      
# FILE: server/processors/greeting_filter.py

# --- OLD (INCORRECT) CODE ---
# from pipecat.processors.frame_processor import FrameProcessor
# from pipecat.frames.frames import Frame, TextFrame
# from loguru import logger
# 
# class GreetingFilterProcessor(FrameProcessor):
#     def __init__(self, greeting_text: str):
#         # MISSING super().__init__()
#         self._greeting_text = greeting_text
#         self._greeting_filtered = False
#         self._buffer = ""
# 
#     # ... (rest of the class was correct)

# --- NEW (CORRECT) CODE ---
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, TextFrame
from loguru import logger
import re
import unicodedata

class GreetingFilterProcessor(FrameProcessor):
    """
    This processor filters out the initial greeting text from the assistant
    to prevent it from being added to the LLM context history. This avoids
    issues with strict prompt templates that expect the first turn to be from
    the user.
    """

    def __init__(self, greeting_text: str):
        # CORRECT: Always call the parent constructor.
        super().__init__()
        self._greeting_text = greeting_text
        self._greeting_filtered = False
        self._buffer = ""
        # Regex to catch common intro variants like:
        # "Hello! I'm Slowcat, and I'm here to help. How can I assist you today?"
        # with arbitrary spaces/punctuation between tokens.
        self._intro_regex = re.compile(
            r"^(?:\s*(hello|hi|hey)[\s!.,']*)?\s*i\s*['’`]?\s*m\s*(slow\s*cat|slowcat)\b.*?(help|assist).*$",
            re.IGNORECASE,
        )

    def _normalize(self, s: str) -> str:
        """Normalize text for robust greeting detection."""
        try:
            s = unicodedata.normalize('NFKD', s)
            s = s.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            pass
        s = s.replace("\u200b", "")
        s = re.sub(r"\s+", " ", s)
        # Collapse spaced punctuation and spaced letters in 'slow cat'
        s = re.sub(r"s\s*l\s*o\s*w\s*c\s*a\s*t", "slowcat", s, flags=re.IGNORECASE)
        return s.strip()

    async def process_frame(self, frame: Frame, direction):
        # No change needed here, but call to super().process_frame is also critical
        await super().process_frame(frame, direction)

        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        self._buffer += frame.text or ""

        # Try robust removal using normalization + regex
        normalized = self._normalize(self._buffer)
        removed = False

        # Exact configured greeting, if present
        if self._greeting_text and self._greeting_text in normalized:
            logger.debug("GreetingFilter: removing configured greeting text")
            normalized = normalized.replace(self._greeting_text, "").strip()
            removed = True

        # Regex intro removal (works for variants)
        if self._intro_regex.match(normalized):
            logger.debug("GreetingFilter: stripping intro boilerplate")
            # Remove up to the first sentence end
            normalized = re.sub(r"^.*?(?:\.|!|\?)\s*", "", normalized, count=1).strip()
            removed = True

        if removed:
            if normalized:
                await self.push_frame(TextFrame(normalized), direction)
            self._greeting_filtered = True
            self._buffer = ""
        else:
            # If buffer no longer matches the start of greeting, forward it
            if self._greeting_filtered or not (self._greeting_text and self._greeting_text.startswith(self._buffer)):
                await self.push_frame(TextFrame(self._buffer), direction)
                self._buffer = ""

    
