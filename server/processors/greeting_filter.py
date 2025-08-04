      
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

    async def process_frame(self, frame: Frame, direction):
        # No change needed here, but call to super().process_frame is also critical
        await super().process_frame(frame, direction)

        if self._greeting_filtered or not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        self._buffer += frame.text

        if self._greeting_text in self._buffer:
            logger.debug(f"Filtering initial greeting: '{self._greeting_text}'")
            remaining_text = self._buffer.replace(self._greeting_text, "").strip()
            if remaining_text:
                await self.push_frame(TextFrame(remaining_text), direction)
            
            self._greeting_filtered = True
        elif not self._greeting_text.startswith(self._buffer):
            await self.push_frame(TextFrame(self._buffer), direction)
            self._greeting_filtered = True
        else:
            # Buffer matches start of greeting, wait for more frames.
            pass

    