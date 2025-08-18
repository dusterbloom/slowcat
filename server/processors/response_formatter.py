"""
Response formatter processor to fix LLM formatting issues
Converts markdown links to clickable HTML automatically - Qwen2.5 workaround
"""

import re
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ResponseFormatterProcessor(FrameProcessor):
    """Automatically converts markdown links to HTML for stubborn LLMs"""
    
    def __init__(self):
        super().__init__()
        # Regex to match markdown links [text](url)
        self._markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Convert markdown links to HTML in text frames"""
        
        # Call parent first (required for Pipecat)
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            original_text = frame.text
            
            # Convert [text](url) to <a href="url" target="_blank">text</a>
            converted_text = self._markdown_link_pattern.sub(
                r'<a href="\2" target="_blank" rel="noopener">\1</a>',
                original_text
            )
            
            # Log conversion for debugging
            if converted_text != original_text:
                link_count = len(self._markdown_link_pattern.findall(original_text))
                logger.debug(f"🔗 Converted {link_count} markdown links to HTML")
                
                # Create new frame with converted text and forward it
                converted_frame = TextFrame(converted_text)
                await self.push_frame(converted_frame, direction)
            else:
                # No conversion needed, forward original frame
                await self.push_frame(frame, direction)
        else:
            # Pass through non-text frames unchanged
            await self.push_frame(frame, direction)