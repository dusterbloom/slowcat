"""
Response formatter processor to fix LLM formatting issues
Converts markdown links to clickable HTML automatically - Qwen2.5 workaround
"""

import re
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ResponseFormatterProcessor(FrameProcessor):
    """Optionally normalizes assistant text. Modes:
    - off: pass-through
    - minimal: only collapse letter-split proper nouns (e.g., 'P eppy' -> 'Peppy')
    - full: minimal + spacing fixes + markdown link conversion
    """
    
    def __init__(self, mode: str = 'off'):
        super().__init__()
        self.mode = (mode or 'off').lower()
        # Regex to match markdown links [text](url)
        self._markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def _fix_spacing(self, s: str) -> str:
        """Normalize spacing and collapse common tokenization artifacts from small models.

        - Remove spaces before punctuation (", . ! ? : ;").
        - Normalize possessives: " 's" -> "'s".
        - Collapse single-letter splits in Proper Nouns: "J ules" -> "Jules".
        """
        if not s:
            return s
        # Only collapse patterns like "J ules", "P eppy" -> "Jules", "Peppy"
        s = re.sub(r"\b([A-Z])\s+([a-z]{2,})\b", r"\1\2", s)
        return s.strip()

    def _ensure_post_punct_space(self, s: str) -> str:
        """Ensure a space after common punctuation when followed immediately by a letter.

        This improves readability and TTS without aggressively rewriting sentences.
        - Comma/colon/semicolon/exclamation/question: add space if next char is a letter.
        - Period: add space only when followed by an uppercase letter (likely sentence start).
        """
        if not s:
            return s
        # Add space after , ; : ! ? when followed by a letter
        s = re.sub(r"([,;:!?])(?!\s|$)([A-Za-z])", r"\1 \2", s)
        # Add space after . when followed by an uppercase letter (avoid decimals)
        s = re.sub(r"([.])(?!\s|$)([A-Z])", r"\1 \2", s)
        return s

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Convert markdown links to HTML in text frames"""
        
        # Call parent first (required for Pipecat)
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            if self.mode == 'off':
                await self.push_frame(frame, direction)
                return
            original_text = frame.text
            text = original_text
            if self.mode in ('minimal', 'full'):
                text2 = self._fix_spacing(text)
                if text2 != text:
                    logger.debug("✂️ Fixed letter-split artifacts in assistant text")
                    text = text2
                # Ensure a space after punctuation like "," when followed by a letter
                text3 = self._ensure_post_punct_space(text)
                if text3 != text:
                    logger.debug("✂️ Added missing space after punctuation for readability/TTS")
                    text = text3
            if self.mode == 'full':
                # Convert [text](url) to HTML
                text3 = self._markdown_link_pattern.sub(
                    r'<a href="\2" target="_blank" rel="noopener">\1</a>',
                    text
                )
                if text3 != text:
                    link_count = len(self._markdown_link_pattern.findall(text))
                    logger.debug(f"🔗 Converted {link_count} markdown links to HTML")
                    text = text3
            if text != original_text:
                await self.push_frame(TextFrame(text), direction)
            else:
                await self.push_frame(frame, direction)
        else:
            # Pass through non-text frames unchanged
            await self.push_frame(frame, direction)
