"""
Response formatter processor to fix LLM formatting issues
Converts markdown links to clickable HTML automatically - Qwen2.5 workaround
"""

import re
import json
import os
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger
import os
import json


class ResponseFormatterProcessor(FrameProcessor):
    """Optionally normalizes assistant text. Modes:
    - off: pass-through
    - minimal: only collapse letter-split proper nouns (e.g., 'P eppi' -> 'Peppi')
    - full: minimal + spacing fixes + markdown link conversion
    """
    
    def __init__(self, mode: str = 'off', aggressive_spacing: bool = False):
        super().__init__()
        self.mode = "off"
        self.aggressive_spacing = aggressive_spacing  # New flag for enhanced processing
        self._markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        # Initialize name patterns cache
        self._name_patterns = {}
        self._load_name_patterns()
        
        # Compile regex patterns for better performance
        self._spacing_patterns = [
            (re.compile(r"\b([A-Z])\s+([a-z]{2,})\b"), r"\1\2"),  # Simple splits
            (re.compile(r"\b([A-Z][a-z]{0,2})\s+([a-z]{1,3})\s+([a-z]{1,3})\b"),
            lambda m: self._merge_spaced_name(m)),  # Multi-letter splits
            (re.compile(r"\b([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])\s+([A-Z])\b"),
            lambda m: m.group(1) + m.group(2) + m.group(3) + m.group(4) + m.group(5)),  # All caps
        ]
    

    def _load_name_patterns(self):
        """Load known name patterns from a cache file."""
        try:
            cache_file = "data/name_patterns_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self._name_patterns = json.load(f)
        except Exception:
            self._name_patterns = {}

    def _save_name_patterns(self):
        """Save learned name patterns to cache."""
        try:
            cache_file = "data/name_patterns_cache.json"
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(self._name_patterns, f)
        except Exception:
            pass  # Fail silently

    def _learn_name_pattern(self, spaced_name: str, corrected_name: str):
        """Learn a new name pattern for future correction."""
        if spaced_name != corrected_name and len(spaced_name.split()) > 1:
            self._name_patterns[spaced_name.lower()] = corrected_name
            # Save periodically (not on every call to avoid I/O)
            if len(self._name_patterns) % 10 == 0:
                self._save_name_patterns()





    def _fix_spacing(self, s: str) -> str:
        """Normalize spacing and collapse tokenization artifacts from LLM output.
        
        Handles:
        - Single-letter splits: "J ules" → "Jules"
        - Multi-letter splits: "Pe p py" → "Peppy"
        - Mixed case patterns: "P E P P Y" → "Peppy"
        - Preserves legitimate spacing in words
        """
        if not s:
            return s
        
        # Process the text by finding specific patterns and replacing them
        # This approach is more direct and avoids complex regex boundary issues
        
        # Pattern 1: Simple single-letter splits "J ules" → "Jules"
        s = re.sub(r'([A-Z])\s+([a-z]{2,})', r'\1\2', s)
        
        # Pattern 2: Multi-letter splits "Pe p py" → "Peppy"
        # Look for patterns like "Pe p py" and merge them
        s = re.sub(r'([A-Z][a-z]{0,2})\s+([a-z]{1,3})\s+([a-z]{1,3})',
                  lambda m: m.group(1) + m.group(2) + m.group(3), s)
        
        # Pattern 3: All-caps spaced letters "P E P P Y" → "Peppy"
        # Find sequences of single capital letters separated by spaces
        def fix_spaced_caps(match):
            text = match.group(0)
            # Extract all capital letters
            caps = re.findall(r'[A-Z]', text)
            if len(caps) >= 2:
                return ''.join(caps).capitalize()
            return text
        
        s = re.sub(r'\b[A-Z](?:\s+[A-Z])+\b', fix_spaced_caps, s)
        
        # Pattern 4: Hyphenated caps "P-E-P-P-Y" → "Peppy"
        def fix_hyphenated_caps(match):
            text = match.group(0)
            # Remove hyphens and capitalize
            clean = text.replace('-', '')
            if len(clean) >= 2:
                return clean.capitalize()
            return text
        
        s = re.sub(r'\b[A-Z](?:-[A-Z])+\b', fix_hyphenated_caps, s)
        
        # Final cleanup: ensure proper spacing after punctuation
        s = re.sub(r'([,.!?;:])([A-Za-z])', r'\1 \2', s)
        
        return s.strip()

    def _merge_spaced_name(self, match) -> str:
        """Helper method to intelligently merge spaced name parts."""
        parts = [match.group(1), match.group(2), match.group(3)]
        merged = ''.join(parts)
        
        # Capitalize properly (handles cases like "pe p py" → "Peppy")
        if merged:
            merged = merged[0].upper() + merged[1:].lower()
        
        return merged


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
        """Convert markdown links to HTML in text frames with enhanced spacing fix"""
        
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            if self.mode == 'off':
                await self.push_frame(frame, direction)
                return
                
            original_text = frame.text
            text = original_text
            
            if self.mode in ('minimal', 'full'):
                # Apply learned patterns first
                text_lower = text.lower()
                for spaced_pattern, corrected in self._name_patterns.items():
                    if spaced_pattern in text_lower:
                        text = text.replace(spaced_pattern.title(), corrected)
                
                # Apply regex patterns
                for pattern, replacement in self._spacing_patterns:
                    new_text = pattern.sub(replacement, text)
                    if new_text != text:
                        logger.debug("✂️ Fixed complex spacing artifacts in assistant text")
                        text = new_text
                
                # Ensure proper spacing after punctuation
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
            await self.push_frame(frame, direction)