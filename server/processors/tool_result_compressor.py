"""
Tool Result Compressor

Lightweight processor focused solely on compressing verbose tool results
while preserving essential information for the LLM context.
"""

import json
import re
from typing import Dict, Any
from loguru import logger

from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class ToolResultCompressor(FrameProcessor):
    """
    Compresses tool call results to essential facts only.
    
    Reduces verbose API responses, web content, file listings, etc.
    to concise summaries that preserve key information for the LLM.
    """
    
    def __init__(self, max_result_tokens: int = 150):
        super().__init__()
        self.max_result_tokens = max_result_tokens
        self.min_compress_tokens = 200  # Don't compress results under this size
        logger.info(f"🗜️ ToolResultCompressor initialized: max_tokens={max_result_tokens}")

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def _compress_json_result(self, data: Any) -> str:
        """Extract essential info from JSON tool results"""
        if isinstance(data, dict):
            # Priority fields that usually contain the useful info
            key_fields = [
                "result", "content", "text", "message", "answer", "data",
                "status", "title", "summary", "value", "output"
            ]
            
            # Extract the most important field
            for field in key_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, dict):
                        # Nested result object - recurse once
                        return self._compress_json_result(value)
                    elif isinstance(value, str) and value.strip():
                        # Found useful text content
                        return self._compress_text_content(value.strip())
            
            # Fallback: look for any string values
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10:
                    return self._compress_text_content(value.strip())
                    
            # Last resort: indicate success/failure
            if "status" in data or "success" in data:
                return f"Operation completed: {data.get('status', 'success')}"
            
            return "Tool executed successfully"
            
        elif isinstance(data, list):
            if len(data) == 0:
                return "Empty result"
            elif len(data) == 1:
                return self._compress_json_result(data[0])
            else:
                return f"Found {len(data)} items: {str(data[0])[:50]}..."
                
        else:
            return str(data)[:100]

    def _compress_text_content(self, content: str) -> str:
        """Compress long text content to essential facts, but smartly"""
        estimated_tokens = self._estimate_tokens(content)
        
        # Don't compress if it's already reasonable size
        if estimated_tokens <= self.min_compress_tokens:
            return content
            
        # Only compress if significantly over limit
        if estimated_tokens <= self.max_result_tokens:
            return content
            
        # For web content, try to extract key information
        if self._looks_like_web_content(content):
            return self._compress_web_content(content)
        
        # Smart sentence-based compression
        sentences = content.split('. ')
        if len(sentences) > 1:
            result = sentences[0]
            if not result.endswith('.'):
                result += '.'
                
            # Add more sentences while staying under limit
            for i in range(1, len(sentences)):
                sentence = sentences[i].strip()
                if not sentence:
                    continue
                    
                # Ensure sentence ends properly
                if not sentence.endswith('.') and i < len(sentences) - 1:
                    sentence += '.'
                    
                test_result = result + " " + sentence
                if self._estimate_tokens(test_result) <= self.max_result_tokens:
                    result = test_result
                else:
                    # Can't fit more, stop here
                    break
            
            return result
        
        # For single long paragraph, try word-boundary truncation
        words = content.split()
        if len(words) > 10:
            result_words = []
            for word in words:
                test_result = " ".join(result_words + [word])
                if self._estimate_tokens(test_result) <= self.max_result_tokens:
                    result_words.append(word)
                else:
                    break
            
            if result_words:
                return " ".join(result_words)
        
        # Last resort: character truncation but at word boundary
        max_chars = self.max_result_tokens * 4 - 20  # Leave room for ellipsis
        if len(content) > max_chars:
            # Find the last space before the limit
            truncated = content[:max_chars]
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.7:  # If we found a reasonable word boundary
                return truncated[:last_space] + "..."
        
        return content[:max_chars] + "..."

    def _looks_like_web_content(self, content: str) -> bool:
        """Detect if content looks like web page text"""
        web_indicators = [
            "Skip to", "Log in", "Subscribe", "Cookie", "Privacy", 
            "ADVERTISEMENT", "©", "Terms of Service", "navbar"
        ]
        return any(indicator.lower() in content.lower() for indicator in web_indicators)

    def _compress_web_content(self, content: str) -> str:
        """Extract meaningful content from web pages"""
        lines = content.split('\n')
        meaningful_lines = []
        
        skip_patterns = [
            r'^(skip to|log in|subscribe|search|menu|navigation)',
            r'(privacy|cookie|terms|advertisement|©)',
            r'^[A-Z\s]{10,}$',  # All caps lines (usually navigation)
            r'^\s*$'  # Empty lines
        ]
        
        for line in lines[:20]:  # Only check first 20 lines
            line = line.strip()
            if not line or len(line) < 10:
                continue
                
            # Skip navigation/boilerplate
            if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                continue
                
            # Skip very short lines (likely navigation)
            if len(line) < 30:
                continue
                
            meaningful_lines.append(line)
            
            # Stop if we have enough content
            if len(' '.join(meaningful_lines)) > self.max_result_tokens * 3:
                break
        
        if meaningful_lines:
            result = ' '.join(meaningful_lines)
            if len(result) > self.max_result_tokens * 4:
                return result[:self.max_result_tokens * 4-15] + " [web content...]"
            return result
        
        # Fallback: just take first meaningful chunk
        return content[:self.max_result_tokens * 3] + " [web content...]"

    def _compress_tool_result(self, content: str) -> str:
        """Main compression logic for tool results"""
        if not content or len(content.strip()) == 0:
            return "Empty result"
        
        content = content.strip()
        
        # Try parsing as JSON first
        if content.startswith(('{', '[', '"')):
            try:
                if content.startswith('"') and content.endswith('"'):
                    # Quoted string result
                    unquoted = json.loads(content)
                    return self._compress_text_content(str(unquoted))
                else:
                    # JSON object/array
                    data = json.loads(content)
                    return self._compress_json_result(data)
            except json.JSONDecodeError:
                pass
        
        # Plain text result
        return self._compress_text_content(content)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            compressed_messages = []
            messages_modified = False
            
            for msg in frame.messages:
                if msg.get("role") == "tool":
                    # Compress tool results
                    original_content = msg.get("content", "")
                    compressed_content = self._compress_tool_result(original_content)
                    
                    if compressed_content != original_content:
                        # Log compression stats
                        original_tokens = self._estimate_tokens(original_content)
                        compressed_tokens = self._estimate_tokens(compressed_content)
                        reduction = ((original_tokens - compressed_tokens) / original_tokens) * 100 if original_tokens > 0 else 0
                        
                        logger.info(f"🗜️ Tool result compressed: {original_tokens}→{compressed_tokens} tokens ({reduction:.1f}% reduction)")
                        
                        compressed_messages.append({**msg, "content": compressed_content})
                        messages_modified = True
                    else:
                        compressed_messages.append(msg)
                else:
                    # Pass through non-tool messages unchanged
                    compressed_messages.append(msg)
            
            if messages_modified:
                # Send compressed frame
                compressed_frame = LLMMessagesFrame(messages=compressed_messages)
                await self.push_frame(compressed_frame, direction)
                return
        
        # Pass through all other frames unchanged
        await self.push_frame(frame, direction)