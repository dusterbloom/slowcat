"""
Context Compressor

Replaces AssistantContextGate to eliminate redundant storage and compress context.
Handles tool result compression and prevents double assistant message storage.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class ContextCompressor(FrameProcessor):
    """
    Advanced context processor that:
    1. Eliminates redundant assistant message storage 
    2. Compresses tool results aggressively
    3. Maintains conversation continuity
    4. Prevents context explosion
    """
    
    def __init__(
        self,
        max_tool_result_tokens: int = 50,
        max_assistant_tokens: int = 200,
        compress_tool_results: bool = True,
        store_assistant_messages: bool = False  # Disable since realtime_streamer handles this
    ):
        super().__init__()
        self.max_tool_result_tokens = max_tool_result_tokens
        self.max_assistant_tokens = max_assistant_tokens
        self.compress_tool_results = compress_tool_results
        self.store_assistant_messages = store_assistant_messages
        
        # JSON and protocol detection patterns
        self._json_fence_re = re.compile(r"```(?:json|JSON)?\s*[\s\S]*?```", re.MULTILINE)
        self._protocol_signatures = {
            "tts", "protocol", "message_id", "chunk_index", "total_chunks", 
            "message_type", "is_final", "timestamp"
        }
        
        logger.info(f"🗜️ ContextCompressor initialized: tool_tokens≤{max_tool_result_tokens}, "
                   f"assistant_storage={store_assistant_messages}")

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def _looks_like_protocol_message(self, obj: Any) -> bool:
        """Detect if object looks like a protocol message"""
        if not isinstance(obj, dict):
            return False
            
        # Check for protocol field
        protocol = str(obj.get("protocol", "")).lower()
        if protocol.startswith("tts"):
            return True
            
        # Check for multiple protocol signatures
        obj_keys = set(obj.keys())
        signature_matches = len(self._protocol_signatures.intersection(obj_keys))
        return signature_matches >= 3

    def _extract_json_objects(self, text: str) -> List[Tuple[int, int, Dict]]:
        """Extract JSON objects from text with their positions"""
        objects = []
        
        # Remove JSON code fences first
        cleaned_text = self._json_fence_re.sub("", text)
        
        # Find JSON object boundaries
        i = 0
        while i < len(cleaned_text):
            if cleaned_text[i] == '{':
                # Find matching closing brace
                brace_count = 0
                start = i
                in_string = False
                escaped = False
                
                for j in range(i, len(cleaned_text)):
                    char = cleaned_text[j]
                    
                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == '\\':
                            escaped = True
                        elif char == '"':
                            in_string = False
                    else:
                        if char == '"':
                            in_string = True
                        elif char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                json_str = cleaned_text[start:j+1]
                                try:
                                    obj = json.loads(json_str)
                                    objects.append((start, j+1, obj))
                                except json.JSONDecodeError:
                                    pass
                                i = j + 1
                                break
                else:
                    # No closing brace found
                    i += 1
            else:
                i += 1
                
        return objects

    def _clean_assistant_text(self, text: str) -> str:
        """Extract clean text from assistant response, aggressively removing TTS protocols"""
        if not text or not text.strip():
            return ""
        
        logger.debug(f"🧹 Cleaning text: {text[:100]}...")
        
        # First, remove all TTS protocol JSON objects using regex (more aggressive)
        # Pattern matches: {"protocol": "tts_v2", ...} with any content
        tts_protocol_pattern = r'\{\s*"protocol"\s*:\s*"tts_v2"[^}]*"timestamp"\s*:\s*[\d.]+\s*\}'
        cleaned_text = re.sub(tts_protocol_pattern, '', text)
        
        # Also remove any remaining JSON objects that look like protocols
        general_protocol_pattern = r'\{\s*"[^"]*"\s*:\s*"[^"]*",\s*"message_id"[^}]+\}'
        cleaned_text = re.sub(general_protocol_pattern, '', cleaned_text)
        
        # Remove any remaining standalone JSON objects (backup cleanup)
        json_pattern = r'\{[^{}]*"(?:protocol|message_id|chunk_index|timestamp)"[^{}]*\}'
        cleaned_text = re.sub(json_pattern, '', cleaned_text)
        
        # Clean up extra whitespace and special characters
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())
        cleaned_text = re.sub(r'[^\w\s.,!?;:()-]', ' ', cleaned_text)  # Remove special chars
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())  # Final cleanup
        
        logger.debug(f"🧹 Cleaned result: {cleaned_text}")
        
        return cleaned_text if cleaned_text and len(cleaned_text) > 5 else ""

    def _extract_json_content(self, obj: Dict) -> str:
        """Extract meaningful text from non-protocol JSON"""
        # Common fields that contain readable content
        content_fields = ["content", "text", "message", "response", "result", "answer"]
        
        for field in content_fields:
            if field in obj:
                value = str(obj[field]).strip()
                if len(value) > 5:
                    return value[:100] + ("..." if len(value) > 100 else "")
                    
        return ""

    def _compress_tool_result(self, content: str) -> str:
        """Aggressively compress tool results to key facts"""
        if not content or not content.strip():
            return ""
            
        try:
            # Try parsing as JSON
            if content.strip().startswith(('{', '[')):
                data = json.loads(content)
                return self._compress_json_tool_result(data)
        except json.JSONDecodeError:
            pass
            
        # Handle non-JSON tool results
        return self._compress_text_tool_result(content)

    def _compress_json_tool_result(self, data: Any) -> str:
        """Compress JSON tool results to essential facts"""
        if isinstance(data, dict):
            # Extract key fields in order of priority
            priority_fields = [
                "result", "answer", "data", "content", "message", "text",
                "status", "value", "response", "output", "summary"
            ]
            
            extracted = {}
            for field in priority_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (dict, list)):
                        # Recursively compress nested structures
                        value = self._compress_nested_structure(value)
                    else:
                        value = str(value)
                        
                    # Truncate long values
                    if len(value) > 40:
                        value = value[:37] + "..."
                        
                    extracted[field] = value
                    
                    # Stop after 2-3 key fields to stay within token limit
                    if len(extracted) >= 3:
                        break
                        
            if extracted:
                return json.dumps(extracted, separators=(',', ':'))  # Compact JSON
            else:
                # Fallback: just indicate tool executed
                return f"tool_result: {type(data).__name__}"
                
        elif isinstance(data, list):
            if len(data) > 3:
                return f"list[{len(data)} items]: {str(data[:2])}..."
            else:
                return str(data)
        else:
            value = str(data)
            return value[:47] + "..." if len(value) > 50 else value

    def _compress_nested_structure(self, obj: Any, depth: int = 0) -> str:
        """Compress nested dict/list structures"""
        if depth > 2:  # Prevent deep recursion
            return f"{type(obj).__name__}(...)"
            
        if isinstance(obj, dict):
            if len(obj) > 2:
                key_sample = list(obj.keys())[:2]
                return f"dict({key_sample}...)"
            else:
                return str(obj)
        elif isinstance(obj, list):
            if len(obj) > 3:
                return f"list[{len(obj)}]"
            else:
                return str(obj[:3])
        else:
            return str(obj)

    def _compress_text_tool_result(self, content: str) -> str:
        """Compress plain text tool results"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        if len(content) <= self.max_tool_result_tokens * 4:  # Rough token conversion
            return content
            
        # Try to find natural break points
        sentences = content.split('. ')
        if len(sentences) > 1 and len(sentences[0]) < self.max_tool_result_tokens * 3:
            return sentences[0] + ". [...]"
            
        # Find paragraph breaks
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1 and len(paragraphs[0]) < self.max_tool_result_tokens * 3:
            return paragraphs[0] + " [...]"
            
        # Hard truncation as last resort
        max_chars = self.max_tool_result_tokens * 4
        return content[:max_chars-4] + " ..."

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle both directions - tool results can flow in either direction
        if isinstance(frame, LLMMessagesFrame):
            # Process LLM messages for compression
            compressed_messages = []
            messages_modified = False
            
            for msg in frame.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "assistant":
                    # ALWAYS clean assistant messages, regardless of storage setting
                    cleaned_content = self._clean_assistant_text(content)
                    logger.debug(f"🧹 Cleaned assistant: {len(content)} → {len(cleaned_content)} chars")
                    
                    if self.store_assistant_messages and cleaned_content:
                        # Apply token limit for assistant messages
                        if self._estimate_tokens(cleaned_content) > self.max_assistant_tokens:
                            # Truncate to keep within limit
                            max_chars = self.max_assistant_tokens * 4
                            cleaned_content = cleaned_content[:max_chars-4] + " ..."
                            
                        compressed_messages.append({**msg, "content": cleaned_content})
                        messages_modified = True
                        logger.debug(f"✅ Stored cleaned assistant message")
                    else:
                        # Skip assistant messages or empty cleaned content
                        messages_modified = True
                        logger.debug("🚫 Skipped assistant message")
                        
                elif role == "tool" and self.compress_tool_results:
                    # Compress tool results aggressively
                    logger.debug(f"🔧 Processing tool result: {len(content)} chars")
                    compressed_content = self._compress_tool_result(content)
                    if compressed_content != content:
                        compressed_messages.append({**msg, "content": compressed_content})
                        messages_modified = True
                        
                        # Log compression stats
                        original_tokens = self._estimate_tokens(content)
                        compressed_tokens = self._estimate_tokens(compressed_content)
                        reduction = ((original_tokens - compressed_tokens) / original_tokens) * 100
                        logger.info(f"🗜️ Tool result: {original_tokens}→{compressed_tokens} tokens "
                                   f"({reduction:.1f}% reduction)")
                    else:
                        compressed_messages.append(msg)
                        logger.debug(f"🔧 Tool result passed through unchanged")
                        
                elif role == "tool":
                    # Tool results with compression disabled - pass through
                    compressed_messages.append(msg)
                    logger.debug(f"🔧 Tool result passed through (compression disabled)")
                else:
                    # Keep other message types unchanged
                    compressed_messages.append(msg)
            
            if messages_modified and compressed_messages:
                # Send compressed frame
                compressed_frame = LLMMessagesFrame(messages=compressed_messages)
                await self.push_frame(compressed_frame, direction)
                return
            elif not compressed_messages:
                # All messages were filtered out, don't send frame
                logger.debug("🚫 All messages filtered out, skipping frame")
                return
                
        # For all other cases, pass through unchanged
        await self.push_frame(frame, direction)