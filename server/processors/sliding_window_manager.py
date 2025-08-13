"""
Sliding Window Context Manager

Replaces OpenAILLMContext with smart context pruning to maintain ultra-low latency.
Implements progressive compression and long-term memory integration.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


class SlidingWindowManager(OpenAILLMContext):
    """
    Context manager with sliding window and progressive compression.
    
    Maintains conversation continuity while preventing context explosion:
    - Recent turns (1-3): Full text, no compression
    - Middle turns (4-7): Compressed to key points  
    - Old turns (8-10): Ultra-compressed summaries
    - Ancient turns (11+): Pushed to MCP memory, recalled on demand
    """
    
    def __init__(
        self,
        messages: List[Dict[str, Any]] = None,
        tools=None,
        max_turns: int = 10,
        compression_start_turn: int = 4,
        ultra_compression_turn: int = 8,
        memory_tool_name: str = "memory_store",
        **kwargs
    ):
        super().__init__(messages or [], tools, **kwargs)
        
        self.max_turns = max_turns
        self.compression_start_turn = compression_start_turn  
        self.ultra_compression_turn = ultra_compression_turn
        self.memory_tool_name = memory_tool_name
        
        # Track turn boundaries for compression
        self.turn_count = 0
        self.turn_messages: List[List[Dict]] = []  # Group messages by turn
        
        logger.info(f"🪟 SlidingWindow initialized: max_turns={max_turns}, compression_at={compression_start_turn}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def _compress_message(self, message: Dict[str, Any], compression_level: str = "medium") -> Dict[str, Any]:
        """Compress a message based on compression level"""
        content = message.get("content", "")
        role = message.get("role", "")
        
        if role == "system":
            return message  # Never compress system messages
            
        if role == "user":
            # User messages are usually short, minimal compression
            if compression_level == "high" and len(content) > 200:
                return {**message, "content": content[:150] + "... [user message continues]"}
            return message
            
        if role == "assistant":
            if compression_level == "medium":
                # Keep first sentence + key points
                sentences = content.split('. ')
                if len(sentences) > 1:
                    compressed = sentences[0] + '. [assistant response continues]'
                    return {**message, "content": compressed}
            elif compression_level == "high":
                # Ultra compress to essence
                if len(content) > 100:
                    return {**message, "content": content[:80] + "... [response summary]"}
                    
        if role == "tool":
            # Aggressively compress tool results
            if compression_level == "medium":
                # Extract key facts from tool results
                compressed = self._compress_tool_result(content)
                return {**message, "content": compressed}
            elif compression_level == "high":
                # Ultra compress to minimal facts
                compressed = self._ultra_compress_tool_result(content)
                return {**message, "content": compressed}
                
        return message

    def _compress_tool_result(self, content: str) -> str:
        """Medium compression of tool results - extract key facts"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                data = json.loads(content)
                # Extract key fields commonly used in tool responses
                key_fields = ['result', 'answer', 'data', 'content', 'message', 'status', 'value']
                extracted = {}
                for field in key_fields:
                    if field in data:
                        value = str(data[field])
                        if len(value) > 100:
                            extracted[field] = value[:97] + "..."
                        else:
                            extracted[field] = value
                        if len(extracted) >= 3:  # Limit to 3 key fields
                            break
                return json.dumps(extracted)
        except:
            pass
            
        # For non-JSON content, truncate intelligently
        if len(content) > 200:
            # Try to find first complete sentence or logical break
            if '. ' in content[:200]:
                first_sentence = content.split('. ')[0] + '. [tool result truncated]'
                return first_sentence
            else:
                return content[:150] + "... [tool result continues]"
        return content

    def _ultra_compress_tool_result(self, content: str) -> str:
        """High compression of tool results - minimal essential facts"""
        try:
            if content.strip().startswith('{'):
                data = json.loads(content)
                # Keep only the most essential field
                key_fields = ['result', 'answer', 'data', 'message']
                for field in key_fields:
                    if field in data:
                        value = str(data[field])[:50]
                        return f"{field}: {value}..."
                return "tool_result: [summary]"
        except:
            pass
            
        # For non-JSON, keep only first 50 chars
        return content[:47] + "..." if len(content) > 50 else content

    async def _push_to_memory(self, old_messages: List[Dict]) -> bool:
        """Push old messages to MCP memory tool for later recall"""
        try:
            # Create conversation summary for memory storage
            summary_parts = []
            for msg in old_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]  # Brief content
                summary_parts.append(f"{role}: {content}")
            
            conversation_summary = "\n".join(summary_parts)
            timestamp = datetime.now().isoformat()
            
            # Format for memory tool
            memory_entry = {
                "timestamp": timestamp,
                "type": "conversation_turn", 
                "content": conversation_summary,
                "message_count": len(old_messages)
            }
            
            # TODO: Integrate with MCP memory tool
            # For now, just log that we would store it
            logger.debug(f"📥 Would push to memory: {len(old_messages)} messages")
            logger.debug(f"📝 Memory entry: {json.dumps(memory_entry, indent=2)}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to push to memory: {e}")
            return False

    def _group_messages_by_turns(self, messages: List[Dict]) -> List[List[Dict]]:
        """Group messages into conversation turns for better management"""
        turns = []
        current_turn = []
        
        for msg in messages:
            role = msg.get("role", "")
            
            # Start new turn on user message (unless it's the first message)
            if role == "user" and current_turn:
                turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        
        # Add final turn if it exists
        if current_turn:
            turns.append(current_turn)
            
        return turns

    async def _prune_and_compress(self) -> bool:
        """Apply sliding window with progressive compression"""
        if len(self.messages) <= self.max_turns * 2:  # Rough turn estimate
            return False
            
        # Group messages by conversation turns
        turns = self._group_messages_by_turns(self.messages)
        self.turn_count = len(turns)
        
        if self.turn_count <= self.max_turns:
            return False
            
        logger.info(f"🪟 Pruning context: {self.turn_count} turns, max={self.max_turns}")
        
        # Separate system messages (always keep)
        system_messages = [msg for msg in self.messages if msg.get("role") == "system"]
        conversation_messages = [msg for msg in self.messages if msg.get("role") != "system"]
        
        # Group conversation by turns
        conv_turns = self._group_messages_by_turns(conversation_messages)
        
        if len(conv_turns) <= self.max_turns:
            return False
            
        # Calculate how many turns to keep
        turns_to_remove = len(conv_turns) - self.max_turns
        old_turns = conv_turns[:turns_to_remove]
        remaining_turns = conv_turns[turns_to_remove:]
        
        # Push old turns to memory
        old_messages = [msg for turn in old_turns for msg in turn]
        await self._push_to_memory(old_messages)
        
        # Apply progressive compression to remaining turns
        compressed_messages = []
        
        for turn_idx, turn in enumerate(remaining_turns):
            turn_age = len(remaining_turns) - turn_idx - 1  # 0 = newest
            
            if turn_age < self.compression_start_turn:
                # Recent turns: no compression
                compressed_messages.extend(turn)
            elif turn_age < self.ultra_compression_turn:
                # Middle turns: medium compression
                for msg in turn:
                    compressed_msg = self._compress_message(msg, "medium")
                    compressed_messages.append(compressed_msg)
            else:
                # Old turns: high compression
                for msg in turn:
                    compressed_msg = self._compress_message(msg, "high")
                    compressed_messages.append(compressed_msg)
        
        # Rebuild messages with system + compressed conversation
        # Can't directly assign to self.messages (read-only property), so use internal list
        new_messages = system_messages + compressed_messages
        self._messages = new_messages  # Access internal storage directly
        
        # Log compression results
        original_tokens = sum(self._estimate_tokens(msg.get("content", "")) 
                            for turn in conv_turns for msg in turn)
        compressed_tokens = sum(self._estimate_tokens(msg.get("content", "")) 
                              for msg in compressed_messages)
        
        logger.info(f"🗜️ Compression: {original_tokens} → {compressed_tokens} tokens "
                   f"({((1 - compressed_tokens/original_tokens) * 100):.1f}% reduction)")
        logger.info(f"📝 Context now: {len(new_messages)} messages, ~{compressed_tokens} tokens")
        
        return True

    async def add_messages(self, messages: List[Dict]) -> None:
        """Override to add messages and trigger pruning if needed"""
        # Add messages normally
        if hasattr(super(), 'add_messages'):
            await super().add_messages(messages)
        else:
            # Fallback if parent doesn't have add_messages
            self.messages.extend(messages)
        
        # Check if pruning is needed
        await self._prune_and_compress()

    def add_message(self, message: Dict) -> None:
        """Override to add single message and trigger pruning if needed"""
        super().add_message(message) if hasattr(super(), 'add_message') else self.messages.append(message)
        
        # Schedule pruning check (async)
        asyncio.create_task(self._prune_and_compress())

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        total_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in self.messages)
        
        role_counts = {}
        for msg in self.messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            
        return {
            "total_messages": len(self.messages),
            "estimated_tokens": total_tokens,
            "turn_count": self.turn_count,
            "role_distribution": role_counts,
            "compression_active": self.turn_count > self.max_turns
        }