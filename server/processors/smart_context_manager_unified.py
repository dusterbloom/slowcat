#!/usr/bin/env python3
"""
Smart Context Manager - Unified Schema Version

This is an updated version of SmartContextManager that uses the new unified schema:
- conversation table instead of tape table
- session_meta table for session tracking
- fact_extraction table for processing tracking
- Enhanced session boundary detection
- Proper turn numbering and session management

Key improvements:
- Cleaner session tracking with proper boundaries
- Enhanced metadata collection
- Better fact extraction tracking
- Unified conversation storage with normalization
- Performance optimizations with proper indexes
"""

import asyncio
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import warnings

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, FrameDirection
from pipecat.processors.frame_processor import FrameProcessor
from processors.token_counter import TokenCounter, TokenBudget

# DEPRECATION NOTICE
warnings.warn(
    "processors.smart_context_manager_unified is deprecated and not used by the current pipeline. "
    "Use processors.smart_context_manager (and SmartContextManagerGraph for persistence) instead.",
    DeprecationWarning,
    stacklevel=2,
)
logger.warning(
    "DEPRECATED: smart_context_manager_unified — prefer smart_context_manager + smart_context_manager_graph"
)

def create_smart_context_manager_unified(*args, **kwargs):
    """Compatibility shim: delegate to the maintained SmartContextManager factory.

    This preserves imports for older code/tests while we phase out this module.
    """
    logger.warning(
        "DEPRECATED: create_smart_context_manager_unified → create_smart_context_manager"
    )
    try:
        from processors.smart_context_manager import create_smart_context_manager
        return create_smart_context_manager(*args, **kwargs)
    except Exception as e:
        logger.error(f"Shim call failed: {e}")
        raise


class SmartContextManagerUnified(FrameProcessor):
    """
    Smart Context Manager using unified schema
    
    Maintains EXACTLY 4096 tokens of context while using the new
    unified database schema for better session and conversation management.
    """
    
    def __init__(
        self,
        context,
        tape_store=None,
        max_tokens: int = 4096,
        assistant_id: str = "slowcat",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.context = context
        self.tape_store = tape_store
        self.max_tokens = max_tokens
        self.assistant_id = assistant_id
        
        # Token budget management
        self.token_counter = TokenCounter()
        self.budget = TokenBudget(total_tokens=max_tokens)
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.session_turn_number: int = 0
        self.session_start_time: Optional[datetime] = None
        self.last_activity_time: Optional[datetime] = None
        
        # Memory and processing
        self.memory_system = None
        self.fact_extractions = 0
        self.total_conversations = 0
        
        # Content normalization (same as other processors)
        self._normalization_enabled = True
        
        logger.info(f"🧠 SmartContextManagerUnified initialized with {max_tokens} token budget")
        logger.info(f"📊 Token allocation: system={self.budget.system_prompt}, facts={self.budget.facts}, recent={self.budget.recent_conversation}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            await self._handle_user_input(frame)
        elif isinstance(frame, TextFrame):
            await self._handle_assistant_response(frame)
        
        # Always forward frames
        await self.push_frame(frame, direction)
    
    async def _handle_user_input(self, frame: TranscriptionFrame):
        """
        Handle user input with unified schema
        
        Enhanced with:
        - Session boundary detection
        - Turn number tracking
        - Conversation table storage
        - Fact extraction triggering
        """
        try:
            # Normalize user input
            user_text = self._normalize_user_input(frame.text)
            logger.debug(f"🎤 Processing user input: '{user_text[:50]}...'")
            
            # Detect session boundaries and update session management
            await self._manage_session_boundaries(user_text)
            
            # Store in unified conversation table
            await self._store_conversation_entry('user', user_text, frame.text if frame.text != user_text else None)
            
            # Extract facts if content is rich enough (non-blocking)
            if self._should_extract_facts(user_text):
                asyncio.create_task(self._extract_facts_async(user_text))
            
            # Update context with fixed token budget
            await self._update_context_with_budget(user_text)
            
            # Update session activity
            await self._update_session_activity()
            
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
    
    async def _handle_assistant_response(self, frame: TextFrame):
        """
        Handle assistant response with unified schema
        """
        try:
            if not frame.text or not frame.text.strip():
                return
            
            response_text = frame.text.strip()
            
            # Store assistant response in conversation table
            await self._store_conversation_entry('assistant', response_text)
            
            # Update session activity
            await self._update_session_activity()
            
        except Exception as e:
            logger.error(f"Error handling assistant response: {e}")
    
    def _generate_session_id(self, speaker_id: str = "default", timestamp: Optional[datetime] = None) -> str:
        """Generate consistent session ID"""
        if not timestamp:
            timestamp = datetime.now()
        
        date_str = timestamp.strftime('%Y%m%d')
        return f"{speaker_id}_{date_str}"
    
    async def _manage_session_boundaries(self, user_text: str):
        """
        Manage session boundaries and metadata
        
        Enhanced session management with:
        - Automatic session creation
        - Session boundary detection
        - Turn number tracking
        - Session metadata updates
        """
        current_time = datetime.now()
        speaker_id = self._get_speaker_id()
        
        # Generate session ID for current day
        new_session_id = self._generate_session_id(speaker_id, current_time)
        
        # Check if we need to start a new session
        session_changed = False
        
        if self.current_session_id != new_session_id:
            # New session (new day or first interaction)
            await self._finalize_current_session()
            await self._start_new_session(new_session_id, speaker_id, current_time)
            session_changed = True
        
        # Check for inactivity-based session boundary (e.g., > 30 minutes)
        elif self.last_activity_time:
            inactive_duration = (current_time - self.last_activity_time).total_seconds()
            if inactive_duration > 1800:  # 30 minutes
                await self._finalize_current_session()
                await self._start_new_session(f"{new_session_id}_{int(time.time())}", speaker_id, current_time)
                session_changed = True
        
        # Increment turn number
        if not session_changed:
            self.session_turn_number += 1
        
        # Update activity time
        self.last_activity_time = current_time
    
    async def _start_new_session(self, session_id: str, speaker_id: str, start_time: datetime):
        """Start a new session with unified schema"""
        self.current_session_id = session_id
        self.session_turn_number = 1
        self.session_start_time = start_time
        
        try:
            if self.tape_store and hasattr(self.tape_store, 'query'):
                # Create session_meta record
                await self.tape_store.query('''
                    CREATE session_meta SET
                        session_id = $session_id,
                        speaker_id = $speaker_id,
                        start_time = $start_time,
                        last_activity = $start_time,
                        turn_count = 0,
                        user_turns = 0,
                        assistant_turns = 0,
                        total_words = 0,
                        total_chars = 0,
                        facts_extracted = 0,
                        quality_score = 0.0,
                        status = 'active',
                        agent_id = $agent_id
                ''', {
                    'session_id': session_id,
                    'speaker_id': speaker_id,
                    'start_time': start_time,
                    'agent_id': self.assistant_id
                })
        except Exception as e:
            logger.debug(f"Failed to create session_meta record: {e}")
        
        logger.info(f"🎯 Started new session: {session_id}")
    
    async def _finalize_current_session(self):
        """Finalize current session with summary statistics"""
        if not self.current_session_id or not self.tape_store:
            return
        
        try:
            # Update session_meta with final statistics
            end_time = datetime.now()
            duration_s = 0
            
            if self.session_start_time:
                duration_s = int((end_time - self.session_start_time).total_seconds())
            
            await self.tape_store.query('''
                UPDATE session_meta 
                SET 
                    end_time = $end_time,
                    duration_s = $duration_s,
                    status = 'completed',
                    updated_at = time::now()
                WHERE session_id = $session_id
            ''', {
                'session_id': self.current_session_id,
                'end_time': end_time,
                'duration_s': duration_s
            })
            
            logger.info(f"✅ Finalized session: {self.current_session_id} ({duration_s}s)")
            
        except Exception as e:
            logger.debug(f"Failed to finalize session: {e}")
    
    async def _store_conversation_entry(self, role: str, content: str, raw_content: Optional[str] = None):
        """
        Store conversation entry in unified conversation table
        
        Enhanced with:
        - Turn number tracking
        - Session metadata
        - Content analysis
        - Fact extraction tracking
        """
        if not self.tape_store or not content.strip():
            return
        
        try:
            current_time = datetime.now()
            speaker_id = self._get_speaker_id()
            
            # Ensure we have a valid session
            if not self.current_session_id:
                await self._manage_session_boundaries(content)
            
            # Store in conversation table
            await self.tape_store.query('''
                CREATE conversation SET
                    ts = $ts,
                    speaker_id = $speaker_id,
                    role = $role,
                    content = $content,
                    raw_content = $raw_content,
                    session_id = $session_id,
                    turn_number = $turn_number,
                    session_start = $session_start,
                    agent_id = $agent_id,
                    facts_extracted = false,
                    processing_meta = $processing_meta,
                    content_length = $content_length,
                    word_count = $word_count
            ''', {
                'ts': current_time,
                'speaker_id': speaker_id,
                'role': role,
                'content': content,
                'raw_content': raw_content,
                'session_id': self.current_session_id,
                'turn_number': self.session_turn_number,
                'session_start': self.session_start_time if self.session_turn_number == 1 else None,
                'agent_id': self.assistant_id if role == 'assistant' else None,
                'processing_meta': {
                    'normalized': raw_content is not None,
                    'processor': 'smart_context_manager_unified'
                },
                'content_length': len(content),
                'word_count': len(content.split())
            })
            
            # Update session statistics
            await self._update_session_statistics(role, content)
            
            self.total_conversations += 1
            
        except Exception as e:
            logger.error(f"Failed to store conversation entry: {e}")
    
    async def _update_session_statistics(self, role: str, content: str):
        """Update session_meta statistics"""
        if not self.current_session_id or not self.tape_store:
            return
        
        try:
            word_count = len(content.split())
            char_count = len(content)
            
            # Increment appropriate counters
            if role == 'user':
                await self.tape_store.query('''
                    UPDATE session_meta SET
                        turn_count += 1,
                        user_turns += 1,
                        total_words += $word_count,
                        total_chars += $char_count,
                        last_activity = time::now(),
                        updated_at = time::now()
                    WHERE session_id = $session_id
                ''', {
                    'session_id': self.current_session_id,
                    'word_count': word_count,
                    'char_count': char_count
                })
            elif role == 'assistant':
                await self.tape_store.query('''
                    UPDATE session_meta SET
                        assistant_turns += 1,
                        total_words += $word_count,
                        total_chars += $char_count,
                        last_activity = time::now(),
                        updated_at = time::now()
                    WHERE session_id = $session_id
                ''', {
                    'session_id': self.current_session_id,
                    'word_count': word_count,
                    'char_count': char_count
                })
                
        except Exception as e:
            logger.debug(f"Failed to update session statistics: {e}")
    
    async def _update_session_activity(self):
        """Update last activity timestamp for current session"""
        if not self.current_session_id or not self.tape_store:
            return
        
        try:
            await self.tape_store.query('''
                UPDATE session_meta SET 
                    last_activity = time::now(),
                    updated_at = time::now()
                WHERE session_id = $session_id
            ''', {'session_id': self.current_session_id})
        except Exception as e:
            logger.debug(f"Failed to update session activity: {e}")
    
    async def _extract_facts_async(self, text: str):
        """
        Extract facts and create tracking record
        
        Enhanced with fact_extraction table integration
        """
        try:
            if not self.memory_system:
                return
            
            start_time = time.time()
            
            # Extract facts
            facts_count = await self.memory_system.store_facts(text)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Get the latest conversation entry to link fact extraction
            if self.tape_store and self.current_session_id:
                conversation_entries = await self.tape_store.query('''
                    SELECT id FROM conversation 
                    WHERE session_id = $session_id AND turn_number = $turn_number
                    ORDER BY ts DESC LIMIT 1
                ''', {
                    'session_id': self.current_session_id,
                    'turn_number': self.session_turn_number
                })
                
                if conversation_entries:
                    conversation_id = conversation_entries[0]['id']
                    
                    # Create fact extraction tracking record
                    await self.tape_store.query('''
                        CREATE fact_extraction SET
                            conversation_id = $conversation_id,
                            session_id = $session_id,
                            extracted_at = time::now(),
                            extraction_method = 'spacy',
                            facts_found = $facts_found,
                            entities_found = 0,
                            relationships_found = 0,
                            confidence_score = $confidence_score,
                            processing_time_ms = $processing_time_ms,
                            extraction_version = '1.0',
                            source_content_length = $content_length,
                            source_word_count = $word_count,
                            source_language = 'auto',
                            entity_types = [],
                            relationship_types = [],
                            agent_id = $agent_id,
                            processing_context = $context
                    ''', {
                        'conversation_id': conversation_id,
                        'session_id': self.current_session_id,
                        'facts_found': facts_count,
                        'confidence_score': min(1.0, facts_count / 5.0),  # Heuristic
                        'processing_time_ms': processing_time_ms,
                        'content_length': len(text),
                        'word_count': len(text.split()),
                        'agent_id': self.assistant_id,
                        'context': {'session_turn': self.session_turn_number}
                    })
                    
                    # Update conversation entry
                    await self.tape_store.query('''
                        UPDATE $conversation_id SET facts_extracted = true
                    ''', {'conversation_id': conversation_id})
                    
                    # Update session facts count
                    await self.tape_store.query('''
                        UPDATE session_meta SET 
                            facts_extracted += $facts_count
                        WHERE session_id = $session_id
                    ''', {
                        'session_id': self.current_session_id,
                        'facts_count': facts_count
                    })
            
            self.fact_extractions += 1
            logger.debug(f"🔍 Extracted {facts_count} facts in {processing_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
    
    async def _update_context_with_budget(self, user_text: str):
        """
        Update context maintaining exactly 4096 tokens using unified schema
        """
        try:
            # Build context with fixed token budget
            messages = await self._build_fixed_context_unified(user_text)
            
            # Update context
            if hasattr(self.context, 'messages'):
                self.context.messages = messages
            else:
                # Fallback for different context types
                for msg in messages:
                    self.context.add_message(msg)
            
        except Exception as e:
            logger.error(f"Context update failed: {e}")
    
    async def _build_fixed_context_unified(self, current_input: str) -> List[Dict[str, Any]]:
        """
        Build fixed 4096-token context using unified conversation table
        
        Enhanced with:
        - Query optimization using indexes
        - Better session-aware context building
        - Improved fact integration
        """
        try:
            # Get system prompt
            system_prompt = await self._build_dynamic_system_prompt()
            system_tokens = self.token_counter.count_tokens(system_prompt)
            
            # Get facts context
            facts_context = ""
            if self.memory_system:
                facts_context = await self._get_relevant_facts_context(current_input)
            facts_tokens = self.token_counter.count_tokens(facts_context)
            
            # Get recent conversation using unified schema
            recent_context = await self._get_recent_conversation_unified()
            recent_tokens = self.token_counter.count_tokens(' '.join([msg['content'] for msg in recent_context]))
            
            # Count current input tokens
            input_tokens = self.token_counter.count_tokens(current_input)
            
            # Build final messages
            messages = []
            
            # System message with facts if available
            system_content = system_prompt
            if facts_context:
                system_content += f"\n\n<facts>\n{facts_context}\n</facts>"
            
            messages.append({"role": "system", "content": system_content})
            
            # Add recent conversation
            messages.extend(recent_context)
            
            # Add current user input
            messages.append({"role": "user", "content": current_input})
            
            # Verify token count
            total_tokens = system_tokens + facts_tokens + recent_tokens + input_tokens
            logger.debug(f"🧮 Context: {total_tokens} tokens (system: {system_tokens}, facts: {facts_tokens}, recent: {recent_tokens}, input: {input_tokens})")
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to build fixed context: {e}")
            return [{"role": "user", "content": current_input}]
    
    async def _get_recent_conversation_unified(self) -> List[Dict[str, Any]]:
        """
        Get recent conversation using unified conversation table with performance optimization
        """
        if not self.tape_store or not self.current_session_id:
            return []
        
        try:
            # Use optimized query with session-aware context
            conversations = await self.tape_store.query('''
                SELECT role, content, ts, turn_number
                FROM conversation
                WHERE session_id = $session_id
                ORDER BY ts DESC, turn_number DESC
                LIMIT $limit
            ''', {
                'session_id': self.current_session_id,
                'limit': 20  # Recent conversation limit
            })
            
            # Convert to message format
            messages = []
            for conv in reversed(conversations):  # Reverse to get chronological order
                messages.append({
                    "role": conv['role'],
                    "content": conv['content']
                })
            
            # Trim to fit token budget
            available_tokens = self.budget.recent_conversation
            trimmed_messages = self._trim_messages_to_budget(messages, available_tokens)
            
            logger.debug(f"📚 Recent context: {len(trimmed_messages)} messages from session {self.current_session_id}")
            return trimmed_messages
            
        except Exception as e:
            logger.debug(f"Failed to get recent conversation: {e}")
            return []
    
    def _trim_messages_to_budget(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Trim messages to fit within token budget"""
        if not messages:
            return []
        
        trimmed = []
        current_tokens = 0
        
        # Add messages from newest to oldest until budget is exhausted
        for message in reversed(messages):
            message_tokens = self.token_counter.count_tokens(message['content'])
            if current_tokens + message_tokens <= max_tokens:
                trimmed.insert(0, message)  # Insert at beginning to maintain order
                current_tokens += message_tokens
            else:
                break
        
        return trimmed
    
    async def _build_dynamic_system_prompt(self) -> str:
        """Build dynamic system prompt with session context"""
        base_prompt = f"You are {self.assistant_id}, a helpful AI assistant."
        
        # Add session context if available
        if self.current_session_id:
            session_info = f"\nCurrent session: {self.current_session_id} (Turn {self.session_turn_number})"
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()
                session_info += f"\nSession duration: {int(duration/60)}m {int(duration%60)}s"
            
            base_prompt += session_info
        
        return base_prompt
    
    async def _get_relevant_facts_context(self, query: str) -> str:
        """Get relevant facts for current query"""
        if not self.memory_system:
            return ""
        
        try:
            # Use available token budget for facts
            available_tokens = self.budget.facts
            
            # Get relevant facts (implementation depends on memory system)
            # This is a placeholder - actual implementation would query the facts
            return ""
            
        except Exception as e:
            logger.debug(f"Failed to get facts context: {e}")
            return ""
    
    def _get_speaker_id(self) -> str:
        """Get current speaker ID (placeholder - would integrate with voice recognition)"""
        return "default_user"
    
    def _normalize_user_input(self, text: str) -> str:
        """Normalize user input (same as existing implementation)"""
        if not text or not self._normalization_enabled:
            return text or ""
        
        s = text.strip()
        
        # Apply same normalization as other components
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', s)
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', s)
        s = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', s)
        s = re.sub(r'\b([A-Z])\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]{1,4})\s+([a-z]{2,6})\b', r'\1\2', s)
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        s = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s
    
    def _should_extract_facts(self, text: str) -> bool:
        """Determine if facts should be extracted from text"""
        if not text:
            return False
        
        words = text.split()
        return len(words) >= 3 and len(text) >= 15
    
    async def cleanup(self):
        """Cleanup when processor is destroyed"""
        await self._finalize_current_session()
        logger.info("🧹 SmartContextManagerUnified cleaned up")


# Export for use in pipeline
__all__ = ['SmartContextManagerUnified']
