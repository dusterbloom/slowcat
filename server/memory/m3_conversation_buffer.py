#!/usr/bin/env python3
"""
M3 Conversation Buffer
Implements M3-Agent conversation buffering for batch fact extraction

Based on M3-Agent principles:
- Buffer 5 turns or 30 seconds of conversation
- Process chunks with conversational context
- Extract relationships across turns
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    text: str
    timestamp: float
    speaker_id: str
    turn_index: int


@dataclass
class ConversationChunk:
    """Buffered conversation chunk ready for processing"""
    turns: List[ConversationTurn]
    combined_text: str
    start_time: float
    end_time: float
    total_turns: int
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class M3ConversationBuffer:
    """
    M3-style conversation buffer for batch processing
    
    Follows M3-Agent principles:
    - Accumulate conversation turns
    - Process in chunks of 5 turns OR 30 seconds
    - Maintain context across turns
    """
    
    def __init__(self, max_turns: int = 5, max_seconds: float = 30.0):
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.buffer: List[ConversationTurn] = []
        self.turn_counter = 0
        
        logger.info(f"🔄 M3 Conversation Buffer initialized: {max_turns} turns OR {max_seconds}s")
    
    def add_turn(self, text: str, speaker_id: str, timestamp: Optional[float] = None) -> bool:
        """
        Add a conversation turn to buffer
        
        Returns True if buffer should be processed (full or timeout)
        """
        if not text.strip():
            return False
            
        if timestamp is None:
            timestamp = time.time()
            
        turn = ConversationTurn(
            text=text.strip(),
            timestamp=timestamp,
            speaker_id=speaker_id,
            turn_index=self.turn_counter
        )
        
        self.buffer.append(turn)
        self.turn_counter += 1
        
        logger.debug(f"🔄 Added turn {self.turn_counter}: '{text[:50]}...' speaker={speaker_id}")
        
        return self.should_process()
    
    def should_process(self) -> bool:
        """Check if buffer should be processed based on M3 criteria"""
        if not self.buffer:
            return False
            
        # Criterion 1: Maximum turns reached
        if len(self.buffer) >= self.max_turns:
            logger.debug(f"🔄 Buffer ready: {len(self.buffer)} turns (max: {self.max_turns})")
            return True
            
        # Criterion 2: Maximum time elapsed
        if len(self.buffer) > 1:  # Need at least 2 turns for time check
            duration = self.buffer[-1].timestamp - self.buffer[0].timestamp
            if duration >= self.max_seconds:
                logger.debug(f"🔄 Buffer ready: {duration:.1f}s elapsed (max: {self.max_seconds}s)")
                return True
        
        return False
    
    def get_chunk(self) -> Optional[ConversationChunk]:
        """Extract current buffer as a conversation chunk"""
        if not self.buffer:
            return None
            
        # Combine all turn texts
        combined_text = " ".join([turn.text for turn in self.buffer])
        
        chunk = ConversationChunk(
            turns=self.buffer.copy(),
            combined_text=combined_text,
            start_time=self.buffer[0].timestamp,
            end_time=self.buffer[-1].timestamp,
            total_turns=len(self.buffer)
        )
        
        logger.info(f"🔄 Created chunk: {chunk.total_turns} turns, {chunk.duration_seconds:.1f}s, {len(chunk.combined_text)} chars")
        
        return chunk
    
    def clear_buffer(self) -> List[ConversationTurn]:
        """Clear buffer and return processed turns"""
        processed = self.buffer.copy()
        self.buffer.clear()
        
        logger.debug(f"🔄 Buffer cleared: {len(processed)} turns processed")
        return processed
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status for debugging"""
        if not self.buffer:
            return {
                'turns_count': 0,
                'duration_seconds': 0.0,
                'should_process': False,
                'next_trigger': f"{self.max_turns} turns OR {self.max_seconds}s"
            }
            
        duration = self.buffer[-1].timestamp - self.buffer[0].timestamp
        
        return {
            'turns_count': len(self.buffer),
            'duration_seconds': duration,
            'should_process': self.should_process(),
            'turns_until_full': max(0, self.max_turns - len(self.buffer)),
            'seconds_until_timeout': max(0.0, self.max_seconds - duration),
            'buffer_texts': [turn.text[:30] + '...' if len(turn.text) > 30 else turn.text 
                           for turn in self.buffer[-3:]]  # Show last 3 turns
        }
    
    def force_process(self) -> Optional[ConversationChunk]:
        """Force processing of current buffer regardless of thresholds"""
        if not self.buffer:
            return None
            
        logger.info(f"🔄 Force processing buffer: {len(self.buffer)} turns")
        return self.get_chunk()


# Performance test
if __name__ == "__main__":
    
    def test_m3_buffer():
        """Test M3 conversation buffer behavior"""
        
        print("🚀 M3 Conversation Buffer Test")
        print("=" * 50)
        
        buffer = M3ConversationBuffer(max_turns=3, max_seconds=5.0)  # Small thresholds for testing
        
        test_turns = [
            ("I work at Google as a software engineer", "peppi"),
            ("I'm really passionate about AI and machine learning", "peppi"),  
            ("My dog's name is Potola and she's a golden retriever", "peppi"),
            ("We love going on hikes together in the mountains", "peppi"),
            ("What's your favorite hiking trail?", "peppi")
        ]
        
        chunks_processed = 0
        
        for i, (text, speaker) in enumerate(test_turns, 1):
            print(f"\n📝 Turn {i}: '{text}' (speaker: {speaker})")
            
            should_process = buffer.add_turn(text, speaker)
            status = buffer.get_buffer_status()
            
            print(f"   Buffer: {status['turns_count']} turns, {status['duration_seconds']:.1f}s")
            print(f"   Should process: {should_process}")
            
            if should_process:
                chunk = buffer.get_chunk()
                if chunk:
                    chunks_processed += 1
                    print(f"\n🔄 PROCESSING CHUNK {chunks_processed}:")
                    print(f"   Turns: {chunk.total_turns}")
                    print(f"   Duration: {chunk.duration_seconds:.1f}s") 
                    print(f"   Combined: '{chunk.combined_text[:100]}...'")
                    
                    # Clear buffer after processing
                    buffer.clear_buffer()
        
        # Process any remaining turns
        if buffer.buffer:
            print(f"\n🔄 FINAL CHUNK (remaining turns):")
            chunk = buffer.force_process()
            if chunk:
                chunks_processed += 1
                print(f"   Turns: {chunk.total_turns}")
                print(f"   Combined: '{chunk.combined_text[:100]}...'")
        
        print(f"\n✅ Test complete: {chunks_processed} chunks processed from {len(test_turns)} turns")
        
        # Test time-based triggering
        print(f"\n⏰ Testing time-based processing...")
        buffer2 = M3ConversationBuffer(max_turns=10, max_seconds=2.0)
        
        buffer2.add_turn("First message", "peppi")
        time.sleep(1.0)
        buffer2.add_turn("Second message", "peppi")  
        time.sleep(1.5)  # Should trigger timeout
        
        should_process_time = buffer2.should_process()
        print(f"Time-based trigger: {should_process_time}")
        
        if should_process_time:
            chunk = buffer2.get_chunk()
            print(f"Time chunk: {chunk.total_turns} turns in {chunk.duration_seconds:.1f}s")
        
        return chunks_processed
    
    test_m3_buffer()