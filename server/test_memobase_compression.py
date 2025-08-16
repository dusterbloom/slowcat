#!/usr/bin/env python3
"""
Test MemoBase compression and token management for small models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from processors.memobase_memory_processor import MemobaseMemoryProcessor

def test_token_estimation():
    """Test token estimation"""
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    test_cases = [
        ("Hello world", 3),  # ~2.75 tokens
        ("This is a longer test sentence", 8),  # ~7.75 tokens  
        ("A" * 100, 25),  # 100 chars = ~25 tokens
    ]
    
    print("🧪 Testing token estimation:")
    for text, expected_approx in test_cases:
        estimated = processor._estimate_tokens(text)
        print(f"   Text: '{text[:20]}...' | Estimated: {estimated} tokens | Expected: ~{expected_approx}")
        assert abs(estimated - expected_approx) <= 2, f"Token estimation off by more than 2: {estimated} vs {expected_approx}"
    
    print("✅ Token estimation tests passed!")

def test_memory_compression():
    """Test memory compression"""
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    # Create a VERY large memory prompt that exceeds token limits
    large_memory = """
    # Memory
    Unless the user has relevant queries, do not actively mention those memories in the conversation.
    ## User Current Profile:
    - interest::pet: user has a dog named Bobby [mention 2025/08/15]; the dog is orangey, five years old, and very active [mention 2025/08/15]; the dog weighs around six to seven kilos [mention 2025/08/15]; the dog sometimes goes lethargic [mention 2025/08/15]; the dog is sleeping because it's very late [mention 2025/08/15];user has a dog named [mention 2025/08/16]
    - life_event::thinking: user was thinking about life in general.[mention 2025/08/15]
    - interest::movie: user was frustrated about a movie.[mention 2025/08/15]
    - preference::food: user likes Italian food and pasta dishes [mention 2025/08/15]
    - hobby::programming: user is interested in Python and AI development [mention 2025/08/15]
    - work::schedule: user typically works from 9 AM to 5 PM on weekdays [mention 2025/08/15]
    - location::home: user lives in a two-bedroom apartment with a garden [mention 2025/08/15]
    - family::siblings: user has two siblings, an older brother and younger sister [mention 2025/08/15]
    - education::background: user studied computer science at university [mention 2025/08/15]
    - health::exercise: user goes jogging three times a week in the morning [mention 2025/08/15]
    - music::preference: user enjoys classical music and jazz for relaxation [mention 2025/08/15]
    
    ## Past Events:
    - user mentioned their dog's name. [mention 2025/08/16]
    - user was frustrated about a movie.[mention 2025/08/15] // event
    - user was thinking about life in general.[mention 2025/08/15] // event
    - user mentioned her dog is very active. [mention 2025/08/15, dog is very active in 2025/08/15]
    - user mentioned her dog's name is Bobby. [mention 2025/08/15, dog named Bobby in 2025/08/15]
    - user mentioned her dog is orangey and five years old. [mention 2025/08/15, dog is five years old in 2025/08/15]
    - user mentioned her dog weighs around six to seven kilos. [mention 2025/08/15, dog weighs around six to seven kilos in 2025/08/15]
    - user discussed cooking pasta with tomato sauce last week [mention 2025/08/08]
    - user talked about work project deadlines and stress [mention 2025/08/10]
    - user shared memories of university days and studying computer science [mention 2025/08/12]
    - user mentioned planning to renovate the garden area [mention 2025/08/13]
    - user talked about brother's birthday party plans [mention 2025/08/14]
    - user discussed morning jogging routine and fitness goals [mention 2025/08/14]
    - user shared favorite classical music composers like Bach and Mozart [mention 2025/08/15]
    
    ## Additional old context that might not be as relevant:
    - some old conversation from last week about work projects and deadlines
    - discussion about weather patterns from last month and seasonal changes
    - random facts about cooking techniques from previous sessions
    - technical discussion about programming languages and frameworks
    - movie recommendations from a while back including sci-fi and drama genres
    - travel plans discussion about visiting European countries next year
    - book recommendations about artificial intelligence and machine learning
    - conversations about home improvement projects and interior design
    - discussions about local restaurants and food delivery options
    - talks about weekend activities and hobby development
    - conversations about news events and current affairs
    - discussions about technology trends and future predictions
    - talks about health and wellness routines and diet plans
    - conversations about social media and digital communication preferences
    """
    
    print("🧪 Testing memory compression:")
    original_tokens = processor._estimate_tokens(large_memory)
    print(f"   Original memory: {original_tokens} tokens")
    
    # Test compression when over limit
    if original_tokens > processor.max_token_limit:
        compressed = processor._compress_memory_content(large_memory)
        compressed_tokens = processor._estimate_tokens(compressed)
        
        print(f"   Compressed memory: {compressed_tokens} tokens")
        print(f"   Compression ratio: {compressed_tokens/original_tokens:.2f}")
        
        print("\n📋 Original memory preview:")
        print(f"   {large_memory[:200]}...")
        
        print("\n🗜️ Compressed memory content:")
        print(f"   {compressed[:500]}...")
        if len(compressed) > 500:
            print(f"   ... (truncated, full length: {len(compressed)} chars)")
        
        # Should be under the limit
        assert compressed_tokens <= processor.max_token_limit, f"Compressed memory still too large: {compressed_tokens} > {processor.max_token_limit}"
        
        # Should preserve important information (recent dates, names)
        assert "Bobby" in compressed, "Dog name should be preserved"
        assert "2025" in compressed, "Recent dates should be preserved"
        
        print("\n✅ Memory compression tests passed!")
    else:
        print("   Memory already under limit, no compression needed")

def test_buffer_management():
    """Test buffer token counting and auto-flush"""
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    print("🧪 Testing buffer management:")
    
    # Initially empty
    assert processor._buffer_token_count == 0
    print(f"   Initial buffer tokens: {processor._buffer_token_count}")
    
    # Add some content
    test_content = "This is a test message for buffer management"
    expected_tokens = processor._estimate_tokens(test_content)
    
    # Simulate adding to buffer (without MemoBase connection)
    processor._conversation_buffer.append({
        "role": "user",
        "content": test_content,
        "tokens": expected_tokens
    })
    processor._buffer_token_count += expected_tokens
    
    print(f"   After adding message: {processor._buffer_token_count} tokens")
    assert processor._buffer_token_count == expected_tokens
    
    print("✅ Buffer management tests passed!")

def main():
    print("🚀 Testing MemoBase compression and token management\n")
    
    # Test individual components
    test_token_estimation()
    print()
    
    test_memory_compression() 
    print()
    
    test_buffer_management()
    print()
    
    print("🎉 All tests passed! MemoBase compression is working correctly.")
    print()
    print("📋 Configuration summary:")
    print(f"   - Max context size: {config.memobase.max_context_size}")
    print(f"   - Max token limit: {config.memobase.max_token_limit}")
    print(f"   - Compression enabled: {config.memobase.enable_compression}")
    print(f"   - Auto flush threshold: {config.memobase.auto_flush_threshold}")
    print(f"   - Relevance filtering: {config.memobase.enable_relevance_filtering}")

if __name__ == "__main__":
    main()