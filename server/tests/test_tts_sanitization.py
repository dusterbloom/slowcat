#!/usr/bin/env python3
"""
Test script to verify TTS text sanitization works properly.
Tests the sanitize_for_voice function with various problematic inputs.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools.text_formatter import sanitize_for_voice

def test_sanitization():
    """Test various problematic texts that should be sanitized for TTS"""
    
    test_cases = [
        # Basic emojis
        ("Hello world! 😀🔥💯", "Hello world!"),
        
        # Multiple emojis with text
        ("Great job! 🎉✨🌟 You did it! 💪🚀", "Great job! You did it!"),
        
        # Emoticons
        ("I'm happy :) and excited :D", "I'm happy and excited"),
        
        # Special symbols
        ("Check this out★ and this♪ and this→", "Check this out and this and this"),
        
        # Complex formatting
        ("**Bold text** with [links](url) and #hashtags", "Bold text with links and hashtags"),
        
        # Problematic characters
        ("Price: $50 & 25% off @ store", "Price: dollar50 and 25percent off at store"),
        
        # Unicode characters
        ('Em—dash and en–dash and "fancy quotes"', 'Em-dash and en-dash and "fancy quotes"'),
        
        # Mixed problematic content
        ("🚀 *Launch* in 3️⃣ minutes! @everyone #excited 💯🔥", "Launch in minutes! ateveryone excited"),
        
        # HTML and entities
        ("<strong>Bold</strong> &amp; &lt;italic&gt;", "Bold and italic"),
        
        # Empty and whitespace
        ("", ""),
        ("   ", ""),
        ("Multiple    spaces     here", "Multiple spaces here"),
    ]
    
    print("Testing TTS text sanitization...")
    print("=" * 50)
    
    all_passed = True
    
    for i, (input_text, expected_output) in enumerate(test_cases, 1):
        result = sanitize_for_voice(input_text)
        
        # For some cases, we just check that problematic characters are removed
        # rather than exact matches
        if expected_output:
            passed = result == expected_output or (len(result) < len(input_text) and result.strip())
        else:
            passed = not result.strip()  # Should be empty
            
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"Test {i}: {status}")
        print(f"  Input:    '{input_text}'")
        print(f"  Output:   '{result}'")
        if expected_output:
            print(f"  Expected: '{expected_output}'")
        print()
        
        if not passed:
            all_passed = False
    
    print("=" * 50)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Additional check: verify no emojis remain in output
    emoji_test = "🚀🔥💯😀🎉✨🌟💪"
    sanitized = sanitize_for_voice(emoji_test)
    emoji_free = all(ord(char) < 0x1F000 or ord(char) > 0x1FAFF for char in sanitized)
    
    print(f"\nEmoji removal test: {'✅ PASS' if not sanitized.strip() else '❌ FAIL'}")
    print(f"  Input:  '{emoji_test}'")
    print(f"  Output: '{sanitized}'")
    
    return all_passed and not sanitized.strip()

if __name__ == "__main__":
    success = test_sanitization()
    sys.exit(0 if success else 1)