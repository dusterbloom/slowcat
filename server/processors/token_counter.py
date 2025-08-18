"""
Accurate token counting for memory system with multiple backends
Supports tiktoken for OpenAI models, with fallbacks for other LLMs
"""

import os
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from loguru import logger

class TokenCounter(ABC):
    """Abstract base class for token counting"""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a message list"""
        pass

class TiktokenCounter(TokenCounter):
    """Accurate token counting using tiktoken (OpenAI models)"""
    
    def __init__(self, model_name: str = "gpt-4"):
        try:
            import tiktoken
            # Map common model names to encodings
            model_encodings = {
                "gpt-4": "cl100k_base",
                "gpt-3.5-turbo": "cl100k_base", 
                "text-davinci-003": "p50k_base",
                "code-davinci-002": "p50k_base"
            }
            
            encoding_name = model_encodings.get(model_name, "cl100k_base")
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.available = True
            logger.info(f"✅ Tiktoken initialized with {encoding_name} encoding")
        except ImportError:
            self.available = False
            logger.warning("Tiktoken not available, falling back to estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if not self.available or not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list with OpenAI message format overhead"""
        if not self.available:
            return 0
        
        total_tokens = 0
        
        for message in messages:
            # Count tokens in content
            content = message.get('content', '')
            role = message.get('role', 'user')
            
            # Base tokens for content
            total_tokens += self.count_tokens(content)
            
            # OpenAI format overhead: each message has ~4 tokens overhead
            total_tokens += 4  # <|start|>role<|end|>content<|end|>
            
            # Role-specific overhead
            if role in ['system', 'assistant', 'user']:
                total_tokens += self.count_tokens(role)
        
        # Every reply is primed with <|start|>assistant<|message|>
        total_tokens += 3
        
        return total_tokens

class SimpleCounter(TokenCounter):
    """Simple token estimation using heuristics"""
    
    def __init__(self, multiplier: float = 1.3):
        self.multiplier = multiplier
        logger.info(f"✅ Simple token counter initialized (word_count * {multiplier})")
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens using word count"""
        if not text:
            return 0
        
        # Count words and multiply by average tokens per word
        word_count = len(text.split())
        
        # For non-English text, adjust multiplier
        if any(ord(char) > 127 for char in text):
            # Non-ASCII characters often use more tokens
            return int(word_count * self.multiplier * 1.5)
        
        return int(word_count * self.multiplier)
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate tokens in message list"""
        total_tokens = 0
        
        for message in messages:
            content = message.get('content', '')
            total_tokens += self.count_tokens(content)
            # Add overhead for message structure
            total_tokens += 10
        
        return total_tokens

class CharacterCounter(TokenCounter):
    """Character-based token estimation for multilingual support"""
    
    def __init__(self):
        # Character to token ratios for different language families
        self.ratios = {
            'latin': 4.0,     # English, Spanish, French, etc.
            'cyrillic': 3.5,  # Russian, Ukrainian, etc.
            'cjk': 1.5,       # Chinese, Japanese, Korean
            'arabic': 3.0,    # Arabic, Hebrew, etc.
        }
        logger.info("✅ Character-based token counter initialized")
    
    def _detect_script(self, text: str) -> str:
        """Detect primary script in text"""
        char_counts = {'latin': 0, 'cyrillic': 0, 'cjk': 0, 'arabic': 0}
        
        for char in text:
            code = ord(char)
            if 0x0020 <= code <= 0x007F or 0x00A0 <= code <= 0x00FF:
                char_counts['latin'] += 1
            elif 0x0400 <= code <= 0x04FF:
                char_counts['cyrillic'] += 1
            elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
                char_counts['cjk'] += 1
            elif 0x0600 <= code <= 0x06FF or 0x0590 <= code <= 0x05FF:
                char_counts['arabic'] += 1
        
        return max(char_counts, key=char_counts.get)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens based on character count and detected script"""
        if not text:
            return 0
        
        script = self._detect_script(text)
        ratio = self.ratios.get(script, 4.0)
        
        return max(1, int(len(text) / ratio))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list"""
        total_tokens = 0
        
        for message in messages:
            content = message.get('content', '')
            total_tokens += self.count_tokens(content)
            # Add overhead for message structure
            total_tokens += 8
        
        return total_tokens

class AdaptiveTokenCounter:
    """Adaptive token counter that chooses best available method"""
    
    def __init__(self, model_name: str = "gpt-4", prefer_tiktoken: bool = True):
        self.counters = []
        
        # Try tiktoken first if preferred and available
        if prefer_tiktoken:
            tiktoken_counter = TiktokenCounter(model_name)
            if tiktoken_counter.available:
                self.counters.append(('tiktoken', tiktoken_counter))
                self.primary = tiktoken_counter
                logger.info("🎯 Using tiktoken for accurate token counting")
            else:
                self.primary = None
        else:
            self.primary = None
        
        # Add fallback counters
        self.counters.append(('simple', SimpleCounter()))
        self.counters.append(('character', CharacterCounter()))
        
        # Use primary or fall back to simple
        if not self.primary:
            self.primary = SimpleCounter()
            logger.info("⚡ Using simple estimation for token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using primary counter"""
        return self.primary.count_tokens(text)
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count message tokens using primary counter"""
        return self.primary.count_message_tokens(messages)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available counting methods"""
        return [name for name, _ in self.counters]
    
    def validate_accuracy(self, test_text: str) -> Dict[str, int]:
        """Compare results across all available counters"""
        results = {}
        for name, counter in self.counters:
            try:
                results[name] = counter.count_tokens(test_text)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results

# Global token counter instance
_global_counter: Optional[AdaptiveTokenCounter] = None

def get_token_counter(model_name: str = None, prefer_tiktoken: bool = True) -> AdaptiveTokenCounter:
    """Get global token counter instance"""
    global _global_counter
    
    if _global_counter is None:
        # Initialize with detected model or default
        if not model_name:
            # Try to detect model from environment
            model_name = os.getenv("LLM_MODEL", "gpt-4")
        
        _global_counter = AdaptiveTokenCounter(model_name, prefer_tiktoken)
    
    return _global_counter

def count_tokens(text: str) -> int:
    """Convenience function to count tokens in text"""
    return get_token_counter().count_tokens(text)

def count_message_tokens(messages: List[Dict[str, str]]) -> int:
    """Convenience function to count tokens in messages"""
    return get_token_counter().count_message_tokens(messages)

# Self-test function
if __name__ == "__main__":
    import asyncio
    
    async def test_token_counters():
        """Test all token counting methods"""
        
        test_texts = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "🎯 This is a test with emojis and special characters! 🚀",
            "Привет, как дела? Это тест на русском языке.",
            "你好，今天天气怎么样？这是中文测试。",
            "مرحبا، كيف حالك؟ هذا اختبار باللغة العربية."
        ]
        
        counter = get_token_counter()
        
        print("🧮 Token Counter Test Results")
        print("=" * 50)
        
        for text in test_texts:
            print(f"\nText: {text[:50]}...")
            results = counter.validate_accuracy(text)
            for method, count in results.items():
                print(f"  {method:12}: {count}")
        
        # Test message counting
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather like?"},
            {"role": "assistant", "content": "I don't have access to current weather data."}
        ]
        
        print(f"\n📨 Message Token Count: {counter.count_message_tokens(test_messages)}")
        print(f"Available methods: {counter.get_available_methods()}")
    
    asyncio.run(test_token_counters())