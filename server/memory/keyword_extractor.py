"""
Smart Keyword Extraction for Memory Queries

Extracts meaningful keywords from natural language queries to improve retrieval
without hardcoding or heavy NLP dependencies. Fast and universal approach.
"""

import re
from typing import List, Set
from dataclasses import dataclass


@dataclass
class KeywordExtractionResult:
    """Result of keyword extraction"""
    keywords: List[str]
    original_query: str
    extraction_method: str
    is_simple_query: bool  # True if query was already a simple keyword


class SmartKeywordExtractor:
    """
    Fast keyword extraction for memory queries
    
    Designed for sub-1ms latency with high-quality keyword extraction
    for natural language queries about personal facts.
    """
    
    def __init__(self):
        # Common stop words that don't help with fact retrieval
        self.stop_words: Set[str] = {
            # Question words
            'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
            'what', 'whats', 'when', 'where', 'why', 'who', 'how', 'which',
            
            # Memory/recall words
            'remember', 'recall', 'know', 'tell', 'show', 'find', 'about',
            
            # Pronouns and articles
            'you', 'i', 'me', 'my', 'your', 'our', 'we', 'they', 'them',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            
            # Common verbs/auxiliaries
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'get', 'got', 'give', 'said', 'say', 'says', 'go', 'goes', 'went',
            
            # Common prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once',
            
            # Other common words
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'very', 'just', 'now',
            'any', 'some', 'no', 'not', 'only', 'own', 'same', 'such', 'here', 'there',
            
            # Vague/unhelpful words
            'anything', 'something', 'nothing', 'everything', 'someone', 'anyone',
            'somewhere', 'anywhere', 'somehow', 'anyhow', 'stuff', 'things', 'thing'
        }
        
        # Preserve these important words even if they might seem like stop words
        self.preserve_words: Set[str] = {
            'name', 'color', 'job', 'work', 'live', 'like', 'love', 'favorite',
            'age', 'old', 'young', 'new', 'big', 'small', 'good', 'bad',
            'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening'
        }
        
    def extract_keywords(self, query: str, max_keywords: int = 3) -> KeywordExtractionResult:
        """
        Extract meaningful keywords from a query
        
        Args:
            query: User query text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            KeywordExtractionResult with extracted keywords
        """
        if not query or not query.strip():
            return KeywordExtractionResult(
                keywords=[],
                original_query=query,
                extraction_method='empty',
                is_simple_query=True
            )
        
        original = query.strip()
        
        # Check if it's already a simple keyword/phrase (no extraction needed)
        if self._is_simple_query(original):
            return KeywordExtractionResult(
                keywords=[original.lower()],
                original_query=original,
                extraction_method='passthrough',
                is_simple_query=True
            )
        
        # Extract keywords from natural language
        keywords = self._extract_from_natural_language(original, max_keywords)
        
        return KeywordExtractionResult(
            keywords=keywords,
            original_query=original,
            extraction_method='nlp_extraction',
            is_simple_query=False
        )
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Check if query is already a simple keyword/phrase
        
        Simple queries don't need extraction:
        - Single words: "dog", "Luna"  
        - Simple phrases: "my dog", "software engineer"
        """
        clean = query.lower().strip()
        
        # Single word
        if len(clean.split()) == 1:
            return True
            
        # Short phrase without question words
        words = clean.split()
        if len(words) <= 3:
            # Check if it contains question indicators
            question_indicators = {'what', 'where', 'when', 'why', 'how', 'do', 'does', 'did', 'can', 'could', '?'}
            if not any(word in question_indicators for word in words):
                return True
        
        # Contains question mark or starts with question word
        if '?' in query or any(clean.startswith(q) for q in ['what', 'where', 'when', 'why', 'how', 'do ', 'does ', 'did ']):
            return False
            
        return False
    
    def _extract_from_natural_language(self, query: str, max_keywords: int) -> List[str]:
        """Extract keywords from natural language query"""
        
        # Clean and normalize
        clean = query.lower().strip()
        
        # Remove punctuation except apostrophes (for contractions)
        clean = re.sub(r"[^\w\s']", ' ', clean)
        
        # Handle contractions
        clean = re.sub(r"what's", 'what is', clean)
        clean = re.sub(r"where's", 'where is', clean)  
        clean = re.sub(r"who's", 'who is', clean)
        clean = re.sub(r"that's", 'that is', clean)
        clean = re.sub(r"it's", 'it is', clean)
        clean = re.sub(r"don't", 'do not', clean)
        clean = re.sub(r"doesn't", 'does not', clean)
        clean = re.sub(r"can't", 'can not', clean)
        clean = re.sub(r"won't", 'will not', clean)
        clean = re.sub(r"n't", ' not', clean)  # General contraction handling
        
        # Handle possessives (dog's → dog)
        clean = re.sub(r"(\w+)'s", r'\1', clean)
        
        # Tokenize
        words = clean.split()
        
        # Extract meaningful keywords
        keywords = []
        for word in words:
            word = word.strip()
            
            # Skip if too short or empty
            if len(word) <= 2:
                continue
                
            # Preserve important words
            if word in self.preserve_words:
                keywords.append(word)
                continue
                
            # Skip stop words
            if word in self.stop_words:
                continue
                
            # Skip numbers unless they're meaningful (years, ages, etc.)
            if word.isdigit() and len(word) < 4:
                continue
                
            keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        # Limit to max_keywords
        return unique_keywords[:max_keywords]


# Global instance for efficient reuse
_keyword_extractor = None


def get_keyword_extractor() -> SmartKeywordExtractor:
    """Get shared keyword extractor instance"""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = SmartKeywordExtractor()
    return _keyword_extractor


def extract_keywords(query: str, max_keywords: int = 3) -> List[str]:
    """
    Convenience function to extract keywords from a query
    
    Args:
        query: User query text
        max_keywords: Maximum keywords to return
        
    Returns:
        List of extracted keywords
    """
    extractor = get_keyword_extractor()
    result = extractor.extract_keywords(query, max_keywords)
    return result.keywords


if __name__ == "__main__":
    # Test the extractor
    extractor = get_keyword_extractor()
    
    test_queries = [
        "Do you remember anything about my dog?",
        "What's my dog's name?", 
        "Tell me about Luna",
        "What color is my dog?",
        "Where do I live?",
        "What's my job?",
        "What programming language do I like?",
        "my dog",
        "dog", 
        "Luna",
        "software engineer"
    ]
    
    print("🔍 Testing Smart Keyword Extraction")
    print("=" * 40)
    
    for query in test_queries:
        result = extractor.extract_keywords(query)
        method = "✓" if result.is_simple_query else "→"
        print(f"{method} '{query}' → {result.keywords}")