#!/usr/bin/env python
"""
DSPy Integration Module - Replace hybrid fact extraction with production DSPy extractor
"""

import os
from typing import List, Dict, Any
from loguru import logger

# Import our best production DSPy extractor
from memory.dspy_single_call_extractor import DSPySingleCallExtractor

# Global extractor instance (initialized once)
_dspy_extractor = None


def get_dspy_extractor():
    """Get or create the global DSPy extractor instance"""
    global _dspy_extractor
    
    if _dspy_extractor is None:
        try:
            _dspy_extractor = DSPySingleCallExtractor()
            logger.info("🚀 DSPy fact extractor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DSPy extractor: {e}")
            return None
    
    return _dspy_extractor


def extract_facts_from_text_dspy(text: str) -> List[Dict[str, Any]]:
    """
    Production-ready fact extraction using DSPy + Qwen 2.5 0.5B
    
    This replaces the hybrid extractor with our optimized single-call approach
    that delivers both personal context and factual precision.
    
    Args:
        text: Input text to extract facts from
        
    Returns:
        List of facts in the format expected by the Slowcat system:
        [{"subject": str, "predicate": str, "value": str, "confidence": float}, ...]
    """
    
    extractor = get_dspy_extractor()
    if not extractor:
        logger.warning("⚠️ DSPy extractor not available, using fallback")
        return []
    
    try:
        # Extract facts using our production DSPy system
        facts = extractor.extract_facts(text)
        
        # Convert to Slowcat's expected format
        converted_facts = []
        for fact in facts:
            converted_facts.append({
                'subject': fact.get('subject', ''),
                'predicate': fact.get('predicate', ''),
                'value': fact.get('object', ''),  # DSPy uses 'object', Slowcat expects 'value'
                'confidence': fact.get('confidence', 0.8)
            })
        
        logger.debug(f"🚀 DSPy extracted {len(converted_facts)} facts from: '{text[:50]}...'")
        return converted_facts
        
    except Exception as e:
        logger.error(f"❌ DSPy fact extraction failed: {e}")
        return []


def enable_dspy_extraction():
    """
    Enable DSPy fact extraction by monkey-patching the hybrid extractor
    
    Call this in run_bot.sh or during system initialization to replace
    the hybrid extractor with our optimized DSPy version.
    """
    try:
        # Import and replace in the hybrid module
        import memory.hybrid_fact_extractor as hybrid_module
        
        # Replace the actual function in the module
        if hasattr(hybrid_module, 'extract_facts_from_text'):
            hybrid_module.extract_facts_from_text = extract_facts_from_text_dspy
            logger.info("✅ Replaced hybrid_fact_extractor.extract_facts_from_text")
        
        # Also replace in the memory package's __init__
        import memory
        if hasattr(memory, 'extract_facts_from_text'):
            memory.extract_facts_from_text = extract_facts_from_text_dspy
            logger.info("✅ Replaced memory.extract_facts_from_text")
        
        # Update sys.modules to ensure imports get the new version
        import sys
        if 'memory' in sys.modules:
            sys.modules['memory'].extract_facts_from_text = extract_facts_from_text_dspy
        if 'memory.hybrid_fact_extractor' in sys.modules:
            sys.modules['memory.hybrid_fact_extractor'].extract_facts_from_text = extract_facts_from_text_dspy
        
        logger.info("🎯 DSPy fact extraction enabled - hybrid extractor replaced")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to enable DSPy extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dspy_integration():
    """Test the DSPy integration with Slowcat's expected format"""
    
    logger.info("🧪 Testing DSPy integration...")
    
    test_cases = [
        "My dog Rex is a brown German Shepherd",
        "I work at Google in Seattle",
        "Paris is the capital of France"
    ]
    
    for text in test_cases:
        logger.info(f"📝 Testing: '{text}'")
        
        facts = extract_facts_from_text_dspy(text)
        
        if facts:
            logger.info(f"✅ Extracted {len(facts)} facts:")
            for fact in facts:
                logger.info(f"  • {fact['subject']} → {fact['predicate']} → {fact['value']} (conf: {fact['confidence']})")
        else:
            logger.warning(f"⚠️ No facts extracted from: '{text}'")
    
    logger.info("✅ DSPy integration test complete!")


if __name__ == "__main__":
    # Test the integration
    test_dspy_integration()
    
    # Test enabling the patch
    if enable_dspy_extraction():
        logger.info("✅ DSPy extraction successfully enabled!")
    else:
        logger.error("❌ Failed to enable DSPy extraction!")