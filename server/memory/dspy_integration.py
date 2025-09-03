#!/usr/bin/env python
"""
DSPy Integration Module - Replace hybrid fact extraction with production DSPy extractor
"""

import os
from typing import List, Dict, Any
import re
from loguru import logger

# Import our best production DSPy extractor
from memory.dspy_single_call_extractor import DSPySingleCallExtractor
try:
    from memory.dspy_lmstudio_extractor import LMStudioFactExtractor
except Exception:
    LMStudioFactExtractor = None

# Global extractor instance (initialized once)
_dspy_extractor = None


def get_dspy_extractor():
    """Get or create the global DSPy extractor instance"""
    global _dspy_extractor
    
    if _dspy_extractor is None:
        try:
            # Switch to DSPySingleCallExtractor for better quality
            mode = os.getenv('DSPY_EXTRACTOR_MODE', 'dspy').lower()  # lmstudio | dspy
            if mode == 'lmstudio' and LMStudioFactExtractor is not None:
                _dspy_extractor = LMStudioFactExtractor()
                logger.info("🚀 Using LMStudioFactExtractor (legacy working pipeline)")
            else:
                _dspy_extractor = DSPySingleCallExtractor()
                logger.info("🚀 Using DSPySingleCallExtractor (improved prompts)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize fact extractor: {e}")
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
        # 1) Segment input into short, simple clauses for small models
        t = (text or '').strip()
        if not t:
            return []
        # Sentence boundaries
        parts = re.split(r"(?<=[\.!\?])\s+", t)
        segs: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Further split on comma + coordinator to reduce run-ons
            chunks = re.split(r",\s+(?=(?:and|but|then)\b)", p)
            for c in chunks:
                c = c.strip()
                if len(c.split()) >= 3:
                    segs.append(c)
        if not segs:
            segs = [t]

        # 2) Extract per segment (the extractor prompt already limits to 1 relation)
        triples: List[Dict[str, Any]] = []
        seen = set()
        pronouns = {"i","me","my","mine","we","us","our","ours","you","your","yours","he","him","his","she","her","hers","it","its","they","them","their","theirs"}

        for seg in segs:
            rels = extractor.extract_facts(seg) or []
            for r in rels:
                subj = (r.get('subject') or '').strip() or 'user'
                pred = (r.get('predicate') or '').strip()
                obj = (r.get('object') or '').strip()
                # Basic filtering: predicate short token; object <=3 words; no pronoun object
                if not pred or len(pred.split()) > 2:
                    continue
                if not obj or len(obj.split()) > 3 or obj.lower() in pronouns:
                    continue
                key = (subj.lower(), pred.lower(), obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                triples.append({'subject': subj, 'predicate': pred, 'object': obj, 'confidence': float(r.get('confidence', 0.7))})
            # Keep it compact
            if len(triples) >= 3:
                break

        # 3) Convert to Slowcat expected format (value instead of object)
        converted_facts = [
            {
                'subject': tr['subject'],
                'predicate': tr['predicate'],
                'value': tr['object'],
                'confidence': tr.get('confidence', 0.7),
            }
            for tr in triples
        ]

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
