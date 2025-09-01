"""
Data Validation and Normalization for Knowledge Graph

This module provides functions to ensure the integrity and quality of facts
before they are stored in the knowledge graph.
"""
from loguru import logger

def normalize_entity(text: str) -> str:
    """
    Cleans and normalizes an entity name for consistency.

    - Removes leading/trailing whitespace.
    - Removes possessive markers ('s, s').
    - Removes leading articles (the, a, an).
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    
    # Remove possessive markers (works for English, generally harmless for others)
    if text.endswith("'s"):
        text = text[:-2]
    elif text.endswith("s'"):
        text = text[:-1]

    # Remove leading articles (English-specific but low risk)
    words = text.split()
    if words and words[0].lower() in ['the', 'a', 'an']:
        text = ' '.join(words[1:])

    return text.strip()


def validate_fact(subject: str, predicate: str, obj: str) -> bool:
    """
    Validates a subject-predicate-object triple for semantic and structural integrity.

    Returns:
        bool: True if the fact is valid, False otherwise.
    """
    # 1. Structural validation: Ensure no empty parts
    if not subject or not predicate or not obj:
        logger.warning(f"Validation failed: Empty part in fact ({subject}, {predicate}, {obj})")
        return False

    # 2. Reject circular facts (subject is the same as object)
    if subject.strip().lower() == obj.strip().lower():
        logger.warning(f"Validation failed: Circular fact detected ('{subject}' -> '{predicate}' -> '{obj}')")
        return False

    # 3. Check for leading/trailing whitespace in subject or object
    if subject != subject.strip() or obj != obj.strip():
        logger.warning(f"Validation failed: Leading/trailing whitespace detected in ('{subject}', '{obj}')")
        return False

    # 4. Reject generic, low-value predicates
    # Example: "the space industry" -> "is" -> "the space industry" is caught by circular check.
    # This handles cases like "something" -> "is" -> "it".
    if predicate.strip().lower() in ['is', 'has', 'was', 'are'] and len(obj.split()) < 2:
        # Allow if the object is a multi-word phrase, which has more semantic value.
        logger.warning(f"Validation failed: Generic predicate '{predicate}' with short object '{obj}'")
        return False

    return True
