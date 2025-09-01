"""
Pillar 2: The Intelligent Scribe (Application Layer)

Responsibility: To interpret chaotic LLM output and translate it into clean,
well-formed requests for the Guardian. It prepares data but doesn't validate it.
The database is the final authority on what gets stored.

This layer focuses on:
- Language-agnostic normalization
- Structural pre-validation (prevents obvious garbage)
- Clean entity and predicate preparation
- Graceful error handling when Guardian rejects facts
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from dataclasses import dataclass

@dataclass
class CleanFact:
    """A pre-processed fact ready for Guardian validation"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source: str = 'unknown'
    
    def __post_init__(self):
        """Ensure all fields are clean strings"""
        self.subject = str(self.subject).strip()
        self.predicate = str(self.predicate).strip()
        self.object = str(self.object).strip()


class CognitiveScribe:
    """
    The Intelligent Scribe - Pillar 2 of Cognitive Architecture
    
    Prepares chaotic LLM output for the Guardian's validation.
    Acts as a polite translator, not a rigid validator.
    """
    
    def __init__(self):
        # Language-agnostic patterns for cleaning
        self.possessive_patterns = [r"'s$", r"s'$"]
        self.article_patterns = [r"^(the|a|an|The|A|An)\s+"]
        
        # Predicate mapping for common patterns
        self.predicate_mappings = {
            # Spatial relations
            "lives in": "located_in",
            "is in": "located_in", 
            "from": "located_in",
            "resides in": "located_in",
            
            # Social relations
            "works at": "works_at",
            "works for": "works_at",
            "employed by": "works_at",
            
            # Ownership
            "has": "owns",
            "owns": "owns",
            "possesses": "owns",
            
            # Identity
            "is a": "is_a",
            "is an": "is_a",
            "type of": "is_a",
            
            # Temporal
            "before": "before",
            "after": "after",
            "during": "during"
        }
        
        self.stats = {
            'facts_processed': 0,
            'facts_rejected_prevalidation': 0,
            'facts_rejected_by_guardian': 0,
            'facts_accepted': 0
        }
    
    def normalize_entity(self, text: str) -> str:
        """
        Clean and normalize entity text (language-agnostic)
        
        This is the Scribe's preparation - not validation.
        The Guardian will do final validation.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Basic cleaning
        clean = text.strip()
        
        # Remove possessive markers (English-specific but harmless for others)
        for pattern in self.possessive_patterns:
            clean = re.sub(pattern, "", clean)
        
        # Remove leading articles (English but harmless to check)
        for pattern in self.article_patterns:
            clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
        
        # Clean up whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove quotes and basic punctuation that doesn't belong in entity names
        clean = clean.strip('"\'.,!?;:')
        
        return clean
    
    def normalize_predicate(self, predicate: str) -> str:
        """
        Normalize and map predicates to canonical forms
        """
        if not predicate or not isinstance(predicate, str):
            return ""
            
        clean = predicate.strip().lower()
        
        # Map common variations to canonical forms
        return self.predicate_mappings.get(clean, clean)
    
    def is_structurally_sound(self, subject: str, predicate: str, obj: str) -> Tuple[bool, str]:
        """
        Lightweight pre-validation to catch obvious garbage
        
        This is NOT the final validation - that's the Guardian's job.
        This just prevents wasting database round-trips on obviously bad data.
        
        Returns (is_valid, reason_if_invalid)
        """
        # Check for empty fields
        if not subject or not predicate or not obj:
            return False, "Empty field detected"
        
        # Check minimum lengths
        if len(subject) < 2 or len(obj) < 2:
            return False, "Subject or object too short"
            
        if len(predicate) < 2:
            return False, "Predicate too short"
        
        # Check for obvious circular facts (exact match)
        if subject.lower() == obj.lower():
            return False, "Circular fact detected"
        
        # Check for meaningless generic combinations
        if predicate in ['is', 'has', 'was'] and obj.lower() in ['it', 'that', 'this', 'so', 'there', 'here']:
            return False, "Generic predicate with meaningless object"
        
        # Check for malformed subjects that slipped through normalization
        if "'s" in subject or subject.startswith(" ") or subject.endswith(" "):
            return False, "Malformed subject after normalization"
        
        # All structural checks passed
        return True, ""
    
    def prepare_fact(self, raw_subject: str, raw_predicate: str, raw_object: str, 
                    confidence: float = 0.8, source: str = 'extracted') -> Optional[CleanFact]:
        """
        Prepare a raw fact for Guardian validation
        
        Returns None if the fact fails pre-validation
        """
        # Normalize all components
        subject = self.normalize_entity(raw_subject)
        predicate = self.normalize_predicate(raw_predicate)
        obj = self.normalize_entity(raw_object)
        
        # Pre-validation check
        is_valid, reason = self.is_structurally_sound(subject, predicate, obj)
        
        if not is_valid:
            logger.debug(f"Pre-validation rejected fact: {subject}|{predicate}|{obj} - {reason}")
            self.stats['facts_rejected_prevalidation'] += 1
            return None
        
        return CleanFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source
        )
    
    def process_llm_relations(self, llm_output: List[Dict[str, Any]], 
                            source: str = 'llm') -> List[CleanFact]:
        """
        Process raw LLM output into clean facts ready for Guardian
        
        This is the main entry point for the Scribe's work.
        """
        clean_facts = []
        
        for relation in llm_output:
            self.stats['facts_processed'] += 1
            
            try:
                # Extract components from LLM output
                raw_subject = relation.get('subject', '')
                raw_predicate = relation.get('predicate', '')
                raw_object = relation.get('value') or relation.get('object', '')
                
                confidence = float(relation.get('confidence', 0.8))
                
                # Prepare the fact
                clean_fact = self.prepare_fact(
                    raw_subject=raw_subject,
                    raw_predicate=raw_predicate,
                    raw_object=raw_object,
                    confidence=confidence,
                    source=source
                )
                
                if clean_fact:
                    clean_facts.append(clean_fact)
                    logger.debug(f"Prepared fact: {clean_fact.subject} --{clean_fact.predicate}--> {clean_fact.object}")
                
            except Exception as e:
                logger.warning(f"Failed to process relation {relation}: {e}")
                continue
        
        logger.info(f"Scribe processed {len(llm_output)} raw relations → {len(clean_facts)} clean facts")
        return clean_facts
    
    async def submit_to_guardian(self, facts: List[CleanFact], 
                               store_function) -> Dict[str, int]:
        """
        Submit prepared facts to the Guardian for final validation and storage
        
        The Guardian (database) has the final say on what gets stored.
        We handle rejections gracefully and learn from them.
        """
        results = {
            'accepted': 0,
            'rejected': 0,
            'errors': 0
        }
        
        for fact in facts:
            try:
                # Attempt to store - the Guardian will validate
                success = await store_function(
                    subject_name=fact.subject,
                    predicate=fact.predicate,
                    object_name=fact.object,
                    confidence=fact.confidence,
                    source_message_id=None,  # Can be enhanced later
                    embedding=None  # Can be enhanced with embeddings
                )
                
                if success:
                    results['accepted'] += 1
                    self.stats['facts_accepted'] += 1
                    logger.debug(f"✅ Guardian accepted: {fact.subject} {fact.predicate} {fact.object}")
                else:
                    results['rejected'] += 1
                    self.stats['facts_rejected_by_guardian'] += 1
                    logger.debug(f"🚫 Guardian rejected: {fact.subject} {fact.predicate} {fact.object}")
                    
            except Exception as e:
                results['errors'] += 1
                logger.warning(f"Error submitting to Guardian: {fact.subject} {fact.predicate} {fact.object} - {e}")
        
        logger.info(f"Guardian results: {results['accepted']} accepted, {results['rejected']} rejected, {results['errors']} errors")
        return results
    
    def get_scribe_stats(self) -> Dict[str, Any]:
        """Get statistics about the Scribe's performance"""
        total_processed = self.stats['facts_processed']
        if total_processed == 0:
            return self.stats
        
        return {
            **self.stats,
            'pre_validation_success_rate': (total_processed - self.stats['facts_rejected_prevalidation']) / total_processed,
            'guardian_acceptance_rate': self.stats['facts_accepted'] / max(1, total_processed - self.stats['facts_rejected_prevalidation']),
            'overall_success_rate': self.stats['facts_accepted'] / total_processed
        }


# Global scribe instance
_scribe_instance = None

def get_cognitive_scribe() -> CognitiveScribe:
    """Get the global cognitive scribe instance"""
    global _scribe_instance
    if _scribe_instance is None:
        _scribe_instance = CognitiveScribe()
    return _scribe_instance


# Convenience functions for integration
def normalize_entity_name(text: str) -> str:
    """Convenience function for entity normalization"""
    return get_cognitive_scribe().normalize_entity(text)

def normalize_predicate_name(text: str) -> str:
    """Convenience function for predicate normalization"""
    return get_cognitive_scribe().normalize_predicate(text)

async def process_and_store_facts(llm_relations: List[Dict[str, Any]], 
                                store_function, source: str = 'llm') -> Dict[str, int]:
    """
    Main integration function: Process LLM output and store via Guardian
    
    This is the primary interface for the rest of the application.
    """
    scribe = get_cognitive_scribe()
    
    # Phase 1: Scribe preparation
    clean_facts = scribe.process_llm_relations(llm_relations, source)
    
    if not clean_facts:
        logger.warning("No clean facts produced from LLM output")
        return {'accepted': 0, 'rejected': 0, 'errors': 0}
    
    # Phase 2: Guardian submission
    results = await scribe.submit_to_guardian(clean_facts, store_function)
    
    return results