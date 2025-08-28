"""
Coreference Resolution for spaCy pipeline using dependency parsing and NER
"""

import spacy
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


class CoreferenceResolver:
    """Resolve pronouns to their antecedents using spaCy linguistic features"""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model for coreference resolution"""
        try:
            # Use the same model as fact extractor for consistency
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("✅ Coreference resolver initialized with transformer model")
        except Exception as e:
            logger.error(f"Failed to initialize coreference resolver: {e}")
            self.nlp = None
    
    def resolve_coreferences(self, text: str) -> str:
        """
        Resolve pronouns in text to their antecedents
        
        Args:
            text: Input text with potential pronouns
            
        Returns:
            Text with pronouns resolved to entities where possible
        """
        if not self.nlp:
            return text
            
        try:
            doc = self.nlp(text)
            resolved_tokens = []
            
            # Track entities and their positions
            entities = self._extract_entities_with_positions(doc)
            
            for token in doc:
                if self._is_resolvable_pronoun(token):
                    # Try to resolve this pronoun
                    antecedent = self._resolve_pronoun(token, entities, doc)
                    if antecedent:
                        resolved_tokens.append(antecedent)
                        logger.debug(f"Resolved '{token.text}' → '{antecedent}'")
                    else:
                        resolved_tokens.append(token.text)
                else:
                    resolved_tokens.append(token.text)
                    
                # Add whitespace if it exists
                if token.whitespace_:
                    resolved_tokens.append(token.whitespace_)
            
            resolved_text = "".join(resolved_tokens)
            return resolved_text
            
        except Exception as e:
            logger.debug(f"Coreference resolution failed: {e}")
            return text
    
    def _extract_entities_with_positions(self, doc) -> List[Dict[str, Any]]:
        """Extract named entities with their positions and properties"""
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start,
                'end': ent.end,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
        
        # Also extract proper nouns that might not be caught as entities
        for token in doc:
            if (token.pos_ == "PROPN" and 
                not any(ent['start'] <= token.i < ent['end'] for ent in entities)):
                entities.append({
                    'text': token.text,
                    'label': 'PROPN',
                    'start': token.i,
                    'end': token.i + 1,
                    'start_char': token.idx,
                    'end_char': token.idx + len(token.text)
                })
        
        return entities
    
    def _is_resolvable_pronoun(self, token) -> bool:
        """Check if token is a pronoun we can resolve"""
        return (token.pos_ == "PRON" and 
                token.lemma_.lower() in ["he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"])
    
    def _resolve_pronoun(self, pronoun_token, entities: List[Dict], doc) -> Optional[str]:
        """
        Resolve a pronoun to its most likely antecedent
        
        Uses several heuristics:
        1. Gender agreement (he/she with PERSON entities)
        2. Number agreement (they/them with plural or multiple entities)
        3. Proximity (closer entities are more likely antecedents)
        4. Dependency relationships
        """
        
        # Get pronoun properties
        pronoun_lemma = pronoun_token.lemma_.lower()
        is_singular = pronoun_lemma in ["he", "she", "it", "him", "her", "his", "hers", "its"]
        is_plural = pronoun_lemma in ["they", "them", "their"]
        is_masculine = pronoun_lemma in ["he", "him", "his"]
        is_feminine = pronoun_lemma in ["she", "her", "hers"]
        is_neuter = pronoun_lemma in ["it", "its"]
        
        # Find candidate antecedents
        candidates = []
        
        for entity in entities:
            # Skip entities that come after the pronoun
            if entity['start'] >= pronoun_token.i:
                continue
                
            # Calculate distance (closer is better)
            distance = pronoun_token.i - entity['end']
            
            # Score based on type matching
            type_score = 0
            if entity['label'] == 'PERSON':
                if is_masculine or is_feminine:
                    type_score = 3  # High score for person with gendered pronoun
                elif is_plural:
                    type_score = 2  # Medium score for person with plural pronoun
                else:
                    type_score = 1  # Low score for person with "it"
            elif entity['label'] in ['ORG', 'GPE', 'EVENT']:
                if is_neuter or is_plural:
                    type_score = 2  # Good score for organizations with "it/they"
                else:
                    type_score = 1
            elif entity['label'] == 'PROPN':
                # Could be a person's name
                if is_masculine or is_feminine:
                    type_score = 2
                else:
                    type_score = 1
            else:
                if is_neuter:
                    type_score = 1  # Basic score for other entities with "it"
            
            # Check for dependency relationships
            dependency_score = 0
            entity_tokens = [doc[i] for i in range(entity['start'], entity['end'])]
            for entity_token in entity_tokens:
                if self._has_dependency_relationship(entity_token, pronoun_token):
                    dependency_score = 2
                    break
            
            # Combined score (closer entities and better type matches score higher)
            total_score = type_score + dependency_score - (distance * 0.1)
            
            if total_score > 0:
                candidates.append({
                    'entity': entity,
                    'score': total_score,
                    'distance': distance
                })
        
        # Return the highest scoring candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['score'])
            return best_candidate['entity']['text']
        
        return None
    
    def _has_dependency_relationship(self, entity_token, pronoun_token) -> bool:
        """Check if there's a dependency relationship between entity and pronoun"""
        
        # Check if they're in a subject-object relationship
        for ancestor in pronoun_token.ancestors:
            if ancestor == entity_token:
                return True
        
        for ancestor in entity_token.ancestors:
            if ancestor == pronoun_token:
                return True
        
        # Check if they share a common head
        if (entity_token.head == pronoun_token.head and 
            entity_token.head.pos_ == "VERB"):
            return True
        
        return False
    
    def resolve_text_with_context(self, text: str, conversation_context: List[str] = None) -> str:
        """
        Resolve coreferences with additional conversation context
        
        Args:
            text: Current text to resolve
            conversation_context: Previous sentences for additional context
            
        Returns:
            Text with resolved pronouns
        """
        if not conversation_context:
            return self.resolve_coreferences(text)
        
        # Combine context with current text for better resolution
        full_context = " ".join(conversation_context[-3:] + [text])  # Use last 3 sentences
        resolved_full = self.resolve_coreferences(full_context)
        
        # Extract just the resolved part corresponding to input text
        if len(conversation_context) > 0:
            context_length = len(" ".join(conversation_context[-3:]) + " ")
            resolved_text = resolved_full[context_length:]
        else:
            resolved_text = resolved_full
        
        return resolved_text


# Global instance
_coreference_resolver = None

def get_coreference_resolver() -> CoreferenceResolver:
    """Get singleton coreference resolver instance"""
    global _coreference_resolver
    if _coreference_resolver is None:
        _coreference_resolver = CoreferenceResolver()
    return _coreference_resolver


def resolve_coreferences(text: str, context: List[str] = None) -> str:
    """Convenience function for coreference resolution"""
    resolver = get_coreference_resolver()
    if context:
        return resolver.resolve_text_with_context(text, context)
    else:
        return resolver.resolve_coreferences(text)


if __name__ == "__main__":
    # Test coreference resolution
    test_texts = [
        "Sarah went to the store. She bought milk.",
        "John and Mary are friends. They live in the same neighborhood.",
        "The company announced layoffs. It will affect 100 employees.",
        "My dog Buddy is very smart. He can fetch the newspaper.",
        "I met Dr. Smith yesterday. She is a great doctor."
    ]
    
    resolver = CoreferenceResolver()
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        resolved = resolver.resolve_coreferences(text)
        print(f"Resolved: {resolved}")