"""
Entity Resolution System for deduplicating similar entities across conversations
"""

import spacy
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from loguru import logger
import re


@dataclass
class Entity:
    """Represents a resolved entity"""
    canonical_name: str
    entity_type: str
    aliases: Set[str]
    confidence: float = 1.0
    
    def add_alias(self, alias: str):
        """Add an alias to this entity"""
        self.aliases.add(alias.lower())
    
    def matches(self, name: str, threshold: float = 0.8) -> bool:
        """Check if a name matches this entity"""
        name_lower = name.lower()
        
        # Exact match with canonical name or aliases
        if (name_lower == self.canonical_name.lower() or 
            name_lower in self.aliases):
            return True
        
        # Fuzzy match with canonical name
        if self._fuzzy_match(name_lower, self.canonical_name.lower(), threshold):
            return True
            
        # Fuzzy match with aliases
        for alias in self.aliases:
            if self._fuzzy_match(name_lower, alias, threshold):
                return True
        
        return False
    
    def _fuzzy_match(self, s1: str, s2: str, threshold: float) -> bool:
        """Check if two strings match with fuzzy matching"""
        return SequenceMatcher(None, s1, s2).ratio() >= threshold


class EntityResolver:
    """Resolves and deduplicates entities across conversations"""
    
    def __init__(self):
        self.nlp = None
        self.entities: Dict[str, List[Entity]] = {}  # type -> entities
        self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model for entity analysis"""
        try:
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("✅ Entity resolver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize entity resolver: {e}")
            self.nlp = None
    
    def resolve_entity(self, name: str, entity_type: str = "PERSON") -> str:
        """
        Resolve an entity name to its canonical form
        
        Args:
            name: Entity name to resolve
            entity_type: Type of entity (PERSON, ORG, etc.)
            
        Returns:
            Canonical name for the entity
        """
        if not name or not name.strip():
            return name
            
        name = name.strip()
        
        # Initialize entity type if not seen before
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        # Look for existing entity that matches
        for entity in self.entities[entity_type]:
            if entity.matches(name):
                # Add this name as an alias if it's not already known
                entity.add_alias(name)
                logger.debug(f"Resolved '{name}' → '{entity.canonical_name}' ({entity_type})")
                return entity.canonical_name
        
        # No existing entity found, create new one
        canonical_name = self._canonicalize_name(name, entity_type)
        new_entity = Entity(
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases={name.lower()}
        )
        self.entities[entity_type].append(new_entity)
        
        logger.debug(f"Created new entity: '{canonical_name}' ({entity_type})")
        return canonical_name
    
    def _canonicalize_name(self, name: str, entity_type: str) -> str:
        """Convert name to canonical form"""
        
        if entity_type == "PERSON":
            # For people, use proper case
            return self._proper_case_name(name)
        elif entity_type in ["ORG", "PRODUCT"]:
            # For organizations and products, preserve original casing
            return name.strip()
        else:
            # For other entities, use title case
            return name.strip().title()
    
    def _proper_case_name(self, name: str) -> str:
        """Convert name to proper case (first letter of each word capitalized)"""
        # Handle common name patterns
        words = name.strip().split()
        proper_words = []
        
        for word in words:
            # Handle prefixes like "Dr.", "Mr.", "Mrs."
            if word.lower() in ["dr.", "dr", "mr.", "mr", "mrs.", "mrs", "ms.", "ms"]:
                proper_words.append(word.capitalize() + ("." if not word.endswith(".") else ""))
            # Handle suffixes like "Jr.", "Sr.", "III"
            elif word.lower() in ["jr.", "jr", "sr.", "sr", "iii", "ii", "iv"]:
                proper_words.append(word.upper() + ("." if word.lower().endswith("r") and not word.endswith(".") else ""))
            else:
                proper_words.append(word.capitalize())
        
        return " ".join(proper_words)
    
    def resolve_entities_in_text(self, text: str) -> str:
        """
        Resolve all entities in text to their canonical forms
        
        Args:
            text: Input text with entities to resolve
            
        Returns:
            Text with entities resolved to canonical names
        """
        if not self.nlp or not text:
            return text
            
        try:
            doc = self.nlp(text)
            resolved_text = text
            
            # Process entities in reverse order to maintain character positions
            entities_to_resolve = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                    entities_to_resolve.append((ent.start_char, ent.end_char, ent.text, ent.label_))
            
            # Sort by start position (reverse to maintain positions)
            entities_to_resolve.sort(key=lambda x: x[0], reverse=True)
            
            for start_char, end_char, entity_text, entity_type in entities_to_resolve:
                canonical_name = self.resolve_entity(entity_text, entity_type)
                if canonical_name != entity_text:
                    resolved_text = (resolved_text[:start_char] + 
                                   canonical_name + 
                                   resolved_text[end_char:])
            
            return resolved_text
            
        except Exception as e:
            logger.debug(f"Entity resolution in text failed: {e}")
            return text
    
    def get_entity_info(self, name: str, entity_type: str = None) -> Optional[Entity]:
        """Get information about an entity"""
        if entity_type:
            entity_types = [entity_type]
        else:
            entity_types = list(self.entities.keys())
        
        for etype in entity_types:
            if etype in self.entities:
                for entity in self.entities[etype]:
                    if entity.matches(name):
                        return entity
        
        return None
    
    def merge_entities(self, name1: str, name2: str, entity_type: str) -> bool:
        """
        Manually merge two entities
        
        Args:
            name1: First entity name
            name2: Second entity name  
            entity_type: Type of entities
            
        Returns:
            True if merge was successful
        """
        if entity_type not in self.entities:
            return False
        
        entity1 = None
        entity2 = None
        
        for entity in self.entities[entity_type]:
            if entity.matches(name1):
                entity1 = entity
            if entity.matches(name2):
                entity2 = entity
        
        if entity1 and entity2 and entity1 != entity2:
            # Merge entity2 into entity1
            entity1.aliases.update(entity2.aliases)
            entity1.aliases.add(entity2.canonical_name.lower())
            
            # Remove entity2
            self.entities[entity_type].remove(entity2)
            
            logger.info(f"Merged entities: '{name1}' + '{name2}' → '{entity1.canonical_name}'")
            return True
        
        return False
    
    def list_entities(self, entity_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """List all known entities"""
        result = {}
        
        entity_types = [entity_type] if entity_type else list(self.entities.keys())
        
        for etype in entity_types:
            if etype in self.entities:
                result[etype] = []
                for entity in self.entities[etype]:
                    result[etype].append({
                        'canonical_name': entity.canonical_name,
                        'aliases': list(entity.aliases),
                        'confidence': entity.confidence
                    })
        
        return result


# Global instance
_entity_resolver = None

def get_entity_resolver() -> EntityResolver:
    """Get singleton entity resolver instance"""
    global _entity_resolver
    if _entity_resolver is None:
        _entity_resolver = EntityResolver()
    return _entity_resolver


def resolve_entity(name: str, entity_type: str = "PERSON") -> str:
    """Convenience function for entity resolution"""
    resolver = get_entity_resolver()
    return resolver.resolve_entity(name, entity_type)


def resolve_entities_in_text(text: str) -> str:
    """Convenience function for resolving entities in text"""
    resolver = get_entity_resolver()
    return resolver.resolve_entities_in_text(text)


if __name__ == "__main__":
    # Test entity resolution
    resolver = EntityResolver()
    
    # Test person name resolution
    print("Testing person name resolution:")
    print(f"'sarah' → '{resolver.resolve_entity('sarah', 'PERSON')}'")
    print(f"'Sarah Smith' → '{resolver.resolve_entity('Sarah Smith', 'PERSON')}'")
    print(f"'sarah smith' → '{resolver.resolve_entity('sarah smith', 'PERSON')}'")
    print(f"'Dr. Sarah Smith' → '{resolver.resolve_entity('Dr. Sarah Smith', 'PERSON')}'")
    
    # Test organization resolution
    print(f"\nTesting organization resolution:")
    print(f"'apple' → '{resolver.resolve_entity('apple', 'ORG')}'")
    print(f"'Apple Inc.' → '{resolver.resolve_entity('Apple Inc.', 'ORG')}'")
    print(f"'apple inc' → '{resolver.resolve_entity('apple inc', 'ORG')}'")
    
    # Test text resolution
    print(f"\nTesting text resolution:")
    text = "I met sarah yesterday. Dr. Sarah Smith is a great doctor."
    resolved = resolver.resolve_entities_in_text(text)
    print(f"Original: {text}")
    print(f"Resolved: {resolved}")
    
    # List entities
    print(f"\nKnown entities:")
    entities = resolver.list_entities()
    for etype, elist in entities.items():
        print(f"{etype}:")
        for entity in elist:
            print(f"  {entity['canonical_name']} (aliases: {entity['aliases']})")