#!/usr/bin/env python3
"""
Entity Post-Processor for Sherpa-ONNX STT Results

This module provides post-processing capabilities to improve accuracy 
for names, URLs, and technical terms in STT transcriptions.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityCorrection:
    """Container for entity correction rules"""
    pattern: str  # Regex pattern to match
    replacement: str  # Replacement text
    entity_type: str  # Type of entity (url, name, email, etc.)
    confidence: float = 1.0  # Confidence in this correction

class EntityPostProcessor:
    """Post-processor to improve entity recognition in STT output"""
    
    def __init__(self):
        self.url_patterns = self._create_url_patterns()
        self.email_patterns = self._create_email_patterns()
        self.name_patterns = self._create_name_patterns()
        self.tech_patterns = self._create_tech_patterns()
        
        # Custom correction rules
        self.custom_corrections = []
        
        # Common entity vocabulary
        self.entity_vocabulary = self._load_entity_vocabulary()
    
    def _create_url_patterns(self) -> List[EntityCorrection]:
        """Create patterns for URL corrections"""
        patterns = [
            # Common URL mispronunciations
            EntityCorrection(
                pattern=r'\b(?:bbb|BBB|b b b)[\s\.](?:com|COM)(?:/(?:news|NEWS))?\b',
                replacement='bbc.com/news',
                entity_type='url'
            ),
            EntityCorrection(
                pattern=r'\b(?:cnn|CNN|c n n)[\s\.](?:com|COM)\b',
                replacement='cnn.com',
                entity_type='url'
            ),
            EntityCorrection(
                pattern=r'\b(?:github|GITHUB|git hub)[\s\.](?:com|COM)\b',
                replacement='github.com',
                entity_type='url'
            ),
            EntityCorrection(
                pattern=r'\b(?:google|GOOGLE)[\s\.](?:com|COM)\b',
                replacement='google.com',
                entity_type='url'
            ),
            # Generic patterns for common domains
            EntityCorrection(
                pattern=r'\b([a-zA-Z]+)[\s]+dot[\s]+(?:com|org|net|edu|gov)\b',
                replacement=r'\1.\2',
                entity_type='url'
            ),
            EntityCorrection(
                pattern=r'\b([a-zA-Z]+)[\s]+(?:com|COM)\b',
                replacement=r'\1.com',
                entity_type='url'
            ),
        ]
        return patterns
    
    def _create_email_patterns(self) -> List[EntityCorrection]:
        """Create patterns for email corrections"""
        patterns = [
            # Common email mispronunciations
            EntityCorrection(
                pattern=r'\b([a-zA-Z]+)[\s]*at[\s]*([a-zA-Z]+)[\s]*dot[\s]*(?:com|org|edu|net)\b',
                replacement=r'\1@\2.\3',
                entity_type='email'
            ),
            EntityCorrection(
                pattern=r'\b([a-zA-Z]+)[\s]*[@][\s]*([a-zA-Z]+)[\s]*dot[\s]*([a-zA-Z]+)\b',
                replacement=r'\1@\2.\3',
                entity_type='email'
            ),
        ]
        return patterns
    
    def _create_name_patterns(self) -> List[EntityCorrection]:
        """Create patterns for proper name corrections"""
        patterns = [
            # Common name corrections
            EntityCorrection(
                pattern=r'\bmicro soft\b',
                replacement='Microsoft',
                entity_type='company'
            ),
            EntityCorrection(
                pattern=r'\bopen ai\b',
                replacement='OpenAI',
                entity_type='company'
            ),
            EntityCorrection(
                pattern=r'\banthropic\b',
                replacement='Anthropic',
                entity_type='company'
            ),
            EntityCorrection(
                pattern=r'\bgoogle\b',
                replacement='Google',
                entity_type='company'
            ),
        ]
        return patterns
    
    def _create_tech_patterns(self) -> List[EntityCorrection]:
        """Create patterns for technical term corrections"""
        patterns = [
            # API and technical terms
            EntityCorrection(
                pattern=r'\bapi\b',
                replacement='API',
                entity_type='technical'
            ),
            EntityCorrection(
                pattern=r'\bhttp\b',
                replacement='HTTP',
                entity_type='technical'
            ),
            EntityCorrection(
                pattern=r'\bhttps\b',
                replacement='HTTPS',
                entity_type='technical'
            ),
            EntityCorrection(
                pattern=r'\bjson\b',
                replacement='JSON',
                entity_type='technical'
            ),
        ]
        return patterns
    
    def _load_entity_vocabulary(self) -> Dict[str, Set[str]]:
        """Load common entity vocabularies"""
        return {
            'companies': {
                'Microsoft', 'Google', 'Apple', 'Amazon', 'Meta', 
                'OpenAI', 'Anthropic', 'Tesla', 'NVIDIA', 'IBM'
            },
            'domains': {
                'com', 'org', 'net', 'edu', 'gov', 'io', 'ai', 'co'
            },
            'protocols': {
                'HTTP', 'HTTPS', 'FTP', 'SSH', 'API', 'REST', 'JSON', 'XML'
            }
        }
    
    def add_custom_correction(self, pattern: str, replacement: str, entity_type: str):
        """Add a custom correction rule"""
        correction = EntityCorrection(
            pattern=pattern,
            replacement=replacement,
            entity_type=entity_type
        )
        self.custom_corrections.append(correction)
    
    def process_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Process text and apply entity corrections
        
        Returns:
            tuple: (corrected_text, correction_stats)
        """
        corrected_text = text
        correction_stats = {
            'url': 0,
            'email': 0,
            'name': 0,
            'technical': 0,
            'custom': 0
        }
        
        # Apply URL corrections
        for pattern in self.url_patterns:
            if re.search(pattern.pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(
                    pattern.pattern, 
                    pattern.replacement, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                correction_stats['url'] += 1
        
        # Apply email corrections
        for pattern in self.email_patterns:
            if re.search(pattern.pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(
                    pattern.pattern, 
                    pattern.replacement, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                correction_stats['email'] += 1
        
        # Apply name corrections
        for pattern in self.name_patterns:
            if re.search(pattern.pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(
                    pattern.pattern, 
                    pattern.replacement, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                correction_stats['name'] += 1
        
        # Apply technical corrections
        for pattern in self.tech_patterns:
            if re.search(pattern.pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(
                    pattern.pattern, 
                    pattern.replacement, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                correction_stats['technical'] += 1
        
        # Apply custom corrections
        for pattern in self.custom_corrections:
            if re.search(pattern.pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(
                    pattern.pattern, 
                    pattern.replacement, 
                    corrected_text, 
                    flags=re.IGNORECASE
                )
                correction_stats['custom'] += 1
        
        return corrected_text, correction_stats
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            'urls': [],
            'emails': [],
            'names': [],
            'technical': []
        }
        
        # Extract URLs
        url_pattern = r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        entities['urls'] = re.findall(url_pattern, text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract proper names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
        entities['names'] = re.findall(name_pattern, text)
        
        # Extract technical terms
        tech_words = []
        for word in text.split():
            if word.upper() in self.entity_vocabulary['protocols']:
                tech_words.append(word.upper())
        entities['technical'] = tech_words
        
        return entities
    
    def calculate_entity_accuracy(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate entity-specific accuracy metrics"""
        ref_entities = self.extract_entities(reference)
        hyp_entities = self.extract_entities(hypothesis)
        
        accuracy = {}
        
        for entity_type in ref_entities.keys():
            ref_set = set(ref_entities[entity_type])
            hyp_set = set(hyp_entities[entity_type])
            
            if len(ref_set) == 0:
                accuracy[entity_type] = 1.0 if len(hyp_set) == 0 else 0.0
            else:
                correct = len(ref_set & hyp_set)
                accuracy[entity_type] = correct / len(ref_set)
        
        return accuracy
    
    def save_corrections_log(self, corrections_log: List[Dict], output_file: Path):
        """Save corrections log to file"""
        with open(output_file, 'w') as f:
            json.dump(corrections_log, f, indent=2)
    
    def load_custom_corrections(self, corrections_file: Path):
        """Load custom corrections from JSON file"""
        if corrections_file.exists():
            with open(corrections_file) as f:
                corrections_data = json.load(f)
                for correction_data in corrections_data:
                    self.add_custom_correction(
                        correction_data['pattern'],
                        correction_data['replacement'], 
                        correction_data['entity_type']
                    )

def test_entity_postprocessor():
    """Test the entity post-processor"""
    processor = EntityPostProcessor()
    
    test_cases = [
        "Please visit bbb dot com slash news for updates",
        "Send email to john at company dot com",
        "The micro soft teams meeting is at 3pm",
        "Check the api documentation at github dot com",
        "Contact open ai for more information"
    ]
    
    print("Entity Post-Processor Test Results:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        corrected, stats = processor.process_text(text)
        print(f"\nTest {i}:")
        print(f"Original:  {text}")
        print(f"Corrected: {corrected}")
        print(f"Stats:     {stats}")
        
        # Extract entities
        entities = processor.extract_entities(corrected)
        if any(entities.values()):
            print(f"Entities:  {entities}")

if __name__ == "__main__":
    test_entity_postprocessor()