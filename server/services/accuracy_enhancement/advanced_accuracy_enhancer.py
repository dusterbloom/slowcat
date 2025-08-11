#!/usr/bin/env python3
"""
Advanced Accuracy Enhancer for Sherpa-ONNX STT

Implements multiple language-agnostic post-processing techniques:
1. Phonetic similarity correction (Soundex, Metaphone, Levenshtein)  
2. Dynamic vocabulary extraction from user context
3. NER-based entity identification and correction
4. LLM-powered contextual correction
5. Confidence-based multi-pass processing
6. Real-time vocabulary learning

No static hotwords.txt required - fully adaptive and self-improving.
"""

import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import logging

# Core libraries for accuracy enhancement
try:
    import spacy
    from spacy import displacy
    import jellyfish
    from fuzzywuzzy import fuzz, process
    import editdistance
    HAS_NLP_LIBS = True
except ImportError:
    HAS_NLP_LIBS = False
    logging.warning("NLP libraries not available. Install: pip install spacy jellyfish fuzzywuzzy python-Levenshtein editdistance")

# Phonetic libraries
try:
    from metaphone import dm as double_metaphone
    from phonetics import dmetaphone
    HAS_PHONETIC_LIBS = True
except ImportError:
    HAS_PHONETIC_LIBS = False
    logging.warning("Phonetic libraries not available. Install: pip install metaphone phonetics")

# LLM integration for contextual correction
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.info("OpenAI library not available for LLM correction")

# Ollama fallback
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logging.info("Ollama not available for LLM correction")

logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Result of accuracy enhancement processing"""
    original_text: str
    corrected_text: str
    confidence_score: float
    corrections_applied: List[Dict[str, Any]]
    processing_time_ms: float
    method_used: str
    
@dataclass 
class EntityCandidate:
    """Candidate entity for correction"""
    original: str
    corrected: str
    confidence: float
    method: str
    context_start: int
    context_end: int

class DynamicVocabularyExtractor:
    """Extract vocabulary from user context without maintaining static files"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.vocabulary_cache = {}
        self.last_update = 0
        self.update_interval = 3600  # 1 hour
        
    async def extract_user_vocabulary(self, max_items: int = 10000) -> Set[str]:
        """Extract vocabulary from user's environment"""
        current_time = time.time()
        
        # Use cached vocabulary if recent
        if (current_time - self.last_update) < self.update_interval and self.vocabulary_cache:
            return self.vocabulary_cache.get('words', set())
        
        vocabulary = set()
        
        try:
            # Extract from conversation history
            memory_files = Path("../../../data/memory").glob("*.json")
            for memory_file in memory_files:
                try:
                    with open(memory_file) as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'conversations' in data:
                            for conv in data['conversations']:
                                text = conv.get('user_input', '') + ' ' + conv.get('assistant_response', '')
                                vocabulary.update(self._extract_entities_from_text(text))
                except Exception as e:
                    logger.debug(f"Could not process memory file {memory_file}: {e}")
            
            # Extract from recent transcripts
            transcript_dir = Path("../../../data/transcripts")
            if transcript_dir.exists():
                for transcript_file in transcript_dir.glob("*.txt"):
                    try:
                        with open(transcript_file) as f:
                            text = f.read()
                            vocabulary.update(self._extract_entities_from_text(text))
                    except Exception as e:
                        logger.debug(f"Could not process transcript {transcript_file}: {e}")
            
            # Extract from user documents (safely, common locations)
            user_docs = [
                Path.home() / "Documents",
                Path.home() / "Desktop", 
                Path.home() / "Downloads"
            ]
            
            for doc_dir in user_docs:
                if doc_dir.exists():
                    for text_file in doc_dir.rglob("*.txt"):
                        if text_file.stat().st_size < 100000:  # Only small files
                            try:
                                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    text = f.read()[:5000]  # First 5K chars only
                                    vocabulary.update(self._extract_entities_from_text(text))
                            except Exception:
                                continue
                        
                        if len(vocabulary) > max_items:
                            break
                
                if len(vocabulary) > max_items:
                    break
            
        except Exception as e:
            logger.warning(f"Error extracting user vocabulary: {e}")
        
        # Cache results
        vocabulary = set(list(vocabulary)[:max_items])  # Limit size
        self.vocabulary_cache = {'words': vocabulary, 'timestamp': current_time}
        self.last_update = current_time
        
        logger.info(f"Extracted {len(vocabulary)} terms from user context")
        return vocabulary
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract potential entities from text using heuristics"""
        entities = set()
        
        # URL patterns
        url_pattern = r'https?://[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        entities.update(re.findall(url_pattern, text))
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities.update(re.findall(email_pattern, text))
        
        # Capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]{2,}\b'
        entities.update(re.findall(capitalized_pattern, text))
        
        # Technical terms (camelCase, snake_case)
        tech_pattern = r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z]+\b'
        entities.update(re.findall(tech_pattern, text))
        
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities.update(re.findall(phone_pattern, text))
        
        return {entity for entity in entities if len(entity) > 2}

class PhoneticCorrector:
    """Multi-algorithm phonetic similarity correction"""
    
    def __init__(self):
        self.algorithms = []
        
        if HAS_NLP_LIBS:
            self.algorithms.extend([
                ('soundex', jellyfish.soundex),
                ('metaphone', jellyfish.metaphone),
                ('nysiis', jellyfish.nysiis),
            ])
            
        if HAS_PHONETIC_LIBS:
            self.algorithms.append(('double_metaphone', double_metaphone))
    
    def find_phonetic_matches(self, word: str, vocabulary: Set[str], 
                            threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find phonetically similar words in vocabulary"""
        if not self.algorithms or not vocabulary:
            return []
        
        matches = []
        word_lower = word.lower()
        
        for vocab_word in vocabulary:
            vocab_lower = vocab_word.lower()
            
            if abs(len(word_lower) - len(vocab_lower)) > 3:
                continue  # Skip very different length words
            
            similarities = []
            
            # Calculate phonetic similarities
            for name, algorithm in self.algorithms:
                try:
                    if name == 'double_metaphone':
                        word_codes = algorithm(word_lower)
                        vocab_codes = algorithm(vocab_lower)
                        # Check if any codes match
                        similarity = 1.0 if any(wc in vocab_codes for wc in word_codes if wc) else 0.0
                    else:
                        word_code = algorithm(word_lower)
                        vocab_code = algorithm(vocab_lower)
                        similarity = 1.0 if word_code == vocab_code else 0.0
                    
                    similarities.append(similarity)
                except Exception:
                    continue
            
            if similarities and max(similarities) >= 0.5:
                # Also check edit distance for additional validation
                edit_sim = 1.0 - (editdistance.eval(word_lower, vocab_lower) / max(len(word_lower), len(vocab_lower)))
                
                # Weighted combination
                phonetic_score = max(similarities) if similarities else 0
                combined_score = 0.7 * phonetic_score + 0.3 * edit_sim
                
                if combined_score >= threshold:
                    matches.append((vocab_word, combined_score))
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]

class NEREntityCorrector:
    """Named Entity Recognition based correction"""
    
    def __init__(self):
        self.nlp = None
        self.entity_types = {'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT'}
        
        if HAS_NLP_LIBS:
            try:
                # Try to load transformer model first, fallback to smaller models
                models_to_try = [
                    'en_core_web_sm',  # Most common
                    'en_core_web_md',
                    'en_core_web_trf'  # Best accuracy but larger
                ]
                
                for model_name in models_to_try:
                    try:
                        self.nlp = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model: {model_name}")
                        break
                    except OSError:
                        continue
                        
                if not self.nlp:
                    logger.warning("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
                    
            except Exception as e:
                logger.warning(f"Could not initialize NER: {e}")
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract entities: [(text, label, start, end), ...]"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in self.entity_types and len(ent.text.strip()) > 1:
                    entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
            
            return entities
        except Exception as e:
            logger.debug(f"NER extraction error: {e}")
            return []

class LLMContextualCorrector:
    """Use local LLM for contextual correction"""
    
    def __init__(self, model_name: str = "qwen3-1.7b", max_length: int = 100):
        self.client = None
        self.model_name = model_name
        self.max_length = max_length
        
        # Try LM Studio first (local setup)
        if HAS_OPENAI:
            try:
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                logger.info(f"LLM corrector initialized with LM Studio at http://localhost:1234/v1 using {model_name}")
                return
            except Exception as e:
                logger.warning(f"Could not initialize LM Studio client: {e}")
        
        # Fallback to Ollama
        if HAS_OLLAMA:
            try:
                self.client = ollama.Client()
                # Test if model is available
                models = self.client.list()
                available_models = [m['name'] for m in models.get('models', [])]
                
                if model_name not in available_models:
                    logger.warning(f"Model {model_name} not available in Ollama. Available: {available_models}")
                    self.client = None
                else:
                    logger.info(f"LLM corrector initialized with {model_name}")
                    
            except Exception as e:
                logger.warning(f"Could not initialize Ollama client: {e}")
                self.client = None
    
    async def correct_with_context(self, text: str, context: str = "") -> Optional[str]:
        """Use LLM to correct text with context"""
        if not self.client or len(text) > self.max_length:
            return None
        
        try:
            prompt = f"""Fix speech recognition errors in this text, preserving the original meaning. Focus on proper nouns, technical terms, and URLs.

Context: {context[:200] if context else 'General conversation'}
Text to fix: "{text}"

Return only the corrected text, no explanations:"""

            # Use appropriate client based on type
            if isinstance(self.client, OpenAI):
                # LM Studio/OpenAI compatible API
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    max_tokens=200,
                    stream=False
                )
                corrected = response.choices[0].message.content.strip()
            else:
                # Ollama API
                response = await asyncio.to_thread(
                    self.client.chat,
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1, 'num_ctx': 1024}
                )
                corrected = response['message']['content'].strip()
            
            # Validate the correction (shouldn't be too different)
            if len(corrected) <= len(text) * 2 and editdistance.eval(text, corrected) <= len(text) * 0.5:
                return corrected
            
        except Exception as e:
            logger.debug(f"LLM correction error: {e}")
        
        return None

class AdvancedAccuracyEnhancer:
    """Main accuracy enhancement orchestrator"""
    
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "accuracy_cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.vocab_extractor = DynamicVocabularyExtractor(cache_dir)
        self.phonetic_corrector = PhoneticCorrector()
        self.ner_corrector = NEREntityCorrector()
        self.llm_corrector = LLMContextualCorrector()
        
        # Configuration
        self.confidence_threshold = 0.7  # Below this, apply corrections
        self.max_corrections_per_text = 10
        self.enable_llm_correction = True
        
        # Stats
        self.correction_stats = defaultdict(int)
        
    def _format_urls(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Format URLs from spoken text patterns"""
        corrections = []
        original_text = text
        
        # Pattern: "word dot com slash path" -> "word.com/path"
        url_pattern = r'\b([a-zA-Z0-9]+)\s+dot\s+([a-zA-Z]+)(?:\s+slash\s+([^\s]+))?'
        def url_replacer(match):
            domain = match.group(1)
            tld = match.group(2)
            path = match.group(3) if match.group(3) else ""
            
            # Common TLD corrections
            tld_corrections = {
                'com': 'com', 'org': 'org', 'net': 'net', 'edu': 'edu',
                'gov': 'gov', 'co': 'co', 'io': 'io', 'ai': 'ai'
            }
            
            # Correct common domain names
            domain_corrections = {
                'gee': 'ge', 'goggle': 'google', 'face': 'face',
                'get': 'get', 'stack': 'stack', 'tech': 'tech',
                'git': 'git', 'you': 'you', 'face': 'face'
            }
            
            corrected_domain = domain_corrections.get(domain.lower(), domain)
            corrected_tld = tld_corrections.get(tld.lower(), tld)
            
            formatted_url = f"{corrected_domain}.{corrected_tld}"
            if path:
                # Replace spaces with forward slashes in path
                formatted_path = path.replace(' slash ', '/').replace(' ', '/')
                formatted_url += f"/{formatted_path}"
            
            corrections.append({
                'original': match.group(0),
                'corrected': formatted_url,
                'confidence': 0.95,
                'method': 'url_formatting'
            })
            
            return formatted_url
        
        # Apply URL formatting
        formatted_text = re.sub(url_pattern, url_replacer, text, flags=re.IGNORECASE)
        
        return formatted_text, corrections
    
    def _format_emails(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Format email addresses from spoken text patterns"""
        corrections = []
        
        # Pattern: "user at domain dot com" -> "user@domain.com"
        email_pattern = r'\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9.-]+)\s+dot\s+([a-zA-Z]{2,})\b'
        def email_replacer(match):
            user = match.group(1)
            domain = match.group(2)
            tld = match.group(3)
            
            formatted_email = f"{user}@{domain}.{tld}"
            
            corrections.append({
                'original': match.group(0),
                'corrected': formatted_email,
                'confidence': 0.95,
                'method': 'email_formatting'
            })
            
            return formatted_email
        
        # Apply email formatting
        formatted_text = re.sub(email_pattern, email_replacer, text, flags=re.IGNORECASE)
        
        return formatted_text, corrections
    
    def _format_technical_terms(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Format common technical terms"""
        corrections = []
        
        # Common technical term corrections
        tech_corrections = {
            r'\b(dock her)\b': 'Docker',
            r'\b(react j s)\b': 'React.js',
            r'\b(node j s)\b': 'Node.js',
            r'\b(type script)\b': 'TypeScript',
            r'\b(java script)\b': 'JavaScript',
            r'\b(bootstrap)\b': 'Bootstrap',
            r'\b(github)\b': 'GitHub',
            r'\b(gitlab)\b': 'GitLab',
            r'\b(stack overflow)\b': 'Stack Overflow',
            r'\b(visual studio)\b': 'Visual Studio',
        }
        
        formatted_text = text
        for pattern, replacement in tech_corrections.items():
            matches = re.findall(pattern, formatted_text, re.IGNORECASE)
            if matches:
                formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)
                for match in matches:
                    corrections.append({
                        'original': match,
                        'corrected': replacement,
                        'confidence': 0.9,
                        'method': 'tech_term_formatting'
                    })
        
        return formatted_text, corrections
    
    async def enhance_accuracy(self, text: str, confidence: float = None, 
                             context: str = "") -> CorrectionResult:
        """Main accuracy enhancement pipeline"""
        start_time = time.time()
        original_text = text
        corrections_applied = []
        method_used = "none"
        
        try:
            # Skip if high confidence and short text
            if confidence and confidence > 0.9 and len(text.split()) < 5:
                return CorrectionResult(
                    original_text=original_text,
                    corrected_text=text,
                    confidence_score=confidence or 1.0,
                    corrections_applied=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    method_used="skipped_high_confidence"
                )
            
            # Step 1: Format URLs and emails first (high confidence patterns)
            text, url_corrections = self._format_urls(text)
            corrections_applied.extend(url_corrections)
            
            text, email_corrections = self._format_emails(text)
            corrections_applied.extend(email_corrections)
            
            text, tech_corrections = self._format_technical_terms(text)
            corrections_applied.extend(tech_corrections)
            
            # Update method used if we made formatting corrections
            if url_corrections or email_corrections or tech_corrections:
                method_used = "formatting"
            
            # Step 2: Extract dynamic vocabulary
            user_vocabulary = await self.vocab_extractor.extract_user_vocabulary()
            
            # Step 3: NER-based entity extraction
            entities = self.ner_corrector.extract_entities(text)
            
            # Step 3: Apply corrections in order of confidence
            corrected_text = text
            
            # Phonetic corrections for entities
            if entities and user_vocabulary:
                method_used = "phonetic+ner"
                for entity_text, entity_type, start_pos, end_pos in entities:
                    matches = self.phonetic_corrector.find_phonetic_matches(
                        entity_text, user_vocabulary, threshold=0.75
                    )
                    
                    if matches and matches[0][1] > 0.85:
                        correction = matches[0][0]
                        corrected_text = corrected_text.replace(entity_text, correction, 1)
                        corrections_applied.append({
                            'original': entity_text,
                            'corrected': correction,
                            'method': 'phonetic_ner',
                            'confidence': matches[0][1],
                            'entity_type': entity_type
                        })
                        self.correction_stats['phonetic_ner'] += 1
            
            # Step 4: General phonetic correction for remaining words
            if user_vocabulary and len(corrections_applied) < self.max_corrections_per_text:
                words = corrected_text.split()
                for i, word in enumerate(words):
                    if len(word) > 3 and word.isalpha():  # Skip short words and numbers
                        matches = self.phonetic_corrector.find_phonetic_matches(
                            word, user_vocabulary, threshold=0.8
                        )
                        
                        if matches and matches[0][1] > 0.9:
                            correction = matches[0][0]
                            words[i] = correction
                            corrections_applied.append({
                                'original': word,
                                'corrected': correction,
                                'method': 'phonetic_general',
                                'confidence': matches[0][1]
                            })
                            self.correction_stats['phonetic_general'] += 1
                            
                            if len(corrections_applied) >= self.max_corrections_per_text:
                                break
                
                corrected_text = ' '.join(words)
            
            # Step 5: LLM contextual correction (for low confidence or many corrections)
            if (self.enable_llm_correction and self.llm_corrector.client and 
                (confidence is None or confidence < self.confidence_threshold or len(corrections_applied) > 2)):
                
                llm_corrected = await self.llm_corrector.correct_with_context(
                    corrected_text, context
                )
                
                if llm_corrected and llm_corrected != corrected_text:
                    corrections_applied.append({
                        'original': corrected_text,
                        'corrected': llm_corrected,
                        'method': 'llm_contextual',
                        'confidence': 0.85  # Assumed confidence for LLM
                    })
                    corrected_text = llm_corrected
                    method_used = f"{method_used}+llm"
                    self.correction_stats['llm_contextual'] += 1
            
            # Calculate final confidence
            final_confidence = confidence or 0.5
            if corrections_applied:
                # Boost confidence based on corrections applied
                confidence_boost = min(0.3, len(corrections_applied) * 0.1)
                final_confidence = min(1.0, final_confidence + confidence_boost)
            
        except Exception as e:
            logger.error(f"Error in accuracy enhancement: {e}")
            corrected_text = text  # Fallback to original
            corrections_applied.append({
                'error': str(e),
                'method': 'error_fallback'
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        result = CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            confidence_score=final_confidence,
            corrections_applied=corrections_applied,
            processing_time_ms=processing_time,
            method_used=method_used if corrections_applied else "none"
        )
        
        # Log significant improvements
        if len(corrections_applied) > 0:
            logger.info(f"Applied {len(corrections_applied)} corrections in {processing_time:.1f}ms: {original_text} -> {corrected_text}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get correction statistics"""
        return {
            'total_corrections': sum(self.correction_stats.values()),
            'corrections_by_method': dict(self.correction_stats),
            'vocabulary_size': len(self.vocab_extractor.vocabulary_cache.get('words', set())),
            'available_methods': {
                'phonetic': bool(self.phonetic_corrector.algorithms),
                'ner': bool(self.ner_corrector.nlp),
                'llm': bool(self.llm_corrector.client),
                'dynamic_vocab': True
            }
        }

# Convenience functions for integration
async def enhance_transcription(text: str, confidence: float = None, 
                              context: str = "") -> str:
    """Simple function to enhance transcription accuracy"""
    enhancer = AdvancedAccuracyEnhancer()
    result = await enhancer.enhance_accuracy(text, confidence, context)
    return result.corrected_text

def create_enhancer(cache_dir: Path = None) -> AdvancedAccuracyEnhancer:
    """Factory function to create accuracy enhancer"""
    return AdvancedAccuracyEnhancer(cache_dir)

# CLI for testing
if __name__ == "__main__":
    import argparse
    
    async def test_enhancement():
        parser = argparse.ArgumentParser(description='Test Advanced Accuracy Enhancer')
        parser.add_argument('--text', required=True, help='Text to enhance')
        parser.add_argument('--confidence', type=float, help='Confidence score')
        parser.add_argument('--context', default='', help='Context for correction')
        args = parser.parse_args()
        
        enhancer = AdvancedAccuracyEnhancer()
        result = await enhancer.enhance_accuracy(args.text, args.confidence, args.context)
        
        print(f"Original:  {result.original_text}")
        print(f"Enhanced:  {result.corrected_text}")
        print(f"Method:    {result.method_used}")
        print(f"Time:      {result.processing_time_ms:.1f}ms")
        print(f"Changes:   {len(result.corrections_applied)}")
        
        for correction in result.corrections_applied:
            print(f"  {correction.get('original', 'N/A')} -> {correction.get('corrected', 'N/A')} ({correction.get('method', 'unknown')})")
        
        stats = enhancer.get_stats()
        print(f"\nStats: {stats}")
    
    asyncio.run(test_enhancement())