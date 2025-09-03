#!/usr/bin/env python
"""
Single-call DSPy fact extraction optimized for Qwen 2.5 0.5B
Maximum performance + quality in one HTTP request
"""

import json
import time
import requests
from typing import List, Dict, Any


class DSPySingleCallExtractor:
    """Single-call DSPy extraction optimized for Qwen 2.5 0.5B performance"""
    
    def __init__(self):
        self.name = "DSPy-SingleCall-Qwen"
        # Allow overriding via env to try more capable local models (e.g., qwen3-4b)
        import os
        self.base_url = os.getenv("DSPY_EXTRACTION_BASE_URL", "http://localhost:1234/v1/chat/completions")
        # Two-model strategy:
        # - relations model (strict JSON schema)
        # - general model (looser JSON object), used to backfill when strict fails
        self.model_rel = os.getenv("DSPY_REL_MODEL", "qwen/qwen3-4b")
        self.model_facts = os.getenv("DSPY_FACTS_MODEL", "qwen3-4b-instruct-2507")
    
    def _call_chat(self, model: str, messages: List[Dict[str, str]], response_format: Dict[str, Any], max_tokens: int = 300) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.05,
            "stream": False,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 0.9,
            "response_format": response_format,
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        return response.json()

    def _parse_strict_relations(self, raw: str) -> List[Dict[str, Any]]:
        """Parse the simple array format from qwen2.5-0.5b-instruct-mlx:2"""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                # Convert string facts to relation objects
                relations = []
                for fact_str in parsed:
                    if isinstance(fact_str, str) and len(fact_str.strip()) > 0:
                        # Parse "subject predicate object" format
                        parts = fact_str.strip().split(None, 2)  # Split on first 2 spaces
                        if len(parts) >= 3:
                            relations.append({
                                'subject': parts[0],
                                'predicate': parts[1], 
                                'object': parts[2],
                                'confidence': 0.8
                            })
                return relations
            return []
        except json.JSONDecodeError:
            # Try to extract array from truncated JSON
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed2 = json.loads(raw[start:end+1])
                    if isinstance(parsed2, list):
                        relations = []
                        for fact_str in parsed2:
                            if isinstance(fact_str, str) and len(fact_str.strip()) > 0:
                                parts = fact_str.strip().split(None, 2)
                                if len(parts) >= 3:
                                    relations.append({
                                        'subject': parts[0],
                                        'predicate': parts[1],
                                        'object': parts[2], 
                                        'confidence': 0.8
                                    })
                        return relations
                except json.JSONDecodeError:
                    return []
            return []
    
    def _parse_general_relations(self, raw: str) -> List[Dict[str, Any]]:
        """Parse the complex object format from qwen2.5-0.5b-instruct-mlx"""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Extract relationships from the complex schema
                relationships = parsed.get("relationships", [])
                if isinstance(relationships, list):
                    # Convert to our standard format with confidence
                    relations = []
                    for rel in relationships:
                        if isinstance(rel, dict) and all(key in rel for key in ['subject', 'predicate', 'object']):
                            relations.append({
                                'subject': rel['subject'],
                                'predicate': rel['predicate'],
                                'object': rel['object'],
                                'confidence': 0.7  # Default confidence
                            })
                    return relations
            return []
        except json.JSONDecodeError:
            # Try to extract object from truncated JSON
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed2 = json.loads(raw[start:end+1])
                    if isinstance(parsed2, dict):
                        relationships = parsed2.get("relationships", [])
                        if isinstance(relationships, list):
                            relations = []
                            for rel in relationships:
                                if isinstance(rel, dict) and all(key in rel for key in ['subject', 'predicate', 'object']):
                                    relations.append({
                                        'subject': rel['subject'],
                                        'predicate': rel['predicate'],
                                        'object': rel['object'],
                                        'confidence': 0.7
                                    })
                            return relations
                except json.JSONDecodeError:
                    return []
            return []

    def extract_relations_strict(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations using strict JSON schema (simple array format for qwen2.5-0.5b-instruct-mlx:2)."""
        prompt = f"""[STRICT_MODEL:{self.model_rel}] Extract knowledge facts from this text in 'subject predicate object' format.

Rules:
- For personal statements (I, my, we): use 'user' as subject
- Use simple predicates: likes, prefers, has_pet, works_at, lives_in
- Keep objects specific and short (1-3 words)
- Return array of fact strings like: ["user likes samples", "user prefers hip-hop"]
- If no clear facts, return empty array: []

Text: "{text}"

Return JSON array of fact strings:"""

        messages = [
            {"role": "system", "content": f"You are a precise fact extractor using {self.model_rel}. Extract only what is clearly stated in the text. Return valid JSON array."},
            {"role": "user", "content": prompt},
        ]
        # Use the correct schema for qwen2.5-0.5b-instruct-mlx:2 (simple array format)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "speaker_facts",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of speaker facts in 'subject predicate object' format"
                }
            }
        }
        result = self._call_chat(self.model_rel, messages, response_format)
        content = result["choices"][0]["message"]["content"]
        return self._parse_strict_relations(content)

    def extract_relations_general(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations using general model (complex object format for qwen2.5-0.5b-instruct-mlx)."""
        prompt = f"""[GENERAL_MODEL:{self.model_facts}] Analyze this text and extract structured knowledge.

Text: "{text}"

Return JSON with:
- facts: concrete statements from the text
- concepts: important topics mentioned
- relationships: connections between entities (subject/predicate/object)
- domain: classify as personal, technical, general, or creative

Focus on what's actually stated in the text. Be conservative."""

        messages = [
            {"role": "system", "content": f"You are a knowledge extraction system using {self.model_facts}. Analyze the text and return structured JSON. Be conservative and accurate."},
            {"role": "user", "content": prompt},
        ]
        # Use the correct schema for qwen2.5-0.5b-instruct-mlx (complex object format)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "knowledge_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Concrete factual statements from the text"
                        },
                        "concepts": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Important topics, entities, or ideas mentioned"
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {"type": "string"},
                                    "predicate": {"type": "string"},
                                    "object": {"type": "string"}
                                },
                                "required": ["subject", "predicate", "object"],
                                "additionalProperties": False
                            },
                            "description": "Relationships between entities"
                        },
                        "domain": {
                            "type": "string",
                            "enum": ["personal", "technical", "general", "creative"],
                            "description": "Knowledge domain classification"
                        }
                    },
                    "required": ["facts", "concepts", "relationships", "domain"],
                    "additionalProperties": False
                }
            }
        }
        result = self._call_chat(self.model_facts, messages, response_format)
        content = result["choices"][0]["message"]["content"]
        return self._parse_general_relations(content)

    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract both factual and personal relations using two-model strategy."""
        
        start_time = time.perf_counter()
        print(f"🔍 DSPy extract_facts called with: '{text}' (len={len(text)})")
        
        # Minimal, small-model-friendly prompt: open-vocab predicate, one relation max
        prompt = f"""Extract at most ONE knowledge relation from the text below.

Constraints:
- subject: 'user' for first-person (I, me, my, we); otherwise a proper name if present
- predicate: ONE short action/relationship word (lemma; lowercase/underscores; not a sentence)
- object: 1-3 words; specific noun phrase; no pronouns
- If nothing meaningful, return {{"relations": []}}

Text: "{text}"

Return ONLY JSON for the schema."""

# prompt = f"""Extract knowledge relations from this text: "{text}"

# RULES:
# - Extract ONLY from the text above
# - Ignore any previous conversations
# - Be conservative - if unsure, don't extract
# - Return max 3 relations

# For personal statements (my, I, we): use "user" as subject
# For facts about entities: use actual names as subjects

# Return JSON format:"""



        try:
            # 1) Strict relations (relations model)
            relations = self.extract_relations_strict(text) or []
            # 2) If empty, try general model (looser json_object)
            if not relations:
                relations = self.extract_relations_general(text) or []

            extraction_time = time.perf_counter() - start_time
            
            # Add metadata to each fact with anti-hallucination validation
            facts = []
            for relation in relations:
                if self._validate_relation(relation, text):
                    facts.append({
                        'subject': relation['subject'].strip(),
                        'predicate': relation['predicate'].strip(),
                        'object': relation['object'].strip(),
                        'confidence': float(relation['confidence']),
                        'source': self.name,
                        'extraction_time': extraction_time
                    })
            
            print(f"✅ Single-call extraction: {len(facts)} facts in {extraction_time:.3f}s")
            return facts
            
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {content[:300]}...")
            return []
        except Exception as e:
            print(f"❌ Single-call extraction failed: {e}")
            return []
    
    def _validate_relation(self, relation: Dict, input_text: str) -> bool:
        """Validate relation quality and check for hallucinations"""
        required_fields = ['subject', 'predicate', 'object', 'confidence']
        
        # Check required fields
        for field in required_fields:
            if field not in relation:
                return False
            if not str(relation[field]).strip():
                return False
        
        # Validate confidence range
        try:
            conf = float(relation['confidence'])
            if conf < 0.0 or conf > 1.0:
                return False
        except (ValueError, TypeError):
            return False
        
        # Block vague predicates
        predicate = relation['predicate'].lower().strip()
        vague_predicates = {'is_place', 'has_name', 'is_called', 'is_thing', 'has_thing', 'is_named'}
        
        if predicate in vague_predicates:
            return False
        
        # ANTI-HALLUCINATION: Check if entities actually appear in input text
        subject = relation['subject'].lower().strip()
        obj = relation['object'].lower().strip()
        text_lower = input_text.lower()
        
        # Allow "user" as subject for personal relations
        if subject == "user":
            # Personal relation - check if text has personal indicators
            personal_indicators = ['my ', 'i ', 'we ', 'our ', 'me ', 'myself']
            if not any(indicator in text_lower for indicator in personal_indicators):
                return False
        else:
            # Factual relation - subject must appear in text
            if subject not in text_lower:
                return False
        
        # Object must appear in text (unless it's a reasonable inference)
        if obj not in text_lower:
            # Allow some common inferences
            reasonable_inferences = {
                'german shepherd': 'german shepherd' in text_lower,
                'brown': 'brown' in text_lower,
                'google': 'google' in text_lower,
                'seattle': 'seattle' in text_lower
            }
            
            if obj not in reasonable_inferences or not reasonable_inferences[obj]:
                return False
        
        return True

    def extract_facts_from_chunk(self, chunk_text: str, previous_context: str = "") -> List[Dict[str, Any]]:
        """Extract facts from conversation chunk with context awareness (M3-style)"""
        
        start_time = time.perf_counter()
        print(f"🔍 M3 chunk extraction called: chunk_len={len(chunk_text)}, context_len={len(previous_context)}")
        
        # M3-inspired prompt for conversation chunk processing
        context_section = f"Previous context: {previous_context}\n\n" if previous_context else ""
        
        prompt = f"""[CHUNK_MODEL:{self.model_rel}] Extract knowledge relations from this conversation segment.

{context_section}New conversation: "{chunk_text}"

Extract relationships considering the full conversational context:
- Look for connections between statements across turns
- Extract entity relationships (people, places, things)
- Capture temporal and logical connections
- Use 'user' as subject for first-person statements (I, me, my, we)
- Return array of relation objects: [{{"subject": "X", "predicate": "Y", "object": "Z"}}]

Focus on meaningful relationships that build understanding of the speaker."""

        messages = [
            {"role": "system", "content": f"You are a conversation-aware fact extractor using {self.model_rel}. Extract relationships from conversation chunks, understanding context across turns."},
            {"role": "user", "content": prompt},
        ]
        
        # Use strict relations format for consistency
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "chunk_relations",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["subject", "predicate", "object"],
                        "additionalProperties": False
                    },
                    "description": "Array of knowledge relations from conversation chunk"
                }
            }
        }
        
        try:
            result = self._call_chat(self.model_rel, messages, response_format, max_tokens=500)
            content = result["choices"][0]["message"]["content"]
            
            # Parse chunk relations
            relations = self._parse_chunk_relations(content)
            extraction_time = time.perf_counter() - start_time
            
            # Validate and format facts
            facts = []
            for relation in relations:
                if self._validate_relation(relation, chunk_text):
                    facts.append({
                        'subject': relation['subject'].strip(),
                        'predicate': relation['predicate'].strip(),
                        'object': relation['object'].strip(),
                        'confidence': relation.get('confidence', 0.8),
                        'source': f"{self.name}-Chunk",
                        'extraction_time': extraction_time
                    })
            
            print(f"✅ M3 chunk extraction: {len(facts)} facts from conversation in {extraction_time:.3f}s")
            return facts
            
        except Exception as e:
            extraction_time = time.perf_counter() - start_time
            print(f"❌ M3 chunk extraction failed: {e}")
            return []
    
    def _parse_chunk_relations(self, raw: str) -> List[Dict[str, Any]]:
        """Parse chunk relations from JSON response"""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                relations = []
                for item in parsed:
                    if isinstance(item, dict) and all(key in item for key in ['subject', 'predicate', 'object']):
                        relations.append({
                            'subject': item['subject'],
                            'predicate': item['predicate'],
                            'object': item['object'],
                            'confidence': item.get('confidence', 0.8)
                        })
                return relations
        except json.JSONDecodeError:
            # Try to extract array from truncated JSON
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed2 = json.loads(raw[start:end+1])
                    if isinstance(parsed2, list):
                        relations = []
                        for item in parsed2:
                            if isinstance(item, dict) and all(key in item for key in ['subject', 'predicate', 'object']):
                                relations.append({
                                    'subject': item['subject'],
                                    'predicate': item['predicate'], 
                                    'object': item['object'],
                                    'confidence': item.get('confidence', 0.8)
                                })
                        return relations
                except json.JSONDecodeError:
                    pass
        return []


# Performance test
if __name__ == "__main__":
    
    def test_single_call_performance():
        """Test single-call DSPy performance and quality"""
        
        print("🚀 DSPy Single-Call Qwen 2.5 0.5B Optimization Test")
        print("=" * 60)
        
        extractor = DSPySingleCallExtractor()
        
        test_cases = [
            "My dog Rex is a brown German Shepherd",
            "I work at Google in Seattle as a software engineer", 
            "My girlfriend Sarah is a doctor at Children's Hospital",
            "Paris is the capital of France and has 2 million people",
            "I drive a red Tesla Model 3 that I bought last year"
        ]
        
        total_time = 0
        total_facts = 0
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: '{text}'")
            print("-" * 50)
            
            start = time.perf_counter()
            facts = extractor.extract_facts(text)
            elapsed = time.perf_counter() - start
            
            total_time += elapsed
            total_facts += len(facts)
            
            # Categorize facts
            personal_facts = [f for f in facts if f['subject'] == 'user']
            factual_facts = [f for f in facts if f['subject'] != 'user']
            
            print(f"⏱️  Time: {elapsed:.3f}s")
            print(f"📊 Facts: {len(facts)} total ({len(personal_facts)} personal, {len(factual_facts)} factual)")
            
            if personal_facts:
                print("👤 Personal:")
                for f in personal_facts:
                    print(f"  • {f['subject']} → {f['predicate']} → {f['object']} (conf: {f['confidence']})")
            
            if factual_facts:
                print("📚 Factual:")
                for f in factual_facts:
                    print(f"  • {f['subject']} → {f['predicate']} → {f['object']} (conf: {f['confidence']})")
        
        # Overall performance
        avg_time = total_time / len(test_cases)
        facts_per_second = total_facts / total_time
        
        print(f"\n" + "="*60)
        print("🏆 SINGLE-CALL PERFORMANCE RESULTS")
        print("="*60)
        print(f"Average time per extraction: {avg_time:.3f}s")
        print(f"Total facts extracted: {total_facts}")
        print(f"Facts per second: {facts_per_second:.2f}")
        print(f"Total time: {total_time:.3f}s")
        
        # Performance assessment
        if avg_time < 1.0:
            performance = "🚀 EXCELLENT - Perfect for real-time conversation"
        elif avg_time < 2.0:
            performance = "✅ VERY GOOD - Suitable for voice interaction"
        elif avg_time < 3.0:
            performance = "✅ GOOD - Acceptable for most use cases"
        else:
            performance = "⚠️ NEEDS OPTIMIZATION - May impact user experience"
        
        print(f"\nPerformance: {performance}")
        
        # Quality assessment
        personal_coverage = sum(1 for i, text in enumerate(test_cases) if any(f['subject'] == 'user' for f in extractor.extract_facts(text)))
        factual_coverage = sum(1 for i, text in enumerate(test_cases) if any(f['subject'] != 'user' for f in extractor.extract_facts(text)))
        
        print(f"\nQuality Coverage:")
        print(f"Personal context: {personal_coverage}/{len(test_cases)} sentences")
        print(f"Factual relations: {factual_coverage}/{len(test_cases)} sentences")
        
        return {
            'avg_time': avg_time,
            'total_facts': total_facts,
            'facts_per_second': facts_per_second,
            'performance': performance
        }
    
    test_single_call_performance()
