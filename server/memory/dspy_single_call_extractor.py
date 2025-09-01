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
        self.model = os.getenv("DSPY_EXTRACTION_MODEL", "qwen2.5-0.5b-instruct-mlx")
    
    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract both factual and personal relations in ONE optimized call"""
        
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
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a stateless assistant. Forget everything from previous conversations. Treat each user message independently. Extract ONLY relations that appear in the current input text. Do not use information from previous examples or other contexts. Return precise JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                # Allow a bit more room to prevent truncation
                "max_tokens": 300,
                "temperature": 0.05,  # Very low for consistency
                "stream": False,  # CRITICAL: Disable streaming to prevent JSON truncation
                "presence_penalty": 0.0,  # No penalty for repeating concepts
                "frequency_penalty": 0.0,  # No penalty for repeating words
                "top_p": 0.9,  # Focus on high-probability tokens
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "comprehensive_extraction",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {"type": "string"},
                                            "predicate": {"type": "string"},
                                            "object": {"type": "string"},
                                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                        },
                                        "required": ["subject", "predicate", "object", "confidence"]
                                    }
                                }
                            },
                            "required": ["relations"]
                        }
                    }
                }
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15  # Reasonable timeout
            )
            
            if response.status_code != 200:
                print(f"❌ HTTP error {response.status_code}: {response.text}")
                return []
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse structured response with error handling
            # Robust JSON parse with trimming if LM trails off
            def _parse_relations(raw: str):
                try:
                    parsed = json.loads(raw)
                    return parsed.get("relations", [])
                except json.JSONDecodeError:
                    # Try trimming to last closing brace
                    start = raw.find('{')
                    end = raw.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        try:
                            parsed2 = json.loads(raw[start:end+1])
                            return parsed2.get("relations", [])
                        except json.JSONDecodeError:
                            return []
                    return []

            relations = _parse_relations(content)

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
