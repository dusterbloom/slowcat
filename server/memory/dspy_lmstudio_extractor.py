#!/usr/bin/env python
"""
DSPy fact extraction optimized specifically for LM Studio setup
Uses direct HTTP calls to bypass JSON schema issues
"""

import json
import time
import requests
from typing import List, Dict, Any


class LMStudioFactExtractor:
    """Direct LM Studio integration for fact extraction without DSPy JSON issues"""
    
    def __init__(self):
        self.name = "LMStudio-Direct"
        self.base_url = "http://localhost:1234/v1/chat/completions"
        self.model = "qwen2.5-0.5b-instruct-mlx"
    
    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract facts using direct LM Studio calls"""
        
        facts = []
        start_time = time.perf_counter()
        
        try:
            # Extract factual relations
            factual_facts = self._extract_factual_relations(text)
            facts.extend(factual_facts)
            
            # Extract personal relations
            personal_facts = self._extract_personal_relations(text)
            facts.extend(personal_facts)
            
            extraction_time = time.perf_counter() - start_time
            
            # Add metadata
            for fact in facts:
                fact['source'] = self.name
                fact['extraction_time'] = extraction_time
                
        except Exception as e:
            print(f"❌ LM Studio extraction failed: {e}")
        
        return facts
    
    def _extract_factual_relations(self, text: str) -> List[Dict]:
        """Extract factual relations about entities"""
        
        prompt = f"""Extract factual knowledge about named entities from: "{text}"

RULES:
- Extract facts about specific entities (people, places, things) mentioned in the text
- Use precise predicates: breed_of, located_in, capital_of, has_color, works_at, profession  
- Subject must be a specific entity name (Rex, Paris, Sarah), NOT "user" or generic terms
- Object should be specific information about that entity

Examples:
"My dog Rex is a brown German Shepherd" → 
- Rex breed_of German Shepherd (Rex is the specific dog mentioned)
- Rex has_color brown (Rex has the color brown)

"Sarah works at Google in Seattle" →
- Sarah works_at Google (Sarah is employed at Google)
- Google located_in Seattle (Google's location)

Text: "{text}"

Extract facts and return JSON:"""

        return self._call_lm_studio(prompt, "factual")
    
    def _extract_personal_relations(self, text: str) -> List[Dict]:
        """Extract personal user relations"""
        
        prompt = f"""Extract the speaker's personal relationships from: "{text}"

RULES:
- Focus on what the speaker/user personally owns, does, or is connected to
- Subject is always "user" (representing the speaker)
- Use specific personal predicates: has_pet, owns, drives, works_at, lives_in
- Object should be the specific entity the user relates to

Examples:
"My dog Rex is brown" → user has_pet Rex (user owns/has a pet named Rex)
"I work at Google" → user works_at Google (user is employed at Google)  
"I drive a Tesla Model 3" → user drives Tesla Model 3 (user owns/drives this car)

Text: "{text}"

Extract personal connections and return JSON:"""

        return self._call_lm_studio(prompt, "personal")
    
    def _call_lm_studio(self, prompt: str, extraction_type: str) -> List[Dict]:
        """Make direct HTTP call to LM Studio"""
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a precise knowledge extraction system. Return only valid JSON in the exact format requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.1,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fact_extraction",
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
                                            "confidence": {"type": "number"}
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
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ LM Studio HTTP error {response.status_code}: {response.text}")
                return []
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            parsed = json.loads(content)
            relations = parsed.get("relations", [])
            
            print(f"✅ {extraction_type} extraction: {len(relations)} relations")
            return relations
            
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed for {extraction_type}: {e}")
            print(f"Raw response: {content[:200]}...")
            return []
        except Exception as e:
            print(f"❌ {extraction_type} extraction error: {e}")
            return []


# Test the direct LM Studio approach
if __name__ == "__main__":
    
    def test_lmstudio_direct():
        """Test direct LM Studio integration"""
        
        print("🧪 Testing Direct LM Studio Fact Extraction")
        print("=" * 50)
        
        extractor = LMStudioFactExtractor()
        
        test_cases = [
            "My dog Rex is a brown German Shepherd",
            "I work at Google in Seattle as a software engineer",
            "Paris is the capital of France"
        ]
        
        for text in test_cases:
            print(f"\n📝 Testing: '{text}'")
            print("-" * 40)
            
            facts = extractor.extract_facts(text)
            
            if facts:
                personal_facts = [f for f in facts if f.get('subject') == 'user']
                factual_facts = [f for f in facts if f.get('subject') != 'user']
                
                if personal_facts:
                    print(f"👤 Personal relations ({len(personal_facts)}):")
                    for f in personal_facts:
                        print(f"  • {f['subject']} → {f['predicate']} → {f['object']}")
                
                if factual_facts:
                    print(f"📚 Factual relations ({len(factual_facts)}):")
                    for f in factual_facts:
                        print(f"  • {f['subject']} → {f['predicate']} → {f['object']}")
            else:
                print("❌ No facts extracted")
        
        print("\n✅ Direct LM Studio test complete!")
    
    test_lmstudio_direct()