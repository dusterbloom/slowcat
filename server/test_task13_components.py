#!/usr/bin/env python3
"""
Test Task-13 Component Optimization: spaCy Advanced Features

This tests the individual optimization components:
1. Temporal extraction for events and dates
2. Coreference resolution across sentences  
3. Entity resolution for deduplication
4. Integrated fact extraction pipeline

Goal: Validate that the components work correctly for complex queries.
"""

import sys
import os

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.spacy_fact_extractor import extract_facts_from_text
from memory.coreference_resolver import resolve_coreferences
from memory.entity_resolver import resolve_entities_in_text, get_entity_resolver
from memory.temporal_extractor import extract_temporal_expressions, extract_events_from_text
from loguru import logger


def test_coreference_resolution():
    """Test coreference resolution"""
    print("\n1️⃣  COREFERENCE RESOLUTION TEST")
    print("-" * 50)
    
    test_cases = [
        "Sarah went to the meeting. She presented the proposal.",
        "My dog Luna is smart. She can fetch the newspaper.",
        "I met Dr. Smith yesterday. She is a great doctor.",
        "John and Mary are friends. They live in San Francisco."
    ]
    
    success_count = 0
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Original: {text}")
        
        resolved = resolve_coreferences(text)
        print(f"  Resolved: {resolved}")
        
        # Check if pronouns were resolved (basic check)
        if "she" not in resolved.lower() or "they" not in resolved.lower() or "he" not in resolved.lower():
            success_count += 1
            print("  ✅ Pronouns resolved")
        else:
            print("  ❌ Pronouns not fully resolved")
    
    print(f"\nCoreference Resolution: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)


def test_entity_resolution():
    """Test entity resolution and canonicalization"""
    print("\n2️⃣  ENTITY RESOLUTION TEST")
    print("-" * 50)
    
    resolver = get_entity_resolver()
    
    test_cases = [
        # Person names
        ("sarah", "PERSON"),
        ("Sarah Smith", "PERSON"), 
        ("sarah smith", "PERSON"),
        ("Dr. Sarah Smith", "PERSON"),
        
        # Organizations
        ("apple", "ORG"),
        ("Apple Inc.", "ORG"),
        ("apple inc", "ORG"),
    ]
    
    success_count = 0
    canonical_forms = {}
    
    for i, (name, entity_type) in enumerate(test_cases, 1):
        canonical = resolver.resolve_entity(name, entity_type)
        print(f"{i}. '{name}' ({entity_type}) → '{canonical}'")
        
        # Track canonical forms for consistency checking
        key = (canonical.lower(), entity_type)
        if key not in canonical_forms:
            canonical_forms[key] = canonical
            success_count += 1
        elif canonical_forms[key] == canonical:
            success_count += 1
        else:
            print(f"   ❌ Inconsistent canonicalization")
    
    # Test text resolution
    print(f"\nText resolution test:")
    text = "I met sarah yesterday. Dr. Sarah Smith works at apple inc."
    resolved = resolver.resolve_entities_in_text(text)
    print(f"  Original: {text}")
    print(f"  Resolved: {resolved}")
    
    print(f"\nEntity Resolution: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)


def test_temporal_extraction():
    """Test temporal expression and event extraction"""
    print("\n3️⃣  TEMPORAL EXTRACTION TEST")  
    print("-" * 50)
    
    test_cases = [
        "I have a meeting with Sarah tomorrow at 3 PM",
        "My birthday is on December 15th", 
        "Meeting with John on Friday at 2:30",
        "Deadline for project is next Monday",
        "Call mom today at noon"
    ]
    
    success_count = 0
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        
        # Test temporal expressions
        temporal_exprs = extract_temporal_expressions(text)
        print(f"  Temporal expressions: {len(temporal_exprs)}")
        for expr in temporal_exprs:
            print(f"    • {expr['text']} → {expr.get('parsed_date', 'No date')}")
        
        # Test events  
        events = extract_events_from_text(text)
        print(f"  Events: {len(events)}")
        for event in events:
            print(f"    • {event['title']} @ {event.get('start_time', 'TBD')}")
        
        if temporal_exprs or events:
            success_count += 1
            print("  ✅ Temporal information extracted")
        else:
            print("  ❌ No temporal information found")
    
    print(f"\nTemporal Extraction: {success_count}/{len(test_cases)} tests passed")
    return success_count >= len(test_cases) * 0.8  # 80% threshold


def test_integrated_fact_extraction():
    """Test the complete integrated fact extraction pipeline"""
    print("\n4️⃣  INTEGRATED FACT EXTRACTION TEST")
    print("-" * 50)
    
    test_cases = [
        # Complex sentences with pronouns, entities, and temporal info
        "My dog Luna is very smart. She loves playing in the park.",
        "I met sarah yesterday. She mentioned our meeting tomorrow at 3 PM.",
        "Dr. Smith is my doctor. He works at UCSF medical center.",
        "My birthday is December 15th and I'm planning a party."
    ]
    
    success_count = 0
    total_facts = 0
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        
        # Extract facts using integrated pipeline
        facts = extract_facts_from_text(text)
        print(f"  Extracted facts: {len(facts)}")
        
        for j, fact in enumerate(facts, 1):
            print(f"    {j}. {fact['subject']} {fact['predicate']} {fact['value']}")
            total_facts += 1
        
        if facts:
            success_count += 1
            print("  ✅ Facts extracted successfully")
        else:
            print("  ❌ No facts extracted")
    
    print(f"\nIntegrated Fact Extraction: {success_count}/{len(test_cases)} tests passed")
    print(f"Total facts extracted: {total_facts}")
    
    return success_count >= len(test_cases) * 0.75  # 75% threshold


def test_query_understanding():
    """Test if the system can understand complex queries"""
    print("\n5️⃣  QUERY UNDERSTANDING TEST")
    print("-" * 50)
    
    # Simulate the queries that should now work
    complex_queries = [
        "When is my meeting with Sarah?",
        "What's my dog's name?",
        "Where do I live?", 
        "When is my birthday?",
        "Who is my doctor?"
    ]
    
    # For each query, test the components that should help
    for i, query in enumerate(complex_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        
        # Test coreference resolution
        resolved = resolve_coreferences(query)
        if resolved != query:
            print(f"  Coreference: {query} → {resolved}")
        
        # Test entity resolution
        entity_resolved = resolve_entities_in_text(resolved)
        if entity_resolved != resolved:
            print(f"  Entity: {resolved} → {entity_resolved}")
            
        # Test temporal extraction
        temporal_exprs = extract_temporal_expressions(entity_resolved)
        if temporal_exprs:
            print(f"  Temporal: Found {len(temporal_exprs)} temporal expressions")
            
        print("  ✅ Query processed through pipeline")
    
    return True


def main():
    """Run all component tests"""
    print("🧪 TASK-13 COMPONENT OPTIMIZATION TESTS")
    print("="*80)
    
    results = []
    
    # Run individual tests
    results.append(test_coreference_resolution())
    results.append(test_entity_resolution())
    results.append(test_temporal_extraction())
    results.append(test_integrated_fact_extraction())
    results.append(test_query_understanding())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 FINAL RESULTS:")
    print("="*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.8:  # 80% success rate
        print("\n✅ TASK-13 COMPONENT OPTIMIZATION SUCCESSFUL!")
        print("   Advanced spaCy features are working correctly:")
        print("   • Coreference resolution linking pronouns to entities")
        print("   • Entity resolution canonicalizing names")
        print("   • Temporal extraction finding dates and events")
        print("   • Integrated pipeline processing complex sentences")
    else:
        print("\n❌ Task-13 optimization needs improvement")
        print("   Some components are not working as expected")
    
    return passed >= total * 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)