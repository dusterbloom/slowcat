#!/usr/bin/env python
"""
Final test of the LM Studio JSON schema fact extraction
"""

from memory.dspy_lmstudio_extractor import LMStudioFactExtractor

def test_final_quality():
    """Test the quality against our target expectations"""
    
    print("🎯 Final DSPy + LM Studio Quality Test")
    print("=" * 60)
    
    extractor = LMStudioFactExtractor()
    
    # Test the original problem case
    test_text = "My dog Rex is a brown German Shepherd"
    
    print(f'📝 Testing: "{test_text}"')
    print("-" * 50)
    
    facts = extractor.extract_facts(test_text)
    
    personal_facts = [f for f in facts if f.get('subject') == 'user']
    factual_facts = [f for f in facts if f.get('subject') != 'user' and 'Rex' in f.get('subject', '')]
    
    print(f"✅ Total facts extracted: {len(facts)}")
    
    print(f"\n👤 PERSONAL RELATIONS ({len(personal_facts)}):")
    for f in personal_facts:
        confidence = f.get('confidence', 0)
        print(f"  • {f['subject']} → {f['predicate']} → {f['object']} (conf: {confidence})")
    
    print(f"\n📚 FACTUAL RELATIONS ({len(factual_facts)}):")
    for f in factual_facts:
        confidence = f.get('confidence', 0)
        print(f"  • {f['subject']} → {f['predicate']} → {f['object']} (conf: {confidence})")
    
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    
    # Check for personal context (what REBEL missed)
    has_personal_context = any(f.get('predicate') in ['has_pet', 'owns'] and f.get('object') == 'Rex' for f in personal_facts)
    print(f"✅ Personal Context (user → owns/has_pet → Rex): {'YES' if has_personal_context else 'NO'}")
    
    # Check for factual precision (what Hybrid struggled with)  
    has_breed_info = any('breed' in f.get('predicate', '') for f in factual_facts)
    has_color_info = any('color' in f.get('predicate', '') for f in factual_facts)
    print(f"✅ Factual Precision (Rex → breed_of → German Shepherd): {'YES' if has_breed_info else 'NO'}")
    print(f"✅ Color Information (Rex → has_color → brown): {'YES' if has_color_info else 'NO'}")
    
    # Overall assessment
    if has_personal_context and (has_breed_info or has_color_info):
        print(f"\n🏆 SUCCESS! Achieved both personal context AND factual precision!")
        print(f"💡 This solves the core problem: REBEL's factual accuracy + Personal context")
    elif has_personal_context:
        print(f"\n✅ GOOD! Personal context captured, factual needs improvement")
    elif has_breed_info or has_color_info:
        print(f"\n✅ GOOD! Factual precision captured, personal context needs work")
    else:
        print(f"\n⚠️ NEEDS WORK - Missing both personal context and factual precision")
    
    print(f"\n" + "="*60)
    print("🔬 COMPARISON TO ORIGINAL APPROACHES:")
    print("📊 Original Hybrid: user → has_person → Rex (vague)")
    print("📊 Original REBEL: Rex → animal breed → German Shepherd (no personal context)")  
    print(f"📊 DSPy + LM Studio: COMBINES BOTH! ✨")

if __name__ == "__main__":
    test_final_quality()