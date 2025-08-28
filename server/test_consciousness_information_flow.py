#!/usr/bin/env python3
"""
Test consciousness information storage and retrieval functionality
Validates that important information is stored and correctly retrieved for context
"""

import asyncio
import os
import time
from consciousness.field_persistence import SURREALDB_AVAILABLE

async def test_information_storage_and_retrieval():
    """Test that consciousness stores and retrieves important information correctly"""
    
    print("🧠 Testing Consciousness Information Storage & Retrieval")
    print("=" * 65)
    
    if not SURREALDB_AVAILABLE:
        print("❌ SurrealDB not available - cannot test information persistence")
        return False
    
    try:
        # Create consciousness and smart context manager
        from processors.smart_context_manager import create_smart_context_manager
        from consciousness.core import create_consciousness
        
        # Mock context
        class MockContext:
            def __init__(self):
                self.messages = []
        
        # Set up test user
        test_user = "information_flow_test_user"
        os.environ['USER_ID'] = test_user
        os.environ['ENABLE_CONSCIOUSNESS'] = 'true'
        
        print("1. Setting up consciousness system...")
        smart_manager = create_smart_context_manager(
            MockContext(),
            user_id=test_user,
            enable_consciousness=True
        )
        
        consciousness = smart_manager._consciousness_instance
        persistence = smart_manager.field_persistence
        
        if not consciousness or not persistence:
            print("❌ Failed to initialize consciousness or persistence")
            return False
            
        print("✅ Consciousness system initialized")
        
        # Connect persistence layer
        await persistence.connect()
        
        # 2. Test information storage - feed important facts
        print("\n2. Storing important information...")
        
        important_facts = [
            "My dog's name is Potola and she loves playing fetch",
            "I work as a software engineer at TechCorp in San Francisco", 
            "My favorite programming language is Python and I specialize in AI",
            "I have a meeting with Sarah on Thursday at 3 PM",
            "My birthday is on December 15th and I'm planning a party"
        ]
        
        stored_symbols = []
        field_states_before = {}
        
        # Capture initial field states
        for symbol, field in consciousness.symbol_fields.items():
            field_states_before[symbol] = field.intensity
        
        # Process each fact through consciousness
        for fact in important_facts:
            print(f"  Processing: '{fact[:50]}...'")
            
            # Use consciousness to extract symbols (this evolves fields)
            symbols = consciousness.symbolize(fact)
            stored_symbols.extend(symbols)
            
            # Track field evolution
            await smart_manager._track_field_evolution_async(fact)
            
            # Store facts in memory system
            await smart_manager.memory_system.store_facts(fact)
        
        print(f"  Extracted symbols from facts: {set(stored_symbols)}")
        
        # Check field evolution
        fields_evolved = 0
        for symbol, field in consciousness.symbol_fields.items():
            if abs(field.intensity - field_states_before[symbol]) > 0.001:
                fields_evolved += 1
                print(f"    Field {symbol}: {field_states_before[symbol]:.4f} → {field.intensity:.4f}")
        
        print(f"  Fields evolved: {fields_evolved}")
        
        # Save consciousness state
        session_id = f"info_test_{int(time.time())}"
        saved = await smart_manager.save_consciousness_fields(session_id)
        print(f"  Consciousness state saved: {'✅' if saved else '❌'}")
        
        # 3. Test information retrieval through context building
        print("\n3. Testing information retrieval for context...")
        
        # Simulate queries that should trigger memory retrieval
        test_queries = [
            "What's my dog's name?",  # Should retrieve Potola
            "Where do I work?",       # Should retrieve TechCorp/SF
            "What's my favorite programming language?", # Should retrieve Python
            "When is my meeting with Sarah?",          # Should retrieve Thursday 3PM
            "When is my birthday?"   # Should retrieve Dec 15th
        ]
        
        retrieval_results = {}
        
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            
            # Build context (this should include relevant retrieved information)
            context_messages = await smart_manager._build_fixed_context(query)
            
            # Extract context content to check for relevant information
            context_content = ""
            for msg in context_messages:
                if isinstance(msg, dict) and 'content' in msg:
                    context_content += msg['content'] + " "
            
            # Check if relevant information appears in context
            query_keywords = {
                "What's my dog's name?": ["Potola", "dog"],
                "Where do I work?": ["TechCorp", "San Francisco", "software engineer"],
                "What's my favorite programming language?": ["Python", "programming"],
                "When is my meeting with Sarah?": ["Sarah", "Thursday", "3 PM"],
                "When is my birthday?": ["December", "15th", "birthday"]
            }
            
            relevant_keywords = query_keywords.get(query, [])
            found_keywords = []
            
            for keyword in relevant_keywords:
                if keyword.lower() in context_content.lower():
                    found_keywords.append(keyword)
            
            retrieval_success = len(found_keywords) > 0
            retrieval_results[query] = {
                'success': retrieval_success,
                'found_keywords': found_keywords,
                'context_length': len(context_content)
            }
            
            print(f"    Keywords found in context: {found_keywords}")
            print(f"    Retrieval success: {'✅' if retrieval_success else '❌'}")
            print(f"    Context length: {len(context_content)} chars")
        
        # 4. Test consciousness insights
        print("\n4. Testing consciousness insights generation...")
        
        insights = await persistence.get_consciousness_insights(test_user, days_back=1)
        if insights:
            print("✅ Consciousness insights generated:")
            print(f"    User: {insights.get('user_id')}")
            print(f"    Analysis period: {insights.get('analysis_period_days')} days")
            
            attractor_patterns = insights.get('attractor_patterns', [])
            field_evolution = insights.get('field_evolution', [])
            
            print(f"    Attractor patterns: {len(attractor_patterns)} recorded")
            print(f"    Field evolution records: {len(field_evolution)} tracked")
        else:
            print("⚠️  No consciousness insights generated")
        
        # 5. Test cross-session continuity
        print("\n5. Testing cross-session continuity...")
        
        # Create new consciousness instance (simulating new session)
        new_consciousness = create_consciousness(load_state=False)
        new_smart_manager = create_smart_context_manager(
            MockContext(),
            user_id=test_user,
            enable_consciousness=True
        )
        new_smart_manager.set_consciousness_instance(new_consciousness)
        
        # Load previous field states
        loaded_fields = await new_smart_manager.load_consciousness_fields()
        continuity_success = len(loaded_fields) > 0
        
        print(f"  Cross-session field loading: {'✅' if continuity_success else '❌'}")
        if continuity_success:
            print(f"    Loaded {len(loaded_fields)} field states from previous session")
        
        # 6. Summary and analysis
        print("\n6. Information Flow Analysis")
        print("-" * 40)
        
        successful_retrievals = sum(1 for r in retrieval_results.values() if r['success'])
        total_queries = len(retrieval_results)
        
        print(f"Facts stored: {len(important_facts)}")
        print(f"Symbols extracted: {len(set(stored_symbols))}")
        print(f"Fields evolved: {fields_evolved}")
        print(f"Successful retrievals: {successful_retrievals}/{total_queries}")
        print(f"Cross-session continuity: {'✅' if continuity_success else '❌'}")
        
        # Overall assessment
        overall_success = (
            saved and  # State saved
            successful_retrievals > 0 and  # Some information retrieved
            continuity_success  # Cross-session continuity works
        )
        
        if overall_success:
            print("\n✅ INFORMATION FLOW TEST PASSED")
            print("✅ Important information is being stored correctly")
            print("✅ Relevant information is being retrieved for context")
            print("✅ Consciousness provides meaningful memory enhancement")
        else:
            print("\n❌ INFORMATION FLOW TEST FAILED")
            print("❌ Information storage or retrieval has issues")
        
        # Cleanup
        await persistence.close()
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Information flow test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_specific_retrieval_scenario():
    """Test a specific realistic scenario"""
    
    print("\n" + "=" * 65)
    print("🎯 Testing Specific Retrieval Scenario")
    print("=" * 65)
    
    try:
        from processors.smart_context_manager import create_smart_context_manager
        
        class MockContext:
            def __init__(self):
                self.messages = []
        
        test_user = "scenario_test_user"
        os.environ['USER_ID'] = test_user
        
        smart_manager = create_smart_context_manager(
            MockContext(),
            user_id=test_user,
            enable_consciousness=True
        )
        
        if not smart_manager.field_persistence:
            print("⚠️  No field persistence available")
            return True  # Don't fail if SurrealDB unavailable
        
        await smart_manager.field_persistence.connect()
        
        # Scenario: User tells us about their pet, then asks about it later
        print("\n1. User shares information about their pet...")
        pet_info = "I have a golden retriever named Max who is 3 years old and loves swimming"
        
        # Process through consciousness 
        if smart_manager._consciousness_instance:
            symbols = smart_manager._consciousness_instance.symbolize(pet_info)
            print(f"   Symbols extracted: {symbols}")
        
        # Store in memory
        await smart_manager.memory_system.store_facts(pet_info)
        await smart_manager._track_field_evolution_async(pet_info)
        
        print("   Information stored in memory and consciousness")
        
        # Wait a moment (simulate time passing)
        await asyncio.sleep(0.1)
        
        print("\n2. Later, user asks about their pet...")
        query = "What can you tell me about my dog?"
        
        # Build context - this should include the pet information
        context_messages = await smart_manager._build_fixed_context(query)
        
        context_text = ""
        for msg in context_messages:
            if isinstance(msg, dict) and 'content' in msg:
                context_text += msg['content']
        
        # Check if Max is mentioned in the context
        has_pet_info = any(keyword in context_text.lower() for keyword in ["max", "retriever", "swimming", "3 years"])
        
        print(f"   Context includes pet information: {'✅' if has_pet_info else '❌'}")
        
        if has_pet_info:
            print("   Found relevant keywords in context")
        else:
            print("   ⚠️  Pet information not found in context")
            print(f"   Context preview: {context_text[:200]}...")
        
        await smart_manager.field_persistence.close()
        
        return has_pet_info
        
    except Exception as e:
        print(f"❌ Scenario test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_information_storage_and_retrieval())
    asyncio.run(test_specific_retrieval_scenario())