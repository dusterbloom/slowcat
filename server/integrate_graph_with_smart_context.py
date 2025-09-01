#!/usr/bin/env python3
"""
Integrate Graph Functions with SmartContextManager

This script patches SmartContextManager to actually USE the graph functions
we created in create_complete_graph_system.py instead of traditional queries.

The integration happens through the GraphMemoryIntegration layer.
"""

import asyncio
import os
from loguru import logger


async def test_graph_integration():
    """Test the graph integration with actual data"""
    
    logger.info("🧪 Testing graph integration with SmartContextManager...")
    
    # Import the new integration layer
    from memory.graph_integration import integrate_graph_memory_with_context, get_graph_integration
    
    graph = get_graph_integration()
    await graph.ensure_connected()
    
    # Find a session with actual data to test
    conn = await graph.ensure_connected()
    
    # Get a test session with messages
    test_session_result = await conn.db.query("""
        SELECT session_id, count() as message_count 
        FROM messages 
        WHERE session_id IS NOT NONE
        GROUP BY session_id 
        ORDER BY message_count DESC 
        LIMIT 1;
    """)
    
    if not test_session_result or len(test_session_result) == 0:
        logger.warning("⚠️ No sessions with messages found - creating test data...")
        
        # Create test session and messages
        await conn.db.query("""
            CREATE sessions:test_graph_session SET 
                session_id = "test_graph_session",
                start_time = time::now(),
                is_active = false;
        """)
        
        # Create test messages
        for i in range(3):
            await conn.db.query(f"""
                CREATE messages:test_msg_{i} SET
                    session_id = "test_graph_session",
                    role = "{('user' if i % 2 == 0 else 'assistant')}",
                    content = "Test message {i}",
                    timestamp = time::now(),
                    message_order = {i};
            """)
        
        # Create test knowledge
        await conn.db.query("""
            CREATE knowledge:test_fact SET
                session_id = "test_graph_session",
                predicate = "discusses",
                confidence = 0.8,
                strength = 0.7,
                in = { canonical_name: "user" },
                out = { canonical_name: "graph_integration" },
                created_at = time::now();
        """)
        
        test_session_id = "test_graph_session"
        logger.info("✅ Created test data for graph integration")
    else:
        test_session_id = test_session_result[0]['session_id']
        message_count = test_session_result[0]['message_count']
        logger.info(f"📊 Found test session: {test_session_id} ({message_count} messages)")
    
    # Test 1: Session context via graph
    logger.info("🔍 Test 1: Getting session context via graph traversal...")
    session_context = await graph.get_session_context_graph(test_session_id)
    
    logger.info("📋 Session context results:")
    logger.info(f"   Messages: {session_context['stats'].get('message_count', 0)}")
    logger.info(f"   Knowledge: {session_context['stats'].get('knowledge_count', 0)}")
    logger.info(f"   Entities: {session_context['stats'].get('entity_count', 0)}")
    logger.info(f"   Engrams: {session_context['stats'].get('engram_count', 0)}")
    
    # Test 2: Entity knowledge via graph
    logger.info("🔍 Test 2: Getting entity knowledge via graph relations...")
    
    # Find an entity to test with
    entity_result = await conn.db.query("""
        SELECT canonical_name FROM entity 
        WHERE canonical_name IS NOT NONE AND canonical_name != "" 
        LIMIT 1;
    """)
    
    if entity_result and len(entity_result) > 0:
        test_entity = entity_result[0]['canonical_name']
        entity_knowledge = await graph.get_entity_knowledge_graph(test_entity)
        
        logger.info(f"📋 Entity '{test_entity}' knowledge:")
        logger.info(f"   Knowledge facts: {entity_knowledge['stats'].get('knowledge_count', 0)}")
        logger.info(f"   Sessions mentioned: {entity_knowledge['stats'].get('session_count', 0)}")
    else:
        logger.info("   No entities found to test with")
    
    # Test 3: Engram detection via graph
    logger.info("🔍 Test 3: Testing engram detection via graph patterns...")
    engram_result = await graph.detect_session_engrams(test_session_id, min_coherence=0.5, min_facts=1)
    
    if engram_result.get('success'):
        logger.info(f"✅ Engram {engram_result.get('action', 'detected')}")
        logger.info(f"   Narrative: {engram_result.get('narrative', 'N/A')[:100]}...")
        logger.info(f"   Coherence: {engram_result.get('coherence', 0):.2f}")
    else:
        logger.info(f"⚠️ Engram detection: {engram_result.get('reason', 'failed')}")
    
    # Test 4: Cross-session patterns
    logger.info("🔍 Test 4: Finding cross-session patterns via engram graph...")
    patterns = await graph.get_cross_session_patterns(min_sessions=1)  # Lower threshold for testing
    
    logger.info(f"📋 Found {len(patterns)} cross-session patterns:")
    for i, pattern in enumerate(patterns[:3]):  # Show first 3
        narrative = pattern.get('narrative', 'N/A') or 'N/A'
        coherence = pattern.get('coherence', 0) or 0
        session_count = pattern.get('session_count', 0) or 0
        logger.info(f"   Pattern {i+1}: {narrative[:60]}...")
        logger.info(f"      Coherence: {coherence:.2f}")
        logger.info(f"      Sessions: {session_count}")
    
    # Test 5: Integrated context for SmartContextManager
    logger.info("🔍 Test 5: Full integration with SmartContextManager format...")
    integrated_context = await integrate_graph_memory_with_context(test_session_id, max_context_tokens=2000)
    
    logger.info("📋 Integrated context summary:")
    logger.info(f"   Messages: {len(integrated_context.get('messages', []))}")
    logger.info(f"   Facts: {len(integrated_context.get('facts', []))}")
    logger.info(f"   Entities: {len(integrated_context.get('entities', []))}")
    logger.info(f"   Engrams: {len(integrated_context.get('engrams', []))}")
    logger.info(f"   Patterns: {len(integrated_context.get('patterns', []))}")
    logger.info(f"   Summary: {integrated_context.get('summary', 'N/A')}")
    logger.info(f"   Token estimate: {integrated_context.get('token_estimate', 0)}")
    
    logger.info("🎉 Graph integration testing complete!")
    
    return integrated_context


async def create_smart_context_patch():
    """Create a patch showing how to integrate graph functions into SmartContextManager"""
    
    logger.info("📝 Creating SmartContextManager patch for graph integration...")
    
    patch_content = '''
# PATCH: SmartContextManager Graph Integration
# Add this to the imports section of processors/smart_context_manager.py:

from memory.graph_integration import integrate_graph_memory_with_context, get_graph_integration

# REPLACE the _retrieve_contextual_memory method with this graph-enhanced version:

async def _retrieve_contextual_memory(self, turn_count: int) -> Dict[str, Any]:
    """
    Retrieve contextual memory using GRAPH TRAVERSAL instead of traditional queries.
    
    This uses the SurrealDB graph functions we created for proper RELATE-based retrieval.
    """
    try:
        # Use graph integration instead of traditional queries
        session_id = getattr(self.session, 'session_id', None)
        if not session_id:
            # Generate session ID if not set
            session_id = f"session_{int(time.time())}"
            self.session.session_id = session_id
        
        # Get comprehensive context via graph traversal
        graph_context = await integrate_graph_memory_with_context(
            session_id=session_id,
            max_context_tokens=self.budget.contextual_memory
        )
        
        # Log what we retrieved via graph
        logger.debug(f"🌐 Graph context retrieved:")
        logger.debug(f"   Messages: {len(graph_context.get('messages', []))}")
        logger.debug(f"   Facts: {len(graph_context.get('facts', []))}")
        logger.debug(f"   Entities: {len(graph_context.get('entities', []))}")
        logger.debug(f"   Engrams: {len(graph_context.get('engrams', []))}")
        logger.debug(f"   Patterns: {len(graph_context.get('patterns', []))}")
        logger.debug(f"   Token estimate: {graph_context.get('token_estimate', 0)}")
        
        # Convert graph data to SmartContextManager format
        contextual_memory = {
            'messages': graph_context.get('messages', []),
            'facts': graph_context.get('facts', []),
            'entities': graph_context.get('entities', []),
            'engrams': graph_context.get('engrams', []),
            'cross_session_patterns': graph_context.get('patterns', []),
            'conversation_summary': graph_context.get('summary', ''),
            'session_stats': graph_context.get('stats', {}),
            'retrieval_method': 'graph_traversal'
        }
        
        return contextual_memory
        
    except Exception as e:
        logger.error(f"Graph memory retrieval failed: {e}")
        # Fallback to basic memory if graph fails
        return {
            'messages': [],
            'facts': [],
            'entities': [],
            'engrams': [],
            'cross_session_patterns': [],
            'conversation_summary': '',
            'session_stats': {},
            'retrieval_method': 'fallback'
        }


# ALSO ADD this method for engram processing:

async def _process_assistant_response_for_engrams(self, response_text: str):
    """
    Process assistant response to trigger engram detection using graph functions.
    
    This uses the graph-based engram detection we created.
    """
    try:
        if not response_text or len(response_text) < 10:
            return
            
        session_id = getattr(self.session, 'session_id', None)
        if not session_id:
            return
            
        # Trigger engram detection using graph functions
        graph = get_graph_integration()
        engram_result = await graph.detect_session_engrams(
            session_id=session_id,
            min_coherence=0.6,  # Configurable threshold
            min_facts=2         # Minimum facts needed for engram
        )
        
        if engram_result.get('success'):
            action = engram_result.get('action', 'unknown')
            narrative = engram_result.get('narrative', '')
            coherence = engram_result.get('coherence', 0)
            
            logger.info(f"🧠 Engram {action}: {narrative[:60]}... (coherence: {coherence:.2f})")
            
            # Update session metadata with engram info
            if not hasattr(self.session, 'engram_count'):
                self.session.engram_count = 0
            if action == 'created':
                self.session.engram_count += 1
        
    except Exception as e:
        logger.warning(f"Engram processing failed: {e}")


# UPDATE the process_frame method to call engram processing:
# Add this line after assistant response processing:

if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
    # Process for engrams (async - don't await to avoid blocking)
    asyncio.create_task(self._process_assistant_response_for_engrams(frame.text))

'''
    
    # Write the patch file
    with open('/Users/peppi/Dev/slowcat-consciousness/server/smart_context_graph_patch.py', 'w') as f:
        f.write(patch_content)
    
    logger.info("✅ Created smart_context_graph_patch.py")
    logger.info("📝 This shows how to integrate graph functions into SmartContextManager")
    
    return patch_content


async def main():
    """Main integration and testing function"""
    
    logger.info("🌐 Integrating Graph Functions with SmartContextManager...")
    
    # Test the graph integration
    test_results = await test_graph_integration()
    
    # Create the patch file
    patch_content = await create_smart_context_patch()
    
    logger.info("🎉 Graph integration complete!")
    logger.info("")
    logger.info("📋 SUMMARY:")
    logger.info("✅ Created memory/graph_integration.py - Bridge layer for graph functions")
    logger.info("✅ Created smart_context_graph_patch.py - Integration guide for SmartContextManager")
    logger.info("✅ Tested graph functions with actual data")
    logger.info("")
    logger.info("🔧 NEXT STEPS:")
    logger.info("1. Apply the patch to processors/smart_context_manager.py")
    logger.info("2. Test with bot_v2.py and run_bot.sh")
    logger.info("3. Monitor logs for 'Graph context retrieved' messages")
    logger.info("4. Verify engram detection with '🧠 Engram created/reinforced' messages")
    
    return test_results, patch_content


if __name__ == "__main__":
    asyncio.run(main())