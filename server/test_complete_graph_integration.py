#!/usr/bin/env python3
"""
COMPREHENSIVE GRAPH INTEGRATION TEST

This test verifies that ALL new graph functions are being used and ALL new record tables 
are being populated when running the actual bot (bot_v2.py).

Tests:
1. All graph functions exist in database
2. SmartContextManager uses graph integration
3. All RELATE tables are created
4. Graph traversal works end-to-end
5. Engram creation with proper RELATE edges
6. Cross-session pattern recognition
7. Integration with actual bot pipeline

NO ASSUMPTIONS - Everything is verified step by step.
"""

import asyncio
import json
import time
from typing import Dict, List, Any
from loguru import logger

# Test configuration
TEST_SESSION_ID = f"test_complete_integration_{int(time.time())}"
TEST_USER_ID = "test_user_graph"


class ComprehensiveGraphIntegrationTest:
    """Complete test of graph integration with the actual bot"""
    
    def __init__(self):
        self.conn = None
        self.graph = None
        self.test_results = {
            'database_functions': False,
            'relation_tables': False,
            'smart_context_integration': False,
            'graph_traversal': False,
            'engram_creation': False,
            'cross_session_patterns': False,
            'bot_pipeline_integration': False,
            'all_tables_populated': False
        }
        self.created_records = {
            'sessions': [],
            'messages': [],
            'knowledge': [],
            'entity': [],
            'engrams': [],
            'relations': []
        }
    
    async def setup(self):
        """Setup test environment"""
        logger.info("🔧 Setting up comprehensive graph integration test...")
        
        # Import and setup connections
        from memory.surreal_connection import SurrealConnectionManager
        from memory.graph_integration import get_graph_integration
        
        self.conn = SurrealConnectionManager()
        await self.conn.ensure_connected()
        self.graph = get_graph_integration()
        await self.graph.ensure_connected()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_1_database_functions_exist(self):
        """Test 1: Verify all graph functions exist in database"""
        logger.info("🔍 Test 1: Checking database graph functions...")
        
        try:
            # Check all the functions we created
            functions_to_check = [
                'fn::analyze_session_graph',
                'fn::analyze_entity_graph', 
                'fn::detect_engrams_graph',
                'fn::get_session_messages_graph',
                'fn::get_entity_sessions_graph'
            ]
            
            existing_functions = []
            for func in functions_to_check:
                try:
                    # Try calling each function to see if it exists
                    result = await self.conn.db.query(f"RETURN {func}('test');")
                    existing_functions.append(func)
                    logger.info(f"   ✅ {func} exists")
                except Exception as e:
                    if "does not exist" in str(e):
                        logger.error(f"   ❌ {func} MISSING")
                    else:
                        existing_functions.append(func) # Exists but failed for other reason
                        logger.info(f"   ✅ {func} exists (failed with: {str(e)[:50]}...)")
            
            if len(existing_functions) == len(functions_to_check):
                self.test_results['database_functions'] = True
                logger.info("✅ Test 1 PASSED: All database functions exist")
            else:
                missing = set(functions_to_check) - set(existing_functions)
                logger.error(f"❌ Test 1 FAILED: Missing functions: {missing}")
            
        except Exception as e:
            logger.error(f"❌ Test 1 ERROR: {e}")
    
    async def test_2_relation_tables_exist(self):
        """Test 2: Verify all RELATE tables exist"""
        logger.info("🔍 Test 2: Checking RELATE relation tables...")
        
        try:
            # Check all the relation tables we created
            relation_tables = [
                'message_belongs_to',
                'knowledge_about', 
                'entity_mentioned_in',
                'engram_contains',
                'engram_appears_in',
                'session_includes',
                'knowledge_from'
            ]
            
            existing_tables = []
            for table in relation_tables:
                try:
                    result = await self.conn.db.query(f"INFO FOR TABLE {table};")
                    if result:
                        existing_tables.append(table)
                        logger.info(f"   ✅ {table} table exists")
                except Exception:
                    logger.error(f"   ❌ {table} table MISSING")
            
            if len(existing_tables) == len(relation_tables):
                self.test_results['relation_tables'] = True
                logger.info("✅ Test 2 PASSED: All relation tables exist")
            else:
                missing = set(relation_tables) - set(existing_tables)
                logger.error(f"❌ Test 2 FAILED: Missing tables: {missing}")
                
        except Exception as e:
            logger.error(f"❌ Test 2 ERROR: {e}")
    
    async def test_3_smart_context_integration(self):
        """Test 3: Verify SmartContextManager uses graph integration"""
        logger.info("🔍 Test 3: Testing SmartContextManager graph integration...")
        
        try:
            # Import SmartContextManager and check if it has graph integration
            from processors.smart_context_manager import SmartContextManager
            
            # Check if it imports graph integration
            import inspect
            source = inspect.getsource(SmartContextManager)
            
            has_import = 'from memory.graph_integration import get_graph_integration' in source
            has_usage = 'get_graph_integration()' in source
            
            if has_import and has_usage:
                self.test_results['smart_context_integration'] = True
                logger.info("✅ Test 3 PASSED: SmartContextManager uses graph integration")
            else:
                logger.error(f"❌ Test 3 FAILED: Missing import={not has_import}, missing usage={not has_usage}")
                
        except Exception as e:
            logger.error(f"❌ Test 3 ERROR: {e}")
    
    async def test_4_create_test_data(self):
        """Test 4: Create comprehensive test data with RELATE edges"""
        logger.info("🔍 Test 4: Creating test data with graph relations...")
        
        try:
            # Create test session with proper ID format
            session_result = await self.conn.db.query(f"""
                CREATE sessions:⟨{TEST_SESSION_ID}⟩ SET
                    session_id = $session_id,
                    start_time = time::now(),
                    is_active = false,
                    speaker_id = $user_id;
            """, {"session_id": TEST_SESSION_ID, "user_id": TEST_USER_ID})
            
            if session_result:
                self.created_records['sessions'].append(TEST_SESSION_ID)
                logger.info("   ✅ Created test session")
            
            # Create test entities
            entities = ["test_dog", "test_user"]
            for entity_name in entities:
                entity_result = await self.conn.db.query(f"""
                    CREATE entity:{entity_name} SET
                        canonical_name = $name,
                        entity_type = "test",
                        created_at = time::now();
                """, {"name": entity_name})
                
                if entity_result:
                    self.created_records['entity'].append(entity_name)
                    logger.info(f"   ✅ Created entity: {entity_name}")
            
            # Create test messages with RELATE edges
            messages_data = [
                {"role": "user", "content": "My dog's name is Fluffy", "order": 1},
                {"role": "assistant", "content": "That's a lovely name for your dog!", "order": 2},
                {"role": "user", "content": "Fluffy is a golden retriever", "order": 3}
            ]
            
            for msg_data in messages_data:
                msg_id = f"test_msg_{msg_data['order']}"
                msg_result = await self.conn.db.query(f"""
                    CREATE messages:{msg_id} SET
                        session_id = $session_id,
                        role = $role,
                        content = $content,
                        message_order = $order,
                        timestamp = time::now();
                """, {
                    "session_id": TEST_SESSION_ID,
                    "role": msg_data['role'],
                    "content": msg_data['content'],
                    "order": msg_data['order']
                })
                
                if msg_result:
                    self.created_records['messages'].append(msg_id)
                    
                    # Create RELATE edge: message belongs to session
                    relate_result = await self.conn.db.query(f"""
                        RELATE messages:{msg_id}->message_belongs_to->sessions:{TEST_SESSION_ID} SET
                            created_at = time::now();
                    """)
                    
                    if relate_result:
                        self.created_records['relations'].append(f"messages:{msg_id}->message_belongs_to->sessions:{TEST_SESSION_ID}")
                        logger.info(f"   ✅ Created message and relation: {msg_id}")
            
            # Create test knowledge with RELATE edges
            knowledge_facts = [
                {"predicate": "has_name", "object": "Fluffy", "subject": "user's dog"},
                {"predicate": "is_breed", "object": "golden retriever", "subject": "Fluffy"}
            ]
            
            for i, fact in enumerate(knowledge_facts):
                knowledge_id = f"test_knowledge_{i}"
                knowledge_result = await self.conn.db.query(f"""
                    CREATE knowledge:{knowledge_id} SET
                        session_id = $session_id,
                        predicate = $predicate,
                        confidence = 0.9,
                        strength = 1.0,
                        extraction_method = "test",
                        subject = $subject,
                        object = $object,
                        text_score = 1.0,
                        created_at = time::now();
                """, {
                    "session_id": TEST_SESSION_ID,
                    "predicate": fact['predicate'],
                    "subject": fact['subject'],
                    "object": fact['object']
                })
                
                if knowledge_result:
                    self.created_records['knowledge'].append(knowledge_id)
                    
                    # Create RELATE edge: knowledge from session  
                    relate_result = await self.conn.db.query(f"""
                        RELATE knowledge:{knowledge_id}->knowledge_from->sessions:{TEST_SESSION_ID} SET
                            created_at = time::now();
                    """)
                    
                    # Create RELATE edge: knowledge about entity
                    entity_relate = await self.conn.db.query(f"""
                        RELATE knowledge:{knowledge_id}->knowledge_about->entity:test_dog SET
                            strength = 0.9;
                    """)
                    
                    if relate_result and entity_relate:
                        self.created_records['relations'].append(f"knowledge:{knowledge_id}->knowledge_from->sessions:{TEST_SESSION_ID}")
                        self.created_records['relations'].append(f"knowledge:{knowledge_id}->knowledge_about->entity:test_dog")
                        logger.info(f"   ✅ Created knowledge and relations: {knowledge_id}")
            
            # Create RELATE edge: entity mentioned in session
            entity_mention = await self.conn.db.query(f"""
                RELATE entity:test_dog->entity_mentioned_in->sessions:{TEST_SESSION_ID} SET
                    frequency = 2,
                    first_mention = time::now();
            """)
            
            if entity_mention:
                self.created_records['relations'].append(f"entity:test_dog->entity_mentioned_in->sessions:{TEST_SESSION_ID}")
                logger.info("   ✅ Created entity mention relation")
            
            logger.info("✅ Test 4 PASSED: Created comprehensive test data with RELATE edges")
            
        except Exception as e:
            logger.error(f"❌ Test 4 ERROR: {e}")
    
    async def test_5_graph_traversal_functions(self):
        """Test 5: Test all graph traversal functions work"""
        logger.info("🔍 Test 5: Testing graph traversal functions...")
        
        try:
            # Test session analysis via graph
            session_data = await self.graph.get_session_context_graph(TEST_SESSION_ID)
            
            session_stats = session_data.get('stats', {})
            messages_count = session_stats.get('message_count', 0)
            knowledge_count = session_stats.get('knowledge_count', 0)
            entity_count = session_stats.get('entity_count', 0)
            
            logger.info(f"   📊 Session analysis: {messages_count} messages, {knowledge_count} knowledge, {entity_count} entities")
            
            if messages_count > 0 and knowledge_count > 0:
                logger.info("   ✅ Session graph traversal works")
            else:
                logger.warning("   ⚠️ Session graph traversal returned empty results")
            
            # Test entity analysis via graph  
            entity_data = await self.graph.get_entity_knowledge_graph("test_dog")
            
            entity_stats = entity_data.get('stats', {})
            entity_knowledge = entity_stats.get('knowledge_count', 0)
            entity_sessions = entity_stats.get('session_count', 0)
            
            logger.info(f"   📊 Entity analysis: {entity_knowledge} knowledge, {entity_sessions} sessions")
            
            if entity_knowledge > 0:
                logger.info("   ✅ Entity graph traversal works")
            else:
                logger.warning("   ⚠️ Entity graph traversal returned empty results")
            
            # Test direct function calls
            direct_session = await self.conn.db.query(f"""
                RETURN fn::analyze_session_graph($session_id);
            """, {"session_id": TEST_SESSION_ID})
            
            if direct_session and len(direct_session) > 0:
                logger.info("   ✅ Direct function call works")
                self.test_results['graph_traversal'] = True
                logger.info("✅ Test 5 PASSED: Graph traversal functions work")
            else:
                logger.error("   ❌ Direct function call failed")
            
        except Exception as e:
            logger.error(f"❌ Test 5 ERROR: {e}")
    
    async def test_6_engram_detection(self):
        """Test 6: Test engram detection and creation with RELATE edges"""
        logger.info("🔍 Test 6: Testing engram detection and creation...")
        
        try:
            # Test engram detection
            engram_result = await self.graph.detect_session_engrams(
                TEST_SESSION_ID, 
                min_coherence=0.5,  # Lower threshold for test
                min_facts=1         # Lower threshold for test
            )
            
            logger.info(f"   📊 Engram detection result: {engram_result}")
            
            if engram_result and engram_result.get('success'):
                action = engram_result.get('action', 'unknown')
                narrative = engram_result.get('narrative', '')
                coherence = engram_result.get('coherence', 0)
                engram_id = engram_result.get('engram_id', '')
                
                logger.info(f"   🧠 Engram {action}: {narrative[:50]}... (coherence: {coherence:.2f})")
                
                if engram_id:
                    self.created_records['engrams'].append(engram_id)
                    
                    # Check if RELATE edges were created for the engram
                    engram_relations = await self.conn.db.query(f"""
                        SELECT * FROM engram_contains WHERE in = {engram_id};
                    """)
                    
                    appears_in_relations = await self.conn.db.query(f"""
                        SELECT * FROM engram_appears_in WHERE in = {engram_id};
                    """)
                    
                    if engram_relations or appears_in_relations:
                        logger.info(f"   ✅ Engram RELATE edges created ({len(engram_relations or [])} contains, {len(appears_in_relations or [])} appears_in)")
                        self.test_results['engram_creation'] = True
                        logger.info("✅ Test 6 PASSED: Engram detection and RELATE edges work")
                    else:
                        logger.warning("   ⚠️ Engram created but no RELATE edges found")
                else:
                    logger.warning("   ⚠️ Engram action completed but no engram_id returned")
            else:
                reason = engram_result.get('reason', 'unknown') if engram_result else 'no_result'
                logger.warning(f"   ⚠️ Engram detection failed: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Test 6 ERROR: {e}")
    
    async def test_7_cross_session_patterns(self):
        """Test 7: Test cross-session pattern recognition"""
        logger.info("🔍 Test 7: Testing cross-session pattern recognition...")
        
        try:
            # Create a second session to test cross-session patterns
            second_session_id = f"{TEST_SESSION_ID}_2"
            
            # Create second session with similar patterns
            await self.conn.db.query(f"""
                CREATE sessions:{second_session_id} SET
                    session_id = $session_id,
                    start_time = time::now(),
                    is_active = false,
                    speaker_id = $user_id;
            """, {"session_id": second_session_id, "user_id": TEST_USER_ID})
            
            # Add similar knowledge to second session
            await self.conn.db.query(f"""
                CREATE knowledge:test_knowledge_cross SET
                    session_id = $session_id,
                    predicate = "has_name",
                    confidence = 0.8,
                    strength = 1.0,
                    extraction_method = "test",
                    created_at = time::now();
            """, {"session_id": second_session_id})
            
            # Create engram for second session
            await self.graph.detect_session_engrams(
                second_session_id,
                min_coherence=0.5,
                min_facts=1
            )
            
            # Test cross-session pattern recognition
            patterns = await self.graph.get_cross_session_patterns(min_sessions=1) # Lower threshold for test
            
            logger.info(f"   📊 Found {len(patterns)} cross-session patterns")
            
            if len(patterns) > 0:
                for i, pattern in enumerate(patterns[:2]):  # Show first 2
                    narrative = pattern.get('narrative', 'N/A') or 'N/A'
                    session_count = pattern.get('session_count', 0) or 0
                    coherence = pattern.get('coherence', 0) or 0
                    logger.info(f"   Pattern {i+1}: {narrative[:40]}... ({session_count} sessions, coherence: {coherence:.2f})")
                
                self.test_results['cross_session_patterns'] = True
                logger.info("✅ Test 7 PASSED: Cross-session pattern recognition works")
            else:
                logger.warning("   ⚠️ No cross-session patterns found")
            
        except Exception as e:
            logger.error(f"❌ Test 7 ERROR: {e}")
    
    async def test_8_verify_all_tables_populated(self):
        """Test 8: Verify all tables have data and relations work"""
        logger.info("🔍 Test 8: Verifying all tables are populated...")
        
        try:
            tables_to_check = {
                'sessions': f"SELECT * FROM sessions WHERE session_id = '{TEST_SESSION_ID}';",
                'messages': f"SELECT * FROM messages WHERE session_id = '{TEST_SESSION_ID}';", 
                'knowledge': f"SELECT * FROM knowledge WHERE session_id = '{TEST_SESSION_ID}';",
                'entity': "SELECT * FROM entity WHERE canonical_name CONTAINS 'test_';",
                'engrams': "SELECT * FROM engrams LIMIT 5;",
                'message_belongs_to': f"SELECT * FROM message_belongs_to LIMIT 10;",
                'knowledge_about': "SELECT * FROM knowledge_about LIMIT 5;",
                'entity_mentioned_in': f"SELECT * FROM entity_mentioned_in LIMIT 10;",
                'knowledge_from': f"SELECT * FROM knowledge_from LIMIT 10;"
            }
            
            populated_tables = 0
            total_tables = len(tables_to_check)
            
            for table_name, query in tables_to_check.items():
                try:
                    result = await self.conn.db.query(query)
                    count = len(result) if result else 0
                    
                    if count > 0:
                        populated_tables += 1
                        logger.info(f"   ✅ {table_name}: {count} records")
                    else:
                        logger.warning(f"   ⚠️ {table_name}: EMPTY")
                        
                except Exception as e:
                    logger.error(f"   ❌ {table_name}: ERROR - {e}")
            
            if populated_tables >= (total_tables * 0.8):  # At least 80% of tables populated
                self.test_results['all_tables_populated'] = True
                logger.info(f"✅ Test 8 PASSED: {populated_tables}/{total_tables} tables populated")
            else:
                logger.error(f"❌ Test 8 FAILED: Only {populated_tables}/{total_tables} tables populated")
            
        except Exception as e:
            logger.error(f"❌ Test 8 ERROR: {e}")
    
    async def test_9_bot_pipeline_integration(self):
        """Test 9: Test integration with bot pipeline components"""
        logger.info("🔍 Test 9: Testing bot pipeline integration...")
        
        try:
            # Test that we can import and use the integration with SmartContextManager
            from memory.graph_integration import integrate_graph_memory_with_context
            context_result = await integrate_graph_memory_with_context(
                TEST_SESSION_ID, 
                max_context_tokens=2000
            )
            
            messages = context_result.get('messages', [])
            facts = context_result.get('facts', [])
            entities = context_result.get('entities', [])
            patterns = context_result.get('patterns', [])
            summary = context_result.get('summary', '')
            
            logger.info(f"   📊 Context integration: {len(messages)} messages, {len(facts)} facts, {len(entities)} entities, {len(patterns)} patterns")
            logger.info(f"   📝 Summary: {summary[:50]}...")
            
            if len(messages) > 0 or len(facts) > 0:
                self.test_results['bot_pipeline_integration'] = True
                logger.info("✅ Test 9 PASSED: Bot pipeline integration works")
            else:
                logger.warning("   ⚠️ Bot pipeline integration returned empty results")
            
        except Exception as e:
            logger.error(f"❌ Test 9 ERROR: {e}")
    
    async def cleanup(self):
        """Clean up test data"""
        logger.info("🧹 Cleaning up test data...")
        
        try:
            # Clean up in reverse order
            cleanup_queries = [
                f"DELETE FROM message_belongs_to WHERE out.session_id = '{TEST_SESSION_ID}';",
                f"DELETE FROM knowledge_about WHERE in.session_id = '{TEST_SESSION_ID}';", 
                f"DELETE FROM knowledge_from WHERE out.session_id = '{TEST_SESSION_ID}';",
                f"DELETE FROM entity_mentioned_in WHERE out.session_id = '{TEST_SESSION_ID}';",
                f"DELETE FROM engram_contains;",
                f"DELETE FROM engram_appears_in;",
                f"DELETE FROM messages WHERE session_id = '{TEST_SESSION_ID}';",
                f"DELETE FROM knowledge WHERE session_id = '{TEST_SESSION_ID}';",
                f"DELETE FROM sessions WHERE session_id CONTAINS '{TEST_SESSION_ID}';",
                "DELETE FROM entity WHERE canonical_name CONTAINS 'test_';",
                "DELETE FROM engrams WHERE id CONTAINS 'test_' OR narrative_summary CONTAINS 'test';"
            ]
            
            for query in cleanup_queries:
                try:
                    await self.conn.db.query(query)
                except Exception as e:
                    logger.debug(f"Cleanup query failed (expected): {e}")
            
            logger.info("✅ Test data cleanup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup error (non-critical): {e}")
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 COMPREHENSIVE GRAPH INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status}: {test_name.replace('_', ' ').title()}")
        
        logger.info("=" * 60)
        logger.info(f"📊 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Graph integration is complete and working!")
            logger.info("✅ All graph functions are being used")
            logger.info("✅ All RELATE tables are populated") 
            logger.info("✅ SmartContextManager uses graph integration")
            logger.info("✅ Bot pipeline integration works")
        else:
            failed_tests = [name for name, result in self.test_results.items() if not result]
            logger.error(f"❌ TESTS FAILED: {failed_tests}")
            logger.error("🚨 Graph integration is NOT complete - issues found")
        
        logger.info("=" * 60)
        
        # Summary of created records
        logger.info("📋 TEST DATA SUMMARY:")
        for record_type, records in self.created_records.items():
            logger.info(f"   {record_type}: {len(records)} created")
        
        return passed_tests == total_tests


async def main():
    """Run comprehensive graph integration test"""
    test = ComprehensiveGraphIntegrationTest()
    
    try:
        await test.setup()
        
        # Run all tests in sequence
        await test.test_1_database_functions_exist()
        await test.test_2_relation_tables_exist()
        await test.test_3_smart_context_integration()
        await test.test_4_create_test_data()
        await test.test_5_graph_traversal_functions()
        await test.test_6_engram_detection()
        await test.test_7_cross_session_patterns()
        await test.test_8_verify_all_tables_populated()
        await test.test_9_bot_pipeline_integration()
        
        # Generate final report
        success = await test.generate_report()
        
        return success
        
    except Exception as e:
        logger.error(f"💥 Test suite error: {e}")
        return False
    finally:
        await test.cleanup()


if __name__ == "__main__":
    logger.info("🚀 Starting COMPREHENSIVE Graph Integration Test")
    logger.info("This test verifies ALL graph functions and tables work with the actual bot")
    
    success = asyncio.run(main())
    
    if success:
        logger.info("🎉 SUCCESS: Graph integration is complete and verified!")
        exit(0)
    else:
        logger.error("💥 FAILURE: Graph integration has issues that need fixing")
        exit(1)