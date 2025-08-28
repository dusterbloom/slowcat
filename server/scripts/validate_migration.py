#!/usr/bin/env python3
"""
Validate Graph Schema Migration

This script validates that the migration from flat tables to graph schema
was successful and that all data relationships are working correctly.
"""

import asyncio
import os
from typing import Dict, List, Any
from loguru import logger
from dataclasses import dataclass

# Import the graph memory system
import sys
sys.path.append('..')
from memory.graph_surreal_memory import GraphSurrealMemory

@dataclass
class ValidationResult:
    """Results of validation tests"""
    test_name: str
    passed: bool
    details: str
    count: int = 0

class MigrationValidator:
    """Validates the graph schema migration"""
    
    def __init__(self):
        self.memory = GraphSurrealMemory()
        self.results: List[ValidationResult] = []
    
    async def validate_all(self) -> List[ValidationResult]:
        """Run all validation tests"""
        logger.info("🔍 Starting migration validation...")
        
        await self.memory.connect()
        
        # Run all validation tests
        test_methods = [
            self._test_users_exist,
            self._test_sessions_exist,
            self._test_messages_exist,
            self._test_concepts_exist,
            self._test_knowledge_relationships,
            self._test_session_message_relationships,
            self._test_graph_traversal_queries,
            self._test_search_functionality,
            self._test_conversation_context,
            self._test_data_integrity
        ]
        
        for test_method in test_methods:
            try:
                result = await test_method()
                self.results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{status} {result.test_name}: {result.details}")
                
            except Exception as e:
                error_result = ValidationResult(
                    test_name=test_method.__name__,
                    passed=False,
                    details=f"Test failed with error: {e}"
                )
                self.results.append(error_result)
                logger.error(f"❌ ERROR {test_method.__name__}: {e}")
        
        await self.memory.close()
        return self.results
    
    async def _test_users_exist(self) -> ValidationResult:
        """Test that users were created from speakers"""
        try:
            result = await self.memory.db.query("SELECT count() FROM user")
            # Handle SurrealDB result format - result is a list with 'result' key containing the data
            count = 0
            if result and len(result) > 0 and 'result' in result[0]:
                result_data = result[0]['result']
                if result_data and len(result_data) > 0:
                    count = result_data[0].get('count', 0) if isinstance(result_data[0], dict) else result_data[0]
            
            if count > 0:
                # Get sample user
                sample = await self.memory.db.query("SELECT * FROM user LIMIT 1")
                user = {}
                if sample and len(sample) > 0 and 'result' in sample[0]:
                    sample_data = sample[0]['result']
                    if sample_data and len(sample_data) > 0:
                        user = sample_data[0]
                
                return ValidationResult(
                    test_name="Users Created",
                    passed=True,
                    details=f"{count} users created (sample: {user.get('name', 'unknown')})",
                    count=count
                )
            else:
                return ValidationResult(
                    test_name="Users Created", 
                    passed=False,
                    details="No users found in database"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Users Created",
                passed=False, 
                details=f"Query failed: {e}"
            )
    
    async def _test_sessions_exist(self) -> ValidationResult:
        """Test that sessions were created with user relationships"""
        try:
            # Count sessions
            result = await self.memory.db.query("SELECT count() FROM session")
            count = result[0]['result'][0] if result[0].get('result') else 0
            
            if count > 0:
                # Test user relationship
                sample = await self.memory.db.query("""
                    SELECT *, user_id.name as user_name FROM session LIMIT 1
                """)
                
                session = sample[0]['result'][0] if sample[0].get('result') else {}
                user_name = session.get('user_name', 'unknown')
                
                return ValidationResult(
                    test_name="Sessions Created",
                    passed=True,
                    details=f"{count} sessions created with user links (sample user: {user_name})",
                    count=count
                )
            else:
                return ValidationResult(
                    test_name="Sessions Created",
                    passed=False,
                    details="No sessions found"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Sessions Created",
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_messages_exist(self) -> ValidationResult:
        """Test that messages were created"""
        try:
            result = await self.memory.db.query("SELECT count() FROM message")
            count = result[0]['result'][0] if result[0].get('result') else 0
            
            return ValidationResult(
                test_name="Messages Created",
                passed=count > 0,
                details=f"{count} messages created" if count > 0 else "No messages found",
                count=count
            )
                
        except Exception as e:
            return ValidationResult(
                test_name="Messages Created",
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_concepts_exist(self) -> ValidationResult:
        """Test that concepts were created"""
        try:
            result = await self.memory.db.query("SELECT count() FROM concept")
            count = result[0]['result'][0] if result[0].get('result') else 0
            
            if count > 0:
                # Get sample concepts
                sample = await self.memory.db.query("""
                    SELECT name, kind FROM concept LIMIT 3
                """)
                concepts = sample[0].get('result', [])
                concept_names = [c.get('name', 'unknown') for c in concepts]
                
                return ValidationResult(
                    test_name="Concepts Created",
                    passed=True,
                    details=f"{count} concepts created (samples: {', '.join(concept_names[:3])})",
                    count=count
                )
            else:
                return ValidationResult(
                    test_name="Concepts Created",
                    passed=False,
                    details="No concepts found"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Concepts Created",
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_knowledge_relationships(self) -> ValidationResult:
        """Test user->knows->concept relationships"""
        try:
            # Count knowledge relationships
            result = await self.memory.db.query("SELECT count() FROM knows")
            count = result[0]['result'][0] if result[0].get('result') else 0
            
            if count > 0:
                # Test a sample relationship
                sample = await self.memory.db.query("""
                    SELECT 
                        in.name as user_name,
                        relationship,
                        out.name as concept_name,
                        strength,
                        fidelity
                    FROM knows 
                    LIMIT 1
                """)
                
                rel = sample[0]['result'][0] if sample[0].get('result') else {}
                
                return ValidationResult(
                    test_name="Knowledge Relationships",
                    passed=True,
                    details=f"{count} knowledge relations (sample: {rel.get('user_name')} {rel.get('relationship')} {rel.get('concept_name')})",
                    count=count
                )
            else:
                return ValidationResult(
                    test_name="Knowledge Relationships",
                    passed=False,
                    details="No knowledge relationships found"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Knowledge Relationships", 
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_session_message_relationships(self) -> ValidationResult:
        """Test session->contains->message relationships"""
        try:
            # Count contains relationships
            result = await self.memory.db.query("SELECT count() FROM contains")
            count = result[0]['result'][0] if result[0].get('result') else 0
            
            if count > 0:
                # Test traversal
                sample = await self.memory.db.query("""
                    SELECT count() FROM session->contains->message LIMIT 1
                """)
                
                traversal_count = sample[0]['result'][0] if sample[0].get('result') else 0
                
                return ValidationResult(
                    test_name="Session-Message Relationships",
                    passed=traversal_count > 0,
                    details=f"{count} contains relations, {traversal_count} traversable",
                    count=count
                )
            else:
                return ValidationResult(
                    test_name="Session-Message Relationships",
                    passed=False,
                    details="No session-message relationships found"
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Session-Message Relationships",
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_graph_traversal_queries(self) -> ValidationResult:
        """Test various graph traversal queries"""
        try:
            # Test 1: User's knowledge traversal
            knowledge_query = await self.memory.db.query("""
                SELECT count() FROM user->knows->concept LIMIT 1
            """)
            knowledge_count = knowledge_query[0]['result'][0] if knowledge_query[0].get('result') else 0
            
            # Test 2: Session conversation traversal  
            conversation_query = await self.memory.db.query("""
                SELECT count() FROM session->contains->message LIMIT 1
            """)
            conversation_count = conversation_query[0]['result'][0] if conversation_query[0].get('result') else 0
            
            # Test 3: Multi-hop traversal (user -> session -> messages)
            multihop_query = await self.memory.db.query("""
                SELECT count() FROM user<-session->contains->message LIMIT 1
            """)
            multihop_count = multihop_query[0]['result'][0] if multihop_query[0].get('result') else 0
            
            success = all(count > 0 for count in [knowledge_count, conversation_count, multihop_count])
            
            return ValidationResult(
                test_name="Graph Traversals",
                passed=success,
                details=f"Knowledge: {knowledge_count}, Conversations: {conversation_count}, Multi-hop: {multihop_count}"
            )
                
        except Exception as e:
            return ValidationResult(
                test_name="Graph Traversals",
                passed=False,
                details=f"Query failed: {e}"
            )
    
    async def _test_search_functionality(self) -> ValidationResult:
        """Test search capabilities"""
        try:
            # Test FTS search on messages
            fts_query = await self.memory.db.query("""
                SELECT count() FROM message WHERE content @@ 'test' LIMIT 1
            """)
            fts_works = True
            
        except Exception:
            # FTS might not be available
            fts_works = False
        
        # Test contains search
        try:
            contains_query = await self.memory.db.query("""
                SELECT count() FROM message 
                WHERE string::contains(string::lowercase(content), 'hello')
                LIMIT 1
            """)
            contains_count = contains_query[0]['result'][0] if contains_query[0].get('result') else 0
            contains_works = True
            
        except Exception as e:
            contains_works = False
            contains_count = 0
        
        return ValidationResult(
            test_name="Search Functionality",
            passed=contains_works,
            details=f"FTS: {'✓' if fts_works else '✗'}, Contains: {'✓' if contains_works else '✗'} ({contains_count} test results)"
        )
    
    async def _test_conversation_context(self) -> ValidationResult:
        """Test conversation context retrieval"""
        try:
            # Get a sample session
            session_query = await self.memory.db.query("SELECT id FROM session LIMIT 1")
            sessions = session_query[0].get('result', [])
            
            if not sessions:
                return ValidationResult(
                    test_name="Conversation Context",
                    passed=False,
                    details="No sessions available for context test"
                )
            
            session_id = sessions[0]['id']
            
            # Test context retrieval using graph memory
            context = await self.memory.get_conversation_context(session_id)
            
            has_session = bool(context.get('session'))
            has_messages = bool(context.get('messages'))
            has_knowledge = bool(context.get('user_knowledge'))
            
            return ValidationResult(
                test_name="Conversation Context",
                passed=has_session and has_messages,
                details=f"Session: {'✓' if has_session else '✗'}, Messages: {'✓' if has_messages else '✗'}, Knowledge: {'✓' if has_knowledge else '✗'}"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Conversation Context",
                passed=False,
                details=f"Context test failed: {e}"
            )
    
    async def _test_data_integrity(self) -> ValidationResult:
        """Test data integrity and consistency"""
        try:
            issues = []
            
            # Test 1: All sessions have valid user references
            orphaned_sessions = await self.memory.db.query("""
                SELECT count() FROM session WHERE user_id NOT IN (SELECT id FROM user)
            """)
            orphaned_count = orphaned_sessions[0]['result'][0] if orphaned_sessions[0].get('result') else 0
            
            if orphaned_count > 0:
                issues.append(f"{orphaned_count} orphaned sessions")
            
            # Test 2: All messages have valid session references
            orphaned_messages = await self.memory.db.query("""
                SELECT count() FROM message WHERE session_id NOT IN (SELECT id FROM session)
            """)
            orphaned_msg_count = orphaned_messages[0]['result'][0] if orphaned_messages[0].get('result') else 0
            
            if orphaned_msg_count > 0:
                issues.append(f"{orphaned_msg_count} orphaned messages")
            
            # Test 3: All knowledge relationships have valid endpoints
            orphaned_knowledge = await self.memory.db.query("""
                SELECT count() FROM knows 
                WHERE in NOT IN (SELECT id FROM user) 
                   OR out NOT IN (SELECT id FROM concept)
            """)
            orphaned_knowledge_count = orphaned_knowledge[0]['result'][0] if orphaned_knowledge[0].get('result') else 0
            
            if orphaned_knowledge_count > 0:
                issues.append(f"{orphaned_knowledge_count} orphaned knowledge relations")
            
            return ValidationResult(
                test_name="Data Integrity",
                passed=len(issues) == 0,
                details="All relationships valid" if not issues else f"Issues: {'; '.join(issues)}"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Data Integrity",
                passed=False,
                details=f"Integrity check failed: {e}"
            )

async def main():
    """Main validation execution"""
    logger.info("🚀 Starting migration validation...")
    
    validator = MigrationValidator()
    results = await validator.validate_all()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info("MIGRATION VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL" 
        logger.info(f"{status} {result.test_name}")
        logger.info(f"    {result.details}")
        if result.count > 0:
            logger.info(f"    Count: {result.count}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed! Migration successful!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} validation tests failed")
        return False

if __name__ == "__main__":
    import sys
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)