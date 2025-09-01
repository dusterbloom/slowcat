#!/usr/bin/env python3
"""
Graph Integration Layer - Bridges SmartContextManager with SurrealDB Graph Functions

This module provides the integration layer that actually USES the graph functions
we created in create_complete_graph_system.py. Instead of traditional SQL queries,
this uses proper SurrealDB graph traversal.

Key Integration Points:
- Session analysis via graph traversal
- Entity knowledge retrieval via graph relations
- Engram detection and reinforcement
- Cross-session pattern recognition
- Memory consolidation using graph structure
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from memory.surreal_connection import SurrealConnectionManager


class GraphMemoryIntegration:
    """
    Integration layer that uses the SurrealDB graph functions in actual runtime.
    
    This replaces traditional queries with proper graph traversal functions.
    """
    
    def __init__(self):
        self.conn: Optional[SurrealConnectionManager] = None
        
    async def ensure_connected(self):
        """Ensure SurrealDB connection is established"""
        if not self.conn:
            self.conn = SurrealConnectionManager()
            await self.conn.ensure_connected()
        return self.conn
    
    async def get_session_context_graph(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session context using graph traversal.
        Uses: fn::analyze_session_graph() from complete graph system.
        
        Returns:
            {
                'messages': [...],
                'knowledge': [...], 
                'entities': [...],
                'engrams': [...],
                'stats': {...}
            }
        """
        try:
            await self.ensure_connected()
            
            # Use the graph function we created
            result = await self.conn.db.query("""
                RETURN fn::analyze_session_graph($session_id);
            """, {"session_id": session_id})
            
            if result and len(result) > 0:
                session_data = result[0] if isinstance(result, list) else result
                
                # Handle error case
                if isinstance(session_data, dict) and session_data.get('error'):
                    logger.warning(f"Session {session_id} not found in graph")
                    return {
                        'messages': [],
                        'knowledge': [],
                        'entities': [],
                        'engrams': [],
                        'stats': {'message_count': 0, 'knowledge_count': 0, 'entity_count': 0, 'engram_count': 0}
                    }
                
                return session_data
            
        except Exception as e:
            logger.error(f"Error getting session context via graph: {e}")
            
        # Fallback empty result
        return {
            'messages': [],
            'knowledge': [],
            'entities': [],
            'engrams': [],
            'stats': {'message_count': 0, 'knowledge_count': 0, 'entity_count': 0, 'engram_count': 0}
        }
    
    async def get_entity_knowledge_graph(self, entity_name: str) -> Dict[str, Any]:
        """
        Get all knowledge about an entity using graph traversal.
        Uses: fn::analyze_entity_graph() from complete graph system.
        
        Returns:
            {
                'entity': {...},
                'knowledge': [...],
                'sessions': [...],
                'engrams': [...],
                'stats': {...}
            }
        """
        try:
            await self.ensure_connected()
            
            # Use the graph function we created
            result = await self.conn.db.query("""
                RETURN fn::analyze_entity_graph($entity_name);
            """, {"entity_name": entity_name})
            
            if result and len(result) > 0:
                entity_data = result[0] if isinstance(result, list) else result
                
                # Handle error case
                if isinstance(entity_data, dict) and entity_data.get('error'):
                    logger.debug(f"Entity {entity_name} not found in graph")
                    return {
                        'entity': None,
                        'knowledge': [],
                        'sessions': [],
                        'engrams': [],
                        'stats': {'knowledge_count': 0, 'session_count': 0, 'engram_count': 0}
                    }
                
                return entity_data
            
        except Exception as e:
            logger.error(f"Error getting entity knowledge via graph: {e}")
            
        # Fallback empty result
        return {
            'entity': None,
            'knowledge': [],
            'sessions': [],
            'engrams': [],
            'stats': {'knowledge_count': 0, 'session_count': 0, 'engram_count': 0}
        }
    
    async def detect_session_engrams(self, session_id: str, 
                                   min_coherence: float = 0.6, 
                                   min_facts: int = 2) -> Dict[str, Any]:
        """
        Detect engrams for a session using graph-based pattern recognition.
        Uses: fn::detect_engrams_graph() from proper engram graph system.
        
        Returns:
            {
                'success': bool,
                'action': 'created'|'reinforced',
                'engram_id': str,
                'narrative': str,
                'coherence': float,
                'pattern_hash': str
            }
        """
        try:
            await self.ensure_connected()
            
            # Use the graph-based engram detection function
            result = await self.conn.db.query("""
                RETURN fn::detect_engrams_graph($session_id, $min_coherence, $min_facts);
            """, {
                "session_id": session_id,
                "min_coherence": min_coherence,
                "min_facts": min_facts
            })
            
            if result and len(result) > 0:
                engram_result = result[0] if isinstance(result, list) else result
                
                # Handle case where result might be a string or unexpected format
                if isinstance(engram_result, str):
                    return {
                        'success': False,
                        'reason': f'Function returned string: {engram_result[:100]}',
                        'action': 'failed'
                    }
                
                if isinstance(engram_result, dict) and engram_result.get('success'):
                    logger.info(f"✅ Engram {engram_result.get('action', 'processed')} for session {session_id}")
                    logger.debug(f"   Narrative: {engram_result.get('narrative', 'N/A')[:100]}...")
                    logger.debug(f"   Coherence: {engram_result.get('coherence', 0):.2f}")
                
                return engram_result if isinstance(engram_result, dict) else {
                    'success': False,
                    'reason': f'Unexpected result format: {type(engram_result)}',
                    'action': 'failed'
                }
            
        except Exception as e:
            logger.error(f"Error detecting engrams via graph: {e}")
            
        # Fallback failure result
        return {
            'success': False,
            'reason': 'Graph detection failed',
            'action': 'failed'
        }
    
    async def get_session_messages_graph(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get messages for a session using graph traversal.
        Uses: fn::get_session_messages_graph() from complete graph system.
        """
        try:
            await self.ensure_connected()
            
            result = await self.conn.db.query("""
                RETURN fn::get_session_messages_graph($session_id);
            """, {"session_id": session_id})
            
            if result and len(result) > 0:
                messages_data = result[0] if isinstance(result, list) else result
                # Handle the nested structure from graph traversal
                if isinstance(messages_data, list) and len(messages_data) > 0:
                    # Extract messages from graph relation structure
                    messages = []
                    for item in messages_data:
                        if isinstance(item, dict) and 'messages' in item:
                            messages.extend(item['messages'])
                    return messages
                return messages_data if isinstance(messages_data, list) else []
            
        except Exception as e:
            logger.error(f"Error getting session messages via graph: {e}")
            
        return []
    
    async def get_entity_sessions_graph(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get sessions where entity was mentioned using graph traversal.
        Uses: fn::get_entity_sessions_graph() from complete graph system.
        """
        try:
            await self.ensure_connected()
            
            result = await self.conn.db.query("""
                RETURN fn::get_entity_sessions_graph($entity_name);
            """, {"entity_name": entity_name})
            
            if result and len(result) > 0:
                sessions_data = result[0] if isinstance(result, list) else result
                # Handle the nested structure from graph traversal
                if isinstance(sessions_data, list) and len(sessions_data) > 0:
                    # Extract sessions from graph relation structure  
                    sessions = []
                    for item in sessions_data:
                        if isinstance(item, dict) and 'sessions' in item:
                            sessions.extend(item['sessions'])
                    return sessions
                return sessions_data if isinstance(sessions_data, list) else []
            
        except Exception as e:
            logger.error(f"Error getting entity sessions via graph: {e}")
            
        return []
    
    async def get_cross_session_patterns(self, min_sessions: int = 2) -> List[Dict[str, Any]]:
        """
        Find patterns that appear across multiple sessions using engram graph.
        
        Returns patterns discovered by the graph-based engram system.
        """
        try:
            await self.ensure_connected()
            
            # Get all engrams with their session connections
            result = await self.conn.db.query("""
                SELECT 
                    id,
                    narrative_summary,
                    dominant_symbols,
                    pattern_hash,
                    coherence,
                    stability,
                    ->engram_appears_in->sessions[*] AS connected_sessions
                FROM engrams 
                WHERE count(->engram_appears_in->sessions) >= $min_sessions
                ORDER BY coherence DESC, stability DESC
                LIMIT 20;
            """, {"min_sessions": min_sessions})
            
            if result:
                patterns = []
                for engram in result:
                    if engram.get('connected_sessions'):
                        session_count = len(engram['connected_sessions'])
                        patterns.append({
                            'engram_id': engram['id'],
                            'narrative': engram.get('narrative_summary', ''),
                            'symbols': engram.get('dominant_symbols', []),
                            'pattern_hash': engram.get('pattern_hash', ''),
                            'coherence': engram.get('coherence', 0),
                            'stability': engram.get('stability', 0),
                            'session_count': session_count,
                            'sessions': engram['connected_sessions']
                        })
                
                return patterns
            
        except Exception as e:
            logger.error(f"Error getting cross-session patterns: {e}")
            
        return []
    
    async def get_conversation_summary_graph(self, session_id: str, 
                                           max_tokens: int = 500) -> str:
        """
        Generate conversation summary using graph-traversed data.
        
        This uses the comprehensive session analysis to create intelligent summaries.
        """
        try:
            # Get full session context via graph
            session_data = await self.get_session_context_graph(session_id)
            
            if not session_data or session_data['stats']['message_count'] == 0:
                return ""
            
            # Extract key information for summary
            messages = session_data.get('messages', [])
            knowledge = session_data.get('knowledge', [])
            entities = session_data.get('entities', [])
            engrams = session_data.get('engrams', [])
            
            # Build intelligent summary
            summary_parts = []
            
            # Session overview
            stats = session_data.get('stats', {})
            summary_parts.append(f"Session with {stats.get('message_count', 0)} messages")
            
            # Key entities mentioned
            if entities:
                entity_names = [e.get('canonical_name', 'unknown') for e in entities[:3]]
                summary_parts.append(f"discussing {', '.join(entity_names)}")
            
            # Key facts discovered
            if knowledge:
                fact_count = len(knowledge)
                summary_parts.append(f"{fact_count} facts extracted")
            
            # Engrams (patterns) identified
            if engrams:
                engram_count = len(engrams)
                summary_parts.append(f"{engram_count} patterns recognized")
            
            return "; ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating conversation summary via graph: {e}")
            return ""


# Global instance for use by SmartContextManager
_graph_integration = None

def get_graph_integration() -> GraphMemoryIntegration:
    """Get singleton graph integration instance"""
    global _graph_integration
    if _graph_integration is None:
        _graph_integration = GraphMemoryIntegration()
    return _graph_integration


async def integrate_graph_memory_with_context(session_id: str, 
                                            max_context_tokens: int = 1500) -> Dict[str, Any]:
    """
    Main integration function for SmartContextManager.
    
    This function provides graph-based memory retrieval that SmartContextManager
    can use instead of traditional queries.
    
    Returns:
        {
            'messages': [...],          # Recent messages via graph traversal
            'facts': [...],            # Knowledge facts via graph relations  
            'entities': [...],         # Entities mentioned via graph
            'patterns': [...],         # Cross-session patterns via engrams
            'summary': str,            # Graph-based conversation summary
            'token_estimate': int      # Rough token count for context planning
        }
    """
    try:
        graph = get_graph_integration()
        
        # Get comprehensive session data via graph
        session_data = await graph.get_session_context_graph(session_id)
        
        # Get cross-session patterns
        patterns = await graph.get_cross_session_patterns(min_sessions=2)
        
        # Generate summary
        summary = await graph.get_conversation_summary_graph(session_id)
        
        # Estimate token usage (rough heuristic)
        token_estimate = (
            len(str(session_data.get('messages', []))) // 4 +  # ~4 chars per token
            len(str(session_data.get('knowledge', []))) // 4 +
            len(str(patterns)) // 4 +
            len(summary) // 4
        )
        
        return {
            'messages': session_data.get('messages', []),
            'facts': session_data.get('knowledge', []),
            'entities': session_data.get('entities', []),
            'engrams': session_data.get('engrams', []),
            'patterns': patterns,
            'summary': summary,
            'stats': session_data.get('stats', {}),
            'token_estimate': min(token_estimate, max_context_tokens)
        }
        
    except Exception as e:
        logger.error(f"Error integrating graph memory: {e}")
        return {
            'messages': [],
            'facts': [],
            'entities': [],
            'engrams': [],
            'patterns': [],
            'summary': '',
            'stats': {},
            'token_estimate': 0
        }


# Export main functions for use by SmartContextManager
__all__ = [
    'GraphMemoryIntegration',
    'get_graph_integration',
    'integrate_graph_memory_with_context'
]