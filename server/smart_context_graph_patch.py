
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

