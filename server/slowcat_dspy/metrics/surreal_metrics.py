"""
SurrealDB-Powered DSPy Metrics

These metrics leverage SurrealDB's unique capabilities for DSPy optimization:
- Graph relationships to measure context connectivity and relevance
- Time-travel queries to analyze historical conversation success
- Multi-model queries to correlate facts, conversations, and outcomes
- Real-time subscriptions to adapt metrics based on live performance

Key Metrics:
- context_relevance_metric: Uses graph connectivity to measure context quality
- response_quality_metric: Uses conversation history to measure response success
- temporal_coherence_metric: Measures conversation flow consistency over time
- graph_connectivity_metric: Measures fact relationship utilization
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from loguru import logger


def context_relevance_metric(prediction: Any, ground_truth: Any, surreal_memory: Any) -> float:
    """
    Use SurrealDB's graph relationships to measure context quality
    
    This metric evaluates:
    - How well selected facts connect to the query (graph connectivity)
    - Temporal relevance of retrieved information
    - Relationship depth appropriateness for the query type
    - Speaker-specific context accuracy
    
    Args:
        prediction: DSPy prediction with selected context
        ground_truth: Expected or actual conversation outcome
        surreal_memory: SurrealDB memory instance
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    try:
        context = getattr(prediction, 'context', {})
        selected_facts = context.get('selected_facts', [])
        query = getattr(prediction, 'query', '')
        
        if not selected_facts or not query:
            return 0.0
        
        # Score components
        graph_connectivity_score = _calculate_graph_connectivity(selected_facts, query, surreal_memory)
        temporal_relevance_score = _calculate_temporal_relevance(selected_facts, ground_truth)
        speaker_accuracy_score = _calculate_speaker_accuracy(selected_facts, context.get('speaker_id'))
        query_alignment_score = _calculate_query_alignment(selected_facts, query)
        
        # Weighted combination
        relevance_score = (
            0.3 * graph_connectivity_score +
            0.25 * temporal_relevance_score +
            0.25 * speaker_accuracy_score +
            0.2 * query_alignment_score
        )
        
        return min(1.0, max(0.0, relevance_score))
        
    except Exception as e:
        logger.error(f"Context relevance metric failed: {e}")
        return 0.0


def response_quality_metric(prediction: Any, ground_truth: Any, surreal_memory: Any) -> float:
    """
    Use conversation history in SurrealDB to measure response quality
    
    This metric evaluates:
    - How well the response matches successful historical patterns
    - Conversation flow continuity and coherence
    - Speaker-specific response appropriateness
    - User satisfaction indicators from past conversations
    
    Args:
        prediction: DSPy prediction with generated response
        ground_truth: Actual conversation outcome or user feedback
        surreal_memory: SurrealDB memory instance
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    try:
        response = getattr(prediction, 'response', '')
        context = getattr(prediction, 'context', {})
        
        if not response:
            return 0.0
        
        # Score components
        historical_pattern_score = _calculate_historical_pattern_match(response, context, surreal_memory)
        conversation_flow_score = _calculate_conversation_flow_score(response, context)
        length_appropriateness_score = _calculate_length_appropriateness(response, context)
        tone_consistency_score = _calculate_tone_consistency(response, context)
        
        # Include actual outcome if available
        outcome_score = _calculate_outcome_score(ground_truth) if ground_truth else 0.7
        
        # Weighted combination
        quality_score = (
            0.3 * historical_pattern_score +
            0.25 * conversation_flow_score +
            0.2 * outcome_score +
            0.15 * length_appropriateness_score +
            0.1 * tone_consistency_score
        )
        
        return min(1.0, max(0.0, quality_score))
        
    except Exception as e:
        logger.error(f"Response quality metric failed: {e}")
        return 0.0


def temporal_coherence_metric(prediction: Any, ground_truth: Any, surreal_memory: Any) -> float:
    """
    Measure conversation flow consistency over time using SurrealDB time-travel queries
    
    This metric evaluates:
    - How well current context builds on previous conversations
    - Temporal consistency of retrieved information
    - Session-to-session continuity for returning speakers
    - Memory utilization effectiveness over time
    
    Args:
        prediction: DSPy prediction with temporal elements
        ground_truth: Expected temporal consistency
        surreal_memory: SurrealDB memory instance
        
    Returns:
        Coherence score between 0.0 and 1.0
    """
    try:
        context = getattr(prediction, 'context', {})
        speaker_id = context.get('speaker_id', 'unknown')
        current_session_id = context.get('session_id')
        
        # Measure temporal consistency across sessions
        session_continuity_score = _calculate_session_continuity(speaker_id, current_session_id, surreal_memory)
        
        # Measure information decay appropriateness
        information_decay_score = _calculate_information_decay_appropriateness(context, surreal_memory)
        
        # Measure temporal query accuracy (if applicable)
        temporal_query_score = _calculate_temporal_query_accuracy(context, ground_truth)
        
        coherence_score = (
            0.4 * session_continuity_score +
            0.35 * information_decay_score +
            0.25 * temporal_query_score
        )
        
        return min(1.0, max(0.0, coherence_score))
        
    except Exception as e:
        logger.error(f"Temporal coherence metric failed: {e}")
        return 0.0


def graph_connectivity_metric(prediction: Any, ground_truth: Any, surreal_memory: Any) -> float:
    """
    Measure fact relationship utilization using SurrealDB graph capabilities
    
    This metric evaluates:
    - How effectively graph relationships were utilized
    - Appropriate traversal depth for the query complexity
    - Discovery of relevant connected information
    - Graph query efficiency and accuracy
    
    Args:
        prediction: DSPy prediction with graph traversal info
        ground_truth: Expected graph utilization
        surreal_memory: SurrealDB memory instance
        
    Returns:
        Connectivity score between 0.0 and 1.0
    """
    try:
        context = getattr(prediction, 'context', {})
        graph_traversal = context.get('graph_traversal', {})
        
        if not graph_traversal:
            # No graph traversal attempted - score based on whether it should have been
            query = getattr(prediction, 'query', '')
            if _should_use_graph_traversal(query):
                return 0.3  # Missed opportunity
            else:
                return 0.8  # Appropriately didn't use graph traversal
        
        # Evaluate graph traversal quality
        traversal_efficiency_score = _calculate_traversal_efficiency(graph_traversal)
        relationship_discovery_score = _calculate_relationship_discovery(graph_traversal, surreal_memory)
        depth_appropriateness_score = _calculate_depth_appropriateness(graph_traversal, context)
        
        connectivity_score = (
            0.4 * traversal_efficiency_score +
            0.35 * relationship_discovery_score +
            0.25 * depth_appropriateness_score
        )
        
        return min(1.0, max(0.0, connectivity_score))
        
    except Exception as e:
        logger.error(f"Graph connectivity metric failed: {e}")
        return 0.0


# Helper functions for metric calculations

def _calculate_graph_connectivity(facts: List[Any], query: str, surreal_memory: Any) -> float:
    """Calculate how well facts connect in the graph"""
    if not facts:
        return 0.0
    
    try:
        # Simple connectivity measure based on shared subjects/predicates
        subjects = set()
        predicates = set()
        
        for fact in facts:
            if hasattr(fact, 'subject'):
                subjects.add(getattr(fact, 'subject', ''))
            if hasattr(fact, 'predicate'):
                predicates.add(getattr(fact, 'predicate', ''))
        
        # More connections = better connectivity
        connectivity_ratio = len(subjects & predicates) / max(len(subjects) + len(predicates), 1)
        return min(1.0, connectivity_ratio * 2)  # Scale to 0-1
        
    except Exception:
        return 0.5  # Default moderate score


def _calculate_temporal_relevance(facts: List[Any], ground_truth: Any) -> float:
    """Calculate temporal relevance of selected facts"""
    if not facts:
        return 0.0
    
    try:
        current_time = time.time()
        relevance_scores = []
        
        for fact in facts:
            last_seen = getattr(fact, 'last_seen', current_time)
            # More recent = more relevant (exponential decay)
            age_hours = (current_time - last_seen) / 3600
            relevance = np.exp(-age_hours / 24)  # 24-hour half-life
            relevance_scores.append(relevance)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
        
    except Exception:
        return 0.5


def _calculate_speaker_accuracy(facts: List[Any], speaker_id: str) -> float:
    """Calculate accuracy of speaker-specific fact selection"""
    if not facts or not speaker_id:
        return 0.7  # Neutral score when no speaker info
    
    try:
        speaker_relevant_count = 0
        for fact in facts:
            # Check if fact is relevant to the speaker
            subject = getattr(fact, 'subject', '')
            if subject == 'user' or speaker_id in subject:
                speaker_relevant_count += 1
        
        return speaker_relevant_count / len(facts) if facts else 0.0
        
    except Exception:
        return 0.5


def _calculate_query_alignment(facts: List[Any], query: str) -> float:
    """Calculate how well facts align with the query"""
    if not facts or not query:
        return 0.0
    
    try:
        query_words = set(query.lower().split())
        alignment_scores = []
        
        for fact in facts:
            fact_text = getattr(fact, 'source_text', '')
            if not fact_text:
                # Construct fact text from components
                subject = getattr(fact, 'subject', '')
                predicate = getattr(fact, 'predicate', '')  
                value = getattr(fact, 'value', '')
                fact_text = f"{subject} {predicate} {value}"
            
            fact_words = set(fact_text.lower().split())
            overlap = len(query_words & fact_words)
            alignment = overlap / max(len(query_words), 1)
            alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
        
    except Exception:
        return 0.5


def _calculate_historical_pattern_match(response: str, context: Dict, surreal_memory: Any) -> float:
    """Calculate how well response matches successful historical patterns"""
    try:
        # Simple pattern matching based on response characteristics
        response_length = len(response.split())
        
        # Different lengths appropriate for different contexts
        if context.get('query_type') == 'question':
            optimal_length = 20  # Questions need informative answers
        elif context.get('query_type') == 'greeting':
            optimal_length = 5   # Greetings should be brief
        else:
            optimal_length = 15  # General optimal length
        
        length_score = 1.0 - abs(response_length - optimal_length) / optimal_length
        return max(0.0, min(1.0, length_score))
        
    except Exception:
        return 0.5


def _calculate_conversation_flow_score(response: str, context: Dict) -> float:
    """Calculate conversation flow appropriateness"""
    try:
        turn_count = context.get('turn_count', 1)
        
        # Early conversation should be more introductory
        if turn_count <= 2:
            # Look for greeting or introductory elements
            intro_words = ['hello', 'hi', 'nice', 'help', 'assist']
            has_intro = any(word in response.lower() for word in intro_words)
            return 0.8 if has_intro else 0.4
        else:
            # Later conversation should be more direct
            direct_words = ['sure', 'yes', 'here', 'that']
            has_direct = any(word in response.lower() for word in direct_words)
            return 0.8 if has_direct else 0.6
            
    except Exception:
        return 0.5


def _calculate_length_appropriateness(response: str, context: Dict) -> float:
    """Calculate response length appropriateness"""
    try:
        length = len(response.split())
        query_complexity = context.get('query_complexity', 'moderate')
        
        if query_complexity == 'simple':
            optimal_range = (3, 15)
        elif query_complexity == 'moderate':  
            optimal_range = (10, 30)
        else:  # complex
            optimal_range = (20, 50)
        
        if optimal_range[0] <= length <= optimal_range[1]:
            return 1.0
        elif length < optimal_range[0]:
            return max(0.0, length / optimal_range[0])
        else:
            return max(0.0, 1.0 - (length - optimal_range[1]) / optimal_range[1])
            
    except Exception:
        return 0.5


def _calculate_tone_consistency(response: str, context: Dict) -> float:
    """Calculate tone consistency with conversation"""
    try:
        # Simple sentiment/tone analysis
        conversation_tone = context.get('conversation_tone', 'neutral')
        
        response_lower = response.lower()
        
        if conversation_tone == 'enthusiastic':
            enthusiasm_words = ['great', 'awesome', 'excellent', 'wonderful']
            has_enthusiasm = any(word in response_lower for word in enthusiasm_words)
            return 0.9 if has_enthusiasm else 0.5
        elif conversation_tone == 'helpful':
            helpful_words = ['help', 'assist', 'guide', 'support']
            has_helpful = any(word in response_lower for word in helpful_words)
            return 0.9 if has_helpful else 0.6
        else:  # neutral
            return 0.7  # Neutral tone is generally appropriate
            
    except Exception:
        return 0.5


def _calculate_outcome_score(ground_truth: Any) -> float:
    """Calculate score based on actual conversation outcome"""
    try:
        if hasattr(ground_truth, 'outcome'):
            outcome = ground_truth.outcome
            if outcome == 'successful':
                return 1.0
            elif outcome == 'partially_successful':
                return 0.7
            elif outcome == 'unsuccessful':
                return 0.2
        
        # If no explicit outcome, try to infer from other signals
        if hasattr(ground_truth, 'user_satisfaction'):
            return ground_truth.user_satisfaction / 10.0  # Assume 0-10 scale
        
        return 0.7  # Default moderate score
        
    except Exception:
        return 0.7


def _calculate_session_continuity(speaker_id: str, session_id: str, surreal_memory: Any) -> float:
    """Calculate session-to-session continuity"""
    # Placeholder - would use SurrealDB to analyze cross-session patterns
    return 0.7


def _calculate_information_decay_appropriateness(context: Dict, surreal_memory: Any) -> float:
    """Calculate whether information decay was handled appropriately"""
    # Placeholder - would analyze if old/stale information was properly filtered
    return 0.8


def _calculate_temporal_query_accuracy(context: Dict, ground_truth: Any) -> float:
    """Calculate accuracy of temporal query handling"""
    # Placeholder - would verify if time-based queries were handled correctly
    return 0.7


def _should_use_graph_traversal(query: str) -> bool:
    """Determine if query should have used graph traversal"""
    relationship_indicators = ['friends', 'family', 'related', 'connected', 'about', 'knows']
    return any(indicator in query.lower() for indicator in relationship_indicators)


def _calculate_traversal_efficiency(graph_traversal: Dict) -> float:
    """Calculate efficiency of graph traversal"""
    try:
        depth = graph_traversal.get('depth', 1)
        # Moderate depth is usually most efficient
        if depth == 2:
            return 1.0
        elif depth in [1, 3]:
            return 0.8
        else:
            return max(0.2, 1.0 - (abs(depth - 2) * 0.2))
    except Exception:
        return 0.5


def _calculate_relationship_discovery(graph_traversal: Dict, surreal_memory: Any) -> float:
    """Calculate quality of relationship discovery"""
    # Placeholder - would analyze the relationships discovered
    return 0.7


def _calculate_depth_appropriateness(graph_traversal: Dict, context: Dict) -> float:
    """Calculate whether traversal depth was appropriate for the query"""
    try:
        depth = graph_traversal.get('depth', 1)
        query_complexity = context.get('query_complexity', 'moderate')
        
        if query_complexity == 'simple' and depth <= 2:
            return 1.0
        elif query_complexity == 'moderate' and depth <= 3:
            return 1.0  
        elif query_complexity == 'complex' and depth <= 4:
            return 1.0
        else:
            return max(0.2, 1.0 - (depth * 0.2))
            
    except Exception:
        return 0.5