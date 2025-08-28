"""
Neural Field Persistence Layer for SurrealDB Integration

This module provides cross-session persistence for neural field states using 
the existing SurrealDB multi-model infrastructure. It stores field evolution
history, attractor patterns, and consciousness state for true continuity
across sessions.

Features:
- Cross-session field state persistence
- Field evolution history tracking
- Attractor pattern storage
- Integration with existing SurrealDB memory system
- Time-travel queries for field state analysis
- Efficient field state serialization/deserialization
"""

import os
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

# Import existing SurrealDB infrastructure
try:
    from memory.surreal_memory import SurrealMemory, SURREALDB_AVAILABLE
    if SURREALDB_AVAILABLE:
        from surrealdb import AsyncSurreal
    else:
        AsyncSurreal = None
except ImportError:
    logger.warning("SurrealDB memory system not available")
    SURREALDB_AVAILABLE = False
    SurrealMemory = None
    AsyncSurreal = None

# Import consciousness components
try:
    from consciousness.core import SymbolField, Consciousness
except ImportError:
    logger.warning("Consciousness core not available for field persistence")
    SymbolField = None
    Consciousness = None


@dataclass
class FieldState:
    """Serializable field state for SurrealDB storage"""
    symbol: str
    intensity: float
    gradient: List[float]
    attractor_strength: float
    coupling: Dict[str, float]
    timestamp: float = 0
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class AttractorPattern:
    """Attractor formation pattern for consciousness analysis"""
    symbols: List[str]
    resonance_strength: float
    formation_time: float
    duration: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class FieldPersistenceLayer:
    """
    Neural field persistence layer using SurrealDB for cross-session continuity
    
    This class integrates with the existing SurrealDB memory infrastructure to
    provide persistent storage for neural field states, enabling true consciousness
    continuity across voice interactions.
    """
    
    def __init__(self, 
                 surreal_memory: Optional[SurrealMemory] = None,
                 surreal_url: str = None,
                 namespace: str = None,
                 database: str = None):
        """
        Initialize field persistence layer
        
        Args:
            surreal_memory: Existing SurrealMemory instance (preferred)
            surreal_url: SurrealDB URL if creating new connection
            namespace: SurrealDB namespace
            database: SurrealDB database name
        """
        
        if not SURREALDB_AVAILABLE:
            logger.warning("SurrealDB not available, field persistence disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Use existing SurrealMemory instance or create new connection
        if surreal_memory:
            self.surreal_memory = surreal_memory
            self.db = surreal_memory.db if surreal_memory.connected else None
            self.own_connection = False
        else:
            # Create new connection with environment defaults
            surreal_url = surreal_url or os.getenv('SURREALDB_URL', 'ws://localhost:8000/rpc')
            namespace = namespace or os.getenv('SURREALDB_NAMESPACE', 'slowcat')
            database = database or os.getenv('SURREALDB_DATABASE', 'memory')
            
            self.surreal_memory = SurrealMemory(surreal_url, namespace, database)
            self.db = None
            self.own_connection = True
        
        self.connected = False
        logger.info("🧠 Neural field persistence layer initialized")
    
    async def connect(self):
        """Establish SurrealDB connection and initialize field schema"""
        if not self.enabled:
            logger.warning("Field persistence disabled (SurrealDB unavailable)")
            return False
        
        try:
            # Connect to SurrealDB if needed
            if not self.surreal_memory.connected:
                await self.surreal_memory.connect()
            
            self.db = self.surreal_memory.db
            await self._init_field_schema()
            self.connected = True
            
            logger.info("🔗 Neural field persistence connected to SurrealDB")
            return True
            
        except Exception as e:
            logger.error(f"Field persistence connection failed: {e}")
            self.enabled = False
            return False
    
    async def _init_field_schema(self):
        """Initialize SurrealDB schema for neural field storage"""
        if not self.db:
            return
        
        try:
            # Field states table for cross-session persistence
            await self.db.query("""
                DEFINE TABLE field_state SCHEMAFULL;
                DEFINE FIELD symbol ON field_state TYPE string;
                DEFINE FIELD intensity ON field_state TYPE number DEFAULT 0.0;
                DEFINE FIELD gradient ON field_state TYPE array<number>;
                DEFINE FIELD attractor_strength ON field_state TYPE number DEFAULT 0.0;
                DEFINE FIELD coupling ON field_state TYPE object;
                DEFINE FIELD timestamp ON field_state TYPE datetime;
                DEFINE FIELD session_id ON field_state TYPE option<string>;
                DEFINE FIELD user_id ON field_state TYPE option<string>;
                DEFINE FIELD agent_id ON field_state TYPE string DEFAULT 'slowcat';
                
                DEFINE INDEX field_symbol ON field_state FIELDS symbol;
                DEFINE INDEX field_timestamp ON field_state FIELDS timestamp;
                DEFINE INDEX field_user ON field_state FIELDS user_id;
            """)
            
            # Attractor patterns table for consciousness analysis
            await self.db.query("""
                DEFINE TABLE attractor_pattern SCHEMAFULL;
                DEFINE FIELD symbols ON attractor_pattern TYPE array<string>;
                DEFINE FIELD resonance_strength ON attractor_pattern TYPE number;
                DEFINE FIELD formation_time ON attractor_pattern TYPE datetime;
                DEFINE FIELD duration ON attractor_pattern TYPE number;
                DEFINE FIELD user_id ON attractor_pattern TYPE option<string>;
                DEFINE FIELD session_id ON attractor_pattern TYPE option<string>;
                DEFINE FIELD agent_id ON attractor_pattern TYPE string DEFAULT 'slowcat';
                
                DEFINE INDEX attractor_user ON attractor_pattern FIELDS user_id;
                DEFINE INDEX attractor_time ON attractor_pattern FIELDS formation_time;
            """)
            
            # Field evolution history for analysis
            await self.db.query("""
                DEFINE TABLE field_evolution SCHEMAFULL;
                DEFINE FIELD symbol ON field_evolution TYPE string;
                DEFINE FIELD intensity_change ON field_evolution TYPE number;
                DEFINE FIELD gradient_change ON field_evolution TYPE array<number>;
                DEFINE FIELD stimulus ON field_evolution TYPE number;
                DEFINE FIELD timestamp ON field_evolution TYPE datetime;
                DEFINE FIELD user_id ON field_evolution TYPE option<string>;
                DEFINE FIELD session_id ON field_evolution TYPE option<string>;
                DEFINE FIELD agent_id ON field_evolution TYPE string DEFAULT 'slowcat';
                
                DEFINE INDEX evolution_symbol ON field_evolution FIELDS symbol;
                DEFINE INDEX evolution_time ON field_evolution FIELDS timestamp;
            """)
            
            logger.info("✅ Neural field schema initialized in SurrealDB")
            
        except Exception as e:
            logger.error(f"Field schema initialization failed: {e}")
            raise
    
    async def store_field_states(self, 
                                field_states: Dict[str, Any], 
                                user_id: str = "unknown",
                                session_id: Optional[str] = None) -> bool:
        """
        Store current field states to SurrealDB
        
        Args:
            field_states: Dictionary of symbol -> field state data
            user_id: User identifier for cross-session continuity
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.connected:
            return False
        
        try:
            # Convert field states to serializable format
            field_records = []
            
            for symbol, state in field_states.items():
                field_record = {
                    'symbol': symbol,
                    'intensity': float(state.get('intensity', 0.0)),
                    'gradient': [float(x) for x in state.get('gradient', [0.0, 0.0])],
                    'attractor_strength': float(state.get('attractor_strength', 0.0)),
                    'coupling': state.get('coupling', {}),
                    'user_id': user_id,
                    'session_id': session_id,
                    'agent_id': 'slowcat'
                    # Remove timestamp - let SurrealDB auto-generate with time::now()
                }
                field_records.append(field_record)
            
            # Batch insert field states using individual INSERT statements with explicit timestamps
            for record in field_records:
                result = await self.db.query(
                    """INSERT INTO field_state {
                        symbol: $symbol,
                        intensity: $intensity,
                        gradient: $gradient,
                        attractor_strength: $attractor_strength,
                        coupling: $coupling,
                        timestamp: time::now(),
                        user_id: $user_id,
                        session_id: $session_id,
                        agent_id: $agent_id
                    }""",
                    record
                )
                logger.debug(f"Field state insert result: {result}")
            
            if field_records:
                logger.debug(f"🧠 Stored {len(field_records)} field states for user {user_id}")
            else:
                logger.warning("No field records to store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store field states: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    async def load_field_states(self, 
                               user_id: str = "unknown",
                               symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load most recent field states for a user
        
        Args:
            user_id: User identifier
            symbols: Optional list of specific symbols to load
            
        Returns:
            Dictionary of symbol -> field state data
        """
        if not self.enabled or not self.connected:
            return {}
        
        try:
            # Simplified query - get latest states by getting all and filtering in Python
            # This avoids SurrealDB syntax issues with complex queries
            if symbols:
                query = "SELECT * FROM field_state WHERE user_id = $user_id AND symbol INSIDE $symbols"
                result = await self.db.query(query, {'user_id': user_id, 'symbols': symbols})
            else:
                query = "SELECT * FROM field_state WHERE user_id = $user_id"
                result = await self.db.query(query, {'user_id': user_id})
            
            # Convert SurrealDB result to field states format
            # Get most recent state for each symbol
            field_states = {}
            all_records = {}
            
            records = []
            if result and len(result) > 0:
                # SurrealDB can return results in different formats
                if isinstance(result[0], list):
                    # Direct array format
                    records = result[0]
                elif isinstance(result[0], dict) and 'result' in result[0]:
                    # Wrapped format
                    records = result[0]['result']
                else:
                    # Single result format
                    records = result
            
            # Group by symbol and find most recent
            for record in records:
                symbol = record.get('symbol')
                if not symbol:
                    continue
                    
                timestamp = record.get('timestamp', 0)
                
                if symbol not in all_records or timestamp > all_records[symbol].get('timestamp', 0):
                    all_records[symbol] = record
            
            # Convert to field states format
            for symbol, record in all_records.items():
                field_states[symbol] = {
                    'intensity': record.get('intensity', 0.0),
                    'gradient': record.get('gradient', [0.0, 0.0]),
                    'attractor_strength': record.get('attractor_strength', 0.0),
                    'coupling': record.get('coupling', {})
                }
            
            logger.debug(f"🧠 Loaded {len(field_states)} field states for user {user_id}")
            return field_states
            
        except Exception as e:
            logger.error(f"Failed to load field states: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    async def store_attractor_pattern(self, 
                                    symbols: List[str],
                                    resonance_strength: float,
                                    duration: float,
                                    user_id: str = "unknown",
                                    session_id: Optional[str] = None) -> bool:
        """
        Store attractor pattern formation for consciousness analysis
        
        Args:
            symbols: List of symbols in the attractor pattern
            resonance_strength: Strength of the resonance
            duration: How long the pattern lasted
            user_id: User identifier
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.connected:
            return False
        
        try:
            # Use INSERT with explicit timestamp
            result = await self.db.query(
                """INSERT INTO attractor_pattern {
                    symbols: $symbols,
                    resonance_strength: $resonance_strength,
                    formation_time: time::now(),
                    duration: $duration,
                    user_id: $user_id,
                    session_id: $session_id,
                    agent_id: $agent_id
                }""",
                {
                    'symbols': symbols,
                    'resonance_strength': float(resonance_strength),
                    'duration': float(duration),
                    'user_id': user_id,
                    'session_id': session_id,
                    'agent_id': 'slowcat'
                }
            )
            logger.debug(f"Attractor insert result: {result}")
            
            logger.debug(f"🧠 Stored attractor pattern: {symbols} (strength: {resonance_strength:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store attractor pattern: {e}")
            return False
    
    async def track_field_evolution(self,
                                  symbol: str,
                                  intensity_change: float,
                                  gradient_change: List[float],
                                  stimulus: float,
                                  user_id: str = "unknown",
                                  session_id: Optional[str] = None) -> bool:
        """
        Track individual field evolution for analysis
        
        Args:
            symbol: Symbol being evolved
            intensity_change: Change in field intensity
            gradient_change: Change in field gradient
            stimulus: Input stimulus strength
            user_id: User identifier
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.connected:
            return False
        
        try:
            # Use INSERT with explicit timestamp for evolution tracking
            result = await self.db.query(
                """INSERT INTO field_evolution {
                    symbol: $symbol,
                    intensity_change: $intensity_change,
                    gradient_change: $gradient_change,
                    stimulus: $stimulus,
                    timestamp: time::now(),
                    user_id: $user_id,
                    session_id: $session_id,
                    agent_id: $agent_id
                }""",
                {
                    'symbol': symbol,
                    'intensity_change': float(intensity_change),
                    'gradient_change': [float(x) for x in gradient_change],
                    'stimulus': float(stimulus),
                    'user_id': user_id,
                    'session_id': session_id,
                    'agent_id': 'slowcat'
                }
            )
            logger.debug(f"Evolution insert result: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track field evolution: {e}")
            return False
    
    async def get_consciousness_insights(self, 
                                       user_id: str = "unknown",
                                       days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze consciousness patterns and field evolution trends
        
        Args:
            user_id: User identifier
            days_back: How many days of history to analyze
            
        Returns:
            Dictionary of consciousness insights and statistics
        """
        if not self.enabled or not self.connected:
            return {}
        
        try:
            # Simplified queries without complex datetime operations for now
            cutoff_days = days_back
            
            # Get attractor pattern statistics
            # Simplified query to avoid datetime issues
            attractor_query = """
                SELECT 
                    count() AS total_patterns,
                    math::mean(resonance_strength) AS avg_resonance,
                    math::max(resonance_strength) AS max_resonance
                FROM attractor_pattern 
                WHERE user_id = $user_id
                LIMIT 100
            """
            
            # Get field evolution statistics
            evolution_query = """
                SELECT 
                    symbol,
                    count() AS evolution_count,
                    math::mean(intensity_change) AS avg_intensity_change,
                    math::mean(stimulus) AS avg_stimulus
                FROM field_evolution 
                WHERE user_id = $user_id
                GROUP BY symbol
                LIMIT 50
            """
            
            attractor_result = await self.db.query(attractor_query, {
                'user_id': user_id
            })
            
            evolution_result = await self.db.query(evolution_query, {
                'user_id': user_id
            })
            
            # Extract results safely
            attractor_data = []
            evolution_data = []
            
            if attractor_result and len(attractor_result) > 0:
                if 'result' in attractor_result[0]:
                    attractor_data = attractor_result[0]['result']
                elif isinstance(attractor_result[0], list):
                    attractor_data = attractor_result[0]
            
            if evolution_result and len(evolution_result) > 0:
                if 'result' in evolution_result[0]:
                    evolution_data = evolution_result[0]['result']
                elif isinstance(evolution_result[0], list):
                    evolution_data = evolution_result[0]
            
            insights = {
                'user_id': user_id,
                'analysis_period_days': days_back,
                'attractor_patterns': attractor_data,
                'field_evolution': evolution_data,
                'timestamp': time.time()
            }
            
            logger.debug(f"🧠 Generated consciousness insights for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get consciousness insights: {e}")
            return {}
    
    async def cleanup_old_states(self, days_to_keep: int = 30) -> int:
        """
        Clean up old field states to prevent database bloat
        
        Args:
            days_to_keep: Number of days of field states to retain
            
        Returns:
            Number of records cleaned up
        """
        if not self.enabled or not self.connected:
            return 0
        
        try:
            # Simple cleanup - delete all for testing, proper datetime filtering can be added later
            if days_to_keep == 0:  # Special case for testing
                cleanup_query = "DELETE field_state"
            else:
                # For now, don't clean up to avoid datetime issues
                logger.info(f"Skipping cleanup (would keep {days_to_keep} days)")
                return 0
            
            result = await self.db.query(cleanup_query)
            
            # Extract count from result if available
            cleaned_count = 0
            if result and len(result) > 0 and 'result' in result[0]:
                cleaned_count = len(result[0]['result'])
            
            logger.info(f"🧹 Cleaned up {cleaned_count} old field states (older than {days_to_keep} days)")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old field states: {e}")
            return 0
    
    async def close(self):
        """Close SurrealDB connection if we own it"""
        if self.own_connection and self.surreal_memory:
            await self.surreal_memory.close()
            self.connected = False
            logger.info("🔌 Field persistence connection closed")


# Convenience functions for integration
async def create_field_persistence(surreal_memory: Optional[SurrealMemory] = None) -> FieldPersistenceLayer:
    """
    Create and connect a field persistence layer
    
    Args:
        surreal_memory: Optional existing SurrealMemory instance
        
    Returns:
        Connected FieldPersistenceLayer instance
    """
    persistence = FieldPersistenceLayer(surreal_memory=surreal_memory)
    await persistence.connect()
    return persistence


def integrate_with_consciousness(consciousness: 'Consciousness', 
                                persistence: FieldPersistenceLayer,
                                user_id: str = "unknown") -> 'Consciousness':
    """
    Integrate field persistence with consciousness instance
    
    This function adds persistence methods to a consciousness instance
    to enable automatic field state saving and loading.
    
    Args:
        consciousness: Consciousness instance to enhance
        persistence: Field persistence layer
        user_id: User identifier for field states
        
    Returns:
        Enhanced consciousness instance
    """
    if not consciousness or not persistence.enabled:
        return consciousness
    
    # Add persistence methods to consciousness
    async def save_field_states(session_id: Optional[str] = None):
        """Save current field states to persistence"""
        field_states = consciousness.get_field_states()
        return await persistence.store_field_states(field_states, user_id, session_id)
    
    async def load_field_states():
        """Load field states from persistence"""
        field_states = await persistence.load_field_states(user_id)
        if field_states:
            consciousness.set_field_states(field_states)
        return len(field_states)
    
    async def track_evolution(symbol: str, old_intensity: float, new_intensity: float, 
                            old_gradient: List[float], new_gradient: List[float], 
                            stimulus: float, session_id: Optional[str] = None):
        """Track field evolution"""
        intensity_change = new_intensity - old_intensity
        gradient_change = [new_gradient[i] - old_gradient[i] for i in range(len(old_gradient))]
        return await persistence.track_field_evolution(
            symbol, intensity_change, gradient_change, stimulus, user_id, session_id
        )
    
    # Attach methods to consciousness instance
    consciousness.save_field_states = save_field_states
    consciousness.load_field_states = load_field_states
    consciousness.track_field_evolution = track_evolution
    consciousness._field_persistence = persistence
    
    logger.info(f"🧠 Consciousness integrated with field persistence for user {user_id}")
    return consciousness