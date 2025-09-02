"""
M3-SurrealDB Integration - AudioGraph persistence with M3 schema patterns

Provides SurrealDB schema and operations for AudioGraph persistence:
- M3-equivalent node and edge storage
- Equivalence relationship tracking  
- Efficient querying with vector similarity
- Batch operations for performance

Based on M3-Agent's graph persistence with SurrealDB optimizations.
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# SurrealDB connection
try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None
    logger.warning("SurrealDB not available - persistence will be disabled")

# AudioGraph integration
from .audio_graph import AudioGraph


class M3SurrealIntegration:
    """
    M3-inspired SurrealDB integration for AudioGraph persistence.
    
    Provides:
    - M3-equivalent schema for nodes, edges, and equivalences
    - Batch operations for performance
    - Vector similarity search capabilities
    - Transaction support for consistency
    """
    
    def __init__(self, connection_config: Optional[Dict] = None):
        """
        Initialize SurrealDB integration.
        
        Args:
            connection_config: SurrealDB connection configuration
        """
        self.connection_config = connection_config or {
            'url': 'ws://localhost:8000/rpc',
            'namespace': 'slowcat',
            'database': 'm3_memory'
        }
        
        self.db = None
        self.connected = False
        
        # M3's schema version for migrations
        self.schema_version = "1.0.0"
        
        logger.info(f"🗄️  M3SurrealIntegration initialized")
    
    async def connect(self) -> bool:
        """Connect to SurrealDB and initialize schema."""
        if not SURREALDB_AVAILABLE:
            logger.error("SurrealDB not available - cannot connect")
            return False
        
        try:
            self.db = AsyncSurreal(self.connection_config['url'])
            
            # Sign in and select namespace/database
            await self.db.signin({
                'user': 'root',
                'pass': 'root'
            })
            
            await self.db.use(
                self.connection_config['namespace'],
                self.connection_config['database']
            )
            
            # Initialize M3 schema
            await self._initialize_m3_schema()
            
            self.connected = True
            logger.info(f"✅ Connected to SurrealDB: {self.connection_config}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SurrealDB: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from SurrealDB."""
        if self.db and self.connected:
            try:
                await self.db.close()
                self.connected = False
                logger.info("🔌 Disconnected from SurrealDB")
            except Exception as e:
                logger.error(f"Error disconnecting from SurrealDB: {e}")
    
    async def _initialize_m3_schema(self):
        """Initialize M3-equivalent schema in SurrealDB."""
        
        # M3's node table with full complexity
        node_schema = """
        -- M3-equivalent node table for AudioGraph
        DEFINE TABLE node SCHEMAFULL;
        DEFINE FIELD node_id ON node TYPE int;
        DEFINE FIELD node_type ON node TYPE string ASSERT $value IN ['voice', 'episodic', 'semantic'];
        DEFINE FIELD embeddings ON node TYPE array<array<float>>;  -- Multiple embeddings per node
        DEFINE FIELD metadata ON node TYPE object;
        DEFINE FIELD metadata.contents ON node TYPE array<string>;
        DEFINE FIELD metadata.timestamp ON node TYPE option<int>;  -- clip_id
        DEFINE FIELD metadata.speaker_id ON node TYPE option<string>;
        DEFINE FIELD metadata.confidence ON node TYPE float DEFAULT 1.0;
        DEFINE FIELD metadata.first_seen ON node TYPE float;
        DEFINE FIELD metadata.last_updated ON node TYPE float;
        DEFINE FIELD metadata.session_id ON node TYPE option<int>;
        
        -- M3's indexes for performance
        DEFINE INDEX node_type_idx ON node FIELDS node_type;
        DEFINE INDEX timestamp_idx ON node FIELDS metadata.timestamp;
        DEFINE INDEX speaker_idx ON node FIELDS metadata.speaker_id;
        DEFINE INDEX node_id_idx ON node FIELDS node_id UNIQUE;
        """
        
        # M3's edge table with bidirectional weights
        edge_schema = """
        -- M3-equivalent edge table for AudioGraph relationships
        DEFINE TABLE edge SCHEMAFULL;
        DEFINE FIELD source ON edge TYPE record<node>;
        DEFINE FIELD target ON edge TYPE record<node>;
        DEFINE FIELD weight ON edge TYPE float DEFAULT 1.0;
        DEFINE FIELD created_at ON edge TYPE float DEFAULT time::unix();
        
        -- M3's edge indexes
        DEFINE INDEX edge_source_idx ON edge FIELDS source;
        DEFINE INDEX edge_target_idx ON edge FIELDS target;
        DEFINE INDEX edge_weight_idx ON edge FIELDS weight;
        """
        
        # M3's equivalence tracking table
        equivalence_schema = """
        -- M3-equivalent equivalence table for identity resolution
        DEFINE TABLE equivalence SCHEMAFULL;
        DEFINE FIELD canonical_id ON equivalence TYPE string;
        DEFINE FIELD node_ids ON equivalence TYPE array<int>;
        DEFINE FIELD entity_type ON equivalence TYPE string DEFAULT 'speaker';
        DEFINE FIELD created_at ON equivalence TYPE float DEFAULT time::unix();
        DEFINE FIELD last_updated ON equivalence TYPE float DEFAULT time::unix();
        
        -- M3's equivalence indexes
        DEFINE INDEX canonical_idx ON equivalence FIELDS canonical_id UNIQUE;
        DEFINE INDEX entity_type_idx ON equivalence FIELDS entity_type;
        """
        
        # M3's session/clip tracking
        session_schema = """
        -- M3-equivalent session tracking for clip-based organization
        DEFINE TABLE session SCHEMAFULL;
        DEFINE FIELD session_id ON session TYPE int;
        DEFINE FIELD clip_id ON session TYPE int;
        DEFINE FIELD start_time ON session TYPE float;
        DEFINE FIELD end_time ON session TYPE option<float>;
        DEFINE FIELD speaker_count ON session TYPE int DEFAULT 0;
        DEFINE FIELD node_count ON session TYPE int DEFAULT 0;
        
        -- Session indexes
        DEFINE INDEX session_id_idx ON session FIELDS session_id;
        DEFINE INDEX clip_id_idx ON session FIELDS clip_id;
        """
        
        # Schema metadata
        metadata_schema = """
        -- Schema version tracking
        DEFINE TABLE schema_info SCHEMAFULL;
        DEFINE FIELD version ON schema_info TYPE string;
        DEFINE FIELD created_at ON schema_info TYPE float;
        DEFINE FIELD description ON schema_info TYPE string;
        """
        
        # Execute schema definitions
        schemas = [node_schema, edge_schema, equivalence_schema, session_schema, metadata_schema]
        
        for schema in schemas:
            try:
                await self.db.query(schema)
                logger.debug("✅ Schema definition executed successfully")
            except Exception as e:
                logger.error(f"Failed to execute schema: {e}")
                raise
        
        # Insert schema version
        try:
            await self.db.query("""
                CREATE schema_info SET
                    version = $version,
                    created_at = time::unix(),
                    description = 'M3-AudioGraph schema for voice agent memory';
            """, {"version": self.schema_version})
            
            logger.info(f"🏗️  M3 schema v{self.schema_version} initialized successfully")
            
        except Exception as e:
            # Schema info might already exist
            logger.debug(f"Schema info creation skipped: {e}")
    
    async def save_audio_graph(self, audio_graph: AudioGraph) -> bool:
        """
        Save complete AudioGraph to SurrealDB using M3 patterns.
        
        Args:
            audio_graph: AudioGraph instance to persist
            
        Returns:
            Success indicator
        """
        if not self.connected:
            logger.error("Not connected to SurrealDB")
            return False
        
        try:
            # M3's transaction-based save for consistency
            async with self.db.transaction():
                # Save all nodes
                await self._save_nodes(audio_graph.nodes)
                
                # Save all edges  
                await self._save_edges(audio_graph.edges)
                
                # Save equivalence relationships
                await self._save_equivalences(audio_graph.equivalences)
                
                # Update session metadata
                await self._update_session_metadata(audio_graph)
            
            logger.info(f"💾 AudioGraph saved successfully ({len(audio_graph.nodes)} nodes, {len(audio_graph.edges)//2} edges)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save AudioGraph: {e}")
            return False
    
    async def load_audio_graph(self, session_id: Optional[int] = None) -> Optional[AudioGraph]:
        """
        Load AudioGraph from SurrealDB using M3 patterns.
        
        Args:
            session_id: Optional session to load (None for all data)
            
        Returns:
            Loaded AudioGraph instance or None if failed
        """
        if not self.connected:
            logger.error("Not connected to SurrealDB")
            return None
        
        try:
            # Create new AudioGraph instance
            audio_graph = AudioGraph()
            
            # M3's incremental loading
            nodes = await self._load_nodes(session_id)
            edges = await self._load_edges(session_id)
            equivalences = await self._load_equivalences()
            
            # Reconstruct AudioGraph state
            audio_graph.nodes = nodes
            audio_graph.edges = edges
            audio_graph.equivalences = equivalences
            
            # Rebuild node type lists (M3 pattern)
            audio_graph.voice_nodes = [
                nid for nid, node in nodes.items() if node.type == 'voice'
            ]
            audio_graph.text_nodes = [
                nid for nid, node in nodes.items() if node.type in ['episodic', 'semantic']
            ]
            
            # Update next_node_id
            if nodes:
                audio_graph.next_node_id = max(nodes.keys()) + 1
            
            logger.info(f"📂 AudioGraph loaded successfully ({len(nodes)} nodes, {len(edges)//2} edges)")
            return audio_graph
            
        except Exception as e:
            logger.error(f"Failed to load AudioGraph: {e}")
            return None
    
    async def _save_nodes(self, nodes: Dict[int, Any]):
        """Save all nodes using M3's batch pattern."""
        if not nodes:
            return
        
        # M3's batch insert for performance
        batch_size = 100
        node_list = list(nodes.items())
        
        for i in range(0, len(node_list), batch_size):
            batch = node_list[i:i + batch_size]
            
            # Prepare batch insert data
            insert_data = []
            for node_id, node in batch:
                node_data = {
                    'node_id': node_id,
                    'node_type': node.type,
                    'embeddings': node.embeddings,
                    'metadata': node.metadata
                }
                insert_data.append(node_data)
            
            # M3's upsert pattern (insert or update)
            try:
                query = """
                FOR $node IN $batch {
                    UPSERT (SELECT * FROM node WHERE node_id = $node.node_id)[0] 
                    CONTENT $node;
                }
                """
                await self.db.query(query, {"batch": insert_data})
                
            except Exception as e:
                logger.error(f"Failed to save node batch: {e}")
                # Try individual inserts as fallback
                for node_id, node in batch:
                    await self._save_single_node(node_id, node)
    
    async def _save_single_node(self, node_id: int, node: Any):
        """Save single node with error handling."""
        try:
            query = """
            UPSERT (SELECT * FROM node WHERE node_id = $node_id)[0]
            CONTENT {
                node_id: $node_id,
                node_type: $node_type,
                embeddings: $embeddings,
                metadata: $metadata
            };
            """
            
            await self.db.query(query, {
                'node_id': node_id,
                'node_type': node.type,
                'embeddings': node.embeddings,
                'metadata': node.metadata
            })
            
        except Exception as e:
            logger.error(f"Failed to save node {node_id}: {e}")
    
    async def _save_edges(self, edges: Dict[tuple, float]):
        """Save all edges using M3's bidirectional pattern."""
        if not edges:
            return
        
        # M3's bidirectional edge deduplication
        processed_pairs = set()
        unique_edges = []
        
        for (source, target), weight in edges.items():
            edge_pair = tuple(sorted([source, target]))
            if edge_pair not in processed_pairs:
                unique_edges.append({
                    'source': f"node:{source}",
                    'target': f"node:{target}", 
                    'weight': weight
                })
                processed_pairs.add(edge_pair)
        
        # Batch edge insert
        batch_size = 100
        for i in range(0, len(unique_edges), batch_size):
            batch = unique_edges[i:i + batch_size]
            
            try:
                # Clear existing edges and insert new ones
                await self.db.query("DELETE edge;")  # M3's full refresh pattern
                
                query = """
                FOR $edge IN $batch {
                    CREATE edge CONTENT $edge;
                }
                """
                await self.db.query(query, {"batch": batch})
                
            except Exception as e:
                logger.error(f"Failed to save edge batch: {e}")
    
    async def _save_equivalences(self, equivalences: Dict[str, set]):
        """Save equivalence relationships using M3's pattern."""
        if not equivalences:
            return
        
        # Clear existing equivalences
        await self.db.query("DELETE equivalence;")
        
        # Insert new equivalences
        for canonical_id, node_set in equivalences.items():
            try:
                query = """
                CREATE equivalence SET
                    canonical_id = $canonical_id,
                    node_ids = $node_ids,
                    entity_type = 'speaker',
                    last_updated = time::unix();
                """
                
                await self.db.query(query, {
                    'canonical_id': canonical_id,
                    'node_ids': list(node_set)  # Convert set to list
                })
                
            except Exception as e:
                logger.error(f"Failed to save equivalence {canonical_id}: {e}")
    
    async def _load_nodes(self, session_id: Optional[int] = None) -> Dict[int, Any]:
        """Load nodes from SurrealDB with M3's filtering."""
        try:
            if session_id is not None:
                # Load nodes for specific session
                query = "SELECT * FROM node WHERE metadata.session_id = $session_id;"
                result = await self.db.query(query, {"session_id": session_id})
            else:
                # Load all nodes
                query = "SELECT * FROM node;"
                result = await self.db.query(query)
            
            nodes = {}
            for row in result[0].get('result', []):
                # Reconstruct AudioGraph.Node
                from .audio_graph import AudioGraph
                node = AudioGraph.Node(row['node_id'], row['node_type'])
                node.embeddings = row['embeddings']
                node.metadata = row['metadata']
                
                nodes[row['node_id']] = node
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to load nodes: {e}")
            return {}
    
    async def _load_edges(self, session_id: Optional[int] = None) -> Dict[tuple, float]:
        """Load edges from SurrealDB with M3's bidirectional reconstruction."""
        try:
            query = "SELECT * FROM edge;"
            result = await self.db.query(query)
            
            edges = {}
            for row in result[0].get('result', []):
                # Extract node IDs from records
                source_id = int(row['source'].split(':')[1])
                target_id = int(row['target'].split(':')[1])
                weight = row['weight']
                
                # M3's bidirectional edge reconstruction
                edges[(source_id, target_id)] = weight
                edges[(target_id, source_id)] = weight
            
            return edges
            
        except Exception as e:
            logger.error(f"Failed to load edges: {e}")
            return {}
    
    async def _load_equivalences(self) -> Dict[str, set]:
        """Load equivalence relationships from SurrealDB."""
        try:
            query = "SELECT * FROM equivalence;"
            result = await self.db.query(query)
            
            equivalences = {}
            for row in result[0].get('result', []):
                canonical_id = row['canonical_id']
                node_ids = set(row['node_ids'])  # Convert list back to set
                
                equivalences[canonical_id] = node_ids
            
            return equivalences
            
        except Exception as e:
            logger.error(f"Failed to load equivalences: {e}")
            return {}
    
    async def _update_session_metadata(self, audio_graph: AudioGraph):
        """Update session metadata using M3's tracking pattern."""
        try:
            # Get current session statistics
            stats = audio_graph.get_stats()
            
            # Update or create session record
            query = """
            UPSERT (SELECT * FROM session WHERE session_id = 0)[0]
            CONTENT {
                session_id: 0,
                clip_id: 0,
                start_time: time::unix(),
                speaker_count: $speaker_count,
                node_count: $node_count
            };
            """
            
            speaker_count = len(set(
                node.metadata.get('speaker_id') for node in audio_graph.nodes.values()
                if node.type == 'voice' and node.metadata.get('speaker_id')
            ))
            
            await self.db.query(query, {
                'speaker_count': speaker_count,
                'node_count': stats['total_nodes']
            })
            
        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
    
    async def search_similar_nodes(self, query_embeddings: List[float], 
                                  node_type: str = 'semantic',
                                  limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar nodes using vector similarity (M3's pattern).
        
        Args:
            query_embeddings: Query embedding vector
            node_type: Type of nodes to search
            limit: Maximum results to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if not self.connected:
            return []
        
        try:
            # SurrealDB vector similarity search (if available)
            query = """
            SELECT node_id, vector::similarity::cosine(embeddings[0], $query_emb) AS similarity
            FROM node 
            WHERE node_type = $node_type 
            AND array::len(embeddings) > 0
            ORDER BY similarity DESC
            LIMIT $limit;
            """
            
            result = await self.db.query(query, {
                'query_emb': query_embeddings,
                'node_type': node_type,
                'limit': limit
            })
            
            similar_nodes = []
            for row in result[0].get('result', []):
                similar_nodes.append((row['node_id'], row['similarity']))
            
            return similar_nodes
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        if not self.connected:
            return {}
        
        try:
            stats_query = """
            {
                total_nodes: (SELECT count() FROM node GROUP ALL)[0].count,
                voice_nodes: (SELECT count() FROM node WHERE node_type = 'voice' GROUP ALL)[0].count,
                episodic_nodes: (SELECT count() FROM node WHERE node_type = 'episodic' GROUP ALL)[0].count, 
                semantic_nodes: (SELECT count() FROM node WHERE node_type = 'semantic' GROUP ALL)[0].count,
                total_edges: (SELECT count() FROM edge GROUP ALL)[0].count,
                equivalences: (SELECT count() FROM equivalence GROUP ALL)[0].count
            }
            """
            
            result = await self.db.query(stats_query)
            return result[0].get('result', [{}])[0]
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# Convenience functions for AudioGraph integration
async def connect_audio_graph_to_surreal(audio_graph: AudioGraph, 
                                        connection_config: Optional[Dict] = None) -> Optional[M3SurrealIntegration]:
    """
    Connect AudioGraph to SurrealDB with M3 schema.
    
    Args:
        audio_graph: AudioGraph instance
        connection_config: SurrealDB connection configuration
        
    Returns:
        M3SurrealIntegration instance if successful, None otherwise
    """
    integration = M3SurrealIntegration(connection_config)
    
    if await integration.connect():
        # Optionally load existing data
        existing_graph = await integration.load_audio_graph()
        if existing_graph and existing_graph.nodes:
            logger.info(f"🔄 Merging existing data: {len(existing_graph.nodes)} nodes")
            # Simple merge strategy - could be more sophisticated
            audio_graph.nodes.update(existing_graph.nodes)
            audio_graph.edges.update(existing_graph.edges)
            audio_graph.equivalences.update(existing_graph.equivalences)
        
        return integration
    
    return None