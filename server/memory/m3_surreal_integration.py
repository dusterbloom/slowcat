"""M3 Memory Integration with SurrealDB

This module provides the integration layer between the M3 memory system
and SurrealDB, implementing node storage, graph relationships, and 
similarity search capabilities.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .surreal_connection import SurrealConnectionManager

logger = logging.getLogger(__name__)

@dataclass
class M3Node:
    """M3 Memory Node representation"""
    node_id: Optional[int] = None
    node_type: str = "semantic"  # voice, episodic, semantic
    contents: List[str] = None
    embeddings: List[List[float]] = None
    clip_id: int = 1
    source_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.contents is None:
            self.contents = []
        if self.embeddings is None:
            self.embeddings = []
        if self.metadata is None:
            self.metadata = {
                'speaker_id': 'unknown',
                'confidence': 0.8,
                'extraction_method': 'automatic',
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0
            }

@dataclass
class M3Edge:
    """M3 Memory Edge representation"""
    source_node_id: int
    target_node_id: int
    weight: float = 1.0
    edge_type: str = "similarity"

@dataclass 
class M3Clip:
    """M3 Memory Clip representation"""
    clip_id: Optional[int] = None
    session_id: str = "default"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    node_count: int = 0
    dominant_speaker: Optional[str] = None

class M3SurrealIntegration:
    """M3 memory integration with SurrealDB"""
    
    def __init__(self, connection: SurrealConnectionManager):
        self.connection = connection
        self.current_clip_id: Optional[int] = None
    
    @staticmethod
    def _first(result):
        if result is None:
            return None
        return result[0] if isinstance(result, list) else result
    
    async def query(self, query: str, params: Dict[str, Any] = None):
        """Execute raw SurrealDB query through connection manager"""
        await self.connection.ensure_connected()
        return await self.connection.db.query(query, params or {})
        
    async def initialize(self) -> bool:
        """Initialize M3 system and verify schema"""
        try:
            # Try to verify M3 migration was applied
            try:
                result = await self.query("RETURN fn::verify_m3_migration()")
                logger.debug(f"Migration verification result: {result}")
                
                if result:
                    verification_result = self._first(result)
                    if isinstance(verification_result, dict) and verification_result.get('status') == 'success':
                        logger.info("✅ M3 schema verified successfully")
                    elif verification_result == True or verification_result == 'T':
                        # Some SurrealDB versions return boolean true as 'T'
                        logger.info("✅ M3 schema verified (boolean response)")
                    else:
                        logger.warning(f"⚠️ M3 verification unexpected format: {verification_result}")
                else:
                    logger.warning("⚠️ M3 schema verification returned no result")
                    
            except Exception as verify_error:
                logger.warning(f"⚠️ M3 schema verification failed: {verify_error}")
                logger.info("🔄 Attempting to continue with basic M3 table check...")
                
                # Fallback: Check if basic M3 tables exist
                try:
                    table_check = await self.query("SELECT * FROM m3_nodes LIMIT 1")
                    logger.info("✅ M3 tables appear to be accessible")
                except Exception as table_error:
                    logger.error(f"❌ M3 tables not accessible: {table_error}")
                    return False
            
            # Initialize or get current clip
            await self._ensure_current_clip()
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize M3 system: {e}")
            return False
    
    async def _ensure_current_clip(self, session_id: str = "default"):
        """Ensure we have a current clip for storing nodes"""
        try:
            # Get or create current clip
            result = await self.query(
                "RETURN fn::create_m3_clip($session_id)",
                {"session_id": session_id}
            )
            if result:
                self.current_clip_id = self._first(result)
                logger.info(f"Current clip ID: {self.current_clip_id}")
            else:
                logger.warning("Failed to create/get current clip")
                self.current_clip_id = 1  # Fallback
                
        except Exception as e:
            logger.error(f"Failed to ensure current clip: {e}")
            self.current_clip_id = 1  # Fallback
    
    async def store_m3_node(self, 
                           node_type: str, 
                           contents: List[str], 
                           embeddings: List[List[float]], 
                           clip_id: Optional[int] = None,
                           source_message_id: Optional[str] = None,
                           speaker_id: str = "unknown",
                           extraction_method: str = "automatic",
                           confidence: float = 0.8) -> Optional[int]:
        """Store M3 node with content and embeddings
        
        Args:
            node_type: Type of node ('voice', 'episodic', 'semantic')
            contents: List of content strings stored in the node
            embeddings: List of embedding vectors (multiple embeddings per node)
            clip_id: Temporal clip ID (uses current clip if not provided)
            source_message_id: ID of source message if applicable
            speaker_id: Speaker identifier
            extraction_method: Method used to extract this node
            confidence: Confidence score for the extraction
            
        Returns:
            Node ID if successful, None if failed
        """
        try:
            if clip_id is None:
                clip_id = self.current_clip_id or 1
                
            # Prepare metadata
            metadata = {
                'speaker_id': speaker_id,
                'confidence': confidence,
                'extraction_method': extraction_method,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0
            }
            
            # Create node directly with SurrealDB query
            # Generate a simple node ID based on timestamp
            import time
            simple_node_id = int(time.time() * 1000) % 1000000  # Simple ID generation
            
            result = await self.query(
                """
                CREATE m3_nodes CONTENT {
                    node_id: $node_id,
                    node_type: $node_type,
                    contents: $contents,
                    embeddings: $embeddings,
                    clip_id: $clip_id,
                    speaker_id: $speaker_id,
                    confidence: $confidence,
                    metadata: $metadata
                }
                """,
                {
                    "node_id": simple_node_id,
                    "node_type": node_type,
                    "contents": contents,
                    "embeddings": embeddings,
                    "clip_id": clip_id,
                    "speaker_id": speaker_id,
                    "confidence": confidence,
                    "metadata": metadata
                }
            )
            
            if result:
                logger.info(f"Created M3 node {simple_node_id} of type {node_type}")
                
                # Link to source message if provided
                if source_message_id:
                    await self._link_node_to_message(simple_node_id, source_message_id, extraction_method, confidence)
                
                return simple_node_id
            else:
                logger.error("Failed to create M3 node")
                return None
                
        except Exception as e:
            logger.error(f"Failed to store M3 node: {e}")
            return None
    
    async def _link_node_to_message(self, node_id: int, message_id: str, extraction_method: str, confidence: float):
        """Link M3 node to its source message"""
        try:
            await self.query("""
                LET $node = (SELECT * FROM m3_nodes WHERE node_id = $node_id)[0];
                LET $message = (SELECT * FROM messages WHERE id = $message_id)[0];
                IF $node AND $message THEN
                    RELATE $node.id->m3_node_from_message->$message.id SET {
                        extraction_confidence: $confidence,
                        extraction_method: $extraction_method,
                        created_at: time::now()
                    }
                END;
            """, {
                "node_id": node_id,
                "message_id": message_id,
                "confidence": confidence,
                "extraction_method": extraction_method
            })
            
        except Exception as e:
            logger.error(f"Failed to link node {node_id} to message {message_id}: {e}")
    
    async def create_m3_edge(self, 
                           source_node_id: int, 
                           target_node_id: int,
                           weight: float = 1.0, 
                           edge_type: str = 'similarity') -> bool:
        """Create edge between M3 nodes
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID  
            weight: Edge weight (0.0 to 1.0)
            edge_type: Type of relationship
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use direct connection to avoid _extract_records_list issues
            result = await self.connection.db.query(
                """
                LET $source_node = (SELECT * FROM m3_nodes WHERE node_id = $source_id)[0];
                LET $target_node = (SELECT * FROM m3_nodes WHERE node_id = $target_id)[0];
                CREATE m3_edges CONTENT {
                    source: $source_node.id,
                    target: $target_node.id,
                    weight: $weight,
                    edge_type: $edge_type,
                    created_at: time::now(),
                    last_reinforced: time::now()
                };
                """,
                {
                    "source_id": source_node_id,
                    "target_id": target_node_id,
                    "weight": weight,
                    "edge_type": edge_type
                }
            )
            
            # Check if result is successful (should be a list with the created edge)
            if isinstance(result, list) and len(result) > 0:
                logger.info(f"Created edge {source_node_id} -> {target_node_id} (weight: {weight})")
                return True
            else:
                logger.warning(f"Edge creation failed: {result}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to create M3 edge: {e}")
            return False
    
    async def search_similar_nodes(self, 
                                 query_embedding: List[float],
                                 node_type: Optional[str] = None, 
                                 limit: int = 10,
                                 min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Vector similarity search across M3 nodes
        
        Args:
            query_embedding: Query vector for similarity search
            node_type: Filter by node type (optional)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar nodes with similarity scores
        """
        try:
            # Filter out nodes with incorrect embedding dimensions (must be 384D)
            query_sql = """
            SELECT *,
                vector::similarity::cosine($query_embedding, embeddings[0]) AS similarity
            FROM m3_nodes
            WHERE array::len(embeddings) > 0
            AND array::len(embeddings[0]) = 384
            """ + (f" AND node_type = '{node_type}'" if node_type else "") + f"""
            AND vector::similarity::cosine($query_embedding, embeddings[0]) >= {min_similarity}
            ORDER BY similarity DESC 
            LIMIT {limit}
            """
            
            # Use the connection manager directly instead of self.query to avoid _extract_records_list issues
            result = await self.connection.db.query(query_sql, {"query_embedding": query_embedding})
            
            # result is already a list of dict records from SurrealConnectionManager
            if isinstance(result, list) and all(isinstance(r, dict) for r in result):
                logger.debug(f"Found {len(result)} similar nodes with similarity >= {min_similarity}")
                return result
            else:
                logger.warning(f"Unexpected result format: {type(result)}, trying _extract_records_list...")
                records = self._extract_records_list(result)
                return records
                
        except Exception as e:
            logger.error(f"Failed to search similar nodes: {e}")
            # Fallback to function-based approaches (try production 4-arg, then legacy 3-arg)
            # Attempt production signature with threshold
            try:
                result = await self.query(
                    "RETURN fn::search_similar_m3_nodes($query_embedding, $node_type, $limit, $threshold)",
                    {
                        "query_embedding": query_embedding,
                        "node_type": node_type,
                        "limit": limit,
                        "threshold": min_similarity
                    }
                )
                records = self._extract_records_list(result)
                if records:
                    return records
            except Exception:
                pass

            # Attempt legacy signature without threshold
            try:
                result = await self.query(
                    "RETURN fn::search_similar_m3_nodes($query_embedding, $node_type, $limit)",
                    {
                        "query_embedding": query_embedding,
                        "node_type": node_type,
                        "limit": limit
                    }
                )
                records = self._extract_records_list(result)
                if records:
                    filtered_results = [
                        node for node in records if isinstance(node, dict)
                        and node.get('similarity', 0) >= min_similarity
                    ]
                    logger.info(f"Found {len(filtered_results)} similar nodes (legacy function)")
                    return filtered_results
            except Exception as e2:
                logger.error(f"Fallback similarity search also failed: {e2}")
            return []

    def _extract_records_list(self, result):
        """Normalize SurrealDB query outputs into a list of dict records."""
        try:
            if result is None:
                return []
            # If result is a list of dict records already
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict) and 'result' in result[0]:
                    inner = result[0]['result']
                    return inner if isinstance(inner, list) else []
                if all(isinstance(x, dict) for x in result):
                    return result
                if len(result) == 1 and isinstance(result[0], list):
                    inner = result[0]
                    return inner if all(isinstance(x, dict) for x in inner) else []
                return []
            if isinstance(result, dict) and 'result' in result and isinstance(result['result'], list):
                inner = result['result']
                return inner if all(isinstance(x, dict) for x in inner) else []
            return []
        except Exception:
            return []
    
    async def resolve_equivalence(self, 
                                entity_name: str, 
                                node_ids: List[int],
                                entity_type: str = "speaker") -> Optional[str]:
        """Create or update equivalence relation
        
        Args:
            entity_name: Canonical name for the entity
            node_ids: List of node IDs that refer to this entity
            entity_type: Type of entity (speaker, location, concept, etc.)
            
        Returns:
            Canonical entity name if successful, None if failed
        """
        try:
            result = await self.query(
                "RETURN fn::resolve_m3_equivalence($entity_name, $node_ids, $entity_type)",
                {
                    "entity_name": entity_name,
                    "node_ids": node_ids,
                    "entity_type": entity_type
                }
            )
            
            if result:
                canonical_id = self._first(result)
                logger.info(f"Resolved equivalence for {entity_name} with {len(node_ids)} nodes")
                return canonical_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to resolve equivalence: {e}")
            return None
    
    async def get_clip_nodes(self, clip_id: int) -> List[Dict[str, Any]]:
        """Get all nodes in a temporal clip
        
        Args:
            clip_id: Clip ID to retrieve nodes for
            
        Returns:
            List of nodes in the clip, ordered by creation time
        """
        try:
            result = await self.query(
                "RETURN fn::get_clip_nodes($clip_id)",
                {"clip_id": clip_id}
            )
            records = self._extract_records_list(result)
            if records:
                logger.info(f"Retrieved {len(records)} nodes from clip {clip_id}")
                return records
            return []
        except Exception as e:
            logger.error(f"Failed to get clip nodes: {e}")
            return []
    
    async def create_new_clip(self, session_id: str = "default") -> Optional[int]:
        """Start new temporal clip
        
        Args:
            session_id: Session identifier for the clip
            
        Returns:
            New clip ID if successful, None if failed
        """
        try:
            # Direct query approach (more reliable than functions)
            import time
            clip_id = int(time.time() * 1000) % 1000000  # Simple ID generation
            
            result = await self.query(
                """
                CREATE m3_clips CONTENT {
                    clip_id: $clip_id,
                    session_id: $session_id,
                    start_time: time::now(),
                    end_time: NONE,
                    node_count: 0,
                    dominant_speaker: NONE,
                    metadata: {}
                }
                """,
                {
                    "clip_id": clip_id,
                    "session_id": session_id
                }
            )
            
            if result:
                self.current_clip_id = clip_id
                logger.info(f"Created new clip {clip_id} for session {session_id}")
                return clip_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to create new clip: {e}")
            return None
    
    async def close_current_clip(self, create_new: bool = True, session_id: str = "default") -> Optional[int]:
        """Close current clip and optionally create a new one
        
        Args:
            create_new: Whether to create a new clip after closing
            session_id: Session ID for new clip if creating
            
        Returns:
            New clip ID if created, current clip ID if not creating new
        """
        try:
            if self.current_clip_id is None:
                logger.warning("No current clip to close")
                return await self.create_new_clip(session_id) if create_new else None
            
            result = await self.query(
                "RETURN fn::close_m3_clip($clip_id, $create_new, $session_id)",
                {
                    "clip_id": self.current_clip_id,
                    "create_new": create_new,
                    "session_id": session_id if create_new else None
                }
            )
            
            if result:
                new_clip_id = self._first(result)
                if create_new:
                    self.current_clip_id = new_clip_id
                    logger.info(f"Closed clip and created new clip {new_clip_id}")
                else:
                    logger.info(f"Closed clip {self.current_clip_id}")
                
                return new_clip_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to close current clip: {e}")
            return None
    
    async def infer_edges_for_node(self, 
                                 node_id: int, 
                                 similarity_threshold: float = 0.7,
                                 max_edges: int = 5) -> int:
        """Automatically infer edges for a node based on similarity
        
        Args:
            node_id: Node to create edges for
            similarity_threshold: Minimum similarity for edge creation
            max_edges: Maximum number of edges to create
            
        Returns:
            Number of edges created
        """
        try:
            # Get the node and its embeddings
            node_result = await self.query(
                "SELECT * FROM m3_nodes WHERE node_id = $node_id",
                {"node_id": node_id}
            )
            records = self._extract_records_list(node_result)
            if not records:
                logger.warning(f"Node {node_id} not found for edge inference")
                return 0
            
            node = records[0]
            if not node.get('embeddings') or not node['embeddings']:
                logger.warning(f"Node {node_id} has no embeddings for similarity")
                return 0
            
            # Use first embedding for similarity search
            query_embedding = node['embeddings'][0]
            
            # Find similar nodes
            similar_nodes = await self.search_similar_nodes(
                query_embedding, 
                limit=max_edges + 1,  # +1 to account for self
                min_similarity=similarity_threshold
            )
            
            edges_created = 0
            for similar_node in similar_nodes:
                # Skip self
                if similar_node.get('node_id') == node_id:
                    continue
                
                # Create edge
                success = await self.create_m3_edge(
                    node_id,
                    similar_node['node_id'],
                    weight=similar_node.get('similarity', 1.0),
                    edge_type='similarity'
                )
                
                if success:
                    edges_created += 1
                
                # Stop if we've created enough edges
                if edges_created >= max_edges:
                    break
            
            logger.info(f"Created {edges_created} edges for node {node_id}")
            return edges_created
            
        except Exception as e:
            logger.error(f"Failed to infer edges for node {node_id}: {e}")
            return 0
    
    async def decay_edges(self, decay_rate: float = 0.01) -> str:
        """Apply decay to edge weights (memory aging)
        
        Args:
            decay_rate: Rate of decay per time period
            
        Returns:
            Status message
        """
        try:
            result = await self.query(
                "RETURN fn::decay_m3_edges($decay_rate)",
                {"decay_rate": decay_rate}
            )
            
            return (result[0] if isinstance(result, list) else result) if result else "Edge decay failed"
            
        except Exception as e:
            logger.error(f"Failed to decay edges: {e}")
            return f"Edge decay failed: {e}"
    
    async def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific node by its ID
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Node data if found, None otherwise
        """
        try:
            result = await self.query(
                "SELECT * FROM m3_nodes WHERE node_id = $node_id",
                {"node_id": node_id}
            )
            
            if result:
                # Update access tracking
                await self.query("""
                    UPDATE m3_nodes SET 
                        metadata.last_accessed = time::now(),
                        metadata.access_count += 1
                    WHERE node_id = $node_id
                """, {"node_id": node_id})
                
                return self._first(result)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    async def get_graph_context(self, 
                              node_id: int, 
                              max_depth: int = 2, 
                              max_nodes: int = 20) -> Dict[str, Any]:
        """Get graph context around a node (for memory retrieval)
        
        Args:
            node_id: Central node to get context around
            max_depth: Maximum graph traversal depth
            max_nodes: Maximum number of nodes to return
            
        Returns:
            Graph context with nodes and relationships
        """
        try:
            # Get central node
            central_node = await self.get_node_by_id(node_id)
            if not central_node:
                return {"nodes": [], "edges": [], "central_node": None}
            
            # Get connected nodes via graph traversal
            result = await self.query("""
                LET $central = (SELECT * FROM m3_nodes WHERE node_id = $node_id)[0];
                LET $connected = SELECT * FROM (
                    TRAVERSE $central.id->m3_edges->m3_nodes MAXDEPTH $max_depth
                    UNION
                    TRAVERSE $central.id<-m3_edges<-m3_nodes MAXDEPTH $max_depth
                ) LIMIT $max_nodes;
                
                LET $node_ids = array::flatten([[$central.id], $connected[*].id]);
                LET $edges = SELECT * FROM m3_edges 
                    WHERE source IN $node_ids AND target IN $node_ids;
                
                RETURN {
                    central_node: $central,
                    nodes: $connected,
                    edges: $edges
                };
            """, {
                "node_id": node_id,
                "max_depth": max_depth,
                "max_nodes": max_nodes
            })
            
            if result:
                return self._first(result) or {"nodes": [], "edges": [], "central_node": None}
            else:
                return {"nodes": [], "edges": [], "central_node": central_node}
                
        except Exception as e:
            logger.error(f"Failed to get graph context for node {node_id}: {e}")
            return {"nodes": [], "edges": [], "central_node": None}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get M3 system statistics
        
        Returns:
            Dictionary with system statistics
        """
        try:
            result = await self.query("""
                RETURN {
                    total_nodes: count(SELECT * FROM m3_nodes),
                    total_edges: count(SELECT * FROM m3_edges),
                    total_clips: count(SELECT * FROM m3_clips),
                    total_equivalences: count(SELECT * FROM m3_equivalences),
                    nodes_by_type: (
                        SELECT node_type, count() as count FROM m3_nodes GROUP BY node_type
                    ),
                    current_clip_id: $current_clip_id,
                    avg_edges_per_node: math::mean(
                        SELECT count(SELECT * FROM m3_edges WHERE source = $parent.id OR target = $parent.id) as edge_count 
                        FROM m3_nodes
                    )
                };
            """, {"current_clip_id": self.current_clip_id})
            
            return self._first(result) if result else {}
            
        except Exception as e:
            logger.error(f"Failed to get M3 statistics: {e}")
            return {}
    
    async def batch_create_nodes(self, 
                               nodes_data: List[Dict[str, Any]]) -> List[int]:
        """Batch create multiple M3 nodes for performance
        
        Args:
            nodes_data: List of node data dictionaries
            
        Returns:
            List of created node IDs
        """
        try:
            created_ids = []
            
            # Use transaction for consistency
            for node_data in nodes_data:
                node_id = await self.store_m3_node(
                    node_type=node_data.get('node_type', 'semantic'),
                    contents=node_data.get('contents', []),
                    embeddings=node_data.get('embeddings', []),
                    clip_id=node_data.get('clip_id'),
                    source_message_id=node_data.get('source_message_id'),
                    speaker_id=node_data.get('speaker_id', 'unknown'),
                    extraction_method=node_data.get('extraction_method', 'batch'),
                    confidence=node_data.get('confidence', 0.8)
                )
                
                if node_id:
                    created_ids.append(node_id)
            
            logger.info(f"Batch created {len(created_ids)} M3 nodes")
            return created_ids
            
        except Exception as e:
            logger.error(f"Batch node creation failed: {e}")
            return []
    
    async def optimize_for_similarity_search(self):
        """Optimize database for similarity search performance"""
        try:
            logger.info("🚀 Optimizing M3 database for similarity search...")
            
            # Ensure vector similarity indexes exist
            await self.query("""
                DEFINE INDEX embedding_similarity_idx ON m3_nodes FIELDS embeddings SEARCH;
            """)
            
            # Update statistics for query planner
            await self.query("ANALYZE INDEX embedding_similarity_idx ON m3_nodes;")
            
            # Cleanup old weak edges
            cleanup_result = await self.query("""
                DELETE m3_edges WHERE weight < 0.1 OR created_at < time::now() - 30d;
            """)
            
            logger.info(f"✅ M3 database optimization complete. Cleaned up weak/old edges.")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    async def get_recent_nodes_with_embeddings(self, 
                                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent nodes with embeddings for cache warming
        
        Args:
            limit: Number of recent nodes to fetch
            
        Returns:
            List of recent nodes with embeddings
        """
        try:
            result = await self.query("""
                SELECT * FROM m3_nodes 
                WHERE array::len(embeddings) > 0 
                ORDER BY metadata.created_at DESC 
                LIMIT $limit
            """, {"limit": limit})
            
            return result or []
            
        except Exception as e:
            logger.error(f"Failed to get recent nodes: {e}")
            return []
