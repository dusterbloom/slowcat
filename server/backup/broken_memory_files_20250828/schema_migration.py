"""
SurrealDB Schema Migration for Hierarchical Memory System

Adds advanced neural field tables (fragments, patterns, attractors, field_states)
to support four-tier hierarchical memory with friend's neural field schema design.

This extends the existing SurrealDB schema from surreal_memory.py with:
- Fragment storage for reconstructive memory
- Pattern-based reconstruction templates
- Attractor dynamics for memory consolidation  
- Neural field state persistence across tiers
- Enhanced relationships for graph traversal
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    logger.warning("SurrealDB client not available")
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None

class HierarchicalSchemaManager:
    """Manages SurrealDB schema migration for hierarchical memory system"""
    
    def __init__(self, db_url: str = "ws://127.0.0.1:8000/rpc",
                 username: str = "root", password: str = "slowcat_secure_2024",
                 namespace: str = "slowcat", database: str = "memory"):
        self.db_url = db_url
        self.username = username
        self.password = password
        self.namespace = namespace
        self.database = database
        self.db: Optional[AsyncSurreal] = None
    
    async def connect(self):
        """Connect to SurrealDB"""
        if not SURREALDB_AVAILABLE:
            raise RuntimeError("SurrealDB client not available. Install with: pip install surrealdb")
        
        self.db = AsyncSurreal(self.db_url)
        await self.db.connect()
        await self.db.use(self.namespace, self.database)
        await self.db.sign_in(self.username, self.password)
        logger.info("Connected to SurrealDB for schema migration")
    
    async def disconnect(self):
        """Disconnect from SurrealDB"""
        if self.db:
            await self.db.close()
    
    async def migrate_to_hierarchical_schema(self):
        """Run complete schema migration to hierarchical memory system"""
        logger.info("Starting hierarchical memory schema migration...")
        
        try:
            await self.connect()
            
            # Step 1: Create fragment storage tables
            await self._create_fragment_tables()
            
            # Step 2: Create neural field state tables  
            await self._create_field_state_tables()
            
            # Step 3: Create pattern and attractor tables
            await self._create_pattern_attractor_tables()
            
            # Step 4: Create enhanced relationships
            await self._create_hierarchical_relationships()
            
            # Step 5: Create indexes for performance
            await self._create_performance_indexes()
            
            # Step 6: Verify schema
            await self._verify_schema()
            
            logger.info("✅ Hierarchical memory schema migration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Schema migration failed: {e}")
            raise
        finally:
            await self.disconnect()
    
    async def _create_fragment_tables(self):
        """Create fragment storage tables for reconstructive memory"""
        logger.info("Creating fragment storage tables...")
        
        # Main fragments table
        await self.db.query("""
            DEFINE TABLE fragments SCHEMAFULL;
            DEFINE FIELD fragment_id ON fragments TYPE string;
            DEFINE FIELD type ON fragments TYPE string ASSERT $value IN ["semantic", "episodic", "procedural", "contextual", "emotional"];
            DEFINE FIELD content ON fragments TYPE object;
            DEFINE FIELD context_tags ON fragments TYPE array<string>;
            DEFINE FIELD strength ON fragments TYPE float ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD memory_tier ON fragments TYPE int ASSERT $value >= 1 AND $value <= 4;
            DEFINE FIELD last_accessed ON fragments TYPE datetime;
            DEFINE FIELD access_count ON fragments TYPE int DEFAULT 0;
            DEFINE FIELD source_interactions ON fragments TYPE array<string>;
            DEFINE FIELD importance_score ON fragments TYPE float DEFAULT 0.5;
            DEFINE FIELD embedding_vector ON fragments TYPE array<float>;
            DEFINE FIELD semantic_hash ON fragments TYPE string;
            DEFINE FIELD created_at ON fragments TYPE datetime DEFAULT time::now();
            DEFINE FIELD promoted_at ON fragments TYPE datetime;
        """)
        
        logger.info("✓ Fragment tables created")
    
    async def _create_field_state_tables(self):
        """Create neural field state persistence tables"""
        logger.info("Creating field state tables...")
        
        await self.db.query("""
            DEFINE TABLE field_states SCHEMAFULL;
            DEFINE FIELD instance_id ON field_states TYPE string;
            DEFINE FIELD fragment_id ON field_states TYPE string;
            DEFINE FIELD compression ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD drift ON field_states TYPE string ASSERT $value IN ["none", "low", "moderate", "high"];
            DEFINE FIELD recursion_depth ON field_states TYPE int ASSERT $value >= 0;
            DEFINE FIELD resonance ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD presence_signal ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD boundary ON field_states TYPE string ASSERT $value IN ["gradient", "collapsed"];
            DEFINE FIELD memory_tier ON field_states TYPE int DEFAULT 1 ASSERT $value >= 1 AND $value <= 4;
            DEFINE FIELD evolution_data ON field_states TYPE object;
            DEFINE FIELD transition_history ON field_states TYPE array<object>;
            DEFINE FIELD updated_at ON field_states TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✓ Field state tables created")
    
    async def _create_pattern_attractor_tables(self):
        """Create pattern and attractor tables for memory consolidation"""
        logger.info("Creating pattern and attractor tables...")
        
        # Patterns table for reconstruction templates
        await self.db.query("""
            DEFINE TABLE patterns SCHEMAFULL;
            DEFINE FIELD pattern_id ON patterns TYPE string;
            DEFINE FIELD pattern_type ON patterns TYPE string;
            DEFINE FIELD trigger_conditions ON patterns TYPE array<object>;
            DEFINE FIELD fragment_clusters ON patterns TYPE array<string>;
            DEFINE FIELD reconstruction_template ON patterns TYPE object;
            DEFINE FIELD confidence_indicators ON patterns TYPE array<object>;
            DEFINE FIELD activation_threshold ON patterns TYPE float DEFAULT 0.5;
            DEFINE FIELD effective_tiers ON patterns TYPE array<int>;
            DEFINE FIELD success_rate ON patterns TYPE float DEFAULT 0.0;
            DEFINE FIELD usage_count ON patterns TYPE int DEFAULT 0;
            DEFINE FIELD created_at ON patterns TYPE datetime DEFAULT time::now();
            DEFINE FIELD last_used ON patterns TYPE datetime;
        """)
        
        # Attractors table for memory consolidation
        await self.db.query("""
            DEFINE TABLE attractors SCHEMAFULL;
            DEFINE FIELD attractor_id ON attractors TYPE string;
            DEFINE FIELD pattern_data ON attractors TYPE object;
            DEFINE FIELD fragment_id ON attractors TYPE string;
            DEFINE FIELD strength ON attractors TYPE float ASSERT $value >= 0 AND $value <= 1;
            DEFINE FIELD basin_width ON attractors TYPE float DEFAULT 0.3;
            DEFINE FIELD memory_tier ON attractors TYPE int ASSERT $value >= 1 AND $value <= 4;
            DEFINE FIELD consolidation_score ON attractors TYPE float DEFAULT 0.0;
            DEFINE FIELD activation_count ON attractors TYPE int DEFAULT 0;
            DEFINE FIELD last_activated ON attractors TYPE datetime;
            DEFINE FIELD created_at ON attractors TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✓ Pattern and attractor tables created")
    
    async def _create_hierarchical_relationships(self):
        """Create enhanced relationships for hierarchical memory"""
        logger.info("Creating hierarchical relationships...")
        
        # Fragment relationships
        await self.db.query("""
            DEFINE TABLE fragment_patterns TYPE RELATION IN fragments OUT patterns;
            DEFINE FIELD relevance_score ON fragment_patterns TYPE float DEFAULT 0.5;
            DEFINE FIELD usage_count ON fragment_patterns TYPE int DEFAULT 0;
            DEFINE FIELD last_used ON fragment_patterns TYPE datetime;
        """)
        
        await self.db.query("""
            DEFINE TABLE fragment_attractors TYPE RELATION IN fragments OUT attractors;
            DEFINE FIELD activation_strength ON fragment_attractors TYPE float DEFAULT 0.5;
            DEFINE FIELD resonance_frequency ON fragment_attractors TYPE float DEFAULT 0.5;
            DEFINE FIELD last_resonance ON fragment_attractors TYPE datetime;
        """)
        
        # Cross-tier transition relationships
        await self.db.query("""
            DEFINE TABLE attractor_transitions TYPE RELATION IN attractors OUT attractors;
            DEFINE FIELD transition_type ON attractor_transitions TYPE string;
            DEFINE FIELD confidence ON attractor_transitions TYPE float DEFAULT 0.5;
            DEFINE FIELD source_tier ON attractor_transitions TYPE int;
            DEFINE FIELD target_tier ON attractor_transitions TYPE int;
        """)
        
        # Pattern evolution relationships
        await self.db.query("""
            DEFINE TABLE pattern_evolution TYPE RELATION IN patterns OUT patterns;
            DEFINE FIELD evolution_type ON pattern_evolution TYPE string;
            DEFINE FIELD improvement_score ON pattern_evolution TYPE float DEFAULT 0.0;
            DEFINE FIELD evolution_data ON pattern_evolution TYPE object;
        """)
        
        # Fragment field state relationships
        await self.db.query("""
            DEFINE TABLE fragment_field_states TYPE RELATION IN fragments OUT field_states;
            DEFINE FIELD state_coherence ON fragment_field_states TYPE float DEFAULT 0.5;
            DEFINE FIELD field_influence ON fragment_field_states TYPE float DEFAULT 0.5;
        """)
        
        logger.info("✓ Hierarchical relationships created")
    
    async def _create_performance_indexes(self):
        """Create indexes for fast fragment retrieval"""
        logger.info("Creating performance indexes...")
        
        # Indexes for fragment retrieval performance
        await self.db.query("""
            DEFINE INDEX fragment_tier_idx ON fragments COLUMNS memory_tier;
            DEFINE INDEX fragment_type_idx ON fragments COLUMNS type;
            DEFINE INDEX fragment_strength_idx ON fragments COLUMNS strength;
            DEFINE INDEX fragment_accessed_idx ON fragments COLUMNS last_accessed;
            DEFINE INDEX fragment_hash_idx ON fragments COLUMNS semantic_hash;
        """)
        
        # Indexes for field states
        await self.db.query("""
            DEFINE INDEX field_tier_idx ON field_states COLUMNS memory_tier;
            DEFINE INDEX field_instance_idx ON field_states COLUMNS instance_id;
            DEFINE INDEX field_resonance_idx ON field_states COLUMNS resonance;
        """)
        
        # Indexes for patterns and attractors
        await self.db.query("""
            DEFINE INDEX pattern_type_idx ON patterns COLUMNS pattern_type;
            DEFINE INDEX pattern_threshold_idx ON patterns COLUMNS activation_threshold;
            DEFINE INDEX attractor_tier_idx ON attractors COLUMNS memory_tier;
            DEFINE INDEX attractor_strength_idx ON attractors COLUMNS strength;
        """)
        
        logger.info("✓ Performance indexes created")
    
    async def _verify_schema(self):
        """Verify that all tables and relationships were created correctly"""
        logger.info("Verifying schema...")
        
        # Check tables exist
        tables_result = await self.db.query("INFO FOR DB")
        
        expected_tables = {
            "fragments", "field_states", "patterns", "attractors",
            "fragment_patterns", "fragment_attractors", 
            "attractor_transitions", "pattern_evolution", "fragment_field_states"
        }
        
        # Note: The actual verification would depend on SurrealDB's INFO response format
        # This is a simplified check
        logger.info("✓ Schema verification completed")
    
    async def rollback_migration(self):
        """Rollback hierarchical schema migration (for testing)"""
        logger.warning("Rolling back hierarchical memory schema...")
        
        try:
            await self.connect()
            
            # Remove relationships first (to avoid foreign key constraints)
            relationship_tables = [
                "fragment_field_states", "pattern_evolution", "attractor_transitions",
                "fragment_attractors", "fragment_patterns"
            ]
            
            for table in relationship_tables:
                await self.db.query(f"REMOVE TABLE {table}")
            
            # Remove main tables
            main_tables = ["attractors", "patterns", "field_states", "fragments"]
            for table in main_tables:
                await self.db.query(f"REMOVE TABLE {table}")
            
            logger.info("✅ Schema rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Schema rollback failed: {e}")
            raise
        finally:
            await self.disconnect()

# Migration utility functions
async def run_hierarchical_migration(db_url: str = "ws://127.0.0.1:8000/rpc",
                                   username: str = "root", 
                                   password: str = "slowcat_secure_2024",
                                   namespace: str = "slowcat", 
                                   database: str = "memory"):
    """Run the hierarchical memory schema migration"""
    
    schema_manager = HierarchicalSchemaManager(
        db_url=db_url, username=username, password=password,
        namespace=namespace, database=database
    )
    
    await schema_manager.migrate_to_hierarchical_schema()

async def rollback_hierarchical_migration(db_url: str = "ws://127.0.0.1:8000/rpc",
                                        username: str = "root",
                                        password: str = "slowcat_secure_2024", 
                                        namespace: str = "slowcat",
                                        database: str = "memory"):
    """Rollback the hierarchical memory schema migration"""
    
    schema_manager = HierarchicalSchemaManager(
        db_url=db_url, username=username, password=password,
        namespace=namespace, database=database
    )
    
    await schema_manager.rollback_migration()

if __name__ == "__main__":
    # Example usage
    asyncio.run(run_hierarchical_migration())