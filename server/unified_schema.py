#!/usr/bin/env python3
"""
Unified Database Schema for Slowcat Memory System

This module defines the new unified schema that consolidates the current
fragmented table structure into a coherent, efficient design.

Current Problem:
- Multiple overlapping session tables (sessions, session, session_summary)
- Inconsistent session tracking and metadata
- Fragmented conversation storage without clear relationships
- No proper session boundary management

Unified Solution:
- Single conversation table for all user/assistant messages
- Consolidated session_meta table for session tracking
- Clear relationships and efficient indexes
- Backward compatibility during migration
"""

from surrealdb import AsyncSurreal
from loguru import logger
from typing import Dict, List, Optional
import asyncio


class UnifiedSchema:
    """Unified database schema definition and management"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
    
    async def create_unified_schema(self):
        """
        Create the new unified schema tables
        
        This creates new tables alongside existing ones to allow for
        gradual migration without data loss.
        """
        logger.info("🏗️ Creating unified database schema...")
        
        # Core conversation table - replaces tape table
        await self._create_conversation_table()
        
        # Session metadata table - replaces sessions + session_summary tables
        await self._create_session_meta_table()
        
        # Fact extraction tracking table - enhances existing concept table
        await self._create_fact_tracking_table()
        
        # Create indexes for performance
        await self._create_indexes()
        
        logger.info("✅ Unified schema created successfully")
    
    async def _create_conversation_table(self):
        """
        Create unified conversation table
        
        This table replaces the current 'tape' table with enhanced structure:
        - Better session tracking
        - Raw + normalized content storage  
        - Fact extraction tracking
        - Proper indexes for performance
        """
        await self.db.query("""
            DEFINE TABLE conversation SCHEMAFULL;
            
            -- Core conversation fields
            DEFINE FIELD ts ON conversation TYPE datetime VALUE time::now();
            DEFINE FIELD speaker_id ON conversation TYPE string ASSERT $value != NONE;
            DEFINE FIELD role ON conversation TYPE string ASSERT $value INSIDE ['user', 'assistant'];
            DEFINE FIELD content ON conversation TYPE string;  -- Normalized content
            DEFINE FIELD raw_content ON conversation TYPE option<string>;  -- Original STT output
            
            -- Session tracking fields
            DEFINE FIELD session_id ON conversation TYPE string ASSERT $value != NONE;
            DEFINE FIELD turn_number ON conversation TYPE number DEFAULT 0;  -- Turn within session
            DEFINE FIELD session_start ON conversation TYPE datetime VALUE time::now();
            
            -- Agent and processing metadata
            DEFINE FIELD agent_id ON conversation TYPE option<string>;
            DEFINE FIELD embedding ON conversation TYPE option<array<number>>;
            DEFINE FIELD facts_extracted ON conversation TYPE bool DEFAULT false;
            DEFINE FIELD processing_meta ON conversation TYPE object DEFAULT {};
            
            -- Content analysis fields
            DEFINE FIELD content_length ON conversation TYPE number VALUE string::len($parent.content);
            DEFINE FIELD word_count ON conversation TYPE number VALUE array::len(string::split($parent.content, ' '));
            DEFINE FIELD language ON conversation TYPE option<string>;
            
            -- Audit fields
            DEFINE FIELD created_at ON conversation TYPE datetime VALUE time::now();
            DEFINE FIELD updated_at ON conversation TYPE datetime VALUE time::now();
        """)
        
        logger.debug("📊 Created conversation table")
    
    async def _create_session_meta_table(self):
        """
        Create unified session metadata table
        
        This table consolidates sessions, session, and session_summary into
        one coherent structure with all necessary session tracking fields.
        """
        await self.db.query("""
            DEFINE TABLE session_meta SCHEMAFULL;
            
            -- Primary session identification
            DEFINE FIELD session_id ON session_meta TYPE string ASSERT $value != NONE;
            DEFINE FIELD speaker_id ON session_meta TYPE string ASSERT $value != NONE;
            
            -- Session timing and boundaries
            DEFINE FIELD start_time ON session_meta TYPE datetime ASSERT $value != NONE;
            DEFINE FIELD end_time ON session_meta TYPE option<datetime>;
            DEFINE FIELD duration_s ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD last_activity ON session_meta TYPE datetime VALUE time::now();
            
            -- Session content metrics
            DEFINE FIELD turn_count ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD user_turns ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD assistant_turns ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD total_words ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD total_chars ON session_meta TYPE number DEFAULT 0;
            
            -- Session analysis and summaries
            DEFINE FIELD summary ON session_meta TYPE option<string>;
            DEFINE FIELD keywords ON session_meta TYPE array DEFAULT [];
            DEFINE FIELD topics ON session_meta TYPE array DEFAULT [];
            DEFINE FIELD facts_extracted ON session_meta TYPE number DEFAULT 0;
            DEFINE FIELD quality_score ON session_meta TYPE number DEFAULT 0.0;
            
            -- Session state management
            DEFINE FIELD status ON session_meta TYPE string DEFAULT 'active' ASSERT $value INSIDE ['active', 'completed', 'abandoned'];
            DEFINE FIELD agent_id ON session_meta TYPE option<string>;
            
            -- Speaker context
            DEFINE FIELD speaker_session_number ON session_meta TYPE number DEFAULT 1;  -- Nth session for this speaker
            DEFINE FIELD speaker_total_sessions ON session_meta TYPE number DEFAULT 1;  -- Running count for speaker
            
            -- Audit fields
            DEFINE FIELD created_at ON session_meta TYPE datetime VALUE time::now();
            DEFINE FIELD updated_at ON session_meta TYPE datetime VALUE time::now();
        """)
        
        logger.debug("🗂️ Created session_meta table")
    
    async def _create_fact_tracking_table(self):
        """
        Create fact extraction tracking table
        
        This table tracks which conversation entries have had facts extracted
        and provides detailed extraction metadata.
        """
        await self.db.query("""
            DEFINE TABLE fact_extraction SCHEMAFULL;
            
            -- Link to conversation entry
            DEFINE FIELD conversation_id ON fact_extraction TYPE record<conversation>;
            DEFINE FIELD session_id ON fact_extraction TYPE string;
            
            -- Extraction metadata
            DEFINE FIELD extracted_at ON fact_extraction TYPE datetime VALUE time::now();
            DEFINE FIELD extraction_method ON fact_extraction TYPE string DEFAULT 'spacy';
            DEFINE FIELD facts_found ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD entities_found ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD relationships_found ON fact_extraction TYPE number DEFAULT 0;
            
            -- Extraction quality and confidence
            DEFINE FIELD confidence_score ON fact_extraction TYPE number DEFAULT 0.0;
            DEFINE FIELD processing_time_ms ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD extraction_version ON fact_extraction TYPE string DEFAULT '1.0';
            
            -- Source content analysis
            DEFINE FIELD source_content_length ON fact_extraction TYPE number;
            DEFINE FIELD source_word_count ON fact_extraction TYPE number;
            DEFINE FIELD source_language ON fact_extraction TYPE option<string>;
            
            -- Fact categories found
            DEFINE FIELD entity_types ON fact_extraction TYPE array DEFAULT [];  -- ['PERSON', 'ORG', 'GPE']
            DEFINE FIELD relationship_types ON fact_extraction TYPE array DEFAULT [];  -- ['knows', 'mentions']
            
            -- Processing metadata
            DEFINE FIELD agent_id ON fact_extraction TYPE option<string>;
            DEFINE FIELD processing_context ON fact_extraction TYPE object DEFAULT {};
        """)
        
        logger.debug("🔍 Created fact_extraction table")
    
    async def _create_indexes(self):
        """
        Create performance indexes for the unified schema
        
        These indexes optimize common query patterns:
        - Session-based queries
        - Speaker-based queries  
        - Time-range queries
        - Full-text search
        - Fact extraction tracking
        """
        logger.info("📈 Creating performance indexes...")
        
        # Conversation table indexes
        await self.db.query("""
            -- Primary lookup patterns
            DEFINE INDEX idx_conv_speaker_time ON conversation COLUMNS speaker_id, ts;
            DEFINE INDEX idx_conv_session ON conversation COLUMNS session_id, turn_number;
            DEFINE INDEX idx_conv_role_time ON conversation COLUMNS role, ts;
            
            -- Content search
            DEFINE INDEX idx_conv_content_fts ON conversation COLUMNS content SEARCH ANALYZER ascii BM25;
            DEFINE INDEX idx_conv_content_length ON conversation COLUMNS content_length;
            
            -- Processing state
            DEFINE INDEX idx_conv_facts_extracted ON conversation COLUMNS facts_extracted;
            DEFINE INDEX idx_conv_agent ON conversation COLUMNS agent_id;
        """)
        
        # Session metadata indexes
        await self.db.query("""
            -- Primary session lookups
            DEFINE INDEX idx_session_speaker ON session_meta COLUMNS speaker_id, start_time;
            DEFINE INDEX idx_session_time_range ON session_meta COLUMNS start_time, end_time;
            DEFINE INDEX idx_session_status ON session_meta COLUMNS status, last_activity;
            
            -- Session analysis
            DEFINE INDEX idx_session_quality ON session_meta COLUMNS quality_score DESC;
            DEFINE INDEX idx_session_facts ON session_meta COLUMNS facts_extracted DESC;
            DEFINE INDEX idx_session_turns ON session_meta COLUMNS turn_count DESC;
        """)
        
        # Fact extraction tracking indexes  
        await self.db.query("""
            -- Extraction tracking
            DEFINE INDEX idx_fact_conv ON fact_extraction COLUMNS conversation_id;
            DEFINE INDEX idx_fact_session ON fact_extraction COLUMNS session_id, extracted_at;
            DEFINE INDEX idx_fact_method ON fact_extraction COLUMNS extraction_method, extracted_at;
            
            -- Extraction quality analysis
            DEFINE INDEX idx_fact_confidence ON fact_extraction COLUMNS confidence_score DESC;
            DEFINE INDEX idx_fact_counts ON fact_extraction COLUMNS facts_found DESC, entities_found DESC;
        """)
        
        logger.debug("✅ Performance indexes created")
    
    async def validate_schema(self) -> Dict[str, bool]:
        """
        Validate that the unified schema was created correctly
        
        Returns:
            Dict mapping table names to creation success status
        """
        logger.info("🔍 Validating unified schema creation...")
        
        validation_results = {}
        expected_tables = ['conversation', 'session_meta', 'fact_extraction']
        
        for table in expected_tables:
            try:
                # Try to query table info
                result = await self.db.query(f'INFO FOR TABLE {table}')
                validation_results[table] = len(result) > 0
                
                if validation_results[table]:
                    logger.debug(f"✅ Table {table} validated successfully")
                else:
                    logger.error(f"❌ Table {table} validation failed - no schema info")
                    
            except Exception as e:
                logger.error(f"❌ Table {table} validation failed: {e}")
                validation_results[table] = False
        
        # Validate indexes exist
        try:
            indexes_result = await self.db.query('INFO FOR DATABASE')
            validation_results['indexes'] = len(indexes_result) > 0
        except Exception as e:
            logger.error(f"❌ Index validation failed: {e}")
            validation_results['indexes'] = False
        
        success_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        logger.info(f"📊 Schema validation: {success_count}/{total_count} components validated")
        
        return validation_results


async def create_unified_schema_main():
    """
    Main function to create the unified schema
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Create unified schema
    schema_manager = UnifiedSchema(db)
    await schema_manager.create_unified_schema()
    
    # Validate creation
    validation_results = await schema_manager.validate_schema()
    
    # Report results
    if all(validation_results.values()):
        logger.info("🎉 Unified schema created and validated successfully!")
    else:
        failed_components = [k for k, v in validation_results.items() if not v]
        logger.error(f"❌ Schema creation failed for: {failed_components}")
    
    await db.close()
    
    return validation_results


if __name__ == "__main__":
    asyncio.run(create_unified_schema_main())