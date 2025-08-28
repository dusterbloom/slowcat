#!/usr/bin/env python3
"""
Enhance Existing Graph Schema

Instead of creating new tables, this script enhances the existing graph schema:
- message table (for conversation content)
- session table (for session metadata) 
- user table (for speaker information)
- Graph relations: session→contains→message, message→mentions→concept, user→knows→concept

Enhancements:
1. Add missing fields to existing message table
2. Add missing fields to existing session table  
3. Ensure all graph relations are properly defined
4. Create indexes for performance
5. Maintain existing graph structure

This approach leverages the existing well-designed graph schema instead of duplicating it.
"""

import asyncio
from typing import Dict, List, Any
from surrealdb import AsyncSurreal
from loguru import logger


class ExistingSchemaEnhancer:
    """Enhance existing graph schema without creating duplicate tables"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
    
    async def enhance_message_table(self):
        """
        Enhance existing message table with additional fields needed for conversation management
        
        Existing fields from apply_graph_schema.py:
        - id, content, timestamp, sender_id, message_type
        
        New fields to add:
        - raw_content (original STT output before normalization)
        - role (user/assistant) 
        - facts_extracted (boolean flag)
        - word_count, content_length (for analysis)
        - agent_id (which assistant responded)
        - processing_meta (metadata about processing)
        """
        logger.info("📝 Enhancing existing message table...")
        
        await self.db.query("""
            -- Add new fields to existing message table
            DEFINE FIELD raw_content ON message TYPE option<string>;
            DEFINE FIELD role ON message TYPE string DEFAULT 'user' ASSERT $value INSIDE ['user', 'assistant'];
            DEFINE FIELD facts_extracted ON message TYPE bool DEFAULT false;
            DEFINE FIELD word_count ON message TYPE number VALUE array::len(string::split($parent.content, ' '));
            DEFINE FIELD content_length ON message TYPE number VALUE string::len($parent.content);
            DEFINE FIELD agent_id ON message TYPE option<string>;
            DEFINE FIELD processing_meta ON message TYPE object DEFAULT {};
            DEFINE FIELD embedding ON message TYPE option<array<number>>;
            DEFINE FIELD language ON message TYPE option<string>;
            
            -- Update existing fields to ensure proper types
            DEFINE FIELD content ON message TYPE string ASSERT $value != NONE;
            DEFINE FIELD timestamp ON message TYPE datetime VALUE time::now();
            DEFINE FIELD sender_id ON message TYPE string ASSERT $value != NONE;
            DEFINE FIELD message_type ON message TYPE string DEFAULT 'conversation';
        """)
        
        logger.debug("✅ Enhanced message table with conversation management fields")
    
    async def enhance_session_table(self):
        """
        Enhance existing session table with additional session management fields
        
        Existing fields from apply_graph_schema.py:
        - id, user_id, start_time, end_time, summary, turn_count
        
        New fields to add:
        - speaker_id (link to actual speaker, may differ from user_id)
        - last_activity (when session was last active)
        - status (active/completed/abandoned)
        - facts_extracted (count of facts extracted in session)  
        - quality_score (session quality metric)
        - duration_s (calculated duration)
        - total_words, total_chars (content metrics)
        - keywords, topics (analysis results)
        """
        logger.info("🗂️ Enhancing existing session table...")
        
        await self.db.query("""
            -- Add new fields to existing session table
            DEFINE FIELD speaker_id ON session TYPE string;
            DEFINE FIELD last_activity ON session TYPE datetime VALUE time::now();
            DEFINE FIELD status ON session TYPE string DEFAULT 'active' ASSERT $value INSIDE ['active', 'completed', 'abandoned'];
            DEFINE FIELD facts_extracted ON session TYPE number DEFAULT 0;
            DEFINE FIELD quality_score ON session TYPE number DEFAULT 0.0;
            DEFINE FIELD duration_s ON session TYPE number DEFAULT 0;
            DEFINE FIELD total_words ON session TYPE number DEFAULT 0;
            DEFINE FIELD total_chars ON session TYPE number DEFAULT 0;
            DEFINE FIELD keywords ON session TYPE array DEFAULT [];
            DEFINE FIELD topics ON session TYPE array DEFAULT [];
            DEFINE FIELD agent_id ON session TYPE option<string>;
            
            -- Update existing fields to ensure proper constraints
            DEFINE FIELD user_id ON session TYPE string ASSERT $value != NONE;
            DEFINE FIELD start_time ON session TYPE datetime VALUE time::now();
            DEFINE FIELD turn_count ON session TYPE number DEFAULT 0;
        """)
        
        logger.debug("✅ Enhanced session table with session management fields")
    
    async def enhance_user_table(self):
        """
        Enhance existing user table for speaker management
        
        Add fields for speaker recognition and session tracking
        """
        logger.info("👤 Enhancing existing user table...")
        
        await self.db.query("""
            -- Add speaker management fields to existing user table
            DEFINE FIELD speaker_profile ON user TYPE option<object>;
            DEFINE FIELD total_sessions ON user TYPE number DEFAULT 0;
            DEFINE FIELD first_seen ON user TYPE datetime VALUE time::now();
            DEFINE FIELD last_interaction ON user TYPE datetime VALUE time::now();
            DEFINE FIELD preferred_name ON user TYPE option<string>;
            DEFINE FIELD voice_id ON user TYPE option<string>;
            
            -- Ensure existing id field
            DEFINE FIELD id ON user TYPE string ASSERT $value != NONE;
        """)
        
        logger.debug("✅ Enhanced user table with speaker management")
    
    async def ensure_graph_relations(self):
        """
        Ensure all graph relations are properly defined with enhanced fields
        """
        logger.info("🔗 Ensuring graph relations are properly defined...")
        
        await self.db.query("""
            -- Enhance contains relation (session → message)
            DEFINE TABLE contains TYPE RELATION IN session OUT message SCHEMAFULL;
            DEFINE FIELD message_order ON contains TYPE number DEFAULT 0;  -- Order within session
            DEFINE FIELD created_at ON contains TYPE datetime VALUE time::now();
            
            -- Enhance mentions relation (message → concept) 
            DEFINE TABLE mentions TYPE RELATION IN message OUT concept SCHEMAFULL;
            DEFINE FIELD confidence ON mentions TYPE number DEFAULT 1.0;
            DEFINE FIELD mention_type ON mentions TYPE string DEFAULT 'reference';
            DEFINE FIELD created_at ON mentions TYPE datetime VALUE time::now();
            
            -- Enhance knows relation (user → concept)
            DEFINE TABLE knows TYPE RELATION IN user OUT concept SCHEMAFULL;
            DEFINE FIELD strength ON knows TYPE number DEFAULT 1.0;
            DEFINE FIELD first_mentioned ON knows TYPE datetime VALUE time::now();
            DEFINE FIELD last_mentioned ON knows TYPE datetime VALUE time::now();
            DEFINE FIELD mention_count ON knows TYPE number DEFAULT 1;
            
            -- Enhance reflects relation (session → thought)
            DEFINE TABLE reflects TYPE RELATION IN session OUT thought SCHEMAFULL;
            DEFINE FIELD reflection_type ON reflects TYPE string DEFAULT 'summary';
            DEFINE FIELD confidence ON reflects TYPE number DEFAULT 1.0;
            DEFINE FIELD created_at ON reflects TYPE datetime VALUE time::now();
        """)
        
        logger.debug("✅ Enhanced graph relations with metadata")
    
    async def create_fact_extraction_table(self):
        """
        Create fact_extraction table to track processing of messages
        This links to the message table via graph relations
        """
        logger.info("🔍 Creating fact extraction tracking table...")
        
        await self.db.query("""
            DEFINE TABLE fact_extraction SCHEMAFULL;
            
            -- Link to message that was processed
            DEFINE FIELD message_id ON fact_extraction TYPE record<message>;
            DEFINE FIELD session_id ON fact_extraction TYPE record<session>;
            
            -- Extraction metadata
            DEFINE FIELD extracted_at ON fact_extraction TYPE datetime VALUE time::now();
            DEFINE FIELD extraction_method ON fact_extraction TYPE string DEFAULT 'spacy';
            DEFINE FIELD facts_found ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD entities_found ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD relationships_found ON fact_extraction TYPE number DEFAULT 0;
            
            -- Quality metrics
            DEFINE FIELD confidence_score ON fact_extraction TYPE number DEFAULT 0.0;
            DEFINE FIELD processing_time_ms ON fact_extraction TYPE number DEFAULT 0;
            DEFINE FIELD extraction_version ON fact_extraction TYPE string DEFAULT '1.0';
            
            -- Source analysis
            DEFINE FIELD source_content_length ON fact_extraction TYPE number;
            DEFINE FIELD source_word_count ON fact_extraction TYPE number;
            DEFINE FIELD source_language ON fact_extraction TYPE option<string>;
            
            -- Categories found
            DEFINE FIELD entity_types ON fact_extraction TYPE array DEFAULT [];
            DEFINE FIELD relationship_types ON fact_extraction TYPE array DEFAULT [];
            
            -- Processing context
            DEFINE FIELD agent_id ON fact_extraction TYPE option<string>;
            DEFINE FIELD processing_context ON fact_extraction TYPE object DEFAULT {};
        """)
        
        logger.debug("✅ Created fact_extraction tracking table")
    
    async def create_performance_indexes(self):
        """
        Create indexes for optimal query performance on the enhanced graph schema
        """
        logger.info("📈 Creating performance indexes for enhanced schema...")
        
        # Message table indexes
        await self.db.query("""
            -- Message lookup patterns
            DEFINE INDEX idx_message_sender_time ON message COLUMNS sender_id, timestamp;
            DEFINE INDEX idx_message_role_time ON message COLUMNS role, timestamp;  
            DEFINE INDEX idx_message_type ON message COLUMNS message_type;
            DEFINE INDEX idx_message_facts ON message COLUMNS facts_extracted;
            
            -- Content analysis indexes
            DEFINE INDEX idx_message_content_fts ON message COLUMNS content SEARCH ANALYZER ascii BM25;
            DEFINE INDEX idx_message_content_length ON message COLUMNS content_length;
            DEFINE INDEX idx_message_word_count ON message COLUMNS word_count;
            
            -- Agent and processing indexes
            DEFINE INDEX idx_message_agent ON message COLUMNS agent_id;
        """)
        
        # Session table indexes  
        await self.db.query("""
            -- Session lookup patterns
            DEFINE INDEX idx_session_user_time ON session COLUMNS user_id, start_time;
            DEFINE INDEX idx_session_speaker_time ON session COLUMNS speaker_id, start_time;
            DEFINE INDEX idx_session_status ON session COLUMNS status, last_activity;
            DEFINE INDEX idx_session_activity ON session COLUMNS last_activity;
            
            -- Session analysis indexes
            DEFINE INDEX idx_session_quality ON session COLUMNS quality_score;
            DEFINE INDEX idx_session_facts ON session COLUMNS facts_extracted;
            DEFINE INDEX idx_session_turns ON session COLUMNS turn_count;
        """)
        
        # User table indexes
        await self.db.query("""
            -- User/speaker lookup
            DEFINE INDEX idx_user_last_interaction ON user COLUMNS last_interaction;
            DEFINE INDEX idx_user_sessions ON user COLUMNS total_sessions;
            DEFINE INDEX idx_user_voice ON user COLUMNS voice_id;
        """)
        
        # Fact extraction indexes
        await self.db.query("""
            -- Fact extraction tracking
            DEFINE INDEX idx_fact_message ON fact_extraction COLUMNS message_id;
            DEFINE INDEX idx_fact_session ON fact_extraction COLUMNS session_id;
            DEFINE INDEX idx_fact_extracted_at ON fact_extraction COLUMNS extracted_at;
            DEFINE INDEX idx_fact_method ON fact_extraction COLUMNS extraction_method;
            DEFINE INDEX idx_fact_confidence ON fact_extraction COLUMNS confidence_score;
            DEFINE INDEX idx_fact_counts ON fact_extraction COLUMNS facts_found;
        """)
        
        logger.debug("✅ Created performance indexes for enhanced graph schema")
    
    async def validate_enhanced_schema(self) -> Dict[str, Any]:
        """
        Validate the enhanced schema
        """
        logger.info("🔍 Validating enhanced graph schema...")
        
        validation_results = {}
        
        try:
            # Check enhanced tables
            tables_to_check = ['message', 'session', 'user', 'fact_extraction']
            for table in tables_to_check:
                try:
                    info = await self.db.query(f'INFO FOR TABLE {table}')
                    validation_results[f'{table}_enhanced'] = len(info) > 0
                except Exception as e:
                    logger.error(f"Failed to validate {table}: {e}")
                    validation_results[f'{table}_enhanced'] = False
            
            # Check graph relations
            relations_to_check = ['contains', 'mentions', 'knows', 'reflects']
            for relation in relations_to_check:
                try:
                    info = await self.db.query(f'INFO FOR TABLE {relation}')
                    validation_results[f'{relation}_relation'] = len(info) > 0
                except Exception as e:
                    logger.error(f"Failed to validate relation {relation}: {e}")
                    validation_results[f'{relation}_relation'] = False
            
            # Check indexes
            try:
                db_info = await self.db.query('INFO FOR DATABASE')
                validation_results['indexes_created'] = len(db_info) > 0
            except Exception as e:
                logger.error(f"Failed to validate indexes: {e}")
                validation_results['indexes_created'] = False
            
            success_count = sum(validation_results.values())
            total_count = len(validation_results)
            
            logger.info(f"📊 Enhanced schema validation: {success_count}/{total_count} components validated")
            
            if success_count == total_count:
                logger.info("🎉 Enhanced graph schema validated successfully!")
            else:
                failed = [k for k, v in validation_results.items() if not v]
                logger.warning(f"⚠️ Validation failed for: {failed}")
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    async def run_full_enhancement(self) -> Dict[str, Any]:
        """
        Run the complete schema enhancement process
        """
        logger.info("🚀 Starting graph schema enhancement...")
        
        try:
            # Enhance existing tables
            await self.enhance_message_table()
            await self.enhance_session_table()  
            await self.enhance_user_table()
            
            # Ensure graph relations
            await self.ensure_graph_relations()
            
            # Create supporting tables
            await self.create_fact_extraction_table()
            
            # Create indexes
            await self.create_performance_indexes()
            
            # Validate everything
            validation_results = await self.validate_enhanced_schema()
            
            logger.info("✅ Graph schema enhancement completed!")
            
            return {
                'status': 'success',
                'validation': validation_results,
                'enhanced_tables': ['message', 'session', 'user'],
                'new_tables': ['fact_extraction'],
                'enhanced_relations': ['contains', 'mentions', 'knows', 'reflects']
            }
            
        except Exception as e:
            logger.error(f"Schema enhancement failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


async def main():
    """
    Main function to enhance the existing graph schema
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Enhance schema
    enhancer = ExistingSchemaEnhancer(db)
    results = await enhancer.run_full_enhancement()
    
    # Report results
    if results['status'] == 'success':
        logger.info("🎉 Graph schema successfully enhanced!")
        logger.info(f"📊 Enhanced tables: {results['enhanced_tables']}")
        logger.info(f"📊 New tables: {results['new_tables']}")
        logger.info(f"📊 Enhanced relations: {results['enhanced_relations']}")
    else:
        logger.error(f"❌ Schema enhancement failed: {results['error']}")
    
    await db.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())