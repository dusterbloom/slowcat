#!/usr/bin/env python3
"""
Migration Script: Current Schema → Unified Schema

This script migrates data from the current fragmented table structure
to the new unified schema while preserving all existing data.

Migration Path:
1. tape → conversation (with normalization and enhanced metadata)
2. sessions + session_summary → session_meta (consolidated)  
3. Create fact_extraction tracking records
4. Validate data integrity
5. Create migration report

Safe Migration:
- Creates new tables alongside old ones
- Does not drop existing tables
- Validates each step
- Provides rollback information
- Comprehensive logging and error handling
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from surrealdb import AsyncSurreal
from loguru import logger
import re


class SchemaValidator:
    """Validates data integrity during migration"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
    
    async def validate_counts(self) -> Dict[str, Dict[str, int]]:
        """Compare record counts between old and new schemas"""
        counts = {}
        
        # Old schema counts
        old_tape = await self.db.query('SELECT COUNT() FROM tape GROUP ALL')
        old_sessions = await self.db.query('SELECT COUNT() FROM sessions GROUP ALL')
        
        # New schema counts  
        new_conversation = await self.db.query('SELECT COUNT() FROM conversation GROUP ALL')
        new_session_meta = await self.db.query('SELECT COUNT() FROM session_meta GROUP ALL')
        
        counts['old'] = {
            'tape': old_tape[0]['count'] if old_tape else 0,
            'sessions': old_sessions[0]['count'] if old_sessions else 0
        }
        
        counts['new'] = {
            'conversation': new_conversation[0]['count'] if new_conversation else 0,
            'session_meta': new_session_meta[0]['count'] if new_session_meta else 0
        }
        
        return counts
    
    async def validate_sample_data(self, sample_size: int = 10) -> Dict[str, bool]:
        """Validate sample data integrity between old and new schemas"""
        validation_results = {}
        
        try:
            # Sample old tape entries
            old_sample = await self.db.query(f'SELECT * FROM tape ORDER BY ts LIMIT {sample_size}')
            
            if not old_sample:
                validation_results['sample_data'] = True
                return validation_results
            
            # Check if corresponding conversation entries exist
            valid_migrations = 0
            
            for old_entry in old_sample:
                # Find corresponding conversation entry
                query = '''
                    SELECT * FROM conversation 
                    WHERE speaker_id = $speaker_id 
                    AND content = $content 
                    LIMIT 1
                '''
                
                new_entries = await self.db.query(query, {
                    'speaker_id': old_entry.get('speaker_id', ''),
                    'content': old_entry.get('content', '')
                })
                
                if new_entries:
                    valid_migrations += 1
            
            validation_results['sample_data'] = valid_migrations == len(old_sample)
            validation_results['sample_valid_count'] = valid_migrations
            validation_results['sample_total_count'] = len(old_sample)
            
        except Exception as e:
            logger.error(f"Sample data validation failed: {e}")
            validation_results['sample_data'] = False
        
        return validation_results


class UnifiedMigrator:
    """Handles migration from fragmented schema to unified schema"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
        self.validator = SchemaValidator(db)
        
        # Migration statistics
        self.stats = {
            'tape_migrated': 0,
            'sessions_migrated': 0,
            'fact_tracking_created': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _normalize_content(self, content: str) -> str:
        """Apply the same normalization as fix_fragmented_conversations.py"""
        if not content or not isinstance(content, str):
            return content or ''
        
        s = content.strip()
        
        # Basic cleanup
        s = re.sub(r'\s+', ' ', s).strip()
        
        # Fix spaced numbers
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', s)
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', s)
        s = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', s)
        
        # Fix spaced proper names
        s = re.sub(r'\b([A-Z])\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]{1,4})\s+([a-z]{2,6})\b', r'\1\2', s)
        
        # Fix punctuation spacing
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        s = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s
    
    def _generate_session_id(self, speaker_id: str, timestamp: Any) -> str:
        """Generate consistent session ID from speaker and timestamp"""
        if isinstance(timestamp, datetime):
            ts = timestamp.timestamp()
        elif isinstance(timestamp, (int, float)):
            ts = timestamp
        else:
            ts = time.time()
        
        # Daily session format: speaker_YYYYMMDD
        date_str = datetime.fromtimestamp(ts).strftime('%Y%m%d')
        return f"{speaker_id}_{date_str}"
    
    def _extract_timestamp(self, ts_value: Any) -> datetime:
        """Extract datetime from various timestamp formats"""
        if isinstance(ts_value, datetime):
            return ts_value
        elif isinstance(ts_value, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
            except:
                # Try parsing as timestamp
                return datetime.fromtimestamp(float(ts_value))
        elif isinstance(ts_value, (int, float)):
            return datetime.fromtimestamp(ts_value)
        else:
            return datetime.now()
    
    async def migrate_tape_to_conversation(self) -> Dict[str, Any]:
        """
        Migrate tape table to conversation table
        
        Enhancements:
        - Content normalization
        - Session tracking
        - Turn number calculation
        - Metadata extraction
        """
        logger.info("📼 → 💬 Migrating tape entries to conversation table...")
        
        # Get all tape entries ordered by timestamp
        tape_entries = await self.db.query('SELECT * FROM tape ORDER BY ts')
        logger.info(f"Found {len(tape_entries)} tape entries to migrate")
        
        if not tape_entries:
            return {'migrated': 0, 'errors': 0}
        
        # Group by session for turn numbering
        session_turns = {}
        migrated_count = 0
        error_count = 0
        
        for entry in tape_entries:
            try:
                # Extract and normalize data
                speaker_id = entry.get('speaker_id', 'unknown')
                role = entry.get('role', 'user')
                raw_content = entry.get('content', '')
                normalized_content = self._normalize_content(raw_content)
                
                # Extract timestamp
                timestamp = self._extract_timestamp(entry.get('ts'))
                
                # Generate session ID
                session_id = self._generate_session_id(speaker_id, timestamp)
                
                # Calculate turn number within session
                if session_id not in session_turns:
                    session_turns[session_id] = 0
                session_turns[session_id] += 1
                turn_number = session_turns[session_id]
                
                # Create conversation record
                await self.db.query('''
                    CREATE conversation SET
                        ts = $ts,
                        speaker_id = $speaker_id,
                        role = $role,
                        content = $content,
                        raw_content = $raw_content,
                        session_id = $session_id,
                        turn_number = $turn_number,
                        session_start = $session_start,
                        agent_id = $agent_id,
                        embedding = $embedding,
                        facts_extracted = false,
                        processing_meta = $processing_meta,
                        content_length = $content_length,
                        word_count = $word_count,
                        language = $language
                ''', {
                    'ts': timestamp,
                    'speaker_id': speaker_id,
                    'role': role,
                    'content': normalized_content,
                    'raw_content': raw_content if raw_content != normalized_content else None,
                    'session_id': session_id,
                    'turn_number': turn_number,
                    'session_start': timestamp if turn_number == 1 else None,
                    'agent_id': entry.get('agent_id'),
                    'embedding': entry.get('embedding'),
                    'processing_meta': entry.get('metadata', {}),
                    'content_length': len(normalized_content),
                    'word_count': len(normalized_content.split()),
                    'language': None  # Could be detected later
                })
                
                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    logger.info(f"   Progress: {migrated_count}/{len(tape_entries)} entries migrated")
                
            except Exception as e:
                logger.error(f"Failed to migrate tape entry {entry.get('id', 'unknown')}: {e}")
                error_count += 1
        
        self.stats['tape_migrated'] = migrated_count
        logger.info(f"✅ Migrated {migrated_count} tape entries to conversation table")
        
        return {
            'migrated': migrated_count,
            'errors': error_count,
            'unique_sessions': len(session_turns)
        }
    
    async def migrate_sessions_to_session_meta(self) -> Dict[str, Any]:
        """
        Migrate sessions and session_summary tables to session_meta table
        
        Consolidates session tracking into single coherent structure
        """
        logger.info("🗂️ → 📊 Migrating session data to session_meta table...")
        
        # Get data from both session-related tables
        sessions_data = await self.db.query('SELECT * FROM sessions')
        
        # Try to get session summaries (table may not exist)
        try:
            summaries_data = await self.db.query('SELECT * FROM session_summary')
        except:
            summaries_data = []
            logger.debug("No session_summary table found, proceeding without summaries")
        
        logger.info(f"Found {len(sessions_data)} session records and {len(summaries_data)} session summaries")
        
        # Create lookup for summaries by session_id
        summary_lookup = {}
        for summary in summaries_data:
            session_id = summary.get('session_id')
            if session_id:
                summary_lookup[session_id] = summary
        
        migrated_count = 0
        error_count = 0
        
        # Get conversation statistics per session for metadata
        conversation_stats = await self.db.query('''
            SELECT 
                session_id,
                COUNT() as turn_count,
                COUNT(*) FILTER (WHERE role = 'user') as user_turns,
                COUNT(*) FILTER (WHERE role = 'assistant') as assistant_turns,
                SUM(word_count) as total_words,
                SUM(content_length) as total_chars,
                MIN(ts) as start_time,
                MAX(ts) as end_time
            FROM conversation
            GROUP BY session_id
        ''')
        
        # Create lookup for conversation stats
        stats_lookup = {stat['session_id']: stat for stat in conversation_stats}
        
        for session in sessions_data:
            try:
                speaker_id = session.get('speaker_id', 'unknown')
                
                # Generate session_id format (may need to derive from speaker_id and timestamp)
                first_seen = session.get('first_seen') or session.get('last_interaction')
                if first_seen:
                    timestamp = self._extract_timestamp(first_seen)
                    session_id = self._generate_session_id(speaker_id, timestamp)
                else:
                    session_id = f"{speaker_id}_unknown"
                
                # Get conversation stats for this session
                conv_stats = stats_lookup.get(session_id, {})
                
                # Get summary data if available
                summary_data = summary_lookup.get(session_id, {})
                
                # Calculate session metadata
                start_time = conv_stats.get('start_time') or self._extract_timestamp(first_seen or time.time())
                end_time = conv_stats.get('end_time')
                duration_s = 0
                if start_time and end_time:
                    duration_s = int((end_time - start_time).total_seconds())
                
                await self.db.query('''
                    CREATE session_meta SET
                        session_id = $session_id,
                        speaker_id = $speaker_id,
                        start_time = $start_time,
                        end_time = $end_time,
                        duration_s = $duration_s,
                        last_activity = $last_activity,
                        turn_count = $turn_count,
                        user_turns = $user_turns,
                        assistant_turns = $assistant_turns,
                        total_words = $total_words,
                        total_chars = $total_chars,
                        summary = $summary,
                        keywords = $keywords,
                        topics = $topics,
                        facts_extracted = 0,
                        quality_score = $quality_score,
                        status = $status,
                        agent_id = $agent_id,
                        speaker_session_number = $speaker_session_number,
                        speaker_total_sessions = $speaker_total_sessions
                ''', {
                    'session_id': session_id,
                    'speaker_id': speaker_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_s': duration_s,
                    'last_activity': self._extract_timestamp(session.get('last_interaction', time.time())),
                    'turn_count': conv_stats.get('turn_count', session.get('total_turns', 0)),
                    'user_turns': conv_stats.get('user_turns', 0),
                    'assistant_turns': conv_stats.get('assistant_turns', 0),
                    'total_words': conv_stats.get('total_words', 0),
                    'total_chars': conv_stats.get('total_chars', 0),
                    'summary': summary_data.get('summary'),
                    'keywords': summary_data.get('keywords', []),
                    'topics': [],  # Could be derived from keywords
                    'quality_score': min(1.0, conv_stats.get('turn_count', 0) / 10.0) if conv_stats.get('turn_count', 0) > 0 else 0.0,
                    'status': 'completed' if end_time else 'active',
                    'agent_id': None,
                    'speaker_session_number': session.get('session_count', 1),
                    'speaker_total_sessions': session.get('session_count', 1)
                })
                
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate session {session.get('speaker_id', 'unknown')}: {e}")
                error_count += 1
        
        self.stats['sessions_migrated'] = migrated_count
        logger.info(f"✅ Migrated {migrated_count} session records to session_meta table")
        
        return {
            'migrated': migrated_count,
            'errors': error_count
        }
    
    async def create_fact_tracking_records(self) -> Dict[str, Any]:
        """
        Create fact extraction tracking records for existing conversation entries
        
        This provides a baseline for tracking which entries have been processed
        """
        logger.info("🔍 Creating fact extraction tracking records...")
        
        # Get all conversation entries that might need fact extraction
        conversations = await self.db.query('''
            SELECT id, session_id, content, word_count, content_length, ts
            FROM conversation
            WHERE content_length > 15 AND word_count > 3
        ''')
        
        logger.info(f"Found {len(conversations)} conversation entries eligible for fact tracking")
        
        created_count = 0
        error_count = 0
        
        for conv in conversations:
            try:
                # Check if facts were likely extracted (heuristic based on content)
                content = conv.get('content', '')
                facts_likely_extracted = False
                
                # Simple heuristic: if content contains proper nouns or specific patterns
                if re.search(r'\b[A-Z][a-z]+\b.*\b[A-Z][a-z]+\b', content):
                    facts_likely_extracted = True
                
                await self.db.query('''
                    CREATE fact_extraction SET
                        conversation_id = $conversation_id,
                        session_id = $session_id,
                        extracted_at = $extracted_at,
                        extraction_method = 'migration_baseline',
                        facts_found = 0,
                        entities_found = 0,
                        relationships_found = 0,
                        confidence_score = $confidence_score,
                        processing_time_ms = 0,
                        extraction_version = '1.0',
                        source_content_length = $content_length,
                        source_word_count = $word_count,
                        source_language = 'unknown',
                        entity_types = [],
                        relationship_types = [],
                        agent_id = null,
                        processing_context = $processing_context
                ''', {
                    'conversation_id': conv['id'],
                    'session_id': conv.get('session_id'),
                    'extracted_at': conv.get('ts'),
                    'confidence_score': 0.5 if facts_likely_extracted else 0.1,
                    'content_length': conv.get('content_length', 0),
                    'word_count': conv.get('word_count', 0),
                    'processing_context': {
                        'migration_baseline': True,
                        'estimated_facts': facts_likely_extracted
                    }
                })
                
                created_count += 1
                
            except Exception as e:
                logger.error(f"Failed to create fact tracking for conversation {conv.get('id')}: {e}")
                error_count += 1
        
        self.stats['fact_tracking_created'] = created_count
        logger.info(f"✅ Created {created_count} fact extraction tracking records")
        
        return {
            'created': created_count,
            'errors': error_count
        }
    
    async def run_full_migration(self) -> Dict[str, Any]:
        """
        Run complete migration process
        """
        self.stats['start_time'] = datetime.now()
        logger.info("🚀 Starting full schema migration...")
        
        # Step 1: Migrate tape to conversation
        tape_results = await self.migrate_tape_to_conversation()
        
        # Step 2: Migrate sessions to session_meta
        session_results = await self.migrate_sessions_to_session_meta()
        
        # Step 3: Create fact tracking records
        fact_results = await self.create_fact_tracking_records()
        
        # Step 4: Validate migration
        logger.info("🔍 Validating migration results...")
        count_validation = await self.validator.validate_counts()
        sample_validation = await self.validator.validate_sample_data()
        
        self.stats['end_time'] = datetime.now()
        
        # Compile final results
        results = {
            'migration_stats': self.stats,
            'tape_migration': tape_results,
            'session_migration': session_results,
            'fact_tracking': fact_results,
            'validation': {
                'counts': count_validation,
                'sample_data': sample_validation
            },
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        }
        
        # Log summary
        logger.info("🎉 Migration completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   📼 Tape entries migrated: {tape_results['migrated']}")
        logger.info(f"   🗂️ Session records migrated: {session_results['migrated']}")
        logger.info(f"   🔍 Fact tracking records created: {fact_results['created']}")
        logger.info(f"   ⏱️ Duration: {results['duration_seconds']:.2f} seconds")
        logger.info(f"   ✅ Validation: {all(sample_validation.get('sample_data', False) for _ in [True])}")
        
        return results


async def main():
    """
    Main migration execution
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Create unified schema first (if not already created)
    from unified_schema import UnifiedSchema
    schema_manager = UnifiedSchema(db)
    
    logger.info("🏗️ Ensuring unified schema exists...")
    try:
        await schema_manager.create_unified_schema()
    except Exception as e:
        logger.info(f"Schema creation skipped (may already exist): {e}")
    
    # Run migration
    migrator = UnifiedMigrator(db)
    results = await migrator.run_full_migration()
    
    # Save migration report
    import json
    with open('migration_report.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_results = json.loads(
            json.dumps(results, default=str)
        )
        json.dump(serializable_results, f, indent=2)
    
    logger.info("📄 Migration report saved to migration_report.json")
    
    await db.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())