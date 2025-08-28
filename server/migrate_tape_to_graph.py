#!/usr/bin/env python3
"""
Migration Script: tape → Graph Schema (message/session/user + relations)

This script migrates data from the current tape table to the existing graph schema:
- tape → message (enhanced with new fields)
- sessions → session (enhanced with new fields)  
- Create proper graph relations: session→contains→message, message→mentions→concept
- Create user records and user→knows→concept relations
- Fix content fragmentation during migration
- Maintain all existing graph structure and relationships

This leverages the existing well-designed graph schema instead of creating duplicate tables.
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from surrealdb import AsyncSurreal
from loguru import logger
import hashlib


class GraphSchemaMigrator:
    """Migrate tape data to existing graph schema with proper relations"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
        self.stats = {
            'tape_entries_processed': 0,
            'messages_created': 0,
            'sessions_created': 0,
            'users_created': 0,
            'contains_relations': 0,
            'mentions_relations': 0,
            'knows_relations': 0,
            'content_normalized': 0,
            'errors': 0
        }
        
        # Track created entities to avoid duplicates
        self.created_users: Set[str] = set()
        self.created_sessions: Set[str] = set()
        self.session_message_counts: Dict[str, int] = {}
    
    def _normalize_content(self, content: str) -> str:
        """Apply comprehensive content normalization"""
        if not content or not isinstance(content, str):
            return content or ''
        
        original = content
        s = content.strip()
        
        # Phase 1: Basic cleanup
        s = re.sub(r'\s+', ' ', s).strip()
        
        # Phase 2: Fix spaced numbers
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', s)
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', s)  
        s = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', s)
        
        # Phase 3: Fix spaced proper names
        s = re.sub(r'\b([A-Z])\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]{1,4})\s+([a-z]{2,6})\b', r'\1\2', s)
        
        # Phase 4: Enhanced name pattern detection (2-part, 3-part, 4-part)
        parts = s.split()
        reconstructed_parts = []
        i = 0
        
        while i < len(parts):
            current_part = parts[i]
            
            # 2-part fragmented names: "Pe ppy" → "Peppy" or "pepp i" → "Peppi"
            if (i < len(parts) - 1 and 
                ((current_part[0].isupper() and parts[i+1][0].islower()) or
                 (current_part[0].islower() and parts[i+1][0].islower())) and
                len(current_part) <= 6 and len(parts[i+1]) <= 6):
                total_length = len(current_part) + len(parts[i+1])
                if 3 <= total_length <= 10:
                    combined = current_part + parts[i+1]
                    if current_part[0].islower():
                        combined = combined[0].upper() + combined[1:] if combined else combined
                    reconstructed_parts.append(combined)
                    i += 2
                    continue
            
            # 3-part fragmented names: "Pe p py" → "Peppy"
            if (i < len(parts) - 2 and 
                current_part[0].isupper() and
                all(len(parts[i+j]) <= 4 for j in [1, 2]) and
                all(parts[i+j][0].islower() for j in [1, 2])):
                total_length = sum(len(parts[i+j]) for j in [0, 1, 2])
                if 3 <= total_length <= 10:
                    combined = current_part + parts[i+1] + parts[i+2]
                    reconstructed_parts.append(combined)
                    i += 3
                    continue
            
            # 4-part fragmented names: "p e p p i" → "Peppi"
            if (i < len(parts) - 3 and
                all(len(parts[i+j]) <= 3 for j in [0, 1, 2, 3]) and
                all(parts[i+j][0].islower() for j in [0, 1, 2, 3])):
                total_length = sum(len(parts[i+j]) for j in [0, 1, 2, 3])
                if 4 <= total_length <= 12:
                    combined = ''.join(parts[i+j] for j in [0, 1, 2, 3])
                    combined = combined[0].upper() + combined[1:] if combined else combined
                    reconstructed_parts.append(combined)
                    i += 4
                    continue
            
            # No pattern matched
            reconstructed_parts.append(current_part)
            i += 1
        
        s = ' '.join(reconstructed_parts)
        
        # Phase 5: Fix punctuation spacing
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        s = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        if s != original:
            self.stats['content_normalized'] += 1
            
        return s
    
    def _generate_session_id(self, speaker_id: str, timestamp: datetime) -> str:
        """Generate session ID consistent with current system"""
        date_str = timestamp.strftime('%Y%m%d')
        return f"{speaker_id}_{date_str}"
    
    def _extract_timestamp(self, ts_value: Any) -> datetime:
        """Extract datetime from various timestamp formats"""
        if isinstance(ts_value, datetime):
            return ts_value
        elif isinstance(ts_value, str):
            try:
                return datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
            except:
                return datetime.fromtimestamp(float(ts_value))
        elif isinstance(ts_value, (int, float)):
            return datetime.fromtimestamp(ts_value)
        else:
            return datetime.now()
    
    def _generate_record_id(self, table: str, identifier: str) -> str:
        """Generate SurrealDB record ID"""
        # Create a hash-based ID for consistency
        hash_obj = hashlib.md5(identifier.encode())
        return f"{table}:{hash_obj.hexdigest()[:12]}"
    
    async def ensure_user_exists(self, speaker_id: str) -> str:
        """Ensure user record exists and return user record ID"""
        if speaker_id in self.created_users:
            return self._generate_record_id('user', speaker_id)
        
        try:
            user_id = self._generate_record_id('user', speaker_id)
            
            # Check if user already exists
            existing = await self.db.query('SELECT id FROM user WHERE id = $user_id', {
                'user_id': user_id
            })
            
            if not existing:
                # Create user record
                await self.db.query(f'''
                    CREATE {user_id} SET
                        id = $speaker_id,
                        total_sessions = 0,
                        first_seen = time::now(),
                        last_interaction = time::now(),
                        preferred_name = $speaker_id,
                        voice_id = null,
                        speaker_profile = null
                ''', {'speaker_id': speaker_id})
                
                self.stats['users_created'] += 1
                logger.debug(f"👤 Created user: {user_id}")
            
            self.created_users.add(speaker_id)
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to create user {speaker_id}: {e}")
            self.stats['errors'] += 1
            return self._generate_record_id('user', 'unknown')
    
    async def ensure_session_exists(self, session_id: str, user_id: str, start_time: datetime) -> str:
        """Ensure session record exists and return session record ID"""
        if session_id in self.created_sessions:
            return self._generate_record_id('session', session_id)
        
        try:
            session_record_id = self._generate_record_id('session', session_id)
            
            # Check if session already exists
            existing = await self.db.query('SELECT id FROM session WHERE id = $session_id', {
                'session_id': session_record_id
            })
            
            if not existing:
                # Create session record
                await self.db.query(f'''
                    CREATE {session_record_id} SET
                        id = $session_id,
                        user_id = $user_id,
                        speaker_id = $speaker_id,
                        start_time = $start_time,
                        end_time = null,
                        summary = null,
                        turn_count = 0,
                        last_activity = $start_time,
                        status = 'active',
                        facts_extracted = 0,
                        quality_score = 0.0,
                        duration_s = 0,
                        total_words = 0,
                        total_chars = 0,
                        keywords = [],
                        topics = [],
                        agent_id = null
                ''', {
                    'session_id': session_id,
                    'user_id': user_id,
                    'speaker_id': session_id.split('_')[0],  # Extract speaker from session_id
                    'start_time': start_time
                })
                
                self.stats['sessions_created'] += 1
                logger.debug(f"🗂️ Created session: {session_record_id}")
            
            self.created_sessions.add(session_id)
            self.session_message_counts[session_id] = 0
            return session_record_id
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            self.stats['errors'] += 1
            return self._generate_record_id('session', 'unknown')
    
    async def create_message_record(self, tape_entry: Dict[str, Any]) -> Optional[str]:
        """Create message record from tape entry"""
        try:
            # Extract and normalize data
            speaker_id = tape_entry.get('speaker_id', 'unknown')
            role = tape_entry.get('role', 'user')
            raw_content = tape_entry.get('content', '')
            normalized_content = self._normalize_content(raw_content)
            timestamp = self._extract_timestamp(tape_entry.get('ts'))
            
            # Generate IDs
            session_id = self._generate_session_id(speaker_id, timestamp)
            user_id = await self.ensure_user_exists(speaker_id)
            session_record_id = await self.ensure_session_exists(session_id, user_id, timestamp)
            
            # Track message order within session
            if session_id not in self.session_message_counts:
                self.session_message_counts[session_id] = 0
            self.session_message_counts[session_id] += 1
            message_order = self.session_message_counts[session_id]
            
            # Generate message record ID
            message_record_id = self._generate_record_id('message', 
                f"{session_id}_{message_order}_{role}")
            
            # Create message record
            await self.db.query(f'''
                CREATE {message_record_id} SET
                    id = $message_id,
                    content = $content,
                    timestamp = $timestamp,
                    sender_id = $sender_id,
                    message_type = 'conversation',
                    raw_content = $raw_content,
                    role = $role,
                    facts_extracted = false,
                    word_count = $word_count,
                    content_length = $content_length,
                    agent_id = $agent_id,
                    processing_meta = $processing_meta,
                    embedding = $embedding,
                    language = null
            ''', {
                'message_id': f"{session_id}_{message_order}",
                'content': normalized_content,
                'timestamp': timestamp,
                'sender_id': speaker_id,
                'raw_content': raw_content if raw_content != normalized_content else None,
                'role': role,
                'word_count': len(normalized_content.split()),
                'content_length': len(normalized_content),
                'agent_id': tape_entry.get('agent_id'),
                'processing_meta': {
                    'migrated_from_tape': True,
                    'original_tape_id': tape_entry.get('id'),
                    'normalized': raw_content != normalized_content
                },
                'embedding': tape_entry.get('embedding')
            })
            
            self.stats['messages_created'] += 1
            
            # Create contains relation (session → message)
            await self.create_contains_relation(session_record_id, message_record_id, message_order)
            
            # Create mentions relations (message → concept) if concepts exist
            await self.create_mentions_relations(message_record_id, normalized_content)
            
            return message_record_id
            
        except Exception as e:
            logger.error(f"Failed to create message from tape entry: {e}")
            self.stats['errors'] += 1
            return None
    
    async def create_contains_relation(self, session_id: str, message_id: str, message_order: int):
        """Create contains relation: session → message"""
        try:
            await self.db.query('''
                RELATE $session_id -> contains -> $message_id SET
                    message_order = $message_order,
                    created_at = time::now()
            ''', {
                'session_id': session_id,
                'message_id': message_id,
                'message_order': message_order
            })
            
            self.stats['contains_relations'] += 1
            
        except Exception as e:
            logger.debug(f"Failed to create contains relation: {e}")
    
    async def create_mentions_relations(self, message_id: str, content: str):
        """Create mentions relations: message → concept for entities in content"""
        try:
            # Get existing concepts that might be mentioned in this content
            concepts = await self.db.query('''
                SELECT id, name FROM concept 
                WHERE name != null AND name != ""
            ''')
            
            mentioned_concepts = []
            content_lower = content.lower()
            
            for concept in concepts:
                concept_name = concept.get('name', '')
                if concept_name and concept_name.lower() in content_lower:
                    mentioned_concepts.append(concept['id'])
            
            # Create mentions relations
            for concept_id in mentioned_concepts[:10]:  # Limit to avoid too many relations
                try:
                    await self.db.query('''
                        RELATE $message_id -> mentions -> $concept_id SET
                            confidence = 1.0,
                            mention_type = 'reference',
                            created_at = time::now()
                    ''', {
                        'message_id': message_id,
                        'concept_id': concept_id
                    })
                    
                    self.stats['mentions_relations'] += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to create mentions relation: {e}")
                    
        except Exception as e:
            logger.debug(f"Failed to process mentions for message: {e}")
    
    async def create_knows_relations(self):
        """Create knows relations: user → concept based on messages"""
        try:
            logger.info("🔗 Creating user → concept (knows) relations...")
            
            # Get all users and concepts they've mentioned
            user_concepts = await self.db.query('''
                SELECT 
                    m.sender_id as user_id,
                    concepts.id as concept_id,
                    concepts.name as concept_name,
                    COUNT() as mention_count
                FROM message m, mentions, concept concepts
                WHERE m -> mentions -> concepts
                GROUP BY m.sender_id, concepts.id
            ''')
            
            for relation in user_concepts:
                try:
                    user_id = self._generate_record_id('user', relation['user_id'])
                    
                    await self.db.query('''
                        RELATE $user_id -> knows -> $concept_id SET
                            strength = $strength,
                            first_mentioned = time::now(),
                            last_mentioned = time::now(),
                            mention_count = $mention_count
                    ''', {
                        'user_id': user_id,
                        'concept_id': relation['concept_id'],
                        'strength': min(1.0, relation['mention_count'] / 10.0),
                        'mention_count': relation['mention_count']
                    })
                    
                    self.stats['knows_relations'] += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to create knows relation: {e}")
            
            logger.info(f"✅ Created {self.stats['knows_relations']} user → concept relations")
            
        except Exception as e:
            logger.error(f"Failed to create knows relations: {e}")
    
    async def update_session_statistics(self):
        """Update session statistics based on created messages"""
        try:
            logger.info("📊 Updating session statistics...")
            
            # Get message counts per session
            session_stats = await self.db.query('''
                SELECT 
                    s.id as session_id,
                    COUNT(c) as message_count,
                    COUNT(m) FILTER (WHERE m.role = 'user') as user_messages,
                    COUNT(m) FILTER (WHERE m.role = 'assistant') as assistant_messages,
                    SUM(m.word_count) as total_words,
                    SUM(m.content_length) as total_chars,
                    MIN(m.timestamp) as first_message,
                    MAX(m.timestamp) as last_message
                FROM session s, contains c, message m
                WHERE s -> c -> m
                GROUP BY s.id
            ''')
            
            for stats in session_stats:
                try:
                    # Calculate duration
                    duration_s = 0
                    if stats.get('first_message') and stats.get('last_message'):
                        first = self._extract_timestamp(stats['first_message'])
                        last = self._extract_timestamp(stats['last_message'])
                        duration_s = int((last - first).total_seconds())
                    
                    # Calculate quality score
                    quality_score = min(1.0, stats.get('message_count', 0) / 20.0)
                    
                    await self.db.query(f'''
                        UPDATE {stats['session_id']} SET
                            turn_count = $turn_count,
                            total_words = $total_words,
                            total_chars = $total_chars,
                            duration_s = $duration_s,
                            quality_score = $quality_score,
                            last_activity = $last_activity,
                            status = 'completed'
                    ''', {
                        'turn_count': stats.get('message_count', 0),
                        'total_words': stats.get('total_words', 0),
                        'total_chars': stats.get('total_chars', 0),
                        'duration_s': duration_s,
                        'quality_score': quality_score,
                        'last_activity': self._extract_timestamp(stats.get('last_message'))
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed to update session stats: {e}")
            
            logger.info("✅ Updated session statistics")
            
        except Exception as e:
            logger.error(f"Failed to update session statistics: {e}")
    
    async def run_migration(self) -> Dict[str, Any]:
        """
        Run the complete migration from tape to graph schema
        """
        logger.info("🚀 Starting migration: tape → graph schema...")
        start_time = time.time()
        
        try:
            # Step 1: Get all tape entries
            tape_entries = await self.db.query('SELECT * FROM tape ORDER BY ts')
            logger.info(f"📼 Found {len(tape_entries)} tape entries to migrate")
            
            if not tape_entries:
                logger.warning("No tape entries found")
                return {'status': 'success', 'message': 'No data to migrate'}
            
            # Step 2: Process each tape entry
            for i, entry in enumerate(tape_entries):
                try:
                    await self.create_message_record(entry)
                    self.stats['tape_entries_processed'] += 1
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Progress: {i + 1}/{len(tape_entries)} entries processed")
                        
                except Exception as e:
                    logger.error(f"Failed to process tape entry {i}: {e}")
                    self.stats['errors'] += 1
            
            # Step 3: Create user → concept relations
            await self.create_knows_relations()
            
            # Step 4: Update session statistics
            await self.update_session_statistics()
            
            # Step 5: Final statistics
            duration = time.time() - start_time
            
            logger.info("🎉 Migration completed successfully!")
            logger.info(f"📊 Migration summary:")
            logger.info(f"   📼 Tape entries processed: {self.stats['tape_entries_processed']}")
            logger.info(f"   💬 Messages created: {self.stats['messages_created']}")
            logger.info(f"   🗂️ Sessions created: {self.stats['sessions_created']}")
            logger.info(f"   👤 Users created: {self.stats['users_created']}")
            logger.info(f"   🔗 Contains relations: {self.stats['contains_relations']}")
            logger.info(f"   🔗 Mentions relations: {self.stats['mentions_relations']}")
            logger.info(f"   🔗 Knows relations: {self.stats['knows_relations']}")
            logger.info(f"   📝 Content normalized: {self.stats['content_normalized']}")
            logger.info(f"   ⚠️ Errors: {self.stats['errors']}")
            logger.info(f"   ⏱️ Duration: {duration:.2f}s")
            
            return {
                'status': 'success',
                'stats': self.stats,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'stats': self.stats
            }


async def main():
    """
    Main migration execution
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Ensure enhanced schema exists
    from enhance_existing_schema import ExistingSchemaEnhancer
    enhancer = ExistingSchemaEnhancer(db)
    
    logger.info("🏗️ Ensuring enhanced graph schema exists...")
    try:
        await enhancer.run_full_enhancement()
    except Exception as e:
        logger.info(f"Schema enhancement may have already been applied: {e}")
    
    # Run migration
    migrator = GraphSchemaMigrator(db)
    results = await migrator.run_migration()
    
    # Save results
    import json
    with open('graph_migration_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📄 Migration report saved to graph_migration_report.json")
    
    await db.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())