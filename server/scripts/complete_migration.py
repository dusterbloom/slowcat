#!/usr/bin/env python3
"""
Complete Graph Schema Migration

Complete migration with all tables: users, sessions, messages, concepts, 
and all relationships with proper timestamps and schema compliance.
"""

import asyncio
import uuid
from datetime import datetime
from surrealdb import AsyncSurreal
from loguru import logger
from extract_backup_data import SurrealQLParser
from dataclasses import dataclass

@dataclass
class MigrationStats:
    users_created: int = 0
    sessions_created: int = 0
    messages_created: int = 0
    concepts_created: int = 0
    thoughts_created: int = 0
    knowledge_relations: int = 0
    contains_relations: int = 0
    reflects_relations: int = 0
    errors: int = 0

def convert_timestamp(timestamp_str):
    """Convert ISO timestamp string to SurrealDB datetime format"""
    if not timestamp_str or timestamp_str == 'time::now()':
        return 'time::now()'
    
    try:
        # Parse ISO format like "2025-08-22T11:21:14.071981Z"
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        # Format for SurrealDB: d'2025-08-22T11:21:14.071981Z'
        return f"d'{timestamp_str}'"
    except Exception:
        return 'time::now()'

def _safe_id(name: str) -> str:
    """Convert name to safe SurrealDB record ID"""
    import re
    # Remove problematic characters and convert to lowercase
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    safe = re.sub(r'_+', '_', safe).strip('_').lower()
    return safe[:50] if safe else 'unknown'

class CompleteMigrator:
    def __init__(self):
        self.db = None
        self.stats = MigrationStats()
        self.speaker_to_user_map = {}
        self.session_id_map = {}
        self.concept_cache = {}

    async def connect(self):
        """Connect to SurrealDB (env-driven)"""
        import os
        url = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
        ns = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
        db = os.getenv('SURREALDB_DATABASE', 'memory_graph')
        user = os.getenv('SURREALDB_USER', 'root')
        password = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
        self.db = AsyncSurreal(url)
        await self.db.connect()
        try:
            await self.db.signin({"username": user, "password": password})
        except Exception:
            await self.db.signin({"user": user, "pass": password})
        await self.db.use(ns, db)
        logger.info(f"Connected to SurrealDB: ns={ns} db={db}")

    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()

    async def migrate_all(self, extracted_data: dict):
        """Run complete migration"""
        logger.info("🚀 Starting complete graph schema migration...")

        # Clear all existing data
        await self._clear_all_data()

        # Step 1: Create users
        await self._create_users(extracted_data)
        
        # Step 2: Create concepts
        await self._create_concepts(extracted_data)
        
        # Step 3: Create sessions with proper timestamps
        await self._create_sessions(extracted_data)
        
        # Step 4: Create messages and session relationships
        await self._create_messages(extracted_data)
        
        # Step 5: Create thoughts and relationships
        await self._create_thoughts(extracted_data)
        
        # Step 6: Create knowledge relationships
        await self._create_knowledge_relationships(extracted_data)

        logger.info("✅ Complete migration finished!")
        return self.stats

    async def _clear_all_data(self):
        """Clear all existing data"""
        tables = ['user', 'session', 'message', 'concept', 'thought', 'knows', 'contains', 'reflects', 'mentions']
        
        for table in tables:
            try:
                await self.db.query(f"DELETE {table}")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Failed to clear {table}: {e}")

    async def _create_users(self, data: dict):
        """Create user nodes from unique speakers"""
        logger.info("Creating user nodes...")
        
        speakers = set()
        
        # Collect all unique speakers from different sources
        for table_name in ['tape', 'sessions', 'fact']:
            if table_name in data:
                for record in data[table_name]:
                    speaker = record.get('speaker_id') or record.get('subject')
                    if speaker and speaker != 'unknown' and speaker != 'slowcat':
                        speakers.add(speaker)
        
        # Create user records
        for speaker in speakers:
            try:
                user_id = f"user:{_safe_id(speaker)}"
                
                # Use db.create() method which works reliably
                result = await self.db.create(user_id, {
                    "name": speaker,
                    "first_seen": "time::now()",
                    "last_seen": "time::now()",
                    "total_interactions": 0,
                    "metadata": {}
                })
                
                self.speaker_to_user_map[speaker] = user_id
                self.stats.users_created += 1
                logger.debug(f"Created user: {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to create user {speaker}: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.users_created} users")

    async def _create_concepts(self, data: dict):
        """Create concept nodes from facts"""
        logger.info("Creating concept nodes...")
        
        facts_data = data.get('fact', [])
        concepts = []
        
        # Extract concepts from facts
        for fact in facts_data:
            predicate = fact.get('predicate', '')
            obj_value = fact.get('obj', '')
            
            # Map predicates to concept types
            if 'dog' in predicate.lower() or 'pet' in predicate.lower():
                if obj_value:
                    concepts.append((obj_value, 'pet'))
            elif 'name' in predicate.lower():
                if obj_value:
                    concepts.append((obj_value, 'person'))
            elif 'location' in predicate.lower() or 'place' in predicate.lower():
                if obj_value:
                    concepts.append((obj_value, 'location'))
            else:
                if obj_value:
                    concepts.append((obj_value, 'unknown'))
        
        # Create concept records
        for concept_name, kind in concepts:
            try:
                concept_id = f"concept:{_safe_id(concept_name)}"
                
                result = await self.db.create(concept_id, {
                    "name": concept_name,
                    "kind": kind,
                    "properties": {},
                    "mentioned_count": 0,
                    "first_mentioned": "time::now()",
                    "last_mentioned": "time::now()"
                })
                
                self.concept_cache[concept_name] = concept_id
                self.stats.concepts_created += 1
                logger.debug(f"Created concept: {concept_id}")
                
            except Exception as e:
                logger.error(f"Failed to create concept {concept_name}: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.concepts_created} concepts")

    async def _create_sessions(self, data: dict):
        """Create sessions with proper timestamps"""
        logger.info("Creating sessions...")
        
        # Method 1: Sessions from 'sessions' table (3 records)
        sessions_data = data.get('sessions', [])
        for session_record in sessions_data:
            try:
                speaker_id = session_record.get('speaker_id', 'unknown')
                if speaker_id == 'unknown':
                    continue
                    
                user_id = f"user:{_safe_id(speaker_id)}"
                new_session_id = f"session:{uuid.uuid4().hex[:12]}"
                
                # Get original timestamps
                first_seen = convert_timestamp(session_record.get('first_seen'))
                last_interaction = convert_timestamp(session_record.get('last_interaction'))
                
                # Calculate duration
                duration_secs = 0
                if 'first_seen' in session_record and 'last_interaction' in session_record:
                    try:
                        start = datetime.fromisoformat(session_record['first_seen'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(session_record['last_interaction'].replace('Z', '+00:00'))
                        duration_secs = int((end - start).total_seconds())
                    except Exception:
                        pass
                
                # Create session with original timestamps
                result = await self.db.query(f"""
                    CREATE {new_session_id} SET
                        user_id = {user_id},
                        agent_id = 'slowcat',
                        started_at = {first_seen},
                        ended_at = {last_interaction},
                        turn_count = $turn_count,
                        summary = $summary,
                        keywords = $keywords,
                        status = 'ended',
                        duration_secs = $duration_secs
                """, {
                    "turn_count": session_record.get('total_turns', 0),
                    "summary": f"Session with {session_record.get('session_count', 0)} conversations",
                    "keywords": [speaker_id, 'conversation'],
                    "duration_secs": duration_secs
                })
                
                old_id = session_record.get('id', '')
                self.session_id_map[old_id] = new_session_id
                self.stats.sessions_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create session from sessions table: {e}")
                self.stats.errors += 1
        
        # Method 2: Sessions from 'session_summary' table (80 records) 
        session_summary_data = data.get('session_summary', [])
        for summary_record in session_summary_data:
            try:
                old_session_id = summary_record.get('session_id', '')
                
                # Extract speaker from session_id (e.g., "peppi:1756134556")
                if ':' in old_session_id:
                    speaker_id = old_session_id.split(':')[0]
                else:
                    continue
                    
                user_id = f"user:{_safe_id(speaker_id)}"
                new_session_id = f"session:{uuid.uuid4().hex[:12]}"
                
                # Get original timestamp
                timestamp_str = summary_record.get('ts', '')
                session_timestamp = convert_timestamp(timestamp_str)
                
                # Get duration
                duration_secs = summary_record.get('duration_s', 0)
                
                # Create session
                result = await self.db.query(f"""
                    CREATE {new_session_id} SET
                        user_id = {user_id},
                        agent_id = 'slowcat',
                        started_at = {session_timestamp},
                        ended_at = {session_timestamp},
                        turn_count = $turn_count,
                        summary = $summary,
                        keywords = $keywords,
                        status = 'ended',
                        duration_secs = $duration_secs
                """, {
                    "turn_count": summary_record.get('turns', 0),
                    "summary": summary_record.get('summary', ''),
                    "keywords": summary_record.get('keywords', []),
                    "duration_secs": duration_secs
                })
                
                # Map for message creation later
                self.session_id_map[old_session_id] = new_session_id
                self.stats.sessions_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create session from session_summary: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.sessions_created} sessions")

    async def _create_messages(self, data: dict):
        """Create message nodes and session->contains->message relationships"""
        logger.info("Creating messages and relationships...")
        
        tape_data = data.get('tape', [])
        sequence_counters = {}  # Track sequence numbers per session
        
        for tape_record in tape_data:
            try:
                # Get session mapping
                old_session_id = tape_record.get('session_id', '')
                if old_session_id not in self.session_id_map:
                    continue
                
                mapped_session = self.session_id_map[old_session_id]
                message_id = f"message:{uuid.uuid4().hex[:12]}"
                
                # Get sequence number
                if mapped_session not in sequence_counters:
                    sequence_counters[mapped_session] = 0
                sequence_counters[mapped_session] += 1
                
                # Get timestamp
                timestamp_str = tape_record.get('ts', '')
                message_timestamp = convert_timestamp(timestamp_str)
                
                # Create message with all required fields
                result = await self.db.query(f"""
                    CREATE {message_id} SET
                        session_id = {mapped_session},
                        speaker_type = $speaker_type,
                        content = $content,
                        timestamp = {message_timestamp},
                        sequence_num = $sequence,
                        embedding = $embedding,
                        metadata = {{}}
                """, {
                    "speaker_type": tape_record.get('role', 'user'),
                    "content": tape_record.get('content', ''),
                    "sequence": sequence_counters[mapped_session],
                    "embedding": tape_record.get('embedding')
                })
                
                # Create relationship: session -> contains -> message
                await self.db.query(f"""
                    RELATE {mapped_session}->contains->{message_id} SET
                        created_at = {message_timestamp},
                        sequence_order = $sequence
                """, {
                    "sequence": sequence_counters[mapped_session]
                })
                
                self.stats.messages_created += 1
                self.stats.contains_relations += 1
                
                if self.stats.messages_created % 100 == 0:
                    logger.info(f"Created {self.stats.messages_created} messages...")
                
            except Exception as e:
                logger.error(f"Failed to create message: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.messages_created} messages with {self.stats.contains_relations} contains relations")

    async def _create_thoughts(self, data: dict):
        """Create thoughts and session->reflects->thought relationships"""
        logger.info("Creating thoughts...")
        
        # Process thought records
        thought_data = data.get('thought', []) + data.get('emergent_event', [])
        
        for thought_record in thought_data:
            try:
                # Find corresponding session
                session_id = None
                
                # Try to find session from various fields
                if 'session_id' in thought_record:
                    session_id = thought_record['session_id']
                elif 'id' in thought_record:
                    # Extract from thought ID if it contains session info
                    thought_id_str = str(thought_record['id'])
                    # This would need custom logic based on your data structure
                
                if not session_id or session_id not in self.session_id_map:
                    # Create orphaned thought without session relationship
                    session_id = None
                else:
                    session_id = self.session_id_map[session_id]
                
                thought_id = f"thought:{uuid.uuid4().hex[:12]}"
                
                # Get timestamp
                timestamp_str = thought_record.get('ts', thought_record.get('timestamp', ''))
                thought_timestamp = convert_timestamp(timestamp_str)
                
                # Create thought
                result = await self.db.query(f"""
                    CREATE {thought_id} SET
                        agent_id = $agent_id,
                        session_id = $session_id,
                        thought_type = $thought_type,
                        content = $content,
                        timestamp = {thought_timestamp},
                        visibility = 'private',
                        confidence = $confidence
                """, {
                    "agent_id": thought_record.get('agent_id', 'slowcat'),
                    "session_id": session_id,
                    "thought_type": thought_record.get('thought_type', thought_record.get('kind', 'observation')),
                    "content": thought_record.get('content', thought_record.get('content_snippet', '')),
                    "confidence": thought_record.get('confidence', 0.5)
                })
                
                # Create relationship if session exists
                if session_id:
                    await self.db.query(f"""
                        RELATE {session_id}->reflects->{thought_id} SET
                            created_at = {thought_timestamp},
                            thought_type = $thought_type
                    """, {
                        "thought_type": thought_record.get('thought_type', 'reflection')
                    })
                    self.stats.reflects_relations += 1
                
                self.stats.thoughts_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create thought: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.thoughts_created} thoughts with {self.stats.reflects_relations} reflects relations")

    async def _create_knowledge_relationships(self, data: dict):
        """Create user->knows->concept relationships from facts"""
        logger.info("Creating knowledge relationships...")
        
        facts_data = data.get('fact', [])
        
        for fact in facts_data:
            try:
                # Get user
                subject = fact.get('subject', '')
                if not subject or subject not in self.speaker_to_user_map:
                    continue
                
                user_id = self.speaker_to_user_map[subject]
                
                # Get concept
                obj_value = fact.get('obj', '')
                if not obj_value or obj_value not in self.concept_cache:
                    continue
                
                concept_id = self.concept_cache[obj_value]
                
                # Get relationship details
                predicate = fact.get('predicate', 'related_to')
                fidelity = fact.get('fidelity', 3)
                strength = fact.get('strength', 0.8)
                
                # Create knowledge relationship
                await self.db.query(f"""
                    RELATE {user_id}->knows->{concept_id} SET
                        relationship = $relationship,
                        strength = $strength,
                        fidelity = $fidelity,
                        learned_at = time::now(),
                        last_accessed = time::now(),
                        access_count = 1
                """, {
                    "relationship": predicate,
                    "strength": strength,
                    "fidelity": fidelity
                })
                
                self.stats.knowledge_relations += 1
                
            except Exception as e:
                logger.error(f"Failed to create knowledge relationship: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.knowledge_relations} knowledge relationships")

async def main():
    """Main migration execution"""
    logger.info("🚀 Starting complete graph schema migration...")
    
    # Extract data from backup
    backup_file = "/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql"
    parser = SurrealQLParser(backup_file)
    extracted_data = parser.parse_backup()
    
    # Run migration
    migrator = CompleteMigrator()
    await migrator.connect()
    
    try:
        stats = await migrator.migrate_all(extracted_data)
        
        # Print final stats
        logger.info("=" * 60)
        logger.info("COMPLETE MIGRATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Users created: {stats.users_created}")
        logger.info(f"✅ Sessions created: {stats.sessions_created}")
        logger.info(f"✅ Messages created: {stats.messages_created}")
        logger.info(f"✅ Concepts created: {stats.concepts_created}")
        logger.info(f"✅ Thoughts created: {stats.thoughts_created}")
        logger.info(f"✅ Knowledge relations: {stats.knowledge_relations}")
        logger.info(f"✅ Contains relations: {stats.contains_relations}")
        logger.info(f"✅ Reflects relations: {stats.reflects_relations}")
        logger.info(f"❌ Errors: {stats.errors}")
        logger.info("=" * 60)
        
        if stats.errors == 0:
            logger.info("🎉 Migration completed successfully with no errors!")
        else:
            logger.warning(f"⚠️ Migration completed with {stats.errors} errors")
            
    finally:
        await migrator.close()

if __name__ == "__main__":
    asyncio.run(main())
