#!/usr/bin/env python3
"""
Final Complete Migration with Schema Compliance

Complete migration with all schema requirements fixed.
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

class FinalMigrator:
    def __init__(self):
        self.db = None
        self.stats = MigrationStats()
        self.speaker_to_user_map = {}
        self.session_id_map = {}
        self.concept_cache = {}

    async def connect(self):
        """Connect to SurrealDB"""
        self.db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
        await self.db.connect()
        await self.db.signin({
            "username": "root",
            "password": "slowcat_secure_2024"
        })
        await self.db.use("slowcat", "memory")
        logger.info("Connected to SurrealDB")

    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()

    async def migrate_all(self, extracted_data: dict):
        """Run complete migration with schema compliance"""
        logger.info("🚀 Starting final complete migration...")

        # Clear all existing data
        await self._clear_all_data()

        # Step 1: Create users (works fine)
        await self._create_users(extracted_data)
        
        # Step 2: Create concepts (fix required fields)
        await self._create_concepts(extracted_data)
        
        # Step 3: Create sessions (already working)
        await self._create_sessions(extracted_data)
        
        # Step 4: Create messages (fix required fields)
        await self._create_messages(extracted_data)
        
        # Step 5: Create thoughts (fix required fields)
        await self._create_thoughts(extracted_data)
        
        # Step 6: Create knowledge relationships
        await self._create_knowledge_relationships(extracted_data)

        logger.info("✅ Final migration completed!")
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
        """Create user nodes from unique speakers - schema compliant"""
        logger.info("Creating users...")
        
        speakers = set()
        
        # Collect all unique speakers from different sources
        for table_name in ['tape', 'sessions', 'fact']:
            if table_name in data:
                for record in data[table_name]:
                    speaker = record.get('speaker_id') or record.get('subject')
                    if speaker and speaker != 'unknown' and speaker != 'slowcat':
                        speakers.add(speaker)
        
        # Create user records - user schema works fine
        for speaker in speakers:
            try:
                user_id = f"user:{_safe_id(speaker)}"
                
                result = await self.db.query(f"""
                    CREATE {user_id} SET
                        name = $name,
                        first_seen = time::now(),
                        last_seen = time::now(),
                        total_interactions = 0,
                        metadata = {{}}
                """, {
                    "name": speaker
                })
                
                self.speaker_to_user_map[speaker] = user_id
                self.stats.users_created += 1
                logger.debug(f"Created user: {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to create user {speaker}: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.users_created} users")

    async def _create_concepts(self, data: dict):
        """Create concept nodes - fix required 'kind' field"""
        logger.info("Creating concepts...")
        
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
                    concepts.append((obj_value, 'general'))
        
        # Create concept records with ALL required fields
        for concept_name, kind in concepts:
            try:
                concept_id = f"concept:{_safe_id(concept_name)}"
                
                # Include ALL required fields
                result = await self.db.query(f"""
                    CREATE {concept_id} SET
                        name = $name,
                        kind = $kind,
                        properties = {{}},
                        mentioned_count = 0,
                        first_mentioned = time::now(),
                        last_mentioned = time::now()
                """, {
                    "name": concept_name,
                    "kind": kind
                })
                
                self.concept_cache[concept_name] = concept_id
                self.stats.concepts_created += 1
                logger.debug(f"Created concept: {concept_id} (kind: {kind})")
                
            except Exception as e:
                logger.error(f"Failed to create concept {concept_name}: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.concepts_created} concepts")

    async def _create_sessions(self, data: dict):
        """Create sessions - already working"""
        logger.info("Creating sessions...")
        
        # Sessions from session_summary table (80 records) 
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
                
                # Create session with all required fields
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
        """Create messages - fix required 'embedding' field"""
        logger.info("Creating messages...")
        
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
                
                # Handle embedding - required field!
                embedding = tape_record.get('embedding', [])
                if not embedding or not isinstance(embedding, list):
                    embedding = []  # Empty array satisfies array<number> requirement
                
                # Create message with ALL required fields
                result = await self.db.query(f"""
                    CREATE {message_id} SET
                        session_id = {mapped_session},
                        speaker_type = $speaker_type,
                        content = $content,
                        timestamp = {message_timestamp},
                        sequence_num = $sequence_num,
                        embedding = $embedding,
                        metadata = {{}}
                """, {
                    "speaker_type": tape_record.get('role', 'user'),
                    "content": tape_record.get('content', ''),
                    "sequence_num": sequence_counters[mapped_session],
                    "embedding": embedding
                })
                
                # Create relationship: session -> contains -> message
                await self.db.query(f"""
                    RELATE {mapped_session}->contains->{message_id} SET
                        created_at = time::now(),
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
        """Create thoughts - fix required 'agent_id' field"""
        logger.info("Creating thoughts...")
        
        # Process thought records
        thought_data = data.get('thought', []) + data.get('emergent_event', [])
        
        for thought_record in thought_data:
            try:
                thought_id = f"thought:{uuid.uuid4().hex[:12]}"
                
                # Get timestamp
                timestamp_str = thought_record.get('ts', thought_record.get('timestamp', ''))
                thought_timestamp = convert_timestamp(timestamp_str)
                
                # Create thought with ALL required fields
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
                    "session_id": None,
                    "thought_type": thought_record.get('thought_type', thought_record.get('kind', 'observation')),
                    "content": thought_record.get('content', thought_record.get('content_snippet', '')),
                    "confidence": thought_record.get('confidence', 0.5)
                })
                
                self.stats.thoughts_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create thought: {e}")
                self.stats.errors += 1
        
        logger.info(f"Created {self.stats.thoughts_created} thoughts")

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
    logger.info("🚀 Starting final complete migration...")
    
    # Extract data from backup
    backup_file = "/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql"
    parser = SurrealQLParser(backup_file)
    extracted_data = parser.parse_backup()
    
    # Run migration
    migrator = FinalMigrator()
    await migrator.connect()
    
    try:
        stats = await migrator.migrate_all(extracted_data)
        
        # Print final stats
        logger.info("=" * 60)
        logger.info("FINAL MIGRATION RESULTS")
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