#!/usr/bin/env python3
"""
Simple Direct Migration Script

Create sample data directly using SurrealQL INSERT statements to verify the schema works.
Then migrate essential data from the backup.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger
import uuid
from datetime import datetime

class DirectMigration:
    def __init__(self):
        self.db = None
    
    async def connect(self):
        """Connect to SurrealDB"""
        self.db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
        await self.db.connect()
        await self.db.signin({
            "username": "root",
            "password": "slowcat_secure_2024"
        })
        await self.db.use("slowcat", "memory")
        logger.info("✅ Connected to SurrealDB")
    
    async def create_sample_data(self):
        """Create sample data to verify schema works"""
        logger.info("🚀 Creating sample data...")
        
        # Create user
        user_result = await self.db.query("""
            INSERT INTO user {
                id: user:peppi,
                name: 'Peppi',
                first_seen: time::now(),
                last_seen: time::now(),
                total_interactions: 5,
                metadata: {}
            }
        """)
        logger.info(f"Created user: {user_result}")
        
        # Create session
        session_id = f"session:{uuid.uuid4().hex[:12]}"
        session_result = await self.db.query(f"""
            INSERT INTO session {{
                id: {session_id},
                user_id: user:peppi,
                agent_id: 'slowcat',
                started_at: time::now(),
                ended_at: time::now(),
                status: 'completed',
                turn_count: 3,
                duration_secs: 120,
                summary: 'Test conversation',
                keywords: []
            }}
        """)
        logger.info(f"Created session: {session_result}")
        
        # Create messages
        for i in range(3):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Test message {i+1}"
            
            message_id = f"message:{uuid.uuid4().hex[:12]}"
            message_result = await self.db.query(f"""
                INSERT INTO message {{
                    id: {message_id},
                    session_id: {session_id},
                    speaker_type: '{role}',
                    content: '{content}',
                    timestamp: time::now(),
                    sequence_num: {i+1},
                    embedding: [],
                    metadata: {{}}
                }}
            """)
            logger.info(f"Created message {i+1}: {message_result}")
            
            # Create relationship: session contains message
            await self.db.query(f"""
                RELATE {session_id}->contains->{message_id} SET created_at = time::now()
            """)
        
        # Create concepts
        concepts = ["dog", "technology", "conversation"]
        for i, concept_name in enumerate(concepts):
            concept_id = f"concept:{concept_name}"
            concept_result = await self.db.query(f"""
                INSERT INTO concept {{
                    id: {concept_id},
                    name: '{concept_name}',
                    kind: 'general',
                    properties: {{}},
                    mentioned_count: {i+1},
                    first_mentioned: time::now(),
                    last_mentioned: time::now()
                }}
            """)
            logger.info(f"Created concept: {concept_result}")
            
            # Create relationship: user knows concept
            await self.db.query(f"""
                RELATE user:peppi->knows->{concept_id} SET 
                    strength = {0.8 + i*0.1}, 
                    created_at = time::now()
            """)
        
        # Create thoughts
        for i in range(2):
            thought_id = f"thought:{uuid.uuid4().hex[:12]}"
            thought_result = await self.db.query(f"""
                INSERT INTO thought {{
                    id: {thought_id},
                    agent_id: 'slowcat',
                    session_id: {session_id},
                    thought_type: 'observation',
                    content: 'Test thought {i+1}',
                    timestamp: time::now(),
                    visibility: 'private',
                    confidence: 0.8
                }}
            """)
            logger.info(f"Created thought {i+1}: {thought_result}")
            
            # Create relationship: session reflects thought
            await self.db.query(f"""
                RELATE {session_id}->reflects->{thought_id} SET created_at = time::now()
            """)
        
        logger.info("✅ Sample data created successfully!")
    
    async def validate_data(self):
        """Validate the migrated data"""
        logger.info("🔍 Validating migrated data...")
        
        # Count records in each table
        tables = ['user', 'session', 'message', 'concept', 'thought']
        for table in tables:
            result = await self.db.query(f"SELECT count() FROM {table}")
            count = result[0].get('count', 0) if result else 0
            logger.info(f"📊 {table}: {count} records")
        
        # Count relationships
        relationships = ['knows', 'contains', 'reflects']
        for rel in relationships:
            result = await self.db.query(f"SELECT count() FROM {rel}")
            count = result[0].get('count', 0) if result else 0
            logger.info(f"🔗 {rel}: {count} relationships")
        
        # Test graph queries
        logger.info("🧠 Testing graph queries...")
        
        # Get user's known concepts
        concepts_result = await self.db.query("""
            SELECT ->knows->concept.* FROM user:peppi
        """)
        logger.info(f"User knows concepts: {len(concepts_result)} found")
        
        # Get session messages
        messages_result = await self.db.query(f"""
            SELECT ->contains->message.* FROM session 
            WHERE string::contains(string(id), 'session:')
            LIMIT 1
        """)
        logger.info(f"Session messages: {len(messages_result)} found")
        
        # Get session thoughts
        thoughts_result = await self.db.query(f"""
            SELECT ->reflects->thought.* FROM session 
            WHERE string::contains(string(id), 'session:')
            LIMIT 1
        """)
        logger.info(f"Session thoughts: {len(thoughts_result)} found")
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            logger.info("✅ Database connection closed")

async def main():
    migration = DirectMigration()
    try:
        await migration.connect()
        await migration.create_sample_data()
        await migration.validate_data()
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await migration.close()

if __name__ == "__main__":
    asyncio.run(main())