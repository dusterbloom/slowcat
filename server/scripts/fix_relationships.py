#!/usr/bin/env python3
"""
Fix Graph Relationships - Repair the relationships that failed during migration

This script fixes the relationship creation issue in the migration by
providing all required schema fields for knows, contains, and reflects tables.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

class RelationshipFixer:
    def __init__(self, url="ws://127.0.0.1:8000/rpc", ns="slowcat", db="memory_graph", 
                 user="root", password="slowcat_secure_2024"):
        self.url = url
        self.ns = ns
        self.db_name = db
        self.user = user
        self.password = password
        self.db = None
        
    async def connect(self):
        """Connect to SurrealDB"""
        self.db = AsyncSurreal(self.url)
        await self.db.connect()
        await self.db.signin({
            "username": self.user,
            "password": self.password
        })
        await self.db.use(self.ns, self.db_name)
        logger.info(f"✅ Connected to {self.url}/{self.ns}/{self.db_name}")
    
    async def fix_user_concept_relationships(self):
        """Create knows relationships between users and concepts"""
        logger.info("🔗 Creating user->knows->concept relationships...")
        
        # Get all users and concepts
        users = await self.db.query("SELECT * FROM user")
        concepts = await self.db.query("SELECT * FROM concept")
        
        logger.info(f"Found {len(users)} users and {len(concepts)} concepts")
        
        created_count = 0
        for user in users:
            user_id = str(user['id'])
            
            for concept in concepts:
                concept_id = str(concept['id'])
                concept_name = concept.get('name', 'unknown')
                
                try:
                    # Create knows relationship with all required fields
                    result = await self.db.query("""
                        RELATE $user_id->knows->$concept_id SET
                            relationship = $rel_type,
                            strength = 0.7,
                            fidelity = 3,
                            access_count = 1,
                            decay_rate = 1.0,
                            learned_at = time::now(),
                            reinforced_at = time::now(),
                            source_message = NONE
                    """, {
                        "user_id": user_id,
                        "concept_id": concept_id,
                        "rel_type": f"knows_{concept_name}"
                    })
                    
                    created_count += 1
                    logger.debug(f"✅ {user_id} knows {concept_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create knows relationship: {e}")
        
        logger.info(f"✅ Created {created_count} knows relationships")
    
    async def fix_session_message_relationships(self):
        """Create contains relationships between sessions and messages"""
        logger.info("🔗 Creating session->contains->message relationships...")
        
        # Get all sessions and messages
        sessions = await self.db.query("SELECT * FROM session")
        messages = await self.db.query("SELECT * FROM message")
        
        logger.info(f"Found {len(sessions)} sessions and {len(messages)} messages")
        
        created_count = 0
        for session in sessions:
            session_id = str(session['id'])
            
            # Find messages that belong to this session (by session_id field or pattern)
            for message in messages:
                message_id = str(message['id'])
                msg_session_id = message.get('session_id', '')
                
                # Check if message belongs to this session
                if session_id in str(msg_session_id) or str(msg_session_id).endswith(str(session_id).split(':')[-1]):
                    try:
                        # Create contains relationship with all required fields
                        result = await self.db.query("""
                            RELATE $session_id->contains->$message_id SET
                                sequence_num = $seq_num,
                                created_at = time::now()
                        """, {
                            "session_id": session_id,
                            "message_id": message_id,
                            "seq_num": message.get('sequence_num', 1)
                        })
                        
                        created_count += 1
                        logger.debug(f"✅ {session_id} contains {message_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to create contains relationship: {e}")
        
        logger.info(f"✅ Created {created_count} contains relationships")
    
    async def fix_session_thought_relationships(self):
        """Create reflects relationships between sessions and thoughts"""
        logger.info("🔗 Creating session->reflects->thought relationships...")
        
        # Get all sessions and thoughts
        sessions = await self.db.query("SELECT * FROM session")
        thoughts = await self.db.query("SELECT * FROM thought")
        
        logger.info(f"Found {len(sessions)} sessions and {len(thoughts)} thoughts")
        
        created_count = 0
        for session in sessions:
            session_id = str(session['id'])
            
            # Find thoughts that belong to this session
            for thought in thoughts:
                thought_id = str(thought['id'])
                thought_session_id = thought.get('session_id', '')
                
                # Check if thought belongs to this session
                if session_id in str(thought_session_id) or str(thought_session_id).endswith(str(session_id).split(':')[-1]):
                    try:
                        # Create reflects relationship with all required fields
                        result = await self.db.query("""
                            RELATE $session_id->reflects->$thought_id SET
                                trigger_event = 'conversation_end',
                                generated_at = time::now()
                        """, {
                            "session_id": session_id,
                            "thought_id": thought_id
                        })
                        
                        created_count += 1
                        logger.debug(f"✅ {session_id} reflects {thought_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to create reflects relationship: {e}")
        
        logger.info(f"✅ Created {created_count} reflects relationships")
    
    async def validate_relationships(self):
        """Validate that relationships were created correctly"""
        logger.info("🔍 Validating relationships...")
        
        # Check relationship counts
        tables = ['knows', 'contains', 'reflects']
        for table in tables:
            try:
                result = await self.db.query(f"SELECT count() FROM {table}")
                count = result[0].get('count', 0) if result else 0
                logger.info(f"📊 {table}: {count} relationships")
            except Exception as e:
                logger.error(f"❌ Failed to count {table}: {e}")
        
        # Test graph queries
        logger.info("🧠 Testing graph traversals...")
        
        try:
            # Test user->knows->concept
            user_concepts = await self.db.query("SELECT ->knows->concept.* FROM user LIMIT 1")
            logger.info(f"✅ User concepts query: {len(user_concepts)} results")
            
            # Test session->contains->message
            session_messages = await self.db.query("SELECT ->contains->message.* FROM session LIMIT 1") 
            logger.info(f"✅ Session messages query: {len(session_messages)} results")
            
            # Test session->reflects->thought
            session_thoughts = await self.db.query("SELECT ->reflects->thought.* FROM session LIMIT 1")
            logger.info(f"✅ Session thoughts query: {len(session_thoughts)} results")
            
        except Exception as e:
            logger.error(f"❌ Graph query failed: {e}")
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            logger.info("✅ Database connection closed")

async def main():
    fixer = RelationshipFixer()
    
    try:
        await fixer.connect()
        await fixer.fix_user_concept_relationships()
        await fixer.fix_session_message_relationships()
        await fixer.fix_session_thought_relationships()
        await fixer.validate_relationships()
        
        logger.info("🎉 Relationship fixing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Relationship fixing failed: {e}")
        raise
    finally:
        await fixer.close()

if __name__ == "__main__":
    asyncio.run(main())