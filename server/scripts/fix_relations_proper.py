#!/usr/bin/env python3
"""
Fix Graph Relationships Using Proper SurrealDB SDK Methods

Use db.insert_relation() instead of RELATE queries to properly create relationships.
Based on: https://surrealdb.com/docs/sdk/python/methods/insert-relation
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

# Try to import RecordID
try:
    from surrealdb import RecordID
    HAS_RECORD_ID = True
except ImportError:
    HAS_RECORD_ID = False
    logger.warning("RecordID not available, using string IDs")

class ProperRelationshipFixer:
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
    
    def _make_record_id(self, table: str, id_val: str):
        """Create RecordID object if available, otherwise return string"""
        if HAS_RECORD_ID:
            return RecordID(table, id_val)
        else:
            return f"{table}:{id_val}"
    
    async def create_user_concept_relationships(self):
        """Create knows relationships using insert_relation"""
        logger.info("🔗 Creating user->knows->concept relationships...")
        
        # Get all users and concepts
        users = await self.db.query("SELECT * FROM user")
        concepts = await self.db.query("SELECT * FROM concept")
        
        logger.info(f"Found {len(users)} users and {len(concepts)} concepts")
        
        created_count = 0
        for user in users:
            user_id_str = str(user['id'])  # e.g., "user:peppi"
            user_table, user_key = user_id_str.split(':', 1)
            
            for concept in concepts:
                concept_id_str = str(concept['id'])  # e.g., "concept:potola"
                concept_table, concept_key = concept_id_str.split(':', 1)
                concept_name = concept.get('name', 'unknown')
                
                try:
                    # Use proper insert_relation method
                    if HAS_RECORD_ID:
                        # Method 1: Using RecordID objects
                        from_record = RecordID(user_table, user_key)
                        to_record = RecordID(concept_table, concept_key)
                    else:
                        # Method 2: Using string record IDs
                        from_record = user_id_str
                        to_record = concept_id_str
                    
                    # Create the relationship with all required fields
                    result = await self.db.insert_relation(
                        "knows",  # relation table
                        from_record,  # from (user)
                        to_record,   # to (concept)
                        {
                            "relationship": f"knows_{concept_name}",
                            "strength": 0.7,
                            "fidelity": 3,
                            "access_count": 1,
                            "decay_rate": 1.0,
                            "learned_at": "time::now()",
                            "reinforced_at": "time::now()",
                            "source_message": None
                        }
                    )
                    
                    created_count += 1
                    logger.debug(f"✅ {user_id_str} knows {concept_id_str}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create knows relationship {user_id_str} -> {concept_id_str}: {e}")
        
        logger.info(f"✅ Created {created_count} knows relationships using insert_relation")
    
    async def create_session_message_relationships(self):
        """Create contains relationships using insert_relation"""
        logger.info("🔗 Creating session->contains->message relationships...")
        
        # Get all sessions and messages
        sessions = await self.db.query("SELECT * FROM session")
        messages = await self.db.query("SELECT * FROM message")
        
        logger.info(f"Found {len(sessions)} sessions and {len(messages)} messages")
        
        created_count = 0
        for session in sessions:
            session_id_str = str(session['id'])
            session_table, session_key = session_id_str.split(':', 1)
            
            # Find messages that belong to this session
            for message in messages:
                message_id_str = str(message['id'])
                message_table, message_key = message_id_str.split(':', 1)
                msg_session_id = message.get('session_id', '')
                
                # Check if message belongs to this session
                if session_key in str(msg_session_id):
                    try:
                        if HAS_RECORD_ID:
                            from_record = RecordID(session_table, session_key)
                            to_record = RecordID(message_table, message_key)
                        else:
                            from_record = session_id_str
                            to_record = message_id_str
                        
                        # Create contains relationship
                        result = await self.db.insert_relation(
                            "contains",
                            from_record,
                            to_record,
                            {
                                "sequence_num": message.get('sequence_num', 1),
                                "created_at": "time::now()"
                            }
                        )
                        
                        created_count += 1
                        if created_count % 100 == 0:
                            logger.info(f"  Created {created_count} contains relationships...")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to create contains relationship: {e}")
        
        logger.info(f"✅ Created {created_count} contains relationships using insert_relation")
    
    async def create_session_thought_relationships(self):
        """Create reflects relationships using insert_relation"""
        logger.info("🔗 Creating session->reflects->thought relationships...")
        
        # Get all sessions and thoughts
        sessions = await self.db.query("SELECT * FROM session")
        thoughts = await self.db.query("SELECT * FROM thought")
        
        logger.info(f"Found {len(sessions)} sessions and {len(thoughts)} thoughts")
        
        created_count = 0
        for session in sessions:
            session_id_str = str(session['id'])
            session_table, session_key = session_id_str.split(':', 1)
            
            # Find thoughts that belong to this session
            for thought in thoughts:
                thought_id_str = str(thought['id'])
                thought_table, thought_key = thought_id_str.split(':', 1)
                thought_session_id = thought.get('session_id', '')
                
                # Check if thought belongs to this session
                if session_key in str(thought_session_id):
                    try:
                        if HAS_RECORD_ID:
                            from_record = RecordID(session_table, session_key)
                            to_record = RecordID(thought_table, thought_key)
                        else:
                            from_record = session_id_str
                            to_record = thought_id_str
                        
                        # Create reflects relationship
                        result = await self.db.insert_relation(
                            "reflects",
                            from_record,
                            to_record,
                            {
                                "trigger_event": "conversation_end",
                                "generated_at": "time::now()"
                            }
                        )
                        
                        created_count += 1
                        logger.debug(f"✅ {session_id_str} reflects {thought_id_str}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to create reflects relationship: {e}")
        
        logger.info(f"✅ Created {created_count} reflects relationships using insert_relation")
    
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
        
        # Test specific queries
        try:
            # Test user->knows->concept
            peppi_concepts = await self.db.query("SELECT out.* FROM knows WHERE in = user:peppi")
            logger.info(f"✅ user:peppi knows {len(peppi_concepts)} concepts")
            
            # Test session->contains->message
            sample_messages = await self.db.query("SELECT out.content FROM contains LIMIT 3")
            logger.info(f"✅ Sample message query returned {len(sample_messages)} results")
            
        except Exception as e:
            logger.error(f"❌ Graph query failed: {e}")
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            logger.info("✅ Database connection closed")

async def main():
    fixer = ProperRelationshipFixer()
    
    try:
        await fixer.connect()
        await fixer.create_user_concept_relationships()
        await fixer.create_session_message_relationships()
        await fixer.create_session_thought_relationships()
        await fixer.validate_relationships()
        
        logger.info("🎉 Proper relationship creation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Relationship creation failed: {e}")
        raise
    finally:
        await fixer.close()

if __name__ == "__main__":
    asyncio.run(main())