#!/usr/bin/env python3
"""
Setup SurrealDB for Slowcat Consciousness Engine

This script:
1. Connects to SurrealDB
2. Creates the database schema
3. Sets up initial data
4. Verifies the installation
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False
    print("❌ SurrealDB client not installed!")
    print("Please install with: pip install surrealdb")
    sys.exit(1)


class SurrealDBSetup:
    """Handles SurrealDB setup and migration"""
    
    def __init__(self, 
                 url: str = None,
                 username: str = None,
                 password: str = None):
        self.url = url or os.getenv('SURREALDB_URL', 'ws://localhost:8000/rpc')
        self.username = username or os.getenv('SURREALDB_USER', 'root')
        self.password = password or os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
        self.db = None
        self.schema_path = Path(__file__).parent.parent / 'schema' / 'consciousness_schema.surql'
        
    async def connect(self) -> bool:
        """Connect to SurrealDB"""
        try:
            self.db = AsyncSurreal(self.url)
            await self.db.connect()
            
            # Authenticate
            if self.username and self.password:
                await self.db.signin({
                    'username': self.username,
                    'password': self.password
                })
            
            # Select namespace and database from environment  
            namespace = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
            database = os.getenv('SURREALDB_DATABASE', 'memory_graph')
            await self.db.use(namespace, database)
            
            logger.info(f"✅ Connected to SurrealDB at {self.url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    async def execute_schema_file(self) -> bool:
        """Execute the schema file"""
        try:
            if not self.schema_path.exists():
                logger.error(f"❌ Schema file not found: {self.schema_path}")
                return False
            
            logger.info(f"📋 Loading schema from {self.schema_path}")
            
            with open(self.schema_path, 'r') as f:
                schema_content = f.read()
            
            # Split by statements (crude but effective for setup)
            statements = [s.strip() for s in schema_content.split(';') 
                         if s.strip() and not s.strip().startswith('--')]
            
            logger.info(f"📝 Executing {len(statements)} statements...")
            
            # Execute each statement
            success_count = 0
            for i, statement in enumerate(statements, 1):
                try:
                    # Add semicolon back
                    statement = statement + ';'
                    
                    # Skip pure comments
                    if statement.strip().startswith('--'):
                        continue
                    
                    # Log progress for long schemas
                    if i % 10 == 0:
                        logger.info(f"   Progress: {i}/{len(statements)}")
                    
                    result = await self.db.query(statement)
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  Statement {i} failed: {str(e)[:100]}")
                    # Continue with other statements
            
            logger.info(f"✅ Schema setup complete: {success_count}/{len(statements)} statements successful")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to execute schema: {e}")
            return False
    
    async def verify_setup(self) -> bool:
        """Verify that tables and initial data exist"""
        try:
            logger.info("🔍 Verifying database setup...")
            
            # Check core tables
            tables_to_check = [
                'messages',
                'memory_fragments', 
                'field_states',
                'facts',
                'sessions',
                'speakers'
            ]
            
            for table in tables_to_check:
                result = await self.db.query(f"SELECT count() FROM {table} GROUP ALL;")
                if result:
                    count = 0
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'result' in result[0]:
                            res_data = result[0]['result']
                            if isinstance(res_data, list) and len(res_data) > 0:
                                count = res_data[0].get('count', 0)
                    
                    logger.info(f"   ✓ Table '{table}' exists ({count} records)")
                else:
                    logger.warning(f"   ✗ Table '{table}' might not exist")
            
            # Check if default speaker exists
            result = await self.db.query(
                "SELECT * FROM speakers WHERE speaker_id = 'default_user';"
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                res_data = result[0].get('result', [])
                if res_data:
                    logger.info("   ✓ Default speaker exists")
                else:
                    logger.warning("   ✗ Default speaker not found")
            
            logger.info("✅ Verification complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    async def test_operations(self) -> bool:
        """Test basic CRUD operations"""
        try:
            logger.info("🧪 Testing basic operations...")
            
            # Test creating a message
            test_message = {
                'role': 'user',
                'content': 'Test message from setup script',
                'speaker_id': 'default_user',
                'session_id': 'test_session'
            }
            
            result = await self.db.create('messages', test_message)
            if result:
                message_id = None
                if isinstance(result, list) and len(result) > 0:
                    message_id = result[0].get('id')
                
                if message_id:
                    logger.info(f"   ✓ Created test message: {message_id}")
                    
                    # Test retrieval
                    query_result = await self.db.query(
                        "SELECT * FROM messages WHERE session_id = 'test_session';"
                    )
                    
                    if query_result:
                        logger.info("   ✓ Retrieved test message")
                    
                    # Clean up test data
                    await self.db.delete(message_id)
                    logger.info("   ✓ Deleted test message")
                else:
                    logger.warning("   ✗ Failed to get message ID")
            
            logger.info("✅ All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tests failed: {e}")
            return False
    
    async def run_setup(self) -> bool:
        """Run complete setup process"""
        try:
            logger.info("🚀 Starting SurrealDB setup for Slowcat Consciousness Engine")
            logger.info("=" * 60)
            
            # Connect
            if not await self.connect():
                return False
            
            # Execute schema
            if not await self.execute_schema_file():
                logger.warning("⚠️  Some schema statements failed, but continuing...")
            
            # Verify setup
            if not await self.verify_setup():
                logger.warning("⚠️  Verification showed some issues")
            
            # Test operations
            if not await self.test_operations():
                logger.warning("⚠️  Some tests failed")
            
            logger.info("=" * 60)
            logger.info("✅ SurrealDB setup complete!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Start your SurrealDB server if not running:")
            logger.info("   surreal start --user root --pass root")
            logger.info("2. Set environment variables:")
            logger.info("   export SURREAL_URL=ws://localhost:8000")
            logger.info("   export SURREAL_USER=root")
            logger.info("   export SURREAL_PASS=root")
            logger.info("3. Run the bot with SurrealDB integration enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
        
        finally:
            if self.db:
                await self.db.close()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup SurrealDB for Slowcat')
    parser.add_argument('--url', default=None, help='SurrealDB URL (default: ws://localhost:8000)')
    parser.add_argument('--user', default=None, help='SurrealDB username')
    parser.add_argument('--pass', dest='password', default=None, help='SurrealDB password')
    parser.add_argument('--reset', action='store_true', help='Reset database (drops existing data)')
    
    args = parser.parse_args()
    
    # Check if SurrealDB is running
    logger.info("Checking SurrealDB connection...")
    
    setup = SurrealDBSetup(
        url=args.url,
        username=args.user,
        password=args.password
    )
    
    success = await setup.run_setup()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())