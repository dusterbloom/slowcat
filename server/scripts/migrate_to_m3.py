#!/usr/bin/env python3
"""M3 Migration Script

Safely migrate existing SurrealDB schema to M3 memory system.
This script handles backup, schema migration, and verification.
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from memory.surreal_connection import SurrealConnectionManager
from memory.m3_surreal_integration import M3SurrealIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class M3Migration:
    """Safely migrate existing schema to M3"""
    
    def __init__(self, 
                 connection: SurrealConnectionManager,
                 backup_dir: Optional[str] = None,
                 dry_run: bool = False):
        """Initialize migration
        
        Args:
            connection: SurrealDB connection
            backup_dir: Directory for backups (defaults to ./backups)
            dry_run: If True, only simulate migration without making changes
        """
        self.connection = connection
        self.dry_run = dry_run
        
        # Set backup directory
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path(__file__).parent.parent / "backups"
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration state
        self.backup_files: List[str] = []
        self.migration_applied = False
        
        logger.info(f"M3Migration initialized (dry_run={dry_run}, backup_dir={self.backup_dir})")
    
    async def run_full_migration(self, 
                                migrate_existing_data: bool = False,
                                data_limit: int = 1000) -> bool:
        """Run complete migration process
        
        Args:
            migrate_existing_data: Whether to convert existing messages to M3 nodes
            data_limit: Maximum number of existing records to migrate
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("Starting M3 migration process...")
            
            # Step 1: Check current state
            migration_needed = await self._check_migration_needed()
            if not migration_needed:
                logger.info("M3 migration already applied")
                return True
            
            # Step 2: Backup existing data
            logger.info("Backing up existing data...")
            backup_success = await self.backup_existing_data()
            if not backup_success:
                logger.error("Backup failed, aborting migration")
                return False
            
            # Step 3: Apply schema migration
            logger.info("Applying M3 schema migration...")
            schema_success = await self.apply_schema_migration()
            if not schema_success:
                logger.error("Schema migration failed")
                await self._rollback_migration()
                return False
            
            # Step 4: Migrate existing data (optional)
            if migrate_existing_data:
                logger.info(f"Migrating existing data (limit: {data_limit})...")
                data_success = await self.migrate_existing_messages(data_limit)
                if not data_success:
                    logger.warning("Data migration had issues, but schema is applied")
            
            # Step 5: Verify migration
            logger.info("Verifying M3 migration...")
            verify_success = await self.verify_migration()
            if not verify_success:
                logger.error("Migration verification failed")
                return False
            
            logger.info("M3 migration completed successfully!")
            await self._log_migration_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed with error: {e}")
            await self._rollback_migration()
            return False
    
    async def _check_migration_needed(self) -> bool:
        """Check if M3 migration is needed"""
        try:
            # Check if migration marker exists
            result = await self.connection.db.query(
                "SELECT * FROM migration_status WHERE id = 'migration_status:m3'"
            )
            
            if result and result[0].get('status') == 'completed':
                return False
            
            # Check if M3 tables exist
            info_result = await self.connection.db.query("INFO FOR DB")
            if info_result:
                tables = info_result[0].get('tb', {})
                m3_tables = ['m3_nodes', 'm3_edges', 'm3_clips', 'm3_equivalences']
                
                if all(table in tables for table in m3_tables):
                    logger.info("M3 tables exist but no migration marker found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check migration state: {e}")
            return True  # Assume migration needed if we can't check
    
    async def backup_existing_data(self) -> bool:
        """Backup current messages, sessions, entities"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Tables to backup
            tables_to_backup = [
                'messages', 'sessions', 'speakers', 'entity',
                'message_belongs_to', 'session_involves', 'entity_mentioned_in'
            ]
            
            # Legacy tables to backup (if they exist)
            legacy_tables = [
                'memory_fragments', 'field_states', 'engrams', 
                'knowledge', 'engram_appears_in', 'engram_contains'
            ]
            
            backup_summary = {
                'timestamp': timestamp,
                'tables': {},
                'total_records': 0
            }
            
            for table in tables_to_backup + legacy_tables:
                try:
                    if self.dry_run:
                        # Just count records
                        count_result = await self.connection.db.query(f"SELECT count() FROM {table} GROUP ALL")
                        count = count_result[0]['count'] if count_result else 0
                        backup_summary['tables'][table] = {'count': count, 'status': 'simulated'}
                        logger.info(f"[DRY RUN] Would backup {table}: {count} records")
                        continue
                    
                    # Get all records from table
                    result = await self.connection.db.query(f"SELECT * FROM {table}")
                    
                    if result:
                        # Save to backup file
                        backup_file = self.backup_dir / f"{table}_{timestamp}.json"
                        with open(backup_file, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                        
                        self.backup_files.append(str(backup_file))
                        record_count = len(result)
                        backup_summary['tables'][table] = {
                            'count': record_count,
                            'file': str(backup_file),
                            'status': 'success'
                        }
                        backup_summary['total_records'] += record_count
                        
                        logger.info(f"Backed up {table}: {record_count} records -> {backup_file}")
                    else:
                        backup_summary['tables'][table] = {'count': 0, 'status': 'empty'}
                        logger.info(f"Table {table} is empty, skipping backup")
                
                except Exception as e:
                    logger.warning(f"Failed to backup table {table}: {e}")
                    backup_summary['tables'][table] = {'count': 0, 'status': f'error: {e}'}
            
            # Save backup summary
            summary_file = self.backup_dir / f"backup_summary_{timestamp}.json"
            if not self.dry_run:
                with open(summary_file, 'w') as f:
                    json.dump(backup_summary, f, indent=2, default=str)
            
            logger.info(f"Backup completed: {backup_summary['total_records']} total records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup existing data: {e}")
            return False
    
    async def apply_schema_migration(self) -> bool:
        """Apply M3 schema changes"""
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would apply M3 schema migration")
                return True
            
            # Prefer production migration aligned with slowcat/memory_graph schema
            schema_dir = Path(__file__).parent.parent / "schema"
            prod_file = schema_dir / "m3_production_migration.surql"
            legacy_file = schema_dir / "m3_migration.surql"

            if prod_file.exists():
                migration_file = prod_file
            elif legacy_file.exists():
                migration_file = legacy_file
            else:
                logger.error("No M3 migration file found (expected m3_production_migration.surql or m3_migration.surql)")
                return False

            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            logger.info(f"Using migration file: {migration_file.name}
")
            
            # Apply migration
            logger.info("Executing M3 schema migration...")
            try:
                result = await self.connection.db.query(migration_sql)
                logger.info(f"Migration result: {result}")
                
                # SurrealDB returns None for successful DDL operations
                logger.info("M3 schema migration executed successfully")
                self.migration_applied = True
                return True
            except Exception as e:
                logger.error(f"M3 schema migration failed with exception: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to apply schema migration: {e}")
            return False
    
    async def migrate_existing_messages(self, limit: int = 1000) -> bool:
        """Convert existing messages to M3 nodes (optional)
        
        Args:
            limit: Maximum number of messages to migrate
            
        Returns:
            True if migration successful, False if errors occurred
        """
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would migrate up to {limit} existing messages")
                return True
            
            # Get existing messages
            result = await self.connection.db.query(
                "SELECT * FROM messages ORDER BY created_at DESC LIMIT $limit",
                {"limit": limit}
            )
            
            if not result:
                logger.info("No existing messages to migrate")
                return True
            
            messages = result
            logger.info(f"Found {len(messages)} messages to migrate")
            
            # Initialize M3 integration for data migration
            m3_integration = M3SurrealIntegration(self.connection)
            await m3_integration.initialize()
            
            # Initialize embedding service if available
            embedding_service = None
            try:
                # Try to create a simple embedding service
                class SimpleEmbeddingService:
                    async def get_embedding(self, text: str):
                        # Simple hash-based embedding for migration
                        import hashlib
                        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000000
                        return [float(hash_val % 10) / 10.0 for _ in range(384)]  # 384-dim embedding
                embedding_service = SimpleEmbeddingService()
            except Exception as e:
                logger.warning(f"Embedding service not available: {e}")
            
            migrated_count = 0
            error_count = 0
            
            for message in messages:
                try:
                    await self._migrate_single_message(message, m3_integration, embedding_service)
                    migrated_count += 1
                    
                    if migrated_count % 10 == 0:
                        logger.info(f"Migrated {migrated_count}/{len(messages)} messages...")
                        
                except Exception as e:
                    logger.error(f"Failed to migrate message {message.get('id', 'unknown')}: {e}")
                    error_count += 1
            
            logger.info(f"Data migration completed: {migrated_count} migrated, {error_count} errors")
            return error_count == 0
            
        except Exception as e:
            logger.error(f"Failed to migrate existing messages: {e}")
            return False
    
    async def _migrate_single_message(self, 
                                    message: Dict[str, Any], 
                                    m3_integration: M3SurrealIntegration,
                                    embedding_service: Optional[Any]):
        """Migrate a single message to M3 node"""
        try:
            content = message.get('content', '')
            if not content or not content.strip():
                return
            
            role = message.get('role', 'user')
            speaker_id = message.get('speaker_id', 'unknown')
            
            # Determine node type based on role
            if role == 'assistant':
                node_type = 'semantic'
                speaker_id = 'assistant'
            elif role == 'user':
                node_type = 'voice'  # Assume user messages are transcribed speech
            else:
                node_type = 'semantic'
            
            # Generate embedding if service available
            embeddings = []
            if embedding_service:
                try:
                    embedding = await embedding_service.get_embedding(content)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for message: {e}")
            
            # Create M3 node
            node_id = await m3_integration.store_m3_node(
                node_type=node_type,
                contents=[content],
                embeddings=embeddings,
                source_message_id=str(message.get('id', '')),
                speaker_id=speaker_id,
                extraction_method='migration',
                confidence=0.7
            )
            
            if node_id:
                logger.debug(f"Migrated message to node {node_id}")
            
        except Exception as e:
            raise Exception(f"Failed to migrate message: {e}")
    
    async def verify_migration(self) -> bool:
        """Test M3 operations work correctly"""
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would verify M3 migration")
                return True
            
            # Try built-in verification function if available
            try:
                result = await self.connection.db.query("RETURN fn::verify_m3_migration()")
                logger.info(f"Verification query result: {result}")
            except Exception as e:
                logger.info(f"verify_m3_migration() unavailable, proceeding with direct verification: {e}")

            # Test basic M3 operations
            m3_integration = M3SurrealIntegration(self.connection)
            await m3_integration.initialize()
            
            # Test node creation
            test_node_id = await m3_integration.store_m3_node(
                node_type="semantic",
                contents=["Migration verification test"],
                embeddings=[],
                speaker_id="test",
                extraction_method="verification",
                confidence=1.0
            )
            
            if test_node_id:
                logger.info(f"Test node created successfully: {test_node_id}")
                
                # Clean up test node
                await self.connection.db.query(
                    "DELETE FROM m3_nodes WHERE node_id = $node_id",
                    {"node_id": test_node_id}
                )
                
                return True
            else:
                logger.error("Failed to create test node")
                return False
                
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False
    
    async def _rollback_migration(self):
        """Rollback migration if something went wrong"""
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would rollback migration")
                return
            
            if not self.migration_applied:
                logger.info("No migration to rollback")
                return
            
            logger.warning("Rolling back M3 migration...")
            
            # Remove M3 tables
            rollback_queries = [
                "REMOVE TABLE IF EXISTS m3_nodes;",
                "REMOVE TABLE IF EXISTS m3_edges;",
                "REMOVE TABLE IF EXISTS m3_equivalences;",
                "REMOVE TABLE IF EXISTS m3_clips;",
                "REMOVE TABLE IF EXISTS m3_node_from_message;",
                "REMOVE TABLE IF EXISTS m3_node_about_entity;",
                "DELETE migration_status:m3;"
            ]
            
            for query in rollback_queries:
                try:
                    await self.connection.db.query(query)
                except Exception as e:
                    logger.warning(f"Rollback query failed: {query} - {e}")
            
            logger.info("Migration rollback completed")
            
        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
    
    async def _log_migration_summary(self):
        """Log migration summary"""
        try:
            # Get M3 statistics
            m3_integration = M3SurrealIntegration(self.connection)
            await m3_integration.initialize()
            stats = await m3_integration.get_statistics()
            
            logger.info("=== M3 Migration Summary ===")
            logger.info(f"Total M3 nodes: {stats.get('total_nodes', 0)}")
            logger.info(f"Total M3 edges: {stats.get('total_edges', 0)}")
            logger.info(f"Total M3 clips: {stats.get('total_clips', 0)}")
            logger.info(f"Current clip ID: {stats.get('current_clip_id', 'N/A')}")
            
            nodes_by_type = stats.get('nodes_by_type', [])
            if nodes_by_type:
                logger.info("Nodes by type:")
                for type_info in nodes_by_type:
                    logger.info(f"  {type_info.get('node_type', 'unknown')}: {type_info.get('count', 0)}")
            
            logger.info(f"Backup files: {len(self.backup_files)}")
            for backup_file in self.backup_files:
                logger.info(f"  {backup_file}")
            
            logger.info("============================")
            
        except Exception as e:
            logger.error(f"Failed to log migration summary: {e}")

async def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description="Migrate SurrealDB schema to M3 memory system")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without making changes")
    parser.add_argument("--migrate-data", action="store_true", help="Migrate existing messages to M3 nodes")
    parser.add_argument("--data-limit", type=int, default=1000, help="Maximum messages to migrate")
    parser.add_argument("--backup-dir", help="Directory for backups")
    parser.add_argument("--db-url", default="ws://localhost:8000", help="SurrealDB URL")
    parser.add_argument("--namespace", default="slowcat", help="SurrealDB namespace")
    parser.add_argument("--database", default="memory", help="SurrealDB database")
    
    args = parser.parse_args()
    
    try:
        # Initialize connection
        connection = SurrealConnectionManager(
            url=args.db_url,
            namespace=args.namespace,
            database=args.database
        )
        
        await connection.connect()
        logger.info(f"Connected to SurrealDB at {args.db_url}")
        
        # Run migration
        migration = M3Migration(
            connection=connection,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run
        )
        
        success = await migration.run_full_migration(
            migrate_existing_data=args.migrate_data,
            data_limit=args.data_limit
        )
        
        await connection.disconnect()
        
        if success:
            logger.info("Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("Migration failed!")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
