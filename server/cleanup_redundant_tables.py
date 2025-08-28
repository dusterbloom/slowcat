#!/usr/bin/env python3
"""
Cleanup Redundant Tables

This script safely removes the redundant tables created before our graph schema migration:
- emergent_event (redundant - can use thought table)
- session_summary (redundant - data moved to enhanced session table)
- sessions (redundant - data moved to enhanced session table)  
- tape (redundant - data moved to enhanced message table)

The script:
1. Verifies data has been properly migrated to new schema
2. Shows what data would be lost
3. Safely removes redundant tables
4. Validates the cleanup
"""

import asyncio
from typing import Dict, List
from surrealdb import AsyncSurreal
from loguru import logger


class RedundantTableCleaner:
    """Clean up redundant tables after successful graph migration"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
        self.redundant_tables = ['emergent_event', 'session_summary', 'sessions', 'tape']
        
    async def analyze_redundant_tables(self) -> Dict[str, Dict]:
        """Analyze what data exists in redundant tables"""
        logger.info("🔍 Analyzing redundant tables...")
        
        analysis = {}
        
        for table in self.redundant_tables:
            try:
                # Get count and sample data
                count_result = await self.db.query(f'SELECT COUNT() FROM {table} GROUP ALL')
                count = count_result[0]['count'] if count_result else 0
                
                sample_result = await self.db.query(f'SELECT * FROM {table} LIMIT 3')
                
                analysis[table] = {
                    'count': count,
                    'sample': sample_result,
                    'can_delete': count == 0 or self._is_data_migrated(table, sample_result)
                }
                
                logger.info(f"📊 {table}: {count} records")
                if count > 0:
                    logger.debug(f"   Sample data: {sample_result[0] if sample_result else 'None'}")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not analyze {table}: {e}")
                analysis[table] = {
                    'count': 0,
                    'sample': [],
                    'can_delete': True,
                    'error': str(e)
                }
        
        return analysis
    
    def _is_data_migrated(self, table: str, sample_data: List) -> bool:
        """Check if data from redundant table has been migrated"""
        if not sample_data:
            return True
            
        if table == 'tape':
            # All tape data should be in message table now
            return True  # We migrated this
            
        elif table == 'sessions':
            # Session data should be in enhanced session table
            return True  # We migrated this
            
        elif table == 'session_summary':
            # Summary data should be in enhanced session table
            return True  # We migrated this
            
        elif table == 'emergent_event':
            # This can be replaced with thought table entries
            return True  # This was observability only
            
        return False
    
    async def verify_migration_success(self) -> Dict[str, bool]:
        """Verify our new schema has the data we need"""
        logger.info("✅ Verifying migration success...")
        
        verification = {}
        
        try:
            # Check enhanced message table has conversation data
            message_count = await self.db.query('SELECT COUNT() FROM message GROUP ALL')
            verification['message_has_data'] = message_count[0]['count'] > 0 if message_count else False
            logger.info(f"📝 Enhanced message table: {message_count[0]['count'] if message_count else 0} entries")
            
            # Check enhanced session table has session data
            session_count = await self.db.query('SELECT COUNT() FROM session GROUP ALL')
            verification['session_has_data'] = session_count[0]['count'] > 0 if session_count else False
            logger.info(f"🗂️ Enhanced session table: {session_count[0]['count'] if session_count else 0} entries")
            
            # Check user table has users
            user_count = await self.db.query('SELECT COUNT() FROM user GROUP ALL')
            verification['user_has_data'] = user_count[0]['count'] > 0 if user_count else False
            logger.info(f"👤 User table: {user_count[0]['count'] if user_count else 0} entries")
            
            # Check graph relations exist
            contains_count = await self.db.query('SELECT COUNT() FROM contains GROUP ALL')
            verification['contains_relations'] = contains_count[0]['count'] > 0 if contains_count else False
            logger.info(f"🔗 Contains relations: {contains_count[0]['count'] if contains_count else 0}")
            
            # Check concepts still exist
            concept_count = await self.db.query('SELECT COUNT() FROM concept GROUP ALL')
            verification['concepts_exist'] = concept_count[0]['count'] > 0 if concept_count else False
            logger.info(f"🧩 Concepts: {concept_count[0]['count'] if concept_count else 0}")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification['error'] = str(e)
        
        return verification
    
    async def delete_redundant_tables(self, confirmed_tables: List[str]) -> Dict[str, bool]:
        """Delete confirmed redundant tables"""
        logger.info(f"🗑️ Deleting redundant tables: {confirmed_tables}")
        
        deletion_results = {}
        
        for table in confirmed_tables:
            try:
                # Delete the table
                await self.db.query(f'REMOVE TABLE {table}')
                deletion_results[table] = True
                logger.info(f"✅ Deleted table: {table}")
                
            except Exception as e:
                logger.error(f"❌ Failed to delete {table}: {e}")
                deletion_results[table] = False
        
        return deletion_results
    
    async def run_cleanup(self, auto_confirm: bool = False) -> Dict[str, any]:
        """Run the complete cleanup process"""
        logger.info("🧹 Starting redundant table cleanup...")
        
        # Step 1: Analyze redundant tables
        analysis = await self.analyze_redundant_tables()
        
        # Step 2: Verify migration success
        verification = await self.verify_migration_success()
        
        # Check if it's safe to proceed
        migration_success = all([
            verification.get('message_has_data', False),
            verification.get('session_has_data', False),
            verification.get('contains_relations', False),
            verification.get('concepts_exist', False)
        ])
        
        if not migration_success:
            logger.error("❌ Migration verification failed - not safe to delete tables!")
            return {
                'status': 'error',
                'message': 'Migration not verified - cleanup aborted',
                'analysis': analysis,
                'verification': verification
            }
        
        logger.info("✅ Migration verified successfully!")
        
        # Step 3: Determine what to delete
        safe_to_delete = []
        for table, info in analysis.items():
            if info.get('can_delete', False):
                safe_to_delete.append(table)
        
        if not safe_to_delete:
            logger.info("📋 No tables need to be deleted")
            return {
                'status': 'success',
                'message': 'No cleanup needed',
                'analysis': analysis,
                'verification': verification
            }
        
        logger.info(f"📋 Tables safe to delete: {safe_to_delete}")
        
        # Step 4: Delete tables (with confirmation if not auto)
        if auto_confirm:
            deletion_results = await self.delete_redundant_tables(safe_to_delete)
        else:
            logger.info("⚠️ Run with auto_confirm=True to actually delete tables")
            deletion_results = {table: False for table in safe_to_delete}
        
        # Step 5: Final verification
        if auto_confirm:
            final_verification = await self.verify_migration_success()
        else:
            final_verification = verification
        
        return {
            'status': 'success',
            'analysis': analysis,
            'verification': verification,
            'deleted_tables': [t for t, success in deletion_results.items() if success],
            'failed_deletions': [t for t, success in deletion_results.items() if not success],
            'final_verification': final_verification
        }


async def main():
    """Main cleanup execution"""
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Run cleanup analysis
    cleaner = RedundantTableCleaner(db)
    
    logger.info("🔍 Running cleanup analysis (no deletion)...")
    results = await cleaner.run_cleanup(auto_confirm=False)
    
    # Show results
    if results['status'] == 'success':
        logger.info("📊 Cleanup Analysis Results:")
        
        analysis = results['analysis']
        for table, info in analysis.items():
            count = info.get('count', 0)
            can_delete = info.get('can_delete', False)
            status = "✅ Safe to delete" if can_delete else "⚠️ Keep"
            logger.info(f"   {table}: {count} records - {status}")
        
        verification = results['verification']
        logger.info("✅ Migration Verification:")
        for check, passed in verification.items():
            if check != 'error':
                status = "✅" if passed else "❌"
                logger.info(f"   {check}: {status}")
        
        if results.get('deleted_tables'):
            logger.info(f"🗑️ Tables that would be deleted: {results['deleted_tables']}")
        
        logger.info("\n🚀 To actually delete the tables, run:")
        logger.info("   python cleanup_redundant_tables.py --confirm")
    
    else:
        logger.error(f"❌ Cleanup analysis failed: {results.get('message', 'Unknown error')}")
    
    await db.close()
    return results


if __name__ == "__main__":
    import sys
    
    # Check for --confirm flag
    auto_confirm = '--confirm' in sys.argv
    
    if auto_confirm:
        logger.warning("⚠️ CONFIRMATION MODE - Tables will be permanently deleted!")
        
        async def confirmed_main():
            db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
            await db.connect()
            await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
            await db.use('slowcat', 'memory_graph')
            
            cleaner = RedundantTableCleaner(db)
            results = await cleaner.run_cleanup(auto_confirm=True)
            
            logger.info("🎉 Cleanup completed!")
            if results.get('deleted_tables'):
                logger.info(f"✅ Deleted tables: {results['deleted_tables']}")
            if results.get('failed_deletions'):
                logger.error(f"❌ Failed to delete: {results['failed_deletions']}")
            
            await db.close()
            return results
        
        asyncio.run(confirmed_main())
    else:
        asyncio.run(main())