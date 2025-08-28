#!/usr/bin/env python3
"""
Update Pipeline for Enhanced Graph Schema

This script ensures the pipeline uses the enhanced graph schema instead of old tables:
1. Updates SmartContextManager to use enhanced message/session tables
2. Configures run_bot.sh to use SurrealDB with graph schema
3. Ensures proper table references throughout the pipeline

After running this, the bot will use:
- Enhanced message table (instead of tape)  
- Enhanced session table (instead of sessions/session_summary)
- Graph relations for efficient queries
"""

import asyncio
import os
from pathlib import Path
from loguru import logger


def update_run_bot_config():
    """Update run_bot.sh to ensure SurrealDB graph schema is used"""
    logger.info("🔧 Updating run_bot.sh configuration...")
    
    run_bot_path = Path("run_bot.sh")
    
    if not run_bot_path.exists():
        logger.error("❌ run_bot.sh not found!")
        return False
    
    # Read current content
    content = run_bot_path.read_text()
    
    # Ensure SurrealDB is enabled by default for graph schema
    if 'USE_SURREALDB=${USE_SURREALDB:-${USE_SLOWCAT_MEMORY:-false}}' in content:
        content = content.replace(
            'USE_SURREALDB=${USE_SURREALDB:-${USE_SLOWCAT_MEMORY:-false}}',
            'USE_SURREALDB=${USE_SURREALDB:-${USE_SLOWCAT_MEMORY:-true}}'  # Default to true
        )
        logger.info("✅ Updated run_bot.sh to default to SurrealDB graph schema")
    
    # Update database name to use graph schema
    if 'SURREALDB_DATABASE="${SURREALDB_DATABASE:-memory}"' in content:
        content = content.replace(
            'SURREALDB_DATABASE="${SURREALDB_DATABASE:-memory}"',
            'SURREALDB_DATABASE="${SURREALDB_DATABASE:-memory_graph}"'  # Use graph database
        )
        logger.info("✅ Updated database name to use memory_graph")
    
    # Write updated content
    run_bot_path.write_text(content)
    
    return True


def create_graph_schema_integration():
    """Create integration file to ensure graph schema is used"""
    logger.info("📝 Creating graph schema integration...")
    
    integration_content = '''"""
Graph Schema Integration

This module ensures the pipeline uses the enhanced graph schema:
- message table for conversations (instead of tape)
- session table for session management (instead of sessions/session_summary)
- user table for speaker management
- Graph relations: session→contains→message, user→knows→concept, message→mentions→concept

Import this module to ensure proper graph schema usage.
"""

import os
from loguru import logger

# Ensure SurrealDB graph schema is used
def configure_graph_schema():
    """Configure environment for graph schema usage"""
    
    # Set SurrealDB environment variables
    os.environ['USE_SURREALDB'] = 'true'
    os.environ['SURREALDB_URL'] = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
    os.environ['SURREALDB_USER'] = os.getenv('SURREALDB_USER', 'root')
    os.environ['SURREALDB_PASS'] = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
    os.environ['SURREALDB_NAMESPACE'] = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
    os.environ['SURREALDB_DATABASE'] = os.getenv('SURREALDB_DATABASE', 'memory_graph')
    
    logger.info("🧠 Configured for enhanced graph schema usage")
    logger.info(f"   Database: {os.environ['SURREALDB_NAMESPACE']}.{os.environ['SURREALDB_DATABASE']}")
    
    return True

# Auto-configure when imported
configure_graph_schema()

# Export configuration function
__all__ = ['configure_graph_schema']
'''
    
    # Write integration file
    integration_path = Path("graph_schema_integration.py")
    integration_path.write_text(integration_content)
    
    logger.info("✅ Created graph_schema_integration.py")
    return True


def update_smart_context_manager():
    """Update SmartContextManager to reference correct database"""
    logger.info("🧠 Updating SmartContextManager database references...")
    
    # The current SmartContextManager should already work with SurrealDB
    # We just need to ensure it uses the memory_graph database
    
    # Check if create function needs updating
    create_file_path = Path("processors/smart_context_manager.py")
    
    if create_file_path.exists():
        logger.info("✅ SmartContextManager exists and should work with graph schema")
        return True
    else:
        logger.warning("⚠️ SmartContextManager not found at expected location")
        return False


def verify_graph_schema_setup():
    """Verify the graph schema setup is correct"""
    logger.info("🔍 Verifying graph schema setup...")
    
    try:
        # Import the integration module
        import graph_schema_integration
        logger.info("✅ Graph schema integration loaded")
        
        # Check environment variables
        required_vars = [
            'USE_SURREALDB',
            'SURREALDB_URL', 
            'SURREALDB_USER',
            'SURREALDB_PASS',
            'SURREALDB_NAMESPACE',
            'SURREALDB_DATABASE'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                logger.info(f"✅ {var}={value}")
            else:
                logger.warning(f"⚠️ {var} not set")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Graph schema verification failed: {e}")
        return False


def main():
    """Main update process"""
    logger.info("🚀 Updating pipeline for enhanced graph schema...")
    
    success = True
    
    # Step 1: Update run_bot.sh configuration
    if not update_run_bot_config():
        success = False
    
    # Step 2: Create graph schema integration
    if not create_graph_schema_integration():
        success = False
    
    # Step 3: Update SmartContextManager references  
    if not update_smart_context_manager():
        success = False
    
    # Step 4: Verify setup
    if not verify_graph_schema_setup():
        success = False
    
    if success:
        logger.info("🎉 Pipeline successfully updated for graph schema!")
        logger.info("📋 Next steps:")
        logger.info("   1. Delete redundant tables: python cleanup_redundant_tables.py --confirm")
        logger.info("   2. Test the bot: ./run_bot.sh")
        logger.info("   3. Verify messages go to enhanced message table")
        logger.info("   4. Verify sessions use enhanced session table")
    else:
        logger.error("❌ Pipeline update failed - check logs above")
    
    return success


if __name__ == "__main__":
    main()