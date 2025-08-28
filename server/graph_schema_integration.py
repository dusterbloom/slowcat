"""
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
