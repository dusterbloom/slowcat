#!/usr/bin/env python3
"""
Proper SurrealDB Backup Parser

Parse SurrealDB SURQL backup files correctly handling SurrealDB-specific data types.
"""

import re
import json
from typing import Dict, List, Any
from loguru import logger

class SurrealDBParser:
    def __init__(self, backup_file: str):
        self.backup_file = backup_file
    
    def parse_backup(self) -> Dict[str, List[Dict]]:
        """Parse SurrealDB backup file and extract all data"""
        logger.info(f"Parsing SurrealDB backup: {self.backup_file}")
        
        with open(self.backup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all INSERT statements
        insert_pattern = r'INSERT\s+\[\s*(\{.*?\})\s*\];'
        matches = re.findall(insert_pattern, content, re.DOTALL | re.MULTILINE)
        
        logger.info(f"Found {len(matches)} INSERT statements")
        
        # Parse each table's data
        tables_data = {}
        
        # Get table names from the structure
        table_patterns = {
            'emergent_event': r'INSERT\s+\[\s*(\{[^}]*agent_id[^}]*emergent_event[^}]*\}[^]]*)\];',
            'fact': r'INSERT\s+\[\s*(\{[^}]*fact:[^}]*predicate[^}]*\}[^]]*)\];',
            'session_summary': r'INSERT\s+\[\s*(\{[^}]*session_summary:[^}]*session_id[^}]*\}[^]]*)\];',
            'sessions': r'INSERT\s+\[\s*(\{[^}]*sessions:[^}]*speaker_id[^}]*\}[^]]*)\];',
            'tape': r'INSERT\s+\[\s*(\{[^}]*tape:[^}]*content[^}]*\}[^]]*)\];',
            'thought': r'INSERT\s+\[\s*(\{[^}]*thought:[^}]*agent_id[^}]*\}[^]]*)\];'
        }
        
        # Extract table data using more specific patterns
        lines = content.split('\n')
        current_table = None
        current_insert = ""
        in_insert = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('INSERT ['):
                in_insert = True
                current_insert = line
                continue
            elif in_insert:
                current_insert += " " + line
                if line.endswith('];'):
                    # Complete INSERT found
                    in_insert = False
                    table_name = self._identify_table(current_insert)
                    if table_name:
                        records = self._parse_insert_statement(current_insert)
                        if table_name not in tables_data:
                            tables_data[table_name] = []
                        tables_data[table_name].extend(records)
                    current_insert = ""
        
        # Log results
        for table_name, records in tables_data.items():
            logger.info(f"Extracted {len(records)} records from {table_name}")
        
        return tables_data
    
    def _identify_table(self, insert_statement: str) -> str:
        """Identify table name from INSERT statement content"""
        # Look for record IDs to identify table
        if 'emergent_event:' in insert_statement:
            return 'emergent_event'
        elif 'fact:' in insert_statement and 'predicate' in insert_statement:
            return 'fact'
        elif 'session_summary:' in insert_statement and 'session_id' in insert_statement:
            return 'session_summary'
        elif 'sessions:' in insert_statement and 'speaker_id' in insert_statement:
            return 'sessions'
        elif 'tape:' in insert_statement and 'content' in insert_statement:
            return 'tape'
        elif 'thought:' in insert_statement:
            return 'thought'
        return None
    
    def _parse_insert_statement(self, insert_statement: str) -> List[Dict]:
        """Parse a single INSERT statement into records"""
        try:
            # Remove INSERT [ and ];
            content = insert_statement.replace('INSERT [', '').replace('];', '').strip()
            
            # Convert SurrealDB format to JSON-parseable format
            content = self._convert_surreal_to_json(content)
            
            # Parse as JSON array
            records = json.loads(f'[{content}]')
            return records
        except Exception as e:
            logger.error(f"Failed to parse INSERT statement: {e}")
            logger.debug(f"Content: {content[:200]}...")
            return []
    
    def _convert_surreal_to_json(self, content: str) -> str:
        """Convert SurrealDB format to JSON-parseable format"""
        # Convert float literals (0.6f -> 0.6)
        content = re.sub(r'(\d+\.?\d*)f\b', r'\1', content)
        
        # Convert datetime literals (d'2025-08-26T17:39:44.127520Z' -> "2025-08-26T17:39:44.127520Z")
        content = re.sub(r"d'([^']+)'", r'"\1"', content)
        
        # Convert record IDs (emergent_event:abc123 -> "emergent_event:abc123")
        content = re.sub(r'\b(\w+):([a-zA-Z0-9_]+)\b', r'"\1:\2"', content)
        
        # Fix unquoted field names and string values
        content = re.sub(r'\b(\w+):', r'"\1":', content)  # Field names
        content = re.sub(r":\s*([^'\"\[\{0-9\-][^,\}]+)", r': "\1"', content)  # Unquoted strings
        
        # Clean up whitespace in arrays
        content = re.sub(r'\[\s+', '[', content)
        content = re.sub(r'\s+\]', ']', content)
        content = re.sub(r',\s*\]', ']', content)
        
        return content

# Test the parser
if __name__ == "__main__":
    parser = SurrealDBParser("/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql")
    data = parser.parse_backup()
    
    for table_name, records in data.items():
        print(f"\n{table_name}: {len(records)} records")
        if records:
            print(f"Sample: {records[0]}")