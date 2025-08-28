#!/usr/bin/env python3
"""
Extract and parse data from SurrealDB backup file

This script parses the SURQL backup file and extracts all data
into organized Python dictionaries for migration to graph schema.
"""

import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

class SurrealQLParser:
    """Parser for SurrealDB SURQL backup files"""
    
    def __init__(self, backup_path: str):
        self.backup_path = Path(backup_path)
        self.data = {}
        self.tables = []
        
    def parse_backup(self) -> Dict[str, List[Dict]]:
        """Parse the entire backup file and extract all table data"""
        logger.info(f"Parsing backup file: {self.backup_path}")
        
        if not self.backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {self.backup_path}")
        
        with open(self.backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all table sections
        table_pattern = r'-- TABLE: (\w+)'
        tables = re.findall(table_pattern, content)
        logger.info(f"Found tables: {tables}")
        
        # Extract data for each table
        for table_name in tables:
            self.data[table_name] = self._extract_table_data(content, table_name)
            logger.info(f"Extracted {len(self.data[table_name])} records from {table_name}")
        
        return self.data
    
    def _extract_table_data(self, content: str, table_name: str) -> List[Dict]:
        """Extract INSERT data for a specific table"""
        
        # Find the table data section
        data_section_pattern = rf'-- TABLE DATA: {table_name}.*?\n\n(.*?)(?=\n-- |$)'
        match = re.search(data_section_pattern, content, re.DOTALL)
        
        if not match:
            logger.warning(f"No data section found for table {table_name}")
            return []
        
        data_section = match.group(1).strip()
        
        # Find INSERT statements
        insert_pattern = r'INSERT \[ (.*?) \];'
        insert_matches = re.findall(insert_pattern, data_section, re.DOTALL)
        
        records = []
        for insert_data in insert_matches:
            # Parse the array of objects
            try:
                # Clean up the SurrealDB-specific syntax
                cleaned_data = self._clean_surreal_syntax(insert_data)
                # Parse as Python literal
                parsed_objects = ast.literal_eval(f'[{cleaned_data}]')
                records.extend(parsed_objects)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Primary parse failed for {table_name}; using robust fallback: {e}")
                # Try alternative parsing method
                records.extend(self._parse_surreal_objects(insert_data))
        
        return records
    
    def _clean_surreal_syntax(self, data: str) -> str:
        """Clean SurrealDB-specific syntax to make it Python-parseable"""
        
        # Replace SurrealDB record IDs with strings (e.g., user:peppi)
        data = re.sub(r'(\b\w+):([A-Za-z0-9_:-]+)', r'"\1:\2"', data)
        
        # Replace SurrealDB datetime format
        data = re.sub(r"d'([^']+)'", r'"\1"', data)
        
        # Replace SurrealDB float suffix 'f' or 'F' (including negatives)
        data = re.sub(r'(-?\d+(?:\.\d+)?)[fF]\b', r'\1', data)
        
        # Handle empty objects
        data = re.sub(r'\{\s*\}', '{}', data)
        
        return data
    
    def _parse_surreal_objects(self, data: str) -> List[Dict]:
        """Alternative parser for complex SurrealDB objects.

        Robustly splits objects and key/value pairs, handling arrays (e.g., embeddings)
        and float suffixes like 0.1234f.
        """

        def split_objects(s: str) -> List[str]:
            objs: List[str] = []
            depth = 0
            in_str = False
            quote = ''
            buf: List[str] = []
            for ch in s:
                if in_str:
                    buf.append(ch)
                    if ch == quote:
                        in_str = False
                else:
                    if ch in ('"', "'"):
                        in_str = True
                        quote = ch
                        buf.append(ch)
                    elif ch == '{':
                        depth += 1
                        buf.append(ch)
                    elif ch == '}':
                        depth -= 1
                        buf.append(ch)
                        if depth == 0:
                            objs.append(''.join(buf))
                            buf = []
                    else:
                        if depth > 0:
                            buf.append(ch)
            return objs

        def split_top_level_commas(s: str) -> List[str]:
            parts: List[str] = []
            depth = 0
            in_str = False
            quote = ''
            buf: List[str] = []
            for ch in s:
                if in_str:
                    buf.append(ch)
                    if ch == quote:
                        in_str = False
                else:
                    if ch in ('"', "'"):
                        in_str = True
                        quote = ch
                        buf.append(ch)
                    elif ch == '[':
                        depth += 1
                        buf.append(ch)
                    elif ch == ']':
                        depth = max(0, depth - 1)
                        buf.append(ch)
                    elif ch == ',' and depth == 0:
                        parts.append(''.join(buf).strip())
                        buf = []
                    else:
                        buf.append(ch)
            if buf:
                parts.append(''.join(buf).strip())
            return parts

        def parse_value(val: str):
            v = val.strip()
            # Datetime
            if v.startswith("d'") and v.endswith("'"):
                return v[2:-1]
            # Strings
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                return v[1:-1]
            # Booleans
            if v in ('true', 'false'):
                return v == 'true'
            # Array
            if v.startswith('[') and v.endswith(']'):
                inner = v[1:-1].strip()
                if not inner:
                    return []
                items = split_top_level_commas(inner)
                out = []
                for it in items:
                    it_clean = re.sub(r'([fF])\b', '', it.strip())  # remove float suffix
                    try:
                        if '.' in it_clean or 'e' in it_clean.lower() or it_clean.startswith('-'):
                            out.append(float(it_clean))
                        else:
                            out.append(int(it_clean))
                    except Exception:
                        # fallback to raw string without quotes
                        out.append(it_clean.strip("'\""))
                return out
            # Floats with f suffix
            if re.fullmatch(r'-?\d+(?:\.\d+)?[fF]?', v):
                v2 = re.sub(r'[fF]\b', '', v)
                return float(v2) if '.' in v2 else int(v2)
            return v

        objects: List[Dict] = []
        for obj_str in split_objects(data):
            try:
                body = obj_str.strip()
                if not (body.startswith('{') and body.endswith('}')):
                    continue
                body = body[1:-1]
                pairs = split_top_level_commas(body)
                obj: Dict[str, Any] = {}
                for p in pairs:
                    if ':' not in p:
                        continue
                    key, val = p.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    obj[key] = parse_value(val)
                if obj:
                    objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to parse object: {obj_str[:120]}... - {e}")
                continue
        return objects
    
    def get_stats(self) -> Dict[str, int]:
        """Get record counts by table"""
        return {table: len(records) for table, records in self.data.items()}
    
    def export_to_json(self, output_dir: str = None) -> str:
        """Export extracted data to JSON files for inspection"""
        if output_dir is None:
            output_dir = self.backup_path.parent / "extracted_data"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export each table to separate JSON file
        for table_name, records in self.data.items():
            json_file = output_path / f"{table_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, default=str)
        
        # Export combined data
        combined_file = output_path / "all_tables.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        logger.info(f"Exported data to: {output_path}")
        return str(output_path)

def main():
    """Main execution function"""
    import os
    import argparse
    parser_cli = argparse.ArgumentParser(description="Extract data from SurrealDB .surql backup")
    parser_cli.add_argument('--backup', default=os.getenv('BACKUP_FILE', str((Path(__file__).parent.parent / 'memory' / 'localslowcat-2025-08-27.surql').resolve())), help='Path to .surql backup file')
    args = parser_cli.parse_args()

    # Parse the backup file
    parser = SurrealQLParser(args.backup)
    data = parser.parse_backup()
    
    # Show statistics
    stats = parser.get_stats()
    logger.info("Extraction Statistics:")
    for table, count in stats.items():
        logger.info(f"  {table}: {count} records")
    
    # Export to JSON for inspection
    output_dir = parser.export_to_json()
    
    # Sample some data for verification
    logger.info("\nSample Data:")
    for table_name, records in data.items():
        if records:
            logger.info(f"\n{table_name} sample:")
            sample = records[0]
            for key, value in list(sample.items())[:3]:  # Show first 3 fields
                logger.info(f"  {key}: {value}")
    
    return data

if __name__ == "__main__":
    import sys
    try:
        extracted_data = main()
        logger.info("✅ Data extraction completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Data extraction failed: {e}")
        sys.exit(1)
