#!/usr/bin/env python3
"""
Fix Existing Content Fragmentation in Message/Tape Tables

This script fixes STT fragmentation issues in existing message or tape table content.
Can be run on either table depending on what exists in the current database.

The script:
1. Detects which table exists (message or tape) 
2. Applies comprehensive text normalization to content
3. Updates entries where content has changed
4. Reports statistics on fixes applied
5. Preserves original content in raw_content field when available

This can be run before or after migration to ensure all content is properly normalized.
"""

import asyncio
import re
from typing import Dict, List, Tuple
from surrealdb import AsyncSurreal
from loguru import logger


class ContentFragmentationFixer:
    """Fix fragmented content in existing database tables"""
    
    def __init__(self, db: AsyncSurreal):
        self.db = db
        self.stats = {
            'entries_processed': 0,
            'content_fixed': 0,
            'already_clean': 0,
            'errors': 0,
            'table_used': None
        }
    
    def _normalize_content(self, content: str) -> str:
        """Apply comprehensive content normalization (same as migration script)"""
        if not content or not isinstance(content, str):
            return content or ''
        
        original = content
        s = content.strip()
        
        # Phase 1: Basic cleanup
        s = re.sub(r'\s+', ' ', s).strip()
        
        # Phase 2: Fix spaced numbers (2 1 4 7 → 2147)
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', s)
        s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', s)
        s = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', s)
        
        # Phase 3: Fix spaced proper names
        s = re.sub(r'\b([A-Z])\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
        s = re.sub(r'\b([A-Z][a-z]{1,4})\s+([a-z]{2,6})\b', r'\1\2', s)
        
        # Phase 4: Enhanced name pattern detection
        parts = s.split()
        reconstructed_parts = []
        i = 0
        
        while i < len(parts):
            current_part = parts[i]
            
            # 2-part fragmented names: "Pe ppy" → "Peppy" or "pepp i" → "Peppi"
            if (i < len(parts) - 1 and 
                ((current_part[0].isupper() and parts[i+1][0].islower()) or
                 (current_part[0].islower() and parts[i+1][0].islower())) and
                len(current_part) <= 6 and len(parts[i+1]) <= 6):
                total_length = len(current_part) + len(parts[i+1])
                if 3 <= total_length <= 10:
                    combined = current_part + parts[i+1]
                    if current_part[0].islower():
                        combined = combined[0].upper() + combined[1:] if combined else combined
                    reconstructed_parts.append(combined)
                    i += 2
                    continue
            
            # 3-part fragmented names: "Pe p py" → "Peppy"
            if (i < len(parts) - 2 and 
                current_part[0].isupper() and
                all(len(parts[i+j]) <= 4 for j in [1, 2]) and
                all(parts[i+j][0].islower() for j in [1, 2])):
                total_length = sum(len(parts[i+j]) for j in [0, 1, 2])
                if 3 <= total_length <= 10:
                    combined = current_part + parts[i+1] + parts[i+2]
                    reconstructed_parts.append(combined)
                    i += 3
                    continue
            
            # 4-part fragmented names: "p e p p i" → "Peppi"
            if (i < len(parts) - 3 and
                all(len(parts[i+j]) <= 3 for j in [0, 1, 2, 3]) and
                all(parts[i+j][0].islower() for j in [0, 1, 2, 3])):
                total_length = sum(len(parts[i+j]) for j in [0, 1, 2, 3])
                if 4 <= total_length <= 12:
                    combined = ''.join(parts[i+j] for j in [0, 1, 2, 3])
                    combined = combined[0].upper() + combined[1:] if combined else combined
                    reconstructed_parts.append(combined)
                    i += 4
                    continue
            
            # No pattern matched
            reconstructed_parts.append(current_part)
            i += 1
        
        s = ' '.join(reconstructed_parts)
        
        # Phase 5: Fix punctuation spacing
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        s = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s
    
    def _calculate_similarity(self, original: str, normalized: str) -> float:
        """Calculate similarity ratio between original and normalized text"""
        if not original and not normalized:
            return 1.0
        if not original or not normalized:
            return 0.0
        
        # Simple character-level similarity
        orig_chars = set(original.lower().replace(' ', ''))
        norm_chars = set(normalized.lower().replace(' ', ''))
        
        if not orig_chars and not norm_chars:
            return 1.0
        if not orig_chars or not norm_chars:
            return 0.0
        
        intersection = len(orig_chars & norm_chars)
        union = len(orig_chars | norm_chars)
        
        return intersection / union if union > 0 else 0.0
    
    async def _detect_available_table(self) -> Tuple[str, List[str]]:
        """Detect which table exists and what fields are available"""
        
        # Check for message table (graph schema)
        try:
            message_info = await self.db.query('INFO FOR TABLE message')
            if message_info:
                # Get sample to understand structure
                sample = await self.db.query('SELECT * FROM message LIMIT 1')
                if sample:
                    fields = list(sample[0].keys())
                    logger.info(f"🔍 Detected message table with fields: {fields}")
                    return 'message', fields
        except Exception as e:
            logger.debug(f"Message table not found or not accessible: {e}")
        
        # Check for tape table (original schema)
        try:
            tape_info = await self.db.query('INFO FOR TABLE tape')
            if tape_info:
                # Get sample to understand structure
                sample = await self.db.query('SELECT * FROM tape LIMIT 1')
                if sample:
                    fields = list(sample[0].keys())
                    logger.info(f"🔍 Detected tape table with fields: {fields}")
                    return 'tape', fields
        except Exception as e:
            logger.debug(f"Tape table not found or not accessible: {e}")
        
        raise Exception("Neither message nor tape table found in database")
    
    async def _get_content_entries(self, table: str) -> List[Dict]:
        """Get all content entries from the detected table"""
        
        if table == 'message':
            # Graph schema message table
            query = '''
                SELECT id, content, timestamp, sender_id, role, raw_content 
                FROM message 
                ORDER BY timestamp
            '''
        else:
            # Original tape table  
            query = '''
                SELECT id, content, ts, speaker_id, role
                FROM tape
                ORDER BY ts
            '''
        
        entries = await self.db.query(query)
        logger.info(f"📊 Found {len(entries)} entries in {table} table")
        return entries
    
    async def _update_content_entry(self, table: str, entry: Dict, normalized_content: str, original_content: str) -> bool:
        """Update content entry with normalized content"""
        
        try:
            entry_id = entry.get('id')
            if not entry_id:
                logger.error("Entry missing ID, cannot update")
                return False
            
            if table == 'message':
                # For message table, update content and preserve raw_content
                await self.db.query(f'''
                    UPDATE {entry_id} SET
                        content = $normalized_content,
                        raw_content = $raw_content,
                        word_count = array::len(string::split($normalized_content, ' ')),
                        content_length = string::len($normalized_content)
                ''', {
                    'normalized_content': normalized_content,
                    'raw_content': original_content if original_content != normalized_content else entry.get('raw_content')
                })
            else:
                # For tape table, just update content
                await self.db.query(f'''
                    UPDATE {entry_id} SET content = $normalized_content
                ''', {
                    'normalized_content': normalized_content
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entry {entry.get('id', 'unknown')}: {e}")
            return False
    
    async def fix_content_fragmentation(self) -> Dict[str, any]:
        """Main function to fix content fragmentation"""
        
        logger.info("🔧 Starting content fragmentation fix...")
        
        try:
            # Step 1: Detect available table and structure
            table, fields = await self._detect_available_table()
            self.stats['table_used'] = table
            logger.info(f"📋 Using table: {table}")
            
            # Step 2: Get all content entries
            entries = await self._get_content_entries(table)
            
            if not entries:
                logger.warning(f"No entries found in {table} table")
                return {
                    'status': 'success',
                    'message': 'No entries to process',
                    'stats': self.stats
                }
            
            # Step 3: Process each entry
            updates_needed = []
            
            for entry in entries:
                try:
                    original_content = entry.get('content', '')
                    normalized_content = self._normalize_content(original_content)
                    
                    if original_content != normalized_content:
                        # Check similarity to avoid fixing entries that are drastically different
                        similarity = self._calculate_similarity(original_content, normalized_content)
                        
                        if similarity >= 0.7:  # Only fix if normalized version is similar enough
                            updates_needed.append({
                                'entry': entry,
                                'original': original_content,
                                'normalized': normalized_content,
                                'similarity': similarity
                            })
                        else:
                            logger.debug(f"Skipping entry with low similarity ({similarity:.2f}): '{original_content[:50]}...'")
                            self.stats['already_clean'] += 1
                    else:
                        self.stats['already_clean'] += 1
                    
                    self.stats['entries_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process entry: {e}")
                    self.stats['errors'] += 1
            
            logger.info(f"📝 Found {len(updates_needed)} entries needing content normalization")
            logger.info(f"✅ Found {self.stats['already_clean']} entries already normalized")
            
            if not updates_needed:
                logger.info("🎉 All content is already properly normalized!")
                return {
                    'status': 'success',
                    'message': 'All content already normalized',
                    'stats': self.stats
                }
            
            # Step 4: Show sample of changes
            logger.info("📋 Sample of changes to be made:")
            for i, update in enumerate(updates_needed[:5]):
                entry = update['entry']
                role = entry.get('role', 'unknown')
                sender = entry.get('sender_id') or entry.get('speaker_id', 'unknown')
                
                logger.info(f"   {i+1}. [{role}] {sender}")
                logger.info(f"      Before: '{update['original'][:80]}...'")
                logger.info(f"      After:  '{update['normalized'][:80]}...'")
                logger.info(f"      Similarity: {update['similarity']:.2f}")
            
            if len(updates_needed) > 5:
                logger.info(f"   ... and {len(updates_needed) - 5} more entries")
            
            # Step 5: Apply updates
            logger.info("🔄 Applying content normalization updates...")
            
            for i, update in enumerate(updates_needed):
                try:
                    success = await self._update_content_entry(
                        table, 
                        update['entry'], 
                        update['normalized'], 
                        update['original']
                    )
                    
                    if success:
                        self.stats['content_fixed'] += 1
                    else:
                        self.stats['errors'] += 1
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"   Progress: {i + 1}/{len(updates_needed)} entries updated")
                        
                except Exception as e:
                    logger.error(f"Failed to update entry: {e}")
                    self.stats['errors'] += 1
            
            # Step 6: Final verification
            logger.info("🎉 Content fragmentation fix completed!")
            logger.info(f"📊 Final statistics:")
            logger.info(f"   💾 Total entries processed: {self.stats['entries_processed']}")
            logger.info(f"   ✅ Content fixed: {self.stats['content_fixed']}")
            logger.info(f"   📝 Already clean: {self.stats['already_clean']}")
            logger.info(f"   ❌ Errors: {self.stats['errors']}")
            logger.info(f"   📋 Table used: {self.stats['table_used']}")
            
            # Step 7: Show sample of fixed content
            if self.stats['content_fixed'] > 0:
                sample_fixed = await self.db.query(f'''
                    SELECT content, role, {"sender_id" if table == "message" else "speaker_id"} as sender, {"timestamp" if table == "message" else "ts"} as ts
                    FROM {table}
                    ORDER BY {"timestamp" if table == "message" else "ts"} DESC
                    LIMIT 5
                ''')
                
                logger.info("🔍 Sample of normalized content:")
                for entry in sample_fixed:
                    role = entry.get('role', 'unknown')
                    sender = entry.get('sender', 'unknown')
                    content = entry.get('content', '')
                    logger.info(f"   [{role}] {sender}: {content[:100]}...")
            
            return {
                'status': 'success',
                'stats': self.stats,
                'updates_applied': len(updates_needed),
                'table_used': table
            }
            
        except Exception as e:
            logger.error(f"Content fragmentation fix failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'stats': self.stats
            }


async def main():
    """
    Main execution function
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    # Fix content fragmentation
    fixer = ContentFragmentationFixer(db)
    results = await fixer.fix_content_fragmentation()
    
    # Save results
    import json
    with open('content_fix_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📄 Content fix report saved to content_fix_report.json")
    
    await db.close()
    return results


if __name__ == "__main__":
    asyncio.run(main())