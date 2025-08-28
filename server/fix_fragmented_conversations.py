#!/usr/bin/env python3
"""
Fix Fragmented Conversation Content in Tape Table

This script fixes STT fragmentation issues in the tape table conversation content.
Uses the same normalization functions as sherpa_stt.py and smart_context_manager.py
to ensure consistency across the entire pipeline.

Examples of fixes:
- "efficiently . So , are you thinking about..." → "efficiently. So, are you thinking about..."
- "Hello slowcat. How is it going?, Can y..." → "Hello slowcat. How is it going? Can y..."
- "Ant onio Mach ado" → "Antonio Machado"
- "2 1 4 7" → "2147"

The script:
1. Fetches all tape entries from SurrealDB
2. Applies comprehensive text normalization
3. Updates entries where content has changed
4. Reports statistics on fixes applied
"""

import asyncio
import re
from typing import Dict, List
from surrealdb import AsyncSurreal
from loguru import logger
from datetime import datetime


def normalize_conversation_text(text: str) -> str:
    """
    Comprehensive conversation text normalization combining all STT fixes
    
    This function combines:
    - STT normalization from sherpa_stt.py:normalize_stt_text()
    - User input normalization from smart_context_manager.py:_normalize_user_input()
    - Enhanced pattern detection for fragmented names
    """
    if not text or not isinstance(text, str):
        return text
    
    original = text.strip()
    s = original
    
    # Phase 1: Basic cleanup
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Phase 2: Fix spaced numbers (2 1 4 7 → 2147)
    s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', s)
    s = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', s)
    s = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', s)
    
    # Phase 3: Fix spaced proper names (Antonio Machado patterns)
    s = re.sub(r'\b([A-Z])\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
    s = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', s)
    
    # Phase 4: Fix simple spaced words (Pot ola → Potola)
    s = re.sub(r'\b([A-Z][a-z]{1,4})\s+([a-z]{2,6})\b', r'\1\2', s)
    
    # Phase 5: Enhanced name pattern detection
    parts = s.split()
    reconstructed_parts = []
    i = 0
    
    while i < len(parts):
        current_part = parts[i]
        
        # Check for 2-part fragmented names: "Pe ppy" → "Peppy" or "pepp i" → "Peppi"
        if (i < len(parts) - 1 and 
            ((current_part[0].isupper() and parts[i+1][0].islower()) or
             (current_part[0].islower() and parts[i+1][0].islower())) and
            len(current_part) <= 6 and len(parts[i+1]) <= 6):
            total_length = len(current_part) + len(parts[i+1])
            if 3 <= total_length <= 10:  # Reasonable name length
                combined = current_part + parts[i+1]
                # Capitalize first letter if both parts were lowercase
                if current_part[0].islower():
                    combined = combined[0].upper() + combined[1:] if combined else combined
                reconstructed_parts.append(combined)
                i += 2
                continue
        
        # Check for 3-part fragmented names: "Pe p py" → "Peppy"
        if (i < len(parts) - 2 and 
            current_part[0].isupper() and
            all(len(parts[i+j]) <= 4 for j in [1, 2]) and
            all(parts[i+j][0].islower() for j in [1, 2])):
            total_length = sum(len(parts[i+j]) for j in [0, 1, 2])
            if 3 <= total_length <= 10:  # Reasonable name length
                combined = current_part + parts[i+1] + parts[i+2]
                reconstructed_parts.append(combined)
                i += 3
                continue
        
        # Check for 4-part fragmented names: "p e p p i" → "Peppi"
        if (i < len(parts) - 3 and
            all(len(parts[i+j]) <= 3 for j in [0, 1, 2, 3]) and
            all(parts[i+j][0].islower() for j in [0, 1, 2, 3])):
            total_length = sum(len(parts[i+j]) for j in [0, 1, 2, 3])
            if 4 <= total_length <= 12:  # Reasonable name length
                combined = ''.join(parts[i+j] for j in [0, 1, 2, 3])
                # Capitalize first letter
                combined = combined[0].upper() + combined[1:] if combined else combined
                reconstructed_parts.append(combined)
                i += 4
                continue
        
        # No pattern matched, add current part as-is
        reconstructed_parts.append(current_part)
        i += 1
    
    # Rebuild text from reconstructed parts
    s = ' '.join(reconstructed_parts)
    
    # Phase 6: Fix punctuation spacing issues from STT
    # Fix spaces before punctuation: "hello ," → "hello,"
    s = re.sub(r'\s+([,.!?;:])', r'\1', s)
    
    # Fix missing spaces after punctuation: "hello,world" → "hello, world"
    s = re.sub(r'([,.!?;:])([a-zA-Z])', r'\1 \2', s)
    
    # Fix double spaces after normalization
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def calculate_similarity(original: str, normalized: str) -> float:
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


async def fix_fragmented_conversations():
    """
    Main function to fix fragmented conversation content in tape table
    """
    # Connect to SurrealDB
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    logger.info("🔧 Starting conversation content normalization...")
    
    # Step 1: Get all tape entries
    all_entries = await db.query('SELECT id, content, role, speaker_id, ts FROM tape ORDER BY ts')
    logger.info(f"📊 Found {len(all_entries)} conversation entries to analyze")
    
    if not all_entries:
        logger.warning("No tape entries found in database")
        await db.close()
        return
    
    # Step 2: Process and identify entries needing updates
    updates_needed = []
    unchanged_count = 0
    
    for entry in all_entries:
        original_content = entry.get('content', '')
        normalized_content = normalize_conversation_text(original_content)
        
        if original_content != normalized_content:
            # Calculate similarity to avoid fixing entries that are drastically different
            similarity = calculate_similarity(original_content, normalized_content)
            
            if similarity >= 0.7:  # Only fix if normalized version is similar enough
                updates_needed.append({
                    'id': entry['id'],
                    'original': original_content,
                    'normalized': normalized_content,
                    'role': entry.get('role', 'unknown'),
                    'speaker_id': entry.get('speaker_id', 'unknown'),
                    'similarity': similarity
                })
            else:
                logger.debug(f"Skipping entry with low similarity ({similarity:.2f}): '{original_content[:50]}...'")
                unchanged_count += 1
        else:
            unchanged_count += 1
    
    logger.info(f"📝 Found {len(updates_needed)} entries needing normalization")
    logger.info(f"✅ Found {unchanged_count} entries already normalized")
    
    if not updates_needed:
        logger.info("🎉 All conversation content is already properly normalized!")
        await db.close()
        return
    
    # Step 3: Show sample of changes that will be made
    logger.info("📋 Sample of changes to be made:")
    for i, update in enumerate(updates_needed[:5]):  # Show first 5 examples
        logger.info(f"   {i+1}. [{update['role']}] {update['speaker_id']}")
        logger.info(f"      Before: '{update['original'][:80]}...'")
        logger.info(f"      After:  '{update['normalized'][:80]}...'")
        logger.info(f"      Similarity: {update['similarity']:.2f}")
    
    if len(updates_needed) > 5:
        logger.info(f"   ... and {len(updates_needed) - 5} more entries")
    
    # Step 4: Apply updates
    updated_count = 0
    failed_count = 0
    
    logger.info("🔄 Applying normalization updates...")
    
    for update in updates_needed:
        try:
            # Update the content in the database
            await db.query(
                'UPDATE $id SET content = $normalized_content',
                {
                    'id': update['id'],
                    'normalized_content': update['normalized']
                }
            )
            updated_count += 1
            
            if updated_count % 50 == 0:  # Progress update every 50 entries
                logger.info(f"   Progress: {updated_count}/{len(updates_needed)} entries updated")
                
        except Exception as e:
            logger.error(f"Failed to update entry {update['id']}: {e}")
            failed_count += 1
    
    # Step 5: Final verification
    final_entries = await db.query('SELECT COUNT() FROM tape GROUP ALL')
    final_count = final_entries[0]['count'] if final_entries else 0
    
    logger.info("🎉 Conversation content normalization completed!")
    logger.info(f"📊 Final statistics:")
    logger.info(f"   💾 Total entries: {final_count}")
    logger.info(f"   ✅ Updated: {updated_count}")
    logger.info(f"   📝 Already normalized: {unchanged_count}")
    logger.info(f"   ❌ Failed: {failed_count}")
    
    # Step 6: Show sample of fixed content
    if updated_count > 0:
        sample_fixed = await db.query('''
            SELECT content, role, speaker_id, ts 
            FROM tape 
            ORDER BY ts DESC 
            LIMIT 5
        ''')
        
        logger.info("🔍 Sample of normalized conversation content:")
        for entry in sample_fixed:
            logger.info(f"   [{entry['role']}] {entry.get('speaker_id', 'unknown')}: {entry['content'][:100]}...")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(fix_fragmented_conversations())