#!/usr/bin/env python3
"""
Clean Fragmented Entities Script

This script consolidates fragmented entities in the SurrealDB memory graph
caused by STT (Speech-to-Text) spacing issues.

Examples:
- "Ant onio Mach ado" → "Antonio Machado"  
- "Pot ola" → "Potola"
- "2 1 4 7" → "2147"

The script:
1. Identifies fragmented entities
2. Normalizes their names
3. Merges duplicates
4. Updates all relationships
5. Removes old fragmented records
"""

import asyncio
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from surrealdb import AsyncSurreal
from loguru import logger
from datetime import datetime


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity names to fix STT fragmentation
    Enhanced with intelligent name pattern detection
    """
    if not name or not isinstance(name, str):
        return name
        
    original = name.strip()
    text = original
    
    # Step 1: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 2: Fix spaced numbers (2 1 4 7 → 2147)
    text = re.sub(r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\b', r'\1\2\3\4', text)
    text = re.sub(r'\b(\d)\s+(\d)\s+(\d)\b', r'\1\2\3', text)
    text = re.sub(r'\b(\d)\s+(\d)\b', r'\1\2', text)
    
    # Step 3: Fix spaced proper names (like "Ant onio Mach ado" → "Antonio Machado")
    text = re.sub(r'\b([A-Z][a-z]+)\s+([a-z]{1,4})\s+([A-Z][a-z]+)\b', r'\1\2 \3', text)
    
    # Step 4: Enhanced name pattern detection
    parts = text.split()
    
    if len(parts) == 2:
        # Pattern: "Pe ppy" → "Peppy" (Cap + space + lowercase)
        # Pattern: "pepp i" → "Peppi" (lowercase + space + lowercase)
        if ((parts[0][0].isupper() and parts[1][0].islower()) or
            (parts[0][0].islower() and parts[1][0].islower())) and \
            len(parts[0]) <= 6 and len(parts[1]) <= 6:
            # Check if it looks like a fragmented single name
            total_length = len(parts[0]) + len(parts[1])
            if 3 <= total_length <= 10:  # Reasonable name length
                combined = ''.join(parts)
                # Capitalize first letter if both parts were lowercase
                if parts[0][0].islower():
                    text = combined[0].upper() + combined[1:] if combined else text
                else:
                    text = combined
    
    elif len(parts) == 3:
        # Pattern: "Pe p py" → "Peppy" (Cap + short + short)
        if (parts[0][0].isupper() and 
            all(len(part) <= 4 for part in parts[1:]) and
            all(part[0].islower() for part in parts[1:])):
            # Looks like a triple-fragmented name
            total_length = sum(len(part) for part in parts)
            if 3 <= total_length <= 10:  # Reasonable name length
                text = ''.join(parts)
        
        # Pattern: "J ohn S mith" → "John Smith" (preserve surnames)
        elif (parts[0][0].isupper() and parts[2][0].isupper() and
              len(parts[1]) <= 4 and parts[1][0].islower()):
            text = parts[0] + parts[1] + ' ' + parts[2]
    
    elif len(parts) == 4:
        # Pattern: "p e p p i" → "peppi" (all lowercase fragments)  
        if (all(len(part) <= 3 for part in parts) and
            all(part[0].islower() for part in parts)):
            total_length = sum(len(part) for part in parts)
            if 4 <= total_length <= 12:  # Reasonable name length
                # Capitalize first letter of combined name
                combined = ''.join(parts)
                text = combined[0].upper() + combined[1:] if combined else text
    
    # Step 5: Clean up excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def should_remove_entity(name: str, mentioned_count: int) -> bool:
    """
    Determine if an entity should be removed as low-quality
    """
    if not name or len(name.strip()) <= 1:
        return True
        
    text = name.strip()
    
    # Remove single characters
    if len(text) == 1:
        return True
        
    # Remove pure numbers (except reasonable years)
    if text.isdigit():
        num = int(text)
        if not (1900 <= num <= 2100):
            return True
    
    # Remove entities with very low mention count that look like fragments
    if mentioned_count <= 1 and len(text.replace(' ', '')) <= 3:
        return True
        
    # Remove obvious artifacts
    artifacts = {'the', 'and', 'or', 'but', 'a', 'an', 'i', 'you', 'we', 'they'}
    if text.lower() in artifacts:
        return True
    
    return False


async def clean_fragmented_entities():
    """
    Main cleanup function
    """
    # Connect to database
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    logger.info("🧹 Starting fragmented entity cleanup...")
    
    # Step 1: Get all concepts
    all_concepts = await db.query('SELECT id, name, kind, mentioned_count FROM concept')
    logger.info(f"📊 Found {len(all_concepts)} concepts to analyze")
    
    # Step 2: Group entities by normalized name
    normalized_groups = defaultdict(list)
    entities_to_remove = []
    
    for concept in all_concepts:
        name = concept['name']
        normalized_name = normalize_entity_name(name)
        
        # Mark low-quality entities for removal
        if should_remove_entity(name, concept.get('mentioned_count', 0)):
            entities_to_remove.append(concept)
            logger.debug(f"Marked for removal: '{name}' (low quality)")
            continue
            
        # Group by normalized name
        normalized_groups[normalized_name].append(concept)
    
    logger.info(f"🗑️ Found {len(entities_to_remove)} low-quality entities to remove")
    logger.info(f"📦 Grouped entities into {len(normalized_groups)} normalized groups")
    
    # Step 3: Find groups that need merging (multiple entities → same normalized name)
    merge_operations = []
    
    for normalized_name, concepts in normalized_groups.items():
        if len(concepts) > 1:
            # Sort by mentioned_count (descending) to keep the most mentioned one as primary
            concepts.sort(key=lambda c: c.get('mentioned_count', 0), reverse=True)
            primary = concepts[0]
            duplicates = concepts[1:]
            
            merge_operations.append({
                'normalized_name': normalized_name,
                'primary': primary,
                'duplicates': duplicates
            })
            
            logger.info(f"🔀 Merge group '{normalized_name}': {len(concepts)} variants")
            for concept in concepts:
                logger.info(f"   - '{concept['name']}' (mentions: {concept.get('mentioned_count', 0)})")
    
    logger.info(f"🔄 Found {len(merge_operations)} groups requiring merging")
    
    # Step 4: Remove low-quality entities
    removed_count = 0
    for entity in entities_to_remove:
        try:
            # Remove relationships first
            await db.query(f'DELETE knows WHERE out = {entity["id"]}')
            await db.query(f'DELETE knows WHERE in = {entity["id"]}')
            
            # Remove the concept
            await db.query(f'DELETE {entity["id"]}')
            removed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to remove {entity['id']}: {e}")
    
    logger.info(f"✅ Removed {removed_count} low-quality entities")
    
    # Step 5: Perform merges
    merged_count = 0
    
    for merge_op in merge_operations:
        normalized_name = merge_op['normalized_name']
        primary = merge_op['primary']
        duplicates = merge_op['duplicates']
        
        logger.info(f"🔄 Merging '{normalized_name}': keeping {primary['id']}, merging {len(duplicates)} duplicates")
        
        try:
            # Update primary entity name if needed
            if primary['name'] != normalized_name:
                await db.query(f'''
                    UPDATE {primary["id"]} SET
                        name = $normalized_name,
                        last_mentioned = $timestamp
                ''', {
                    'normalized_name': normalized_name,
                    'timestamp': datetime.now()
                })
            
            # Merge duplicates into primary
            total_mentions = primary.get('mentioned_count', 0)
            
            for duplicate in duplicates:
                # Transfer relationships from duplicate to primary
                # Update knows relationships where duplicate is the target
                await db.query(f'''
                    UPDATE knows SET out = {primary["id"]} 
                    WHERE out = {duplicate["id"]}
                ''')
                
                # Update knows relationships where duplicate is the source
                await db.query(f'''
                    UPDATE knows SET in = {primary["id"]} 
                    WHERE in = {duplicate["id"]}
                ''')
                
                # Add mention count
                total_mentions += duplicate.get('mentioned_count', 0)
                
                # Delete the duplicate
                await db.query(f'DELETE {duplicate["id"]}')
                
                logger.debug(f"   Merged {duplicate['id']} → {primary['id']}")
            
            # Update primary with combined mention count
            await db.query(f'''
                UPDATE {primary["id"]} SET mentioned_count = $total_mentions
            ''', {'total_mentions': total_mentions})
            
            merged_count += len(duplicates)
            
        except Exception as e:
            logger.error(f"Failed to merge group '{normalized_name}': {e}")
    
    logger.info(f"✅ Merged {merged_count} duplicate entities")
    
    # Step 6: Final verification
    final_concepts = await db.query('SELECT COUNT() FROM concept GROUP ALL')
    final_relationships = await db.query('SELECT COUNT() FROM knows GROUP ALL')
    
    logger.info("🎉 Cleanup completed!")
    logger.info(f"📊 Final stats:")
    logger.info(f"   🧩 Concepts: {final_concepts[0]['count'] if final_concepts else 0}")
    logger.info(f"   🔗 Relationships: {final_relationships[0]['count'] if final_relationships else 0}")
    logger.info(f"   🗑️ Removed: {removed_count} low-quality entities")
    logger.info(f"   🔄 Merged: {merged_count} duplicate entities")
    
    # Show sample of cleaned entities
    sample_entities = await db.query('''
        SELECT name, kind, mentioned_count 
        FROM concept 
        WHERE mentioned_count > 1
        ORDER BY mentioned_count DESC 
        LIMIT 10
    ''')
    
    logger.info(f"🔝 Top entities after cleanup:")
    for entity in sample_entities:
        logger.info(f"   - {entity['name']}: {entity['kind']} ({entity['mentioned_count']} mentions)")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(clean_fragmented_entities())