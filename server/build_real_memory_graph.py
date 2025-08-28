#!/usr/bin/env python3
"""
Build Real Memory Graph from Conversation Data
Extract facts from actual conversations and populate graph with source links
"""

import asyncio
import spacy
from surrealdb import AsyncSurreal
from loguru import logger
import time
import re
from datetime import datetime

def sanitize_concept_id(entity_name: str) -> str:
    """Sanitize entity name for use as SurrealDB record ID"""
    if not entity_name:
        return "unknown"
    
    # Convert to lowercase and replace spaces with underscores
    sanitized = entity_name.lower().replace(' ', '_')
    
    # Remove or replace problematic characters
    # Keep only alphanumeric, underscore, and hyphen
    sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure it's not empty and doesn't start with a number
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"entity_{sanitized}"
    
    return sanitized

async def extract_facts_from_conversations():
    """Extract facts from ALL conversation data and build proper graph"""
    
    # Connect to database
    db = AsyncSurreal('ws://127.0.0.1:8000/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'slowcat_secure_2024'})
    await db.use('slowcat', 'memory_graph')
    
    logger.info("🧠 Building real memory graph from conversation data...")
    
    # Load SpaCy model
    try:
        nlp = spacy.load('en_core_web_trf')  # Use the transformer model for better accuracy
    except:
        nlp = spacy.load('en_core_web_sm')   # Fallback to smaller model
    
    # Get ALL conversation data
    messages = await db.query('SELECT content, id, created_at FROM message WHERE content IS NOT NONE')
    tape_entries = await db.query('SELECT content, id, created_at FROM tape WHERE content IS NOT NONE')
    
    logger.info(f"📝 Processing {len(messages)} messages + {len(tape_entries)} tape entries")
    
    # Clear artificial graph data
    logger.info("🗑️ Clearing artificial graph data...")
    await db.query("DELETE concept WHERE name IN ['blue', 'Peppy', 'Potola', 'Serramanna, Sardinia, Italy']")
    await db.query("DELETE knows")
    
    # Process all conversation content
    facts_extracted = 0
    concepts_created = 0
    relationships_created = 0
    
    all_content = []
    # Add messages
    for msg in messages:
        content = msg.get('content', '')
        if content and len(content.strip()) > 5:
            timestamp = msg.get('created_at', time.time())
            if timestamp is None:
                timestamp = datetime.now()
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                # Handle string timestamps if any exist
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            all_content.append({
                'content': content,
                'source_type': 'message',
                'source_id': str(msg['id']),
                'timestamp': timestamp
            })
    
    # Add tape entries  
    for tape in tape_entries:
        content = tape.get('content', '')
        if content and len(content.strip()) > 5:
            timestamp = tape.get('created_at', time.time())
            if timestamp is None:
                timestamp = datetime.now()
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                # Handle string timestamps if any exist
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            all_content.append({
                'content': content,
                'source_type': 'tape',
                'source_id': str(tape['id']),
                'timestamp': timestamp
            })
    
    logger.info(f"📊 Total content pieces to analyze: {len(all_content)}")
    
    for i, item in enumerate(all_content):
        content = item['content']
        source_ref = f"{item['source_type']}:{item['source_id']}"
        
        logger.info(f"🔍 [{i+1}/{len(all_content)}] Processing: {content[:60]}...")
        
        try:
            # SpaCy analysis
            doc = nlp(content)
            
            # Extract entities and create concepts
            for ent in doc.ents:
                entity_name = ent.text.strip()
                entity_type = ent.label_
                
                if len(entity_name) > 1 and entity_name.lower() not in ['i', 'you', 'we', 'they']:
                    # Create or update concept with sanitized ID
                    sanitized_id = sanitize_concept_id(entity_name)
                    concept_id = f"concept:{sanitized_id}"
                    
                    try:
                        logger.debug(f"Creating concept: {concept_id} ({entity_name})")
                        result = await db.query(f"""
                            UPSERT {concept_id} SET 
                                name = $name,
                                kind = $kind,
                                first_mentioned = $timestamp,
                                last_mentioned = $timestamp,
                                mentioned_count = (mentioned_count OR 0) + 1,
                                properties = properties OR {{}}
                        """, {
                            'name': entity_name,
                            'kind': entity_type.lower(),
                            'timestamp': item['timestamp']
                        })
                        logger.debug(f"Concept creation result: {result}")
                    except Exception as e:
                        logger.error(f"Failed to create concept {concept_id}: {e}")
                        continue
                    
                    concepts_created += 1
            
            # Extract relationships (subject-verb-object patterns)
            for token in doc:
                if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                    subject = token.text
                    verb = token.head.text
                    
                    # Find object
                    obj = None
                    for child in token.head.children:
                        if child.dep_ in ['dobj', 'attr', 'prep']:
                            obj = child.text
                            break
                    
                    if obj and len(subject) > 1 and len(verb) > 1:
                        # Only create user->concept relationships (schema requirement)
                        if subject.lower() in ['i', 'my', 'me']:
                            subject_id = "user:peppi"  # Use actual user ID
                            
                            obj_sanitized = sanitize_concept_id(obj)
                            obj_id = f"concept:{obj_sanitized}"
                            
                            # Create relationship with sanitized name
                            relationship_raw = f"{verb}_{obj}".replace(' ', '_')
                            relationship_name = sanitize_concept_id(relationship_raw)
                            
                            # Get message ID for source_message (required by schema)
                            # source_id already contains full record ID like "message:12345"
                            source_message_id = item['source_id'] if item['source_type'] == 'message' else None
                            
                            if source_message_id:
                                try:
                                    logger.debug(f"Creating relationship: {subject_id} -> {obj_id} ({relationship_name})")
                                    result = await db.query(f"""
                                        RELATE {subject_id}->knows->{obj_id} SET
                                            relationship = $relationship,
                                            strength = 0.7,
                                            fidelity = 3,
                                            learned_at = $timestamp,
                                            source_message = {source_message_id},
                                            access_count = 1
                                    """, {
                                        'relationship': relationship_name,
                                        'timestamp': item['timestamp']
                                    })
                                    logger.debug(f"Relationship creation result: {result}")
                                    relationships_created += 1
                                    
                                except Exception as e:
                                    logger.error(f"Relationship creation failed ({subject_id} -> {obj_id}): {e}")
                                    continue
            
            facts_extracted += 1
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
    
    # Summary
    logger.info(f"✅ Extraction complete!")
    logger.info(f"   📊 Facts extracted from: {facts_extracted} conversations")
    logger.info(f"   🧩 Concepts created: {concepts_created}")
    logger.info(f"   🔗 Relationships created: {relationships_created}")
    
    # Verify what we built
    logger.info("🔍 Verifying created graph...")
    concepts = await db.query("SELECT name, kind, mentioned_count FROM concept ORDER BY mentioned_count DESC LIMIT 10")
    logger.info(f"📝 Top concepts:")
    for concept in concepts:
        logger.info(f"   - {concept.get('name')}: {concept.get('kind')} (mentioned {concept.get('mentioned_count')} times)")
    
    relationships = await db.query("SELECT relationship, in, out, source_conversation FROM knows LIMIT 10")
    logger.info(f"🔗 Sample relationships:")
    for rel in relationships:
        logger.info(f"   - {rel.get('in')} --{rel.get('relationship')}--> {rel.get('out')} (from {rel.get('source_conversation')})")
    
    await db.close()
    logger.info("🎉 Real memory graph built successfully!")

if __name__ == "__main__":
    asyncio.run(extract_facts_from_conversations())