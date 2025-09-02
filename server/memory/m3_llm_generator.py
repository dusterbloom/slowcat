"""M3 LLM Memory Generator

This module provides LLM-powered memory generation for the M3 system,
including episodic and semantic memory extraction using local LM Studio.
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EpisodicMemory:
    """Represents an episodic memory extracted from conversation"""
    sequence: List[str]  # Sequence of utterances
    summary: str  # LLM-generated summary
    key_events: List[str]  # Important events in the sequence
    participants: List[str]  # Speakers involved
    temporal_markers: List[str]  # Time-related expressions
    confidence: float = 0.8

@dataclass
class SemanticMemory:
    """Represents semantic knowledge extracted from text"""
    facts: List[str]  # Extracted factual statements
    concepts: List[str]  # Key concepts mentioned
    relationships: List[Dict[str, str]]  # Entity relationships
    domain: str  # Knowledge domain (e.g., "personal", "technical")
    confidence: float = 0.8

class M3LLMGenerator:
    """LLM-powered memory generation for M3 system"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:1234/v1",
                 model: str = "qwen2.5-0.5b-instruct-mlx",
                 speaker_facts_model: str = "qwen2.5-0.5b-instruct-mlx:2",
                 timeout: int = 5,
                 max_retries: int = 2):
        """Initialize M3 LLM Generator
        
        Args:
            base_url: LM Studio API endpoint
            model: Model name for semantic/episodic memory extraction
            speaker_facts_model: Model name for speaker facts extraction
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url
        self.model = model
        self.speaker_facts_model = speaker_facts_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Episodic memory prompts
        self.episodic_summary_prompt = """Analyze this conversation sequence and create a concise episodic memory.

CONVERSATION:
{conversation}

Extract:
1. SUMMARY: One sentence summary of what happened
2. KEY_EVENTS: Important actions or decisions (max 3)
3. PARTICIPANTS: Who was involved
4. TEMPORAL_MARKERS: Any time references mentioned

Respond in JSON format:
{{"summary": "...", "key_events": ["...", "..."], "participants": ["...", "..."], "temporal_markers": ["...", "..."]}}"""

        # Semantic memory prompts
        self.semantic_extraction_prompt = """Extract factual knowledge and relationships from this text.

TEXT:
{text}

Extract:
1. FACTS: Concrete factual statements (max 5)
2. CONCEPTS: Key concepts or topics mentioned (max 5)
3. RELATIONSHIPS: Subject-predicate-object relationships (max 5)
4. DOMAIN: Knowledge category (personal/technical/general/creative)

Respond in JSON format:
{{"facts": ["...", "..."], "concepts": ["...", "..."], "relationships": [{{"subject": "...", "predicate": "...", "object": "..."}}], "domain": "..."}}"""

        # Enhanced semantic prompt for better fact extraction
        self.enhanced_semantic_prompt = """You are an expert at extracting structured knowledge. Analyze this text and extract meaningful information.

TEXT: {text}

Your task:
1. Identify concrete FACTS (things that are definitely true based on the text)
2. Extract key CONCEPTS (important topics, entities, ideas mentioned)
3. Find RELATIONSHIPS between entities (who/what relates to whom/what and how)
4. Classify the knowledge DOMAIN

Be precise and only extract information explicitly stated or strongly implied.

Format as JSON:
{{"facts": ["fact1", "fact2"], "concepts": ["concept1", "concept2"], "relationships": [{{"subject": "X", "predicate": "relates_to", "object": "Y"}}], "domain": "personal|technical|general|creative"}}"""

    async def generate_episodic_memory(self, 
                                     conversation_sequence: List[str],
                                     speaker_ids: List[str] = None) -> Optional[EpisodicMemory]:
        """Generate episodic memory from conversation sequence
        
        Args:
            conversation_sequence: List of utterances in temporal order
            speaker_ids: Optional speaker identifiers for each utterance
            
        Returns:
            EpisodicMemory object or None if generation failed
        """
        try:
            if not conversation_sequence or len(conversation_sequence) < 2:
                logger.warning("Insufficient conversation sequence for episodic memory")
                return None
            
            # Format conversation with speaker IDs if available
            formatted_conversation = []
            for i, utterance in enumerate(conversation_sequence):
                speaker = speaker_ids[i] if speaker_ids and i < len(speaker_ids) else f"Speaker{i+1}"
                formatted_conversation.append(f"{speaker}: {utterance}")
            
            conversation_text = "\n".join(formatted_conversation)
            
            # Generate episodic summary using LLM
            prompt = self.episodic_summary_prompt.format(conversation=conversation_text)
            response = await self._call_llm(prompt)
            
            if not response:
                logger.warning("Failed to generate episodic memory via LLM")
                return None
            
            # Parse LLM response
            try:
                parsed = json.loads(response)
                
                return EpisodicMemory(
                    sequence=conversation_sequence,
                    summary=parsed.get("summary", ""),
                    key_events=parsed.get("key_events", []),
                    participants=parsed.get("participants", []),
                    temporal_markers=parsed.get("temporal_markers", []),
                    confidence=0.85  # High confidence for LLM-generated episodic memories
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse episodic memory JSON: {e}")
                # Fallback: create basic episodic memory
                return EpisodicMemory(
                    sequence=conversation_sequence,
                    summary=f"Conversation with {len(set(speaker_ids)) if speaker_ids else 'multiple'} participants",
                    key_events=[utterance for utterance in conversation_sequence[:2]],  # First 2 utterances
                    participants=list(set(speaker_ids)) if speaker_ids else ["unknown"],
                    temporal_markers=[],
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"Error generating episodic memory: {e}")
            return None
    
    async def generate_semantic_memory(self, 
                                     text: str,
                                     enhanced_extraction: bool = True) -> Optional[SemanticMemory]:
        """Generate semantic memory from text content
        
        Args:
            text: Text to extract semantic knowledge from
            enhanced_extraction: Use enhanced prompt for better extraction
            
        Returns:
            SemanticMemory object or None if generation failed
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("Text too short for semantic memory extraction")
                return None
            
            # Choose prompt based on extraction method
            prompt = (self.enhanced_semantic_prompt if enhanced_extraction 
                     else self.semantic_extraction_prompt).format(text=text)
            
            response = await self._call_llm(prompt)
            
            if not response:
                logger.warning("Failed to generate semantic memory via LLM")
                return None
            
            # Parse LLM response
            try:
                parsed = json.loads(response)
                
                return SemanticMemory(
                    facts=parsed.get("facts", []),
                    concepts=parsed.get("concepts", []),
                    relationships=parsed.get("relationships", []),
                    domain=parsed.get("domain", "general"),
                    confidence=0.85  # High confidence for LLM-generated semantic memories
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse semantic memory JSON: {e}")
                # Fallback: basic extraction
                return SemanticMemory(
                    facts=[text[:100] + "..." if len(text) > 100 else text],
                    concepts=[],
                    relationships=[],
                    domain="general",
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"Error generating semantic memory: {e}")
            return None
    
    async def extract_speaker_facts(self, 
                                  text: str, 
                                  speaker_id: str = "user") -> List[Dict[str, Any]]:
        """Extract facts specifically about the speaker
        
        Args:
            text: Text containing speaker information
            speaker_id: Identifier for the speaker
            
        Returns:
            List of fact dictionaries in M3 format
        """
        try:
            speaker_prompt = f"""Extract personal facts about the speaker from this text.

TEXT: {text}

Focus on:
- What the speaker owns, has, or possesses
- Where the speaker lives, works, or goes
- What the speaker likes, does, or experiences
- Personal relationships and connections

Format each fact as: "subject predicate object"
Examples:
- "user has_pet Rex"
- "user lives_in Seattle" 
- "user works_at Google"

Respond with JSON array of facts:
["fact1", "fact2", "fact3"]"""

            response = await self._call_llm(speaker_prompt, model=self.speaker_facts_model)
            
            if not response:
                return []
            
            try:
                facts_list = json.loads(response)
                
                # Convert to M3 fact format
                m3_facts = []
                for fact in facts_list:
                    parts = fact.split(" ", 2)  # Split into 3 parts max
                    if len(parts) >= 3:
                        m3_facts.append({
                            "subject": parts[0],
                            "predicate": parts[1], 
                            "object": " ".join(parts[2:]),
                            "confidence": 0.8,
                            "source": "llm_extraction",
                            "speaker_id": speaker_id,
                            "extracted_at": datetime.now().isoformat()
                        })
                
                logger.info(f"Extracted {len(m3_facts)} speaker facts from text")
                return m3_facts
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse speaker facts JSON")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting speaker facts: {e}")
            return []
    
    async def _call_llm(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Make async HTTP call to LM Studio API
        
        Args:
            prompt: Prompt to send to the LLM
            model: Specific model to use (defaults to self.model)
            
        Returns:
            LLM response text or None if failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model or self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert at extracting structured information. Always respond with valid JSON."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        "max_tokens": 500,
                        "temperature": 0.3,
                        "stream": False
                    }
                    
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        timeout=timeout
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            if content:
                                logger.debug(f"LLM response received: {len(content)} characters")
                                return content.strip()
                            else:
                                logger.warning("Empty response from LLM")
                        else:
                            logger.warning(f"LLM API error: {response.status}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"LLM request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        logger.error(f"All LLM request attempts failed after {self.max_retries + 1} tries")
        return None
    
    async def test_connection(self) -> bool:
        """Test connection to LM Studio API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "This is a connection test. Extract facts from this text: 'connection test ok'"
            logger.info(f"Testing connection to {self.base_url} with model {self.model}")
            response = await self._call_llm(test_prompt)
            
            logger.info(f"Test response: {response}")
            if response and (("ok" in response.lower()) or ("connection" in response.lower()) or ("test" in response.lower())):
                logger.info("LM Studio connection test successful")
                return True
            else:
                logger.warning(f"LM Studio connection test failed - response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"LM Studio connection test error: {e}")
            return False
    
    async def generate_context_summary(self, 
                                     memories: List[Dict[str, Any]], 
                                     query: str = "") -> str:
        """Generate a contextual summary from retrieved memories
        
        Args:
            memories: List of memory nodes from M3 retrieval
            query: Optional query context for focused summarization
            
        Returns:
            Contextual summary for LLM injection
        """
        try:
            if not memories:
                return ""
            
            # Format memories for summarization
            memory_texts = []
            for memory in memories:
                contents = memory.get('contents', [])
                node_type = memory.get('node_type', 'unknown')
                
                for content in contents:
                    memory_texts.append(f"[{node_type.upper()}] {content}")
            
            memories_text = "\n".join(memory_texts[:10])  # Limit to 10 memories
            
            summary_prompt = f"""Create a concise summary of relevant memories for conversation context.

QUERY: {query if query else "General conversation"}

MEMORIES:
{memories_text}

Create a brief summary (2-3 sentences) highlighting the most relevant information for the conversation.
Focus on facts, relationships, and recent context that would be helpful."""

            response = await self._call_llm(summary_prompt)
            
            if response:
                # Clean up the response (remove JSON formatting if present)
                summary = response.strip('"{}[]')
                logger.info(f"Generated context summary: {len(summary)} characters")
                return summary
            else:
                # Fallback: simple concatenation
                return " ".join(memory_texts[:3])
                
        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return ""