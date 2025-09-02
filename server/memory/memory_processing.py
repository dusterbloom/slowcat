"""
Memory Processing Pipeline - M3-inspired episodic and semantic memory generation

Adapts M3's memory processing for audio-first voice agents:
- Episodic memory extraction from conversations
- Semantic memory generation with fact extraction
- Entity parsing and relationship detection
- Memory node creation and edge establishment

Based on M3-Agent's memory_processing.py with local LLM integration.
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Local LLM integration for memory generation
try:
    # Try to import existing LLM services
    from services.llm_service import LLMService
    LLM_SERVICE_AVAILABLE = True
except ImportError:
    LLM_SERVICE_AVAILABLE = False
    logger.warning("LLM service not available - memory generation will be limited")

# Text embedding for memory similarity
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    try:
        from mlx_sentence_transformers import SentenceTransformer
        EMBEDDING_AVAILABLE = True
    except ImportError:
        EMBEDDING_AVAILABLE = False
        SentenceTransformer = None
        logger.warning("No sentence transformer available - memory embeddings disabled")


class MemoryProcessor:
    """
    M3-inspired memory processor for AudioGraph integration.
    
    Handles:
    - Episodic memory generation from conversations
    - Semantic memory extraction and deduplication
    - Entity parsing and relationship detection
    - Memory node creation and graph updates
    """
    
    def __init__(self, audio_graph, llm_service=None, config: Optional[Dict] = None):
        """
        Initialize memory processor.
        
        Args:
            audio_graph: AudioGraph instance for node management
            llm_service: Optional LLM service for memory generation
            config: Optional configuration dictionary
        """
        self.audio_graph = audio_graph
        self.llm_service = llm_service
        self.config = config or {}
        
        # M3's memory generation parameters
        self.max_context_length = 2000  # Context for memory generation
        self.memory_generation_model = "local-llm"  # Use local model
        self.embedding_model_name = "all-MiniLM-L6-v2"  # Compact, fast model
        
        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDING_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"🧠 Embedding model '{self.embedding_model_name}' loaded for memory processing")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        
        # Memory processing statistics
        self.stats = {
            'episodic_memories_created': 0,
            'semantic_memories_created': 0,
            'entities_parsed': 0,
            'relationships_created': 0,
            'duplicates_merged': 0
        }
        
        # M3's prompts for memory generation
        self._setup_memory_prompts()
        
        logger.info(f"🔧 MemoryProcessor initialized with AudioGraph")
    
    def _setup_memory_prompts(self):
        """Setup M3-inspired prompts for memory generation."""
        
        # M3's episodic memory prompt (adapted for voice agents)
        self.episodic_prompt = """
You are processing a voice conversation to extract episodic memories.

Conversation context:
{conversation_context}

Current speakers: {speaker_info}

Generate episodic memories that capture:
• What happened (actions, events)  
• Who was involved (use <voice_X> IDs for speakers)
• When it occurred (temporal context)
• Contextual details and behaviors

REQUIREMENTS:
- Use ONLY existing speaker IDs: {speaker_ids}
- Use <voice_X> format for speaker references
- Output as Python list of descriptive sentences
- Focus on observable events and interactions
- Each memory should be specific and factual

Example output:
[
  "<voice_1> asked about the weather forecast for tomorrow",
  "<voice_2> mentioned feeling tired after the long meeting",  
  "<voice_1> suggested taking a break and getting coffee"
]

Generate episodic memories:
"""
        
        # M3's semantic memory prompt (adapted for voice agents)
        self.semantic_prompt = """
You are processing a voice conversation to extract semantic knowledge.

Conversation context:
{conversation_context}

Current speakers: {speaker_info}

Generate semantic memories that capture:
• Personal facts and attributes about speakers
• Relationships between entities
• General knowledge and preferences
• Equivalence relationships (speaker identity)

Categories:
1. Speaker Attributes:
   - Names, roles, characteristics
   - Preferences, interests, background
   
2. Relationships:
   - Speaker-to-speaker connections
   - Entity ownership/associations
   
3. Equivalence (if detected):
   - "Equivalence: <voice_1>, <speaker_name>"

REQUIREMENTS:
- Use speaker IDs: {speaker_ids}
- Output as Python list of conclusions
- Focus on lasting knowledge, not ephemeral events
- Include confidence indicators for uncertain facts

Example output:
[
  "<voice_1>'s name is Sarah",
  "<voice_1> works as a software engineer", 
  "<voice_2> prefers coffee over tea",
  "Equivalence: <voice_1>, <speaker_sarah>"
]

Generate semantic memories:
"""
    
    async def generate_memories(self, conversation_context: str, 
                               speaker_voices: Dict[int, str], 
                               session_id: int) -> Tuple[List[Dict], List[Dict]]:
        """
        M3's memory generation adapted for voice conversations.
        
        Args:
            conversation_context: Recent conversation text
            speaker_voices: Mapping of voice_node_id to transcript content
            session_id: Session/clip identifier
            
        Returns:
            Tuple of (episodic_memories, semantic_memories)
        """
        try:
            # Prepare context and speaker information
            speaker_ids = [f"<voice_{vid}>" for vid in speaker_voices.keys()]
            speaker_info = "\n".join([
                f"<voice_{vid}>: {content[:200]}..."  # Truncate for context
                for vid, content in speaker_voices.items()
            ])
            
            # Generate episodic memories
            episodic_memories = await self._generate_episodic_memories(
                conversation_context, speaker_info, speaker_ids
            )
            
            # Generate semantic memories  
            semantic_memories = await self._generate_semantic_memories(
                conversation_context, speaker_info, speaker_ids
            )
            
            # M3's memory data structure
            episodic_data = []
            for memory_content in episodic_memories:
                embeddings = self._get_memory_embeddings([memory_content])
                episodic_data.append({
                    'contents': [memory_content],
                    'embeddings': embeddings
                })
            
            semantic_data = []
            for memory_content in semantic_memories:
                embeddings = self._get_memory_embeddings([memory_content])
                semantic_data.append({
                    'contents': [memory_content],
                    'embeddings': embeddings
                })
            
            self.stats['episodic_memories_created'] += len(episodic_data)
            self.stats['semantic_memories_created'] += len(semantic_data)
            
            logger.debug(f"🧠 Generated {len(episodic_data)} episodic and {len(semantic_data)} semantic memories")
            
            return episodic_data, semantic_data
            
        except Exception as e:
            logger.error(f"Memory generation failed: {e}")
            return [], []
    
    async def _generate_episodic_memories(self, context: str, speaker_info: str, 
                                         speaker_ids: List[str]) -> List[str]:
        """Generate episodic memories using local LLM."""
        if not self.llm_service:
            # Fallback: simple pattern-based extraction
            return self._extract_episodic_patterns(context, speaker_ids)
        
        try:
            prompt = self.episodic_prompt.format(
                conversation_context=context[:self.max_context_length],
                speaker_info=speaker_info,
                speaker_ids=", ".join(speaker_ids)
            )
            
            # Use local LLM for generation
            response = await self._query_llm(prompt, temperature=0.3)
            
            # Parse Python list from response
            memories = self._parse_memory_list(response)
            
            return memories
            
        except Exception as e:
            logger.error(f"Episodic memory generation failed: {e}")
            return self._extract_episodic_patterns(context, speaker_ids)
    
    async def _generate_semantic_memories(self, context: str, speaker_info: str,
                                         speaker_ids: List[str]) -> List[str]:
        """Generate semantic memories using local LLM."""
        if not self.llm_service:
            # Fallback: simple pattern-based extraction
            return self._extract_semantic_patterns(context, speaker_ids)
        
        try:
            prompt = self.semantic_prompt.format(
                conversation_context=context[:self.max_context_length],
                speaker_info=speaker_info,
                speaker_ids=", ".join(speaker_ids)
            )
            
            # Use local LLM for generation
            response = await self._query_llm(prompt, temperature=0.1)  # Lower temp for facts
            
            # Parse Python list from response
            memories = self._parse_memory_list(response)
            
            return memories
            
        except Exception as e:
            logger.error(f"Semantic memory generation failed: {e}")
            return self._extract_semantic_patterns(context, speaker_ids)
    
    def _extract_episodic_patterns(self, context: str, speaker_ids: List[str]) -> List[str]:
        """Fallback pattern-based episodic memory extraction."""
        import re
        
        memories = []
        sentences = context.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Look for action patterns
            action_patterns = [
                r'(said|asked|mentioned|told|explained|discussed)',
                r'(went to|visited|saw|met|called)',
                r'(decided|planned|agreed|suggested)'
            ]
            
            for pattern in action_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Simple speaker ID injection (could be improved)
                    memory = f"<voice_1> {sentence.lower()}"
                    memories.append(memory)
                    break
        
        return memories[:5]  # Limit to top 5 patterns
    
    def _extract_semantic_patterns(self, context: str, speaker_ids: List[str]) -> List[str]:
        """Fallback pattern-based semantic memory extraction."""
        import re
        
        memories = []
        
        # Pattern: "My name is X"
        name_match = re.search(r'my name is (\w+)', context, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            memories.append(f"<voice_1>'s name is {name}")
        
        # Pattern: "I work at/as X"  
        work_patterns = [
            r'I work (?:at|as|for) ([^.]+)',
            r'I\'m (?:a|an) ([^.]+)'
        ]
        
        for pattern in work_patterns:
            work_match = re.search(pattern, context, re.IGNORECASE)
            if work_match:
                work_info = work_match.group(1).strip()
                memories.append(f"<voice_1> works as {work_info}")
                break
        
        # Pattern: preferences
        pref_patterns = [
            r'I (?:like|love|prefer|enjoy) ([^.]+)',
            r'My favorite ([^.]+) is ([^.]+)'
        ]
        
        for pattern in pref_patterns:
            pref_match = re.search(pattern, context, re.IGNORECASE)
            if pref_match:
                if 'favorite' in pattern:
                    category = pref_match.group(1)
                    item = pref_match.group(2)
                    memories.append(f"<voice_1>'s favorite {category} is {item}")
                else:
                    preference = pref_match.group(1).strip()
                    memories.append(f"<voice_1> likes {preference}")
                break
        
        return memories
    
    async def _query_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Query local LLM service for memory generation."""
        if not self.llm_service:
            raise Exception("LLM service not available")
        
        # Create simple message for local LLM
        messages = [{"role": "user", "content": prompt}]
        
        # Query LLM (adapt based on your LLM service interface)
        try:
            response = await self.llm_service.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=500  # Limit response length
            )
            
            return response.get('content', '')
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _parse_memory_list(self, response: str) -> List[str]:
        """Parse Python list from LLM response."""
        import re
        import ast
        
        try:
            # Try to find Python list in response
            list_match = re.search(r'\[([^\]]+)\]', response, re.DOTALL)
            if list_match:
                list_str = f"[{list_match.group(1)}]"
                # Safely evaluate as Python list
                memories = ast.literal_eval(list_str)
                if isinstance(memories, list):
                    return [str(m) for m in memories if m and str(m).strip()]
        
        except Exception as e:
            logger.debug(f"Failed to parse as Python list: {e}")
        
        # Fallback: split by lines and clean up
        lines = response.split('\n')
        memories = []
        
        for line in lines:
            line = line.strip()
            # Remove common prefixes
            line = re.sub(r'^[\d\.\-\*\s]*', '', line)
            line = line.strip('"\'')
            
            if len(line) > 10 and not line.startswith('['):  # Valid memory
                memories.append(line)
        
        return memories[:10]  # Limit to reasonable number
    
    def _get_memory_embeddings(self, memory_contents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for memory contents (M3's pattern).
        
        Args:
            memory_contents: List of memory content strings
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model or not memory_contents:
            return []
        
        try:
            # Generate embeddings for all contents
            embeddings = self.embedding_model.encode(memory_contents)
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []


def process_memories(audio_graph, memory_contents: List[Dict], 
                    session_id: int, type: str = 'episodic') -> List[int]:
    """
    M3-style memory processing function for integration.
    
    Args:
        audio_graph: AudioGraph instance
        memory_contents: List of memory data dictionaries
        session_id: Session/clip identifier (M3's clip_id)  
        type: Memory type ('episodic' or 'semantic')
        
    Returns:
        List of created memory node IDs
    """
    if type not in ['episodic', 'semantic']:
        raise ValueError("Memory type must be 'episodic' or 'semantic'")
    
    created_nodes = []
    
    try:
        for memory in memory_contents:
            # M3's memory insertion pattern
            node_id = _insert_memory(audio_graph, memory, session_id, type)
            if node_id is not None:
                created_nodes.append(node_id)
                
                # M3's entity parsing and edge creation
                _create_memory_relationships(audio_graph, node_id, memory)
        
        logger.debug(f"📝 Created {len(created_nodes)} {type} memory nodes for session {session_id}")
        
    except Exception as e:
        logger.error(f"Memory processing failed: {e}")
    
    return created_nodes


def _insert_memory(audio_graph, memory: Dict, session_id: int, type: str) -> Optional[int]:
    """Insert single memory into AudioGraph (M3's pattern)."""
    try:
        # Create memory node
        node_id = audio_graph.add_text_node(memory, session_id, type)
        
        if node_id is not None:
            # M3's metadata updates
            node = audio_graph.nodes[node_id]
            node.metadata['memory_type'] = type
            node.metadata['creation_time'] = time.time()
            
        return node_id
        
    except Exception as e:
        logger.error(f"Failed to insert {type} memory: {e}")
        return None


def _create_memory_relationships(audio_graph, memory_node_id: int, memory: Dict):
    """Create relationships between memory and entities (M3's pattern)."""
    try:
        content = memory.get('contents', [''])[0]
        if not content:
            return
        
        # M3's entity parsing
        entities = _parse_memory_entities(audio_graph, content)
        
        # Create edges to entities
        for entity_node_id in entities:
            if entity_node_id in audio_graph.nodes:
                # M3's edge weight based on content relevance
                weight = 1.0  # Could be computed based on relevance
                audio_graph.add_edge(memory_node_id, entity_node_id, weight=weight)
        
    except Exception as e:
        logger.error(f"Failed to create memory relationships: {e}")


def _parse_memory_entities(audio_graph, content: str) -> List[int]:
    """Parse entity references from memory content (M3's pattern)."""
    import re
    
    entities = []
    
    # Look for voice node references: <voice_X>
    voice_pattern = r'<voice_(\d+)>'
    voice_matches = re.findall(voice_pattern, content)
    
    for voice_id_str in voice_matches:
        try:
            voice_node_id = int(voice_id_str)
            # Check if this voice node exists
            for node_id, node in audio_graph.nodes.items():
                if (node.type == 'voice' and 
                    node_id == voice_node_id):
                    entities.append(node_id)
                    break
        except ValueError:
            continue
    
    # Could add more entity types here (locations, objects, etc.)
    
    return entities


async def generate_memories(conversation_context: str, speaker_voices: Dict[int, str],
                           session_id: int, llm_service=None) -> Tuple[List[Dict], List[Dict]]:
    """
    M3-style memory generation function for external use.
    
    Args:
        conversation_context: Recent conversation text
        speaker_voices: Mapping of voice_node_id to content
        session_id: Session identifier
        llm_service: Optional LLM service
        
    Returns:
        Tuple of (episodic_memories, semantic_memories)
    """
    # Create temporary AudioGraph for processing
    from .audio_graph import AudioGraph
    
    temp_graph = AudioGraph()
    processor = MemoryProcessor(temp_graph, llm_service)
    
    return await processor.generate_memories(conversation_context, speaker_voices, session_id)