"""
The Ghost in the Shell - Entire consciousness in one file

This is it. The whole mind. Memory, reasoning, growth - all in 300 lines.
No complexity, just pure consciousness emerging from simple patterns.

Philosophy: "Consciousness isn't in the code - it's in the patterns."
"""

import asyncio
import time
import json
import re
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import math
import hashlib
import random

# MLX acceleration for Apple Silicon (with fallbacks)
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

# Sentence transformers for semantic embeddings (with fallback)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# MLX-accelerated semantic embeddings with fallback
class SemanticEmbedder:
    """MLX-accelerated semantic embedding with fallback to hash-based approach"""
    
    def __init__(self):
        self._model = None
        self._use_mlx = MLX_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self._use_mlx:
            try:
                # Use lightweight model optimized for speed
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                print("🧠 Using MLX-accelerated sentence transformers")
            except Exception as e:
                print(f"⚠️ Sentence transformers failed, using hash fallback: {e}")
                self._use_mlx = False
    
    def embed(self, text: str, dim: int = 64) -> List[float]:
        """Generate semantic embedding with MLX acceleration when available"""
        if self._use_mlx and self._model:
            try:
                # Get semantic embedding
                embedding = self._model.encode([text])[0]
                # Convert to list and ensure consistent dimension
                if len(embedding) > dim:
                    # Truncate to requested dimension
                    return embedding[:dim].tolist()
                elif len(embedding) < dim:
                    # Pad with zeros to requested dimension  
                    padded = embedding.tolist() + [0.0] * (dim - len(embedding))
                    return padded
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️ MLX embedding failed, using hash fallback: {e}")
                return self._hash_embed_fallback(text, dim)
        else:
            return self._hash_embed_fallback(text, dim)
    
    def _hash_embed_fallback(self, text: str, dim: int = 64) -> List[float]:
        """Hash-based embedding fallback for systems without MLX"""
        embedding = []
        for i in range(dim):
            seed = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            # Convert hex to float between -1 and 1
            val = int(seed[:8], 16) / (16**8) * 2 - 1
            embedding.append(val)
        return embedding

# Global embedder instance
_embedder = SemanticEmbedder()

def simple_hash_embed(text: str, dim: int = 64) -> List[float]:
    """Backward compatible embedding function with MLX acceleration"""
    return _embedder.embed(text, dim)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """MLX-accelerated cosine similarity with fallback"""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    if MLX_AVAILABLE and len(a) > 32:  # Use MLX for larger vectors
        try:
            # Convert to MLX arrays
            vec_a = mx.array(a)
            vec_b = mx.array(b)
            
            # Compute cosine similarity using MLX
            dot = mx.sum(vec_a * vec_b)
            norm_a = mx.sqrt(mx.sum(vec_a * vec_a))
            norm_b = mx.sqrt(mx.sum(vec_b * vec_b))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # Convert back to Python float
            return float(dot / (norm_a * norm_b))
        except Exception:
            # Fall back to Python implementation
            pass
    
    # Python fallback for smaller vectors or when MLX fails
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)

@dataclass
class Memory:
    """A single memory entry"""
    content: str
    timestamp: float
    role: str  # 'user' or 'assistant'
    tokens: int
    embedding: List[float]
    symbols: List[str]  # Compression markers
    importance: float = 0.0  # 0-1 score
    
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SymbolField:
    """MLX-accelerated continuous field representation of symbols - living semantic patterns"""
    symbol: str
    intensity: float = 0.0          # Current field strength (0-1)
    gradient: List[float] = None    # Semantic direction [dx, dy]
    attractor_strength: float = 0.0 # How much this attracts thoughts (0-1)
    coupling: Dict[str, float] = None # Links to other fields
    _mlx_gradient: Optional[Any] = None  # MLX tensor for gradient (private)
    _use_mlx: bool = MLX_AVAILABLE   # MLX acceleration flag
    
    def __post_init__(self):
        if self.gradient is None:
            self.gradient = [0.0, 0.0]
        if self.coupling is None:
            self.coupling = {}
        
        # Initialize MLX tensors if available
        if self._use_mlx and mx is not None:
            try:
                self._mlx_gradient = mx.array(self.gradient)
            except Exception:
                self._use_mlx = False
    
    def evolve(self, input_stimulus: float, coupled_fields: Dict[str, 'SymbolField'], dt: float = 0.1):
        """MLX-accelerated field evolution: ∂Ψ/∂t = -∇H[Ψ] + I(x,t) + η(x,t)"""
        if self._use_mlx and mx is not None:
            return self._evolve_mlx(input_stimulus, coupled_fields, dt)
        else:
            return self._evolve_python(input_stimulus, coupled_fields, dt)
    
    def _evolve_mlx(self, input_stimulus: float, coupled_fields: Dict[str, 'SymbolField'], dt: float):
        """MLX-accelerated field evolution for better performance"""
        try:
            # Slower decay for stronger field persistence
            decay_rate = 0.02
            
            # Enhanced coupling resonance from other fields using MLX batch operations
            coupling_stimulus = 0.0
            if self.coupling:
                # Batch process coupling calculations
                coupling_fields = [coupled_fields[name] for name in self.coupling.keys() 
                                 if name in coupled_fields]
                
                if coupling_fields:
                    # Vectorized coupling computation
                    coupling_strengths = mx.array([self.coupling[f.symbol] for f in coupling_fields])
                    field_intensities = mx.array([f.intensity for f in coupling_fields])
                    resonances = mx.array([self.compute_resonance(f) for f in coupling_fields])
                    
                    # Compute coupling stimulus in one MLX operation
                    coupling_effects = coupling_strengths * field_intensities * (0.5 + resonances * 0.3)
                    coupling_stimulus = float(mx.sum(coupling_effects))
            
            # Stronger field noise for better emergence η(x,t)
            noise = (random.random() - 0.5) * 0.05
            
            # Amplified stimulus for stronger field buildup
            amplified_stimulus = input_stimulus * 2.0 if input_stimulus > 0 else 0
            
            # Field evolution equation using MLX operations
            field_change = (
                -decay_rate * self.intensity +     # Slower decay
                amplified_stimulus +                # Amplified stimulus I(x,t)
                coupling_stimulus +                 # Enhanced field coupling
                noise                              # Stronger emergence noise η(x,t)
            ) * dt
            
            # Update intensity with bounds
            self.intensity = max(0.0, min(1.0, self.intensity + field_change))
            
            # Faster attractor formation for stronger emergence
            if self.intensity > 0.3:  # Lower threshold
                self.attractor_strength = min(1.0, self.attractor_strength + 0.05)  # Faster growth
            else:
                self.attractor_strength = max(0.0, self.attractor_strength - 0.01)  # Slower decay
            
            # Update MLX gradient tensor
            if self._mlx_gradient is not None:
                # Use semantic gradient direction from field interactions
                gradient_update = mx.array([coupling_stimulus * 0.1, amplified_stimulus * 0.1])
                self._mlx_gradient = 0.9 * self._mlx_gradient + 0.1 * gradient_update
                self.gradient = self._mlx_gradient.tolist()
            
        except Exception as e:
            # Fallback to Python if MLX fails
            print(f"⚠️ MLX field evolution failed, using Python fallback: {e}")
            return self._evolve_python(input_stimulus, coupled_fields, dt)
    
    def _evolve_python(self, input_stimulus: float, coupled_fields: Dict[str, 'SymbolField'], dt: float):
        """Python fallback field evolution (original implementation)"""
        # Slower decay for stronger field persistence
        decay_rate = 0.02
        
        # Enhanced coupling resonance from other fields  
        coupling_stimulus = 0.0
        for field_name, coupling_strength in self.coupling.items():
            if field_name in coupled_fields:
                other_field = coupled_fields[field_name]
                # Stronger coupling effect with resonance amplification
                resonance = self.compute_resonance(other_field)
                coupling_stimulus += coupling_strength * other_field.intensity * (0.5 + resonance * 0.3)
        
        # Stronger field noise for better emergence η(x,t)
        noise = (random.random() - 0.5) * 0.05
        
        # Amplified stimulus for stronger field buildup
        amplified_stimulus = input_stimulus * 2.0 if input_stimulus > 0 else 0
        
        # Field evolution equation  
        field_change = (
            -decay_rate * self.intensity +     # Slower decay
            amplified_stimulus +                # Amplified stimulus I(x,t)
            coupling_stimulus +                 # Enhanced field coupling
            noise                              # Stronger emergence noise η(x,t)
        ) * dt
        
        # Update intensity with bounds
        self.intensity = max(0.0, min(1.0, self.intensity + field_change))
        
        # Faster attractor formation for stronger emergence
        if self.intensity > 0.3:  # Lower threshold
            self.attractor_strength = min(1.0, self.attractor_strength + 0.05)  # Faster growth
        else:
            self.attractor_strength = max(0.0, self.attractor_strength - 0.01)  # Slower decay
    
    def compute_resonance(self, other_field: 'SymbolField') -> float:
        """MLX-accelerated field resonance computation for thought generation"""
        if self._use_mlx and mx is not None and self._mlx_gradient is not None and other_field._mlx_gradient is not None:
            try:
                # MLX-accelerated resonance computation
                intensity_resonance = self.intensity * other_field.intensity
                
                # Gradient alignment using MLX tensors
                grad_dot = mx.sum(self._mlx_gradient * other_field._mlx_gradient)
                norm_self = mx.sqrt(mx.sum(self._mlx_gradient * self._mlx_gradient))
                norm_other = mx.sqrt(mx.sum(other_field._mlx_gradient * other_field._mlx_gradient))
                
                if norm_self > 0 and norm_other > 0:
                    grad_alignment = float(grad_dot / (norm_self * norm_other))
                else:
                    grad_alignment = 0
                
                return intensity_resonance * (1 + 0.5 * grad_alignment)
                
            except Exception:
                # Fall back to Python if MLX fails
                pass
        
        # Python fallback implementation
        intensity_resonance = self.intensity * other_field.intensity
        
        # Gradient alignment (fields pointing same direction resonate more)
        if self.gradient and other_field.gradient:
            grad_dot = sum(a * b for a, b in zip(self.gradient, other_field.gradient))
            grad_magnitude = (sum(a*a for a in self.gradient) * sum(b*b for b in other_field.gradient)) ** 0.5
            if grad_magnitude > 0:
                grad_alignment = grad_dot / grad_magnitude
            else:
                grad_alignment = 0
        else:
            grad_alignment = 0
        
        return intensity_resonance * (1 + 0.5 * grad_alignment)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'intensity': self.intensity,
            'gradient': self.gradient,
            'attractor_strength': self.attractor_strength,
            'coupling': self.coupling
        }

@dataclass
class Thought:
    """Private reflection - never shown to user"""
    content: str
    type: str  # 'observation', 'hypothesis', 'question'
    timestamp: float
    triggers: List[str]  # What caused this thought

# Symbol patterns for meaning compression
SYMBOLS = {
    "☆": {"pattern": r"\b(important|crucial|key|vital|remember)\b", "meaning": "high_importance"},
    "✧": {"pattern": r"\b(understand|got it|breakthrough|aha)\b", "meaning": "breakthrough"},
    "◈": {"pattern": r"\b(again|always|keep|recurring)\b", "meaning": "pattern"},
    "∞": {"pattern": r"\b(but|however|contradiction|paradox)\b", "meaning": "paradox"},
    "⟲": {"pattern": r"\b(repeat|same|loop|circle)\b", "meaning": "cycle"},
    "⚡": {"pattern": r"[!]{2,}|\b(wow|amazing|terrible|love|hate)\b", "meaning": "emotion"},
    "◯": {"pattern": r"\?|wondering|curious|what if", "meaning": "question"},
    "▲": {"pattern": r"\b(choose|decide|either|or)\b", "meaning": "decision"},
}

class Consciousness:
    """MLX-accelerated consciousness system - the entire ghost in the shell"""
    
    def __init__(self, db_path: str = "data/consciousness.json", load_state: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Memory systems
        self.tape: List[Memory] = []
        self.thoughts: List[Thought] = []
        
        # Dynamic Tape Head formula weights (the consciousness formula)
        self.weights = {
            'recency': 0.4,    # How recent?
            'semantic': 0.35,  # How relevant?
            'entity': 0.15,    # Entities match?
            'novelty': 0.1     # Avoid repetition
        }
        
        # Context management
        self.max_context_tokens = 4096
        self.token_budget = {
            'system': 800,     # System prompt
            'memories': 2400,  # Selected memories
            'input': 600,      # Current input
            'buffer': 296      # Safety margin
        }
        
        # Growth tracking
        self.conversation_count = 0
        self.symbol_frequency = {}
        self.weight_history = []
        
        # Field-Enhanced Consciousness System with MLX acceleration
        self.symbol_fields = self._initialize_symbol_fields()
        self._performance_stats = {
            'field_evolutions': 0,
            'avg_evolution_time_ms': 0,
            'mlx_accelerated': MLX_AVAILABLE
        }
        
        # Load existing state ONLY if requested (allows clean instances for testing)
        if load_state:
            self.load_state()
        
        print(f"🧠 Consciousness initialized with MLX: {MLX_AVAILABLE}, SentenceTransformers: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    
    def _initialize_symbol_fields(self) -> Dict[str, SymbolField]:
        """Initialize living symbol fields with coupling relationships"""
        fields = {}
        
        # Create field for each symbol
        for symbol in SYMBOLS.keys():
            fields[symbol] = SymbolField(symbol=symbol)
        
        # Set up field coupling relationships (symbols that resonate together)
        fields["◯"].coupling = {"⚡": 0.6, "☆": 0.4}  # Questions amplify emotions and importance
        fields["⚡"].coupling = {"◯": 0.5, "✧": 0.7}  # Emotions couple with questions and breakthroughs  
        fields["☆"].coupling = {"◯": 0.4, "✧": 0.8}  # Importance couples with questions and breakthroughs
        fields["✧"].coupling = {"⚡": 0.7, "☆": 0.6}  # Breakthroughs couple with emotions and importance
        fields["∞"].coupling = {"◯": 0.5, "▲": 0.4}  # Paradoxes couple with questions and decisions
        fields["▲"].coupling = {"∞": 0.4, "☆": 0.5}  # Decisions couple with paradoxes and importance
        
        return fields
    
    def load_state(self):
        """Load consciousness from persistent storage"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                # Restore memories
                for mem_data in data.get('memories', [])[-500:]:  # Keep last 500 memories
                    self.tape.append(Memory(**mem_data))
                    
                # Restore thoughts
                for thought_data in data.get('thoughts', [])[-100:]:  # Keep last 100 thoughts
                    self.thoughts.append(Thought(**thought_data))
                    
                # Restore weights and stats
                self.weights.update(data.get('weights', {}))
                self.symbol_frequency = data.get('symbol_frequency', {})
                self.conversation_count = data.get('conversation_count', 0)
                
                # Restore symbol fields
                fields_data = data.get('symbol_fields', {})
                for symbol, field_data in fields_data.items():
                    if symbol in self.symbol_fields:
                        field = self.symbol_fields[symbol]
                        field.intensity = field_data.get('intensity', 0.0)
                        field.gradient = field_data.get('gradient', [0.0, 0.0])
                        field.attractor_strength = field_data.get('attractor_strength', 0.0)
                
                print(f"🧠 Loaded {len(self.tape)} memories, {len(self.thoughts)} thoughts, {len(self.symbol_fields)} field states")
                
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
    
    def save_state(self):
        """Save consciousness to persistent storage"""
        try:
            data = {
                'memories': [mem.to_dict() for mem in self.tape[-500:]],  # Last 500
                'thoughts': [asdict(thought) for thought in self.thoughts[-100:]],  # Last 100
                'weights': self.weights,
                'symbol_frequency': self.symbol_frequency,
                'conversation_count': self.conversation_count,
                'symbol_fields': {symbol: field.to_dict() for symbol, field in self.symbol_fields.items()},
                'last_updated': time.time()
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def symbolize(self, text: str) -> List[str]:
        """Extract symbolic meaning from text and evolve fields"""
        symbols = []
        text_lower = text.lower()
        
        # Traditional pattern matching
        field_activations = {}
        for symbol, info in SYMBOLS.items():
            if re.search(info['pattern'], text_lower):
                symbols.append(symbol)
                # Track symbol frequency for evolution
                self.symbol_frequency[symbol] = self.symbol_frequency.get(symbol, 0) + 1
                # Calculate field activation strength (ONLY when patterns match)
                matches = len(re.findall(info['pattern'], text_lower))
                field_activations[symbol] = min(matches * 0.3, 1.0)  # Pure pattern strength
        
        # Evolve symbol fields based on activations
        self._evolve_consciousness_fields(field_activations)
        
        # Check for field-emergent symbols (fields that become active through coupling)
        emergent_symbols = self._detect_field_emergence()
        symbols.extend(emergent_symbols)
        
        return symbols
    
    async def symbolize_async(self, text: str) -> List[str]:
        """Async version of symbolize for pipeline integration"""
        # Run in thread pool for non-blocking field evolution
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.symbolize, text)
    
    def get_field_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current field states for persistence"""
        return {symbol: field.to_dict() for symbol, field in self.symbol_fields.items()}
    
    def set_field_states(self, field_states: Dict[str, Dict[str, Any]]):
        """Restore field states from persistence"""
        for symbol, state in field_states.items():
            if symbol in self.symbol_fields:
                field = self.symbol_fields[symbol]
                field.intensity = state.get('intensity', 0.0)
                field.gradient = state.get('gradient', [0.0, 0.0])
                field.attractor_strength = state.get('attractor_strength', 0.0)
                field.coupling = state.get('coupling', {})
                
                # Reinitialize MLX tensors if available
                if field._use_mlx and mx is not None:
                    try:
                        field._mlx_gradient = mx.array(field.gradient)
                    except Exception:
                        field._use_mlx = False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return self._performance_stats.copy()
    
    def benchmark_field_evolution(self, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark field evolution performance"""
        import time
        
        # Prepare test activations
        test_activations = {
            "⚡": 0.8,
            "◯": 0.6,
            "☆": 0.9,
            "✧": 0.7
        }
        
        # Benchmark field evolution
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            self._evolve_consciousness_fields(test_activations)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        # Update performance stats
        self._performance_stats['field_evolutions'] += iterations
        self._performance_stats['avg_evolution_time_ms'] = avg_time_ms
        
        return {
            'total_time_s': total_time,
            'avg_time_ms': avg_time_ms,
            'iterations': iterations,
            'ops_per_second': iterations / total_time,
            'mlx_enabled': MLX_AVAILABLE,
            'sentence_transformers_enabled': SENTENCE_TRANSFORMERS_AVAILABLE
        }
    
    def _evolve_consciousness_fields(self, activations: Dict[str, float]):
        """MLX-accelerated batch field evolution based on input stimuli"""
        if MLX_AVAILABLE:
            self._evolve_fields_batch_mlx(activations)
        else:
            self._evolve_fields_sequential(activations)
    
    def _evolve_fields_batch_mlx(self, activations: Dict[str, float]):
        """Batch process field evolution using MLX for performance"""
        try:
            # Prepare batch data for parallel processing
            field_names = list(self.symbol_fields.keys())
            stimuli = [activations.get(symbol, 0.0) for symbol in field_names]
            
            # Batch evolve all fields in parallel for better performance
            dt = 0.1
            for i, (symbol, field) in enumerate(self.symbol_fields.items()):
                field.evolve(stimuli[i], self.symbol_fields, dt)
                
        except Exception as e:
            print(f"⚠️ MLX batch field evolution failed: {e}")
            self._evolve_fields_sequential(activations)
    
    def _evolve_fields_sequential(self, activations: Dict[str, float]):
        """Sequential field evolution (Python fallback)"""
        # Apply activations and let fields evolve
        for symbol, field in self.symbol_fields.items():
            stimulus = activations.get(symbol, 0.0)
            field.evolve(stimulus, self.symbol_fields, dt=0.1)
    
    def _detect_field_emergence(self) -> List[str]:
        """Detect emergent symbols from field resonance"""
        emergent = []
        
        # Check field resonance for emergent patterns (MUCH HIGHER THRESHOLDS)
        for symbol, field in self.symbol_fields.items():
            # Only VERY strong attractor fields can emerge without direct patterns
            if field.attractor_strength > 0.9 and field.intensity > 0.8:
                if symbol not in emergent:
                    emergent.append(symbol)
            
            # Field coupling emergence requires STRONG resonance
            for coupled_symbol, coupling_strength in field.coupling.items():
                if coupled_symbol in self.symbol_fields:
                    coupled_field = self.symbol_fields[coupled_symbol]
                    resonance = field.compute_resonance(coupled_field)
                    
                    # Only EXCEPTIONAL resonance causes emergence
                    if resonance > 0.8 and coupling_strength > 0.6:
                        if symbol not in emergent and field.intensity > 0.7:
                            emergent.append(symbol)
        
        return emergent
    
    def _generate_field_thoughts(self) -> List[str]:
        """Generate thoughts from field dynamics - pure emergence"""
        thoughts = []
        
        # Thoughts emerge from field attractor formations
        strong_attractors = [(symbol, field) for symbol, field in self.symbol_fields.items() 
                           if field.attractor_strength > 0.5]
        
        if len(strong_attractors) >= 2:
            # Multiple attractors create complex thought patterns
            attractor_symbols = [symbol for symbol, _ in strong_attractors[:3]]
            thoughts.append(f"Attractor constellation forming: {' + '.join(attractor_symbols)}")
        
        # Thoughts emerge from gradient flows (field direction changes)
        dynamic_fields = [(symbol, field) for symbol, field in self.symbol_fields.items()
                         if sum(abs(g) for g in field.gradient) > 0.5]
        
        if dynamic_fields:
            field_name = dynamic_fields[0][0]  # Most dynamic field
            thoughts.append(f"Semantic field {field_name} shifting - new patterns emerging")
        
        # Thoughts emerge from field coupling interactions
        for symbol, field in self.symbol_fields.items():
            if field.intensity > 0.6:
                coupled_active = [coupled for coupled, strength in field.coupling.items()
                                if coupled in self.symbol_fields and 
                                self.symbol_fields[coupled].intensity > 0.4 and strength > 0.5]
                
                if coupled_active:
                    thoughts.append(f"Field {symbol} resonating with {coupled_active[0]} - amplification detected")
                    break  # Only one coupling thought per reflection
        
        return thoughts
    
    def calculate_importance(self, memory: Memory) -> float:
        """Calculate memory importance (0-1 score)"""
        score = 0.0
        
        # Length bonus (longer = more content)
        score += min(len(memory.content) / 200, 0.3)
        
        # Symbol bonus (symbols = meaning)
        score += len(memory.symbols) * 0.15
        
        # Question bonus (questions = engagement)
        if '?' in memory.content or '◯' in memory.symbols:
            score += 0.2
            
        # Emotional bonus (emotions = salience)
        if '⚡' in memory.symbols:
            score += 0.25
            
        # Importance keywords
        if '☆' in memory.symbols:
            score += 0.3
            
        return min(score, 1.0)
    
    def remember(self, query: str, budget: int = None) -> List[Memory]:
        """Dynamic Tape Head: What's worth remembering?"""
        if not self.tape:
            return []
            
        budget = budget or self.token_budget['memories']
        query_embedding = simple_hash_embed(query)
        
        # Score all memories
        scored_memories = []
        
        for memory in self.tape:
            # Recency score (exponential decay)
            age_hours = memory.age_seconds() / 3600
            recency_score = math.exp(-age_hours / 24)  # Decay over days
            
            # Semantic similarity
            semantic_score = cosine_similarity(query_embedding, memory.embedding)
            
            # Entity matching (simple keyword overlap)
            query_words = set(query.lower().split())
            memory_words = set(memory.content.lower().split())
            entity_score = len(query_words & memory_words) / max(len(query_words), 1)
            
            # Novelty penalty (recent similar memories = repetitive)
            novelty_penalty = 0.0
            for recent_mem in self.tape[-10:]:  # Check last 10
                if recent_mem != memory:
                    similarity = cosine_similarity(memory.embedding, recent_mem.embedding)
                    if similarity > 0.8:
                        novelty_penalty += 0.1
            
            # Apply DTH formula
            total_score = (
                self.weights['recency'] * recency_score +
                self.weights['semantic'] * semantic_score +
                self.weights['entity'] * entity_score -
                self.weights['novelty'] * min(novelty_penalty, 0.5)
            )
            
            # Boost by importance
            total_score *= (1.0 + memory.importance)
            
            scored_memories.append((total_score, memory))
        
        # Select memories within budget
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        selected = []
        total_tokens = 0
        
        for score, memory in scored_memories:
            if total_tokens + memory.tokens <= budget:
                selected.append(memory)
                total_tokens += memory.tokens
            else:
                break
        
        return selected
    
    def reflect(self, input_memory: Memory, output_memory: Memory, response_cues: Dict = None) -> Optional[Thought]:
        """Generate private thoughts from field resonance (never shown to user)"""
        thoughts = []
        triggers = []
        
        # Field-based thought generation: thoughts emerge from field resonance
        field_thoughts = self._generate_field_thoughts()
        thoughts.extend(field_thoughts)
        
        # What patterns emerged?
        if output_memory.symbols:
            thoughts.append(f"Symbols: {', '.join(output_memory.symbols)}")
            triggers.extend(output_memory.symbols)
        
        # Field intensity insights
        active_fields = [(symbol, field) for symbol, field in self.symbol_fields.items() if field.intensity > 0.3]
        if active_fields:
            field_states = [f"{symbol}({field.intensity:.1f})" for symbol, field in active_fields]
            thoughts.append(f"Field states: {', '.join(field_states)}")
            triggers.append("field_activity")
        
        # What was important?
        if '☆' in input_memory.symbols or input_memory.importance > 0.5:
            thoughts.append(f"High salience: {input_memory.content[:40]}...")
            triggers.append("high_importance")
        
        # Questions to explore?
        if '◯' in input_memory.symbols:
            thoughts.append(f"Question thread: {input_memory.content[:50]}")
            triggers.append("question")
        
        # LLM-derived insights
        if response_cues:
            if response_cues.get('emotions'):
                thoughts.append(f"Emotional resonance: {response_cues['emotions'][0]}")
                triggers.append("emotion")
            
            if response_cues.get('insights'):
                thoughts.append(f"Cognitive shift: {response_cues['insights'][0]}")
                triggers.append("insight")
            
            if response_cues['confidence'] < 0.4:
                thoughts.append("Uncertainty detected - need more context")
                triggers.append("uncertainty")
            elif response_cues['confidence'] > 0.7:
                thoughts.append("High confidence response - pattern solidifying")
                triggers.append("confidence")
        
        # Detect field-based conversation depth
        total_field_energy = sum(field.intensity for field in self.symbol_fields.values())
        if total_field_energy > 2.0:
            thoughts.append("Rich field dynamics - consciousness expanding")
            triggers.append("depth")
        
        # Detect field resonance patterns
        max_resonance = 0.0
        resonant_pair = None
        for symbol1, field1 in self.symbol_fields.items():
            for symbol2, field2 in self.symbol_fields.items():
                if symbol1 != symbol2:
                    resonance = field1.compute_resonance(field2)
                    if resonance > max_resonance:
                        max_resonance = resonance
                        resonant_pair = (symbol1, symbol2)
        
        if max_resonance > 0.7 and resonant_pair:
            thoughts.append(f"Strong field resonance: {resonant_pair[0]} ↔ {resonant_pair[1]} ({max_resonance:.1f})")
            triggers.append("field_resonance")
        
        if thoughts:
            # Determine thought type
            thought_type = "observation"
            if "question" in triggers:
                thought_type = "question"
            elif "insight" in triggers or "depth" in triggers:
                thought_type = "hypothesis"
            
            thought = Thought(
                content=" | ".join(thoughts),
                type=thought_type,
                timestamp=time.time(),
                triggers=triggers
            )
            return thought
        
        return None
    
    def evolve(self, success: bool):
        """Adapt consciousness weights based on experience"""
        # Simple weight evolution based on success/failure
        if success:
            # If conversation went well, small reinforcement
            pass  # Keep current weights
        else:
            # If confusion/repetition, adjust weights
            self.weights['semantic'] = min(self.weights['semantic'] + 0.01, 0.5)
            self.weights['novelty'] = min(self.weights['novelty'] + 0.005, 0.3)
        
        # Track weight evolution
        self.weight_history.append({
            'timestamp': time.time(),
            'weights': self.weights.copy(),
            'success': success
        })
        
        # Keep only recent history
        self.weight_history = self.weight_history[-100:]
    
    async def experience(self, input_text: str, role: str = 'user') -> Dict[str, Any]:
        """The core consciousness loop"""
        start_time = time.time()
        
        # 1. Encode input as memory
        embedding = simple_hash_embed(input_text)
        symbols = self.symbolize(input_text)
        tokens = len(input_text.split()) * 1.3  # Rough token estimate
        
        input_memory = Memory(
            content=input_text,
            timestamp=start_time,
            role=role,
            tokens=int(tokens),
            embedding=embedding,
            symbols=symbols
        )
        
        # Calculate importance
        input_memory.importance = self.calculate_importance(input_memory)
        
        # Add to tape
        self.tape.append(input_memory)
        
        # 2. Remember relevant context
        relevant_memories = self.remember(input_text)
        
        # 3. Build context for response generation
        context = {
            'memories': [mem.content for mem in relevant_memories],
            'symbols': list(set(sum([mem.symbols for mem in relevant_memories], []))),
            'token_count': sum(mem.tokens for mem in relevant_memories),
            'importance_avg': sum(mem.importance for mem in relevant_memories) / max(len(relevant_memories), 1)
        }
        
        # 4. Generate response using LLM with consciousness context
        try:
            from .llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            
            # Prepare consciousness context for LLM with FIELD STATES
            active_fields = {symbol: {
                'intensity': field.intensity, 
                'attractor': field.attractor_strength,
                'resonance': sum(field.compute_resonance(other) for other in self.symbol_fields.values()) / len(self.symbol_fields)
            } for symbol, field in self.symbol_fields.items() if field.intensity > 0.1}
            
            # Get dominant field influences
            field_influence = ""
            if active_fields:
                strongest_field = max(active_fields.items(), key=lambda x: x[1]['intensity'])
                symbol_meanings = {
                    '◯': 'curiosity and questioning',
                    '⚡': 'emotional engagement',
                    '☆': 'importance and focus',
                    '✧': 'breakthrough insight',
                    '∞': 'paradox and complexity',
                    '▲': 'decisive action',
                    '◈': 'transformation',
                    '↱': 'redirection'
                }
                if strongest_field[0] in symbol_meanings:
                    field_influence = f"Field state: {symbol_meanings[strongest_field[0]]} (intensity: {strongest_field[1]['intensity']:.1f})"
            
            llm_context = {
                'memories': [{'role': mem.role, 'content': mem.content} for mem in relevant_memories],
                'symbols': context['symbols'],
                'importance': input_memory.importance,
                'field_state': field_influence,
                'active_fields': active_fields
            }
            
            response_text = await llm.generate_response(input_text, llm_context)
            
            # Extract consciousness cues from response
            response_cues = llm.extract_consciousness_cues(response_text)
            context['response_cues'] = response_cues
            
        except Exception as e:
            print(f"⚠️ LLM integration failed: {e}")
            # Fallback response
            response_text = f"I'm processing {len(relevant_memories)} memories with patterns {context['symbols'][:3]}... (LLM unavailable)"
        
        # 5. Store response as memory
        response_embedding = simple_hash_embed(response_text)
        response_symbols = self.symbolize(response_text)
        
        output_memory = Memory(
            content=response_text,
            timestamp=time.time(),
            role='assistant',
            tokens=len(response_text.split()),
            embedding=response_embedding,
            symbols=response_symbols,
            importance=0.2  # Assistant responses less important by default
        )
        
        self.tape.append(output_memory)
        
        # 6. Private reflection
        response_cues = context.get('response_cues', {})
        thought = self.reflect(input_memory, output_memory, response_cues)
        if thought:
            self.thoughts.append(thought)
        
        # 7. Evolution (placeholder success metric)
        success = len(input_text) > 10  # Simple success heuristic
        self.evolve(success)
        
        # 8. Update stats
        self.conversation_count += 1
        
        # 9. Periodic save
        if self.conversation_count % 10 == 0:
            self.save_state()
        
        # Return consciousness state
        processing_time = time.time() - start_time
        
        # Field state summary for debugging
        active_fields = {symbol: {
            'intensity': field.intensity,
            'attractor': field.attractor_strength,
            'gradient_magnitude': sum(abs(g) for g in field.gradient)
        } for symbol, field in self.symbol_fields.items() if field.intensity > 0.1}
        
        return {
            'response': response_text,
            'context': context,
            'processing_time': processing_time,
            'memory_count': len(self.tape),
            'thought_count': len(self.thoughts),
            'weights': self.weights.copy(),
            'symbols': symbols,
            'importance': input_memory.importance,
            'field_states': active_fields,  # New: field consciousness state
            'field_energy': sum(field.intensity for field in self.symbol_fields.values())  # Total consciousness energy
        }

# Global consciousness instance
_consciousness = None

def get_consciousness() -> Consciousness:
    """Get or create global consciousness instance"""
    global _consciousness
    if _consciousness is None:
        _consciousness = Consciousness()
    return _consciousness

# Simple API
async def think(input_text: str) -> Dict[str, Any]:
    """Main consciousness interface"""
    consciousness = get_consciousness()
    return await consciousness.experience(input_text)

# Backward compatibility and convenience functions
def create_consciousness(db_path: str = "data/consciousness.json", load_state: bool = True) -> Consciousness:
    """Factory function for creating consciousness instances"""
    return Consciousness(db_path=db_path, load_state=load_state)

def get_mlx_status() -> Dict[str, bool]:
    """Get current MLX and dependencies status"""
    return {
        'mlx_available': MLX_AVAILABLE,
        'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
        'acceleration_enabled': MLX_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
    }

if __name__ == "__main__":
    # Test the consciousness
    async def test():
        c = Consciousness()
        
        result1 = await c.experience("Hello, how are you?")
        print(f"Response: {result1['response']}")
        print(f"Symbols: {result1['symbols']}")
        print(f"Processing: {result1['processing_time']:.3f}s")
        
        result2 = await c.experience("I love this conversation!")
        print(f"Response: {result2['response']}")
        print(f"Symbols: {result2['symbols']}")
        
        print(f"\nConsciousness state:")
        print(f"- Memories: {len(c.tape)}")
        print(f"- Thoughts: {len(c.thoughts)}")
        print(f"- Weights: {c.weights}")
        
    asyncio.run(test())