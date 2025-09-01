"""
Adaptive Knowledge Graph - Self-Organizing Knowledge Structure

This system learns predicate patterns from actual conversations and evolves the knowledge graph organically.
It builds on the existing SurrealDB functions like detect_engrams and memory decay.

Key principles:
1. NO rigid predefined ontology - patterns emerge from usage
2. Predicate clustering using SpaCy embeddings + usage frequency  
3. Automatic predicate normalization based on semantic similarity
4. Self-maintaining graph structure that grows and contracts naturally
5. Uses existing engrams and memory decay systems
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from loguru import logger
import spacy
from sentence_transformers import SentenceTransformer

try:
    from memory.surreal_connection import get_surreal_connection
    SURREAL_AVAILABLE = True
except ImportError:
    logger.error("SurrealDB connection not available")
    SURREAL_AVAILABLE = False

@dataclass
class PredicateCluster:
    """A cluster of semantically similar predicates"""
    canonical_form: str
    variants: Set[str]
    usage_frequency: int
    semantic_center: np.ndarray  # embedding centroid
    confidence: float
    created_at: str
    last_used: str

class AdaptiveKnowledgeGraph:
    """Self-organizing knowledge graph that learns from conversation patterns"""
    
    def __init__(self):
        # Load SpaCy model for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("📚 Loaded SpaCy large model for adaptive KG")
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
            logger.warning("🔸 Using SpaCy small model (less accurate)")
        
        # Load sentence transformer for semantic embeddings
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("🧠 Loaded sentence transformer for predicate clustering")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.embedder = None
            
        self.predicate_clusters = {}  # canonical_form -> PredicateCluster
        self.predicate_to_cluster = {}  # variant -> canonical_form
        self.usage_stats = defaultdict(int)
        
        # Clustering thresholds - these evolve based on usage
        self.semantic_similarity_threshold = 0.75  # Start conservative
        self.min_usage_for_cluster = 3  # Require some evidence
        self.cluster_merge_threshold = 0.85  # High bar for merging
        
    async def analyze_current_predicates(self) -> Dict[str, int]:
        """Analyze current predicate usage patterns in the database"""
        if not SURREAL_AVAILABLE:
            return {}
            
        conn = get_surreal_connection()
        await conn.connect()
        
        try:
            # Get predicate frequency from knowledge table
            result = await conn.db.query(
                "SELECT predicate, count() as frequency FROM knowledge GROUP BY predicate ORDER BY frequency DESC"
            )
            
            predicate_stats = {}
            if result and len(result) > 0 and result[0].get('result'):
                for item in result[0]['result']:
                    predicate_stats[item['predicate']] = item['frequency']
                    
            logger.info(f"📊 Found {len(predicate_stats)} unique predicates in database")
            
            # Also log some sample knowledge for debugging
            try:
                sample_result = await conn.db.query("SELECT * FROM knowledge LIMIT 5")
                if sample_result and len(sample_result) > 0 and sample_result[0].get('result'):
                    logger.info(f"📝 Sample knowledge records found: {len(sample_result[0]['result'])}")
                    for i, record in enumerate(sample_result[0]['result'][:3]):
                        logger.debug(f"  Sample {i+1}: {record.get('predicate', 'N/A')}")
                else:
                    logger.info("📝 No knowledge records found in database")
            except Exception as sample_e:
                logger.debug(f"Sample query failed: {sample_e}")
                    
            return predicate_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze predicates: {e}")
            logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            return {}
        finally:
            await conn.disconnect()
    
    def _compute_predicate_embedding(self, predicate: str) -> np.ndarray:
        """Compute semantic embedding for a predicate"""
        if not self.embedder:
            # Fallback to SpaCy embeddings
            doc = self.nlp(predicate)
            return doc.vector
        
        # Use sentence transformer
        embedding = self.embedder.encode([predicate])
        return embedding[0]
    
    def _find_semantic_clusters(self, predicates: Dict[str, int]) -> List[PredicateCluster]:
        """Find semantic clusters in predicate usage"""
        if not predicates:
            return []
            
        # Compute embeddings for all predicates
        pred_list = list(predicates.keys())
        embeddings = {}
        
        for pred in pred_list:
            embeddings[pred] = self._compute_predicate_embedding(pred)
            
        # Simple clustering based on semantic similarity
        clusters = []
        processed = set()
        
        for pred in pred_list:
            if pred in processed:
                continue
                
            # Start new cluster with this predicate
            cluster_variants = {pred}
            cluster_embedding = embeddings[pred]
            cluster_usage = predicates[pred]
            
            # Find similar predicates
            for other_pred in pred_list:
                if other_pred in processed or other_pred == pred:
                    continue
                    
                # Compute cosine similarity
                similarity = np.dot(embeddings[pred], embeddings[other_pred]) / (
                    np.linalg.norm(embeddings[pred]) * np.linalg.norm(embeddings[other_pred])
                )
                
                if similarity > self.semantic_similarity_threshold:
                    cluster_variants.add(other_pred)
                    cluster_usage += predicates[other_pred]
                    
            # Only create cluster if it has enough usage or multiple variants
            if cluster_usage >= self.min_usage_for_cluster or len(cluster_variants) > 1:
                # Choose canonical form (most frequent or linguistically simplest)
                canonical = max(cluster_variants, key=lambda p: predicates[p])
                
                # Compute cluster centroid
                if len(cluster_variants) > 1:
                    cluster_embeddings = [embeddings[v] for v in cluster_variants]
                    centroid = np.mean(cluster_embeddings, axis=0)
                else:
                    centroid = embeddings[canonical]
                
                cluster = PredicateCluster(
                    canonical_form=canonical,
                    variants=cluster_variants,
                    usage_frequency=cluster_usage,
                    semantic_center=centroid,
                    confidence=min(1.0, cluster_usage / 20.0),  # Confidence grows with usage
                    created_at="now",
                    last_used="now"
                )
                
                clusters.append(cluster)
                processed.update(cluster_variants)
                
        logger.info(f"🎯 Found {len(clusters)} semantic predicate clusters")
        return clusters
    
    def normalize_predicate(self, predicate: str) -> str:
        """Normalize a predicate to its canonical form"""
        # First check direct mapping
        if predicate in self.predicate_to_cluster:
            return self.predicate_to_cluster[predicate]
        
        # If no direct mapping, find closest semantic cluster
        if not self.predicate_clusters or not self.embedder:
            return predicate  # Return as-is
        
        pred_embedding = self._compute_predicate_embedding(predicate)
        best_canonical = predicate
        best_similarity = 0.0
        
        for canonical, cluster in self.predicate_clusters.items():
            similarity = np.dot(pred_embedding, cluster.semantic_center) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(cluster.semantic_center)
            )
            
            if similarity > best_similarity and similarity > self.semantic_similarity_threshold:
                best_similarity = similarity
                best_canonical = canonical
        
        # Update mapping if we found a good match
        if best_canonical != predicate:
            self.predicate_to_cluster[predicate] = best_canonical
            logger.debug(f"🔄 Normalized '{predicate}' → '{best_canonical}' (sim={best_similarity:.3f})")
        
        return best_canonical
    
    async def learn_from_new_knowledge(self, predicate: str) -> str:
        """Learn from a new predicate and return normalized form"""
        # Update usage stats
        self.usage_stats[predicate] += 1
        
        # Normalize the predicate
        canonical = self.normalize_predicate(predicate)
        
        # If this creates a new high-usage predicate, re-cluster
        if self.usage_stats[predicate] % 10 == 0:  # Re-cluster every 10 uses
            await self.refresh_clusters()
        
        return canonical
    
    async def refresh_clusters(self):
        """Refresh predicate clusters based on current database state"""
        logger.info("🔄 Refreshing predicate clusters...")
        
        # Get current predicate usage from database
        predicate_stats = await self.analyze_current_predicates()
        
        # Update our local stats
        for pred, freq in predicate_stats.items():
            self.usage_stats[pred] = freq
        
        # Re-cluster predicates
        clusters = self._find_semantic_clusters(predicate_stats)
        
        # Update our cluster mappings
        self.predicate_clusters.clear()
        self.predicate_to_cluster.clear()
        
        for cluster in clusters:
            self.predicate_clusters[cluster.canonical_form] = cluster
            for variant in cluster.variants:
                self.predicate_to_cluster[variant] = cluster.canonical_form
        
        logger.info(f"✅ Updated {len(clusters)} predicate clusters")
    
    async def detect_emerging_patterns(self) -> List[Dict]:
        """Detect emerging knowledge patterns using existing engrams"""
        if not SURREAL_AVAILABLE:
            return []
            
        conn = get_surreal_connection()
        await conn.connect()
        
        try:
            # Use existing detect_engrams function to find patterns
            result = await conn.db.query(
                "SELECT * FROM engrams ORDER BY coherence_score DESC, activation_count DESC LIMIT 10"
            )
            
            patterns = []
            if result and result[0].get('result'):
                for engram in result[0]['result']:
                    patterns.append({
                        'type': 'engram',
                        'symbols': engram.get('dominant_symbols', []),
                        'coherence': engram.get('coherence_score', 0),
                        'activations': engram.get('activation_count', 0),
                        'summary': engram.get('narrative_summary', '')
                    })
            
            logger.info(f"🌟 Detected {len(patterns)} emerging knowledge patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect patterns: {e}")
            return []
        finally:
            await conn.disconnect()
    
    async def cleanup_weak_patterns(self):
        """Remove weak or unused patterns using existing memory decay"""
        if not SURREAL_AVAILABLE:
            return
            
        conn = get_surreal_connection()
        await conn.connect()
        
        try:
            # Use existing cleanup function
            result = await conn.db.query("SELECT fn::cleanup_fragments(50) as cleaned")
            if result and result[0].get('result'):
                cleaned_count = result[0]['result'][0]['cleaned']
                logger.info(f"🗑️ Cleaned up {cleaned_count} weak knowledge fragments")
            
            # Apply memory decay to all knowledge
            result = await conn.db.query("SELECT fn::decay_background_memories(100) as decayed")
            if result and result[0].get('result'):
                decayed_count = result[0]['result'][0]['decayed']
                logger.info(f"⏳ Applied decay to {decayed_count} knowledge memories")
                
        except Exception as e:
            logger.error(f"Failed to cleanup patterns: {e}")
        finally:
            await conn.disconnect()
    
    async def evolve_graph_structure(self):
        """Main evolution cycle - analyze, cluster, clean up"""
        logger.info("🧬 Starting knowledge graph evolution cycle...")
        
        # Step 1: Refresh predicate clusters
        await self.refresh_clusters()
        
        # Step 2: Detect emerging patterns
        patterns = await self.detect_emerging_patterns()
        
        # Step 3: Clean up weak patterns
        await self.cleanup_weak_patterns()
        
        # Step 4: Adapt thresholds based on graph density
        if len(self.predicate_clusters) > 50:
            # Too many clusters, be more aggressive
            self.semantic_similarity_threshold = max(0.65, self.semantic_similarity_threshold - 0.05)
        elif len(self.predicate_clusters) < 10:
            # Too few clusters, be more conservative
            self.semantic_similarity_threshold = min(0.85, self.semantic_similarity_threshold + 0.05)
        
        logger.info(f"✅ Evolution complete: {len(self.predicate_clusters)} clusters, threshold={self.semantic_similarity_threshold:.3f}")

# Global adaptive knowledge graph instance
adaptive_kg = None

def get_adaptive_kg() -> AdaptiveKnowledgeGraph:
    """Get the global adaptive knowledge graph instance"""
    global adaptive_kg
    if adaptive_kg is None:
        adaptive_kg = AdaptiveKnowledgeGraph()
    return adaptive_kg

async def normalize_predicate_adaptive(predicate: str) -> str:
    """Main function to normalize predicates using adaptive learning"""
    kg = get_adaptive_kg()
    return await kg.learn_from_new_knowledge(predicate)

# Background evolution task
async def run_evolution_cycle():
    """Background task to evolve the knowledge graph"""
    kg = get_adaptive_kg()
    
    while True:
        try:
            await kg.evolve_graph_structure()
            await asyncio.sleep(300)  # Evolve every 5 minutes
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            await asyncio.sleep(60)  # Retry in 1 minute

if __name__ == "__main__":
    # Test the adaptive knowledge graph
    async def test():
        kg = AdaptiveKnowledgeGraph()
        await kg.refresh_clusters()
        
        # Test normalization
        test_predicates = ["dog_name", "cat_name", "pet_name", "works_at", "employed_at", "job_at"]
        for pred in test_predicates:
            normalized = await kg.learn_from_new_knowledge(pred)
            print(f"{pred} → {normalized}")
            
    asyncio.run(test())