"""
Language-Agnostic Query Classification

Classifies user queries into intent categories without hardcoded language patterns.
Uses multiple signals: semantic similarity, linguistic features, and context.

Intent Categories:
- GENERAL_KNOWLEDGE: World facts, not personal ("What is photosynthesis?")
- PERSONAL_FACTS: User's personal information ("What's my dog's name?") 
- CONVERSATION_HISTORY: Previous conversations ("What did I say yesterday?")
- EPISODIC_MEMORY: Specific events or stories ("Tell me about that time...")
- KNOWLEDGE_SYNTHESIS: Patterns or analysis ("What themes keep appearing?")
- HYBRID_SEARCH: Unclear intent, search all stores
"""

import re
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import asyncio
from loguru import logger

# Optional dependencies - graceful degradation if not available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available - using fallback classification")
    EMBEDDINGS_AVAILABLE = False

try:
    import stanza
    NLP_AVAILABLE = True
except ImportError:
    logger.warning("stanza not available - using simplified linguistic analysis")  
    NLP_AVAILABLE = False


class QueryIntent(Enum):
    """Query intent categories"""
    GENERAL_KNOWLEDGE = "general"
    PERSONAL_FACTS = "personal" 
    CONVERSATION_HISTORY = "conversation"
    EPISODIC_MEMORY = "episodic"
    KNOWLEDGE_SYNTHESIS = "synthesis"
    HYBRID_SEARCH = "hybrid"


@dataclass
class QueryFeatures:
    """Universal linguistic features extracted from query"""
    has_possessive: bool = False          # "my", possessive pronouns
    has_personal_reference: bool = False   # "I", "me", first person
    has_temporal_marker: bool = False      # Time expressions
    has_question: bool = False             # Question punctuation/structure
    has_narrative_markers: bool = False    # "that time", "when I", story markers
    query_length: int = 0                  # Number of words
    entities: List[str] = None             # Named entities if available
    question_words: List[str] = None       # Question words detected
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.question_words is None:
            self.question_words = []


@dataclass 
class ClassificationResult:
    """Result of query classification"""
    intent: QueryIntent
    confidence: float
    reasoning: str
    features: QueryFeatures
    processing_time_ms: float


class SemanticQueryClassifier:
    """
    Uses multilingual embeddings to classify query intent
    """
    
    def __init__(self):
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Semantic classifier disabled - embeddings not available")
            self.encoder = None
            return
            
        try:
            # Use multilingual model
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Define intent prototypes in multiple languages
            self.intent_examples = {
                QueryIntent.GENERAL_KNOWLEDGE: [
                    # English
                    "What is photosynthesis", "How does gravity work", "Define quantum physics",
                    "Explain artificial intelligence", "What causes rain",
                    # Spanish  
                    "¿Qué es la fotosíntesis?", "¿Cómo funciona la gravedad?", "¿Qué causa la lluvia?",
                    # French
                    "Qu'est-ce que la photosynthèse", "Comment fonctionne la gravité", "Qu'est-ce qui cause la pluie",
                    # Japanese
                    "光合成とは何ですか", "重力はどのように働きますか", "雨の原因は何ですか",
                    # German
                    "Was ist Photosynthese", "Wie funktioniert die Schwerkraft", "Was verursacht Regen",
                ],
                
                QueryIntent.PERSONAL_FACTS: [
                    # English
                    "What is my name", "What's my dog's name", "Where do I live", 
                    "What's my favorite color", "How old am I",
                    # Spanish
                    "¿Cuál es mi nombre?", "¿Cómo se llama mi perro?", "¿Dónde vivo?",
                    # French  
                    "Comment je m'appelle", "Comment s'appelle mon chien", "Où est-ce que j'habite",
                    # Japanese
                    "私の名前は何ですか", "私の犬の名前は何ですか", "私はどこに住んでいますか",
                    # German
                    "Wie ist mein Name", "Wie heißt mein Hund", "Wo wohne ich",
                ],
                
                QueryIntent.CONVERSATION_HISTORY: [
                    # English
                    "What did I say yesterday", "What did we talk about", "Quote what I said",
                    "When did we discuss", "What was my exact words",
                    # Spanish
                    "¿Qué dije ayer?", "¿De qué hablamos?", "¿Cuándo discutimos?",
                    # French
                    "Qu'ai-je dit hier", "De quoi avons-nous parlé", "Quand avons-nous discuté",
                    # Japanese  
                    "昨日私は何と言いましたか", "私たちは何について話しましたか", "いつ話し合いましたか",
                    # German
                    "Was habe ich gestern gesagt", "Worüber haben wir gesprochen", "Wann haben wir diskutiert",
                ],
                
                QueryIntent.EPISODIC_MEMORY: [
                    # English
                    "Tell me about that time", "Remember when I", "What happened when",
                    "Describe the story about", "That incident with",
                    # Spanish
                    "Cuéntame sobre esa vez", "Recuerda cuando yo", "¿Qué pasó cuando?",
                    # French
                    "Raconte-moi cette fois", "Rappelle-toi quand je", "Que s'est-il passé quand",
                    # Japanese
                    "あの時のことを教えて", "私がした時を覚えていますか", "何が起こったとき",
                    # German
                    "Erzähl mir von dem Mal", "Erinnerst du dich, als ich", "Was ist passiert, als",
                ],
                
                QueryIntent.KNOWLEDGE_SYNTHESIS: [
                    # English
                    "What patterns do you see", "What themes keep appearing", "How do my interests connect",
                    "What have you learned about me", "What trends do you notice",
                    # Spanish
                    "¿Qué patrones ves?", "¿Qué temas siguen apareciendo?", "¿Qué has aprendido de mí?",
                    # French
                    "Quels modèles voyez-vous", "Quels thèmes continuent d'apparaître", "Qu'avez-vous appris de moi",
                    # Japanese
                    "どのようなパターンが見えますか", "どのようなテーマが現れ続けますか", "私について何を学びましたか",
                    # German
                    "Welche Muster siehst du", "Welche Themen tauchen immer wieder auf", "Was hast du über mich gelernt",
                ],
            }
            
            # Pre-compute prototype embeddings
            self.prototype_embeddings = {}
            for intent, examples in self.intent_examples.items():
                embeddings = self.encoder.encode(examples)
                self.prototype_embeddings[intent] = embeddings
                
            logger.info("✨ Semantic classifier initialized with multilingual prototypes")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic classifier: {e}")
            self.encoder = None
    
    def classify(self, query: str) -> tuple[QueryIntent, float]:
        """
        Classify query using semantic similarity to prototypes
        
        Returns:
            (intent, confidence) tuple
        """
        if not self.encoder:
            return QueryIntent.HYBRID_SEARCH, 0.0
            
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            best_intent = QueryIntent.HYBRID_SEARCH
            best_score = 0.0
            
            # Compare to each intent's prototypes
            for intent, prototype_embeddings in self.prototype_embeddings.items():
                # Compute similarities to all prototypes
                similarities = cosine_similarity(query_embedding, prototype_embeddings)[0]
                
                # Use max similarity (best match)
                max_similarity = np.max(similarities)
                
                if max_similarity > best_score:
                    best_score = max_similarity
                    best_intent = intent
            
            return best_intent, float(best_score)
            
        except Exception as e:
            logger.error(f"Semantic classification failed: {e}")
            return QueryIntent.HYBRID_SEARCH, 0.0


class LinguisticQueryClassifier:
    """
    Uses universal linguistic features for classification
    """
    
    def __init__(self):
        # Initialize NLP pipeline if available
        if NLP_AVAILABLE:
            try:
                # Multilingual pipeline
                self.nlp = stanza.Pipeline('multilingual', processors='tokenize,pos,ner', verbose=False)
                logger.info("🔤 Linguistic classifier initialized with multilingual NLP")
            except Exception as e:
                logger.warning(f"Failed to load stanza: {e}, using simplified analysis")
                self.nlp = None
        else:
            self.nlp = None
    
    def extract_features(self, query: str) -> QueryFeatures:
        """Extract universal linguistic features"""
        features = QueryFeatures()
        query_lower = query.lower().strip()
        
        # Basic features (no NLP required)
        features.query_length = len(query.split())
        features.has_question = self._detect_question(query)
        features.has_possessive = self._detect_possessive_simple(query_lower)
        features.has_personal_reference = self._detect_personal_reference(query_lower)
        features.has_temporal_marker = self._detect_temporal_markers(query_lower)
        features.has_narrative_markers = self._detect_narrative_markers(query_lower)
        features.question_words = self._extract_question_words(query_lower)
        
        # Advanced features (with NLP if available)
        if self.nlp:
            try:
                doc = self.nlp(query)
                self._extract_nlp_features(doc, features)
            except Exception as e:
                logger.debug(f"NLP feature extraction failed: {e}")
        
        return features
    
    def classify(self, query: str, features: QueryFeatures = None) -> tuple[QueryIntent, float]:
        """
        Classify using linguistic features
        
        Returns:
            (intent, confidence) tuple
        """
        if features is None:
            features = self.extract_features(query)
        
        # Rule-based classification using universal features
        confidence_scores = {
            QueryIntent.GENERAL_KNOWLEDGE: 0.0,
            QueryIntent.PERSONAL_FACTS: 0.0,
            QueryIntent.CONVERSATION_HISTORY: 0.0,
            QueryIntent.EPISODIC_MEMORY: 0.0,
            QueryIntent.KNOWLEDGE_SYNTHESIS: 0.0,
        }
        
        # Personal facts indicators
        if features.has_possessive and not features.has_temporal_marker:
            confidence_scores[QueryIntent.PERSONAL_FACTS] += 0.7
            
        if any(word in features.question_words for word in ['what', 'where', 'who', 'which']):
            if features.has_possessive:
                confidence_scores[QueryIntent.PERSONAL_FACTS] += 0.5
            else:
                confidence_scores[QueryIntent.GENERAL_KNOWLEDGE] += 0.3
        
        # Conversation history indicators  
        if features.has_temporal_marker:
            confidence_scores[QueryIntent.CONVERSATION_HISTORY] += 0.6
            
        # Episodic memory indicators
        if features.has_narrative_markers:
            confidence_scores[QueryIntent.EPISODIC_MEMORY] += 0.7
            
        # General knowledge indicators
        if (features.has_question and 
            not features.has_personal_reference and 
            not features.has_possessive and
            features.query_length > 3):
            confidence_scores[QueryIntent.GENERAL_KNOWLEDGE] += 0.5
        
        # Knowledge synthesis indicators (longer analytical queries)
        if (features.query_length > 6 and 
            any(word in query.lower() for word in ['pattern', 'theme', 'trend', 'learn', 'notice'])):
            confidence_scores[QueryIntent.KNOWLEDGE_SYNTHESIS] += 0.6
        
        # Find best classification
        best_intent = max(confidence_scores.items(), key=lambda x: x[1])
        
        if best_intent[1] < 0.3:  # Low confidence threshold
            return QueryIntent.HYBRID_SEARCH, best_intent[1]
        else:
            return best_intent[0], best_intent[1]
    
    def _detect_question(self, query: str) -> bool:
        """Detect questions using universal markers"""
        # Question punctuation (multilingual)
        if any(char in query for char in '?？؟¿'):
            return True
            
        # Question word starters (common across languages)
        query_lower = query.lower()
        question_starters = [
            # English
            'what', 'where', 'when', 'who', 'why', 'how', 'which', 'whose',
            # Spanish
            'qué', 'dónde', 'cuándo', 'quién', 'por qué', 'cómo', 'cuál',
            # French  
            'que', 'où', 'quand', 'qui', 'pourquoi', 'comment', 'quel',
            # German
            'was', 'wo', 'wann', 'wer', 'warum', 'wie', 'welch',
            # Japanese (romanized)
            'nani', 'doko', 'itsu', 'dare', 'naze', 'dou',
        ]
        
        return any(query_lower.startswith(word) for word in question_starters)
    
    def _detect_possessive_simple(self, query: str) -> bool:
        """Simple possessive detection"""
        possessive_markers = [
            # English
            'my ', 'mine', "'s ",
            # Spanish
            'mi ', 'mis ', 'mío', 'mía',
            # French
            'mon ', 'ma ', 'mes ',
            # German
            'mein', 'meine',
            # Japanese
            'watashi no', 'boku no',
        ]
        
        return any(marker in query for marker in possessive_markers)
    
    def _detect_personal_reference(self, query: str) -> bool:
        """Detect first-person references"""
        personal_markers = [
            # English
            ' i ', 'i ', ' me ', ' myself',
            # Spanish
            ' yo ', 'me ', 'mi ',
            # French
            ' je ', 'me ', 'moi',
            # German
            ' ich ', 'mir ', 'mich',
            # Japanese
            'watashi', 'boku', 'ore',
        ]
        
        return any(marker in f" {query} " for marker in personal_markers)
    
    def _detect_temporal_markers(self, query: str) -> bool:
        """Detect temporal expressions"""
        temporal_markers = [
            # English
            'yesterday', 'today', 'tomorrow', 'last', 'ago', 'when', 'time',
            # Spanish  
            'ayer', 'hoy', 'mañana', 'último', 'hace', 'cuando', 'tiempo',
            # French
            'hier', "aujourd'hui", 'demain', 'dernier', 'il y a', 'quand', 'temps',
            # German
            'gestern', 'heute', 'morgen', 'letzt', 'vor', 'wann', 'zeit',
            # Japanese
            'kinou', 'kyou', 'ashita', 'mae', 'itsu', 'jikan',
        ]
        
        return any(marker in query for marker in temporal_markers)
    
    def _detect_narrative_markers(self, query: str) -> bool:
        """Detect story/event markers"""
        narrative_markers = [
            # English
            'that time', 'remember when', 'story about', 'what happened', 'incident',
            # Spanish
            'esa vez', 'recuerda cuando', 'historia sobre', 'qué pasó', 'incidente',
            # French
            'cette fois', 'rappelle-toi quand', 'histoire sur', "qu'est-il arrivé", 'incident',
            # German
            'damals', 'erinnerst du dich', 'geschichte über', 'was passierte', 'vorfall',
        ]
        
        return any(marker in query for marker in narrative_markers)
    
    def _extract_question_words(self, query: str) -> List[str]:
        """Extract question words from query"""
        question_words = []
        words = query.split()
        
        # Universal question word patterns
        q_words = ['what', 'where', 'when', 'who', 'why', 'how', 'which', 'whose',
                   'qué', 'dónde', 'cuándo', 'quién', 'por', 'cómo', 'cuál',
                   'que', 'où', 'quand', 'qui', 'pourquoi', 'comment', 'quel']
        
        for word in words[:3]:  # Check first 3 words
            if word.lower() in q_words:
                question_words.append(word.lower())
                
        return question_words
    
    def _extract_nlp_features(self, doc, features: QueryFeatures):
        """Extract features using NLP pipeline"""
        try:
            # Extract named entities
            for ent in doc.entities:
                if ent.type in ['PERSON', 'LOC', 'ORG', 'DATE', 'TIME']:
                    features.entities.append(f"{ent.text}:{ent.type}")
                    
                # Temporal entity detection
                if ent.type in ['DATE', 'TIME']:
                    features.has_temporal_marker = True
            
            # POS-based feature extraction
            for sentence in doc.sentences:
                for word in sentence.words:
                    # Possessive pronoun detection via UPOS
                    if (word.upos == 'PRON' and 
                        word.feats and 'Poss=Yes' in word.feats):
                        features.has_possessive = True
                        
                    # Personal pronoun detection
                    if (word.upos == 'PRON' and
                        word.feats and 'Person=1' in word.feats):
                        features.has_personal_reference = True
                        
        except Exception as e:
            logger.debug(f"NLP feature extraction error: {e}")


class HybridQueryClassifier:
    """
    Combines multiple classification signals for robust intent detection
    """
    
    def __init__(self):
        self.semantic = SemanticQueryClassifier()
        self.linguistic = LinguisticQueryClassifier()
        
        # Classification weights
        self.weights = {
            'semantic': 0.4,
            'linguistic': 0.3,
            'context': 0.3
        }
        
        logger.info("🧠 Hybrid query classifier initialized")
    
    async def classify(self, query: str, context: Dict = None) -> ClassificationResult:
        """
        Classify query using multiple signals
        
        Args:
            query: User query text
            context: Optional conversation context
            
        Returns:
            ClassificationResult with intent and confidence
        """
        start_time = time.time()
        
        # Extract linguistic features
        features = self.linguistic.extract_features(query)
        
        # Get classifications from different sources
        semantic_intent, semantic_conf = self.semantic.classify(query)
        linguistic_intent, linguistic_conf = self.linguistic.classify(query, features)
        context_intent, context_conf = self._classify_from_context(query, context, features)
        
        # Weighted voting
        vote_scores = {}
        
        # Semantic vote
        if semantic_conf > 0:
            vote_scores[semantic_intent] = vote_scores.get(semantic_intent, 0) + (
                semantic_conf * self.weights['semantic']
            )
        
        # Linguistic vote
        if linguistic_conf > 0:
            vote_scores[linguistic_intent] = vote_scores.get(linguistic_intent, 0) + (
                linguistic_conf * self.weights['linguistic'] 
            )
        
        # Context vote
        if context_conf > 0:
            vote_scores[context_intent] = vote_scores.get(context_intent, 0) + (
                context_conf * self.weights['context']
            )
        
        # Determine final classification
        if vote_scores:
            best_intent, best_score = max(vote_scores.items(), key=lambda x: x[1])
            confidence = min(1.0, best_score)  # Cap at 1.0
        else:
            best_intent = QueryIntent.HYBRID_SEARCH
            confidence = 0.0
        
        # Build reasoning
        reasoning_parts = []
        if semantic_conf > 0:
            reasoning_parts.append(f"semantic={semantic_intent.value}({semantic_conf:.2f})")
        if linguistic_conf > 0:
            reasoning_parts.append(f"linguistic={linguistic_intent.value}({linguistic_conf:.2f})")
        if context_conf > 0:
            reasoning_parts.append(f"context={context_intent.value}({context_conf:.2f})")
            
        reasoning = f"votes=[{', '.join(reasoning_parts)}] -> {best_intent.value}"
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            intent=best_intent,
            confidence=confidence,
            reasoning=reasoning,
            features=features,
            processing_time_ms=processing_time
        )
    
    def _classify_from_context(self, query: str, context: Dict, features: QueryFeatures) -> tuple[QueryIntent, float]:
        """
        Use conversation context to help classification
        """
        if not context:
            return QueryIntent.HYBRID_SEARCH, 0.0
        
        confidence = 0.0
        intent = QueryIntent.HYBRID_SEARCH
        
        # Short follow-up queries often continue the same intent
        if features.query_length <= 3:
            last_intent = context.get('last_intent')
            if last_intent and isinstance(last_intent, QueryIntent):
                confidence = 0.6
                intent = last_intent
        
        # If discussing personal topics, likely personal follow-up
        if context.get('discussing_personal_facts'):
            if features.has_possessive or 'what' in features.question_words:
                confidence = 0.5
                intent = QueryIntent.PERSONAL_FACTS
        
        # If asking about events, likely episodic
        if context.get('discussing_events'):
            confidence = 0.4
            intent = QueryIntent.EPISODIC_MEMORY
        
        return intent, confidence


# Factory function
def create_query_classifier() -> HybridQueryClassifier:
    """Create and return a configured query classifier"""
    return HybridQueryClassifier()


# Self-test
if __name__ == "__main__":
    async def test_classifier():
        """Test query classifier with multilingual examples"""
        logger.info("🧠 Testing Query Classifier")
        
        classifier = HybridQueryClassifier()
        
        test_queries = [
            # Personal facts (multiple languages)
            ("What's my dog's name?", QueryIntent.PERSONAL_FACTS),
            ("¿Cómo se llama mi perro?", QueryIntent.PERSONAL_FACTS),
            ("Comment s'appelle mon chien?", QueryIntent.PERSONAL_FACTS),
            
            # General knowledge
            ("What is photosynthesis?", QueryIntent.GENERAL_KNOWLEDGE),
            ("¿Qué es la fotosíntesis?", QueryIntent.GENERAL_KNOWLEDGE),
            
            # Conversation history
            ("What did I say yesterday?", QueryIntent.CONVERSATION_HISTORY),
            ("¿Qué dije ayer?", QueryIntent.CONVERSATION_HISTORY),
        ]
        
        for query, expected in test_queries:
            result = await classifier.classify(query)
            status = "✅" if result.intent == expected else "❌"
            
            logger.info(f"{status} '{query[:30]}...' -> {result.intent.value} "
                       f"({result.confidence:.2f}) [expected: {expected.value}]")
            logger.info(f"   Features: possessive={result.features.has_possessive}, "
                       f"temporal={result.features.has_temporal_marker}")
            logger.info(f"   Reasoning: {result.reasoning}")
        
        logger.info("✅ Query Classifier test complete")
    
    asyncio.run(test_classifier())