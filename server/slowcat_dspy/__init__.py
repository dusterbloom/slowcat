"""
DSPy Integration for Slowcat - Self-optimizing AI components

This module provides DSPy-powered optimization for Slowcat's memory and context systems.
Built on top of SurrealDB for rich training data and real-time optimization.

Key Features:
- SurrealDB-powered training data from conversation history
- Graph-based context optimization using fact relationships  
- Real-time performance optimization via live subscriptions
- Temporal pattern learning from time-travel queries

Components:
- surreal_optimizers: Core DSPy modules using SurrealDB
- metrics: SurrealDB-powered evaluation functions
- trainers: Training data generation from conversation history
- live: Real-time optimization loops

Environment Variables:
    DSPY_OPTIMIZATION_ENABLED: Enable DSPy optimization (default: false)
    DSPY_MODEL_PATH: Path to local optimization model 
    DSPY_TRAINING_DAYS: Days of conversation history for training (default: 30)
    DSPY_OPTIMIZATION_INTERVAL: Seconds between optimization runs (default: 3600)
"""

import os
from loguru import logger

# Check for DSPy v3 availability
try:
    import dspy
    from dspy import LM, Module, ChainOfThought, Predict, Signature, InputField, OutputField
    DSPY_AVAILABLE = True
    logger.info(f"🧠 DSPy v{dspy.__version__} available for optimization")
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None
    logger.info("📦 DSPy not available - install with: pip install -U dspy")

# Configuration
OPTIMIZATION_ENABLED = os.getenv('DSPY_OPTIMIZATION_ENABLED', 'false').lower() == 'true'
MODEL_PATH = os.getenv('DSPY_MODEL_PATH', 'local/optimization-model')
TRAINING_DAYS = int(os.getenv('DSPY_TRAINING_DAYS', '30'))
OPTIMIZATION_INTERVAL = int(os.getenv('DSPY_OPTIMIZATION_INTERVAL', '3600'))
VERBOSE_LOGGING = os.getenv('DSPY_VERBOSE_LOGGING', 'false').lower() == 'true'

def configure_dspy_with_lm_studio():
    """Configure DSPy v3 with LM Studio"""
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, cannot configure")
        return False
    
    try:
        # Get configuration from environment
        model_endpoint = os.getenv('DSPY_MODEL_ENDPOINT', 'http://localhost:1234/v1')
        model_name = os.getenv('DSPY_MODEL_NAME', 'qwen/qwen3-4b')
        
        # Create LM instance for LM Studio
        lm = LM(
            model=f"openai/{model_name}",
            api_base=model_endpoint,
            api_key="lm-studio",  # LM Studio doesn't need real API key
            max_tokens=1024,
            temperature=0.1,  # Low temperature for consistent optimization
            timeout=30.0
        )
        
        # Configure DSPy globally
        dspy.configure(lm=lm)
        
        logger.info(f"🚀 DSPy v3 configured with {model_name} via LM Studio")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure DSPy with LM Studio: {e}")
        return False

if DSPY_AVAILABLE and OPTIMIZATION_ENABLED:
    logger.info(f"🚀 DSPy optimization enabled - training window: {TRAINING_DAYS} days")
    # Auto-configure DSPy on import
    configure_dspy_with_lm_studio()
elif DSPY_AVAILABLE:
    logger.info("🧠 DSPy available but optimization disabled (set DSPY_OPTIMIZATION_ENABLED=true)")

__all__ = [
    'DSPY_AVAILABLE',
    'OPTIMIZATION_ENABLED',
    'MODEL_PATH',
    'TRAINING_DAYS', 
    'OPTIMIZATION_INTERVAL',
    'configure_dspy_with_lm_studio'
]

# Conditional imports for optimization components
if DSPY_AVAILABLE:
    try:
        from .unified_optimizer import UnifiedMemoryOptimizer, create_unified_memory_optimizer
        from .surreal_optimizers import SurrealContextOptimizer, SurrealResponseGenerator
        from .metrics.surreal_metrics import context_relevance_metric, response_quality_metric
        
        __all__.extend([
            'UnifiedMemoryOptimizer',
            'create_unified_memory_optimizer',
            'SurrealContextOptimizer',
            'SurrealResponseGenerator', 
            'context_relevance_metric',
            'response_quality_metric'
        ])
        
        logger.info("🔧 DSPy optimization components loaded successfully")
    except ImportError as e:
        logger.warning(f"DSPy optimization components not fully available: {e}")