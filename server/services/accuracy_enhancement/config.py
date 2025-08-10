"""
Configuration for Sherpa-ONNX Accuracy Enhancement
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AccuracyEnhancementConfig:
    """Configuration for accuracy enhancement service"""
    
    # Enable/disable accuracy enhancement
    enable_accuracy_enhancement: bool = os.getenv("SHERPA_ENABLE_ACCURACY_ENHANCEMENT", "false").lower() == "true"
    
    # LLM model for contextual correction
    accuracy_enhancement_model: str = os.getenv("SHERPA_ACCURACY_MODEL", "qwen3-1.7b:2")
    
    # LLM API endpoint
    llm_base_url: str = os.getenv("SHERPA_ACCURACY_LLM_BASE_URL", "http://localhost:1234/v1")
    
    # Confidence threshold below which enhancement is applied
    confidence_threshold: float = float(os.getenv("SHERPA_ACCURACY_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Enable LLM-based contextual correction
    enable_llm_correction: bool = os.getenv("SHERPA_ENABLE_LLM_CORRECTION", "true").lower() == "true"
    
    # Maximum number of corrections to apply per text
    max_corrections_per_text: int = int(os.getenv("SHERPA_MAX_CORRECTIONS_PER_TEXT", "10"))
    
    # Cache directory for dynamic vocabulary
    cache_dir: Optional[str] = os.getenv("SHERPA_ACCURACY_CACHE_DIR")


# Global configuration instance
config = AccuracyEnhancementConfig()