"""
Core module for Slowcat server
Contains service abstractions, factories, and dependency injection
"""

from .service_factory import ServiceFactory, ServiceRegistry
from .pipeline_builder import PipelineBuilder
from .service_interfaces import (
    BaseSTTService,
    BaseTTSService,
    BaseLLMService,
    BaseProcessor
)

__all__ = [
    'ServiceFactory',
    'ServiceRegistry', 
    'PipelineBuilder',
    'BaseSTTService',
    'BaseTTSService',
    'BaseLLMService',
    'BaseProcessor'
]