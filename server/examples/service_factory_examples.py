"""
Examples demonstrating ServiceFactory usage patterns
"""

import asyncio
from core.service_factory import ServiceFactory, service_factory
from loguru import logger


async def basic_service_usage():
    """Basic service factory usage"""
    print("=== Basic Service Usage ===")
    
    # Get a service (lazy loaded)
    memory_service = await service_factory.get_service("memory_service")
    print(f"Memory service: {memory_service}")
    
    # Services are singletons by default  
    memory_service2 = await service_factory.get_service("memory_service")
    print(f"Same instance: {memory_service is memory_service2}")
    
    # Wait for ML modules to load
    await service_factory.wait_for_ml_modules()
    print("ML modules loaded!")


async def language_specific_services():
    """Create services for specific languages"""
    print("\n=== Language-Specific Services ===")
    
    # English services (optimized models)
    en_services = await service_factory.create_services_for_language("en")
    print(f"English STT: {type(en_services['stt']).__name__}")
    print(f"English TTS: {type(en_services['tts']).__name__}")
    print(f"English LLM: {type(en_services['llm']).__name__}")
    
    # Spanish services (different voice/model)
    es_services = await service_factory.create_services_for_language("es", "custom-spanish-model")
    print(f"Spanish STT: {type(es_services['stt']).__name__}")
    print(f"Spanish TTS voice: {es_services['tts'].voice}")
    print(f"Spanish LLM model: {es_services['llm'].model}")


async def custom_service_registration():
    """Register and use custom services"""
    print("\n=== Custom Service Registration ===")
    
    # Create custom factory
    custom_factory = ServiceFactory()
    
    # Define custom service
    def create_custom_analyzer():
        return {"type": "custom_analyzer", "version": "1.0"}
    
    # Register custom service with dependencies
    custom_factory.registry.register(
        "custom_analyzer",
        create_custom_analyzer,
        dependencies=[],  # No dependencies
        singleton=True,
        lazy=False
    )
    
    # Use custom service
    analyzer = await custom_factory.get_service("custom_analyzer")
    print(f"Custom analyzer: {analyzer}")
    
    # Register dependent service
    def create_dependent_service(analyzer):
        return f"Service using {analyzer['type']} v{analyzer['version']}"
    
    custom_factory.registry.register(
        "dependent_service",
        create_dependent_service,
        dependencies=["custom_analyzer"]
    )
    
    dependent = await custom_factory.get_service("dependent_service")
    print(f"Dependent service: {dependent}")


async def service_lifecycle_management():
    """Demonstrate service lifecycle management"""
    print("\n=== Service Lifecycle Management ===")
    
    # Check which services are registered
    all_services = service_factory.registry.list_services()
    print(f"Registered services: {all_services}")
    
    # Check which services are instantiated
    instantiated = []
    for service_name in all_services:
        if service_factory.registry.has_instance(service_name):
            instantiated.append(service_name)
    
    print(f"Instantiated services: {instantiated}")
    
    # Force instantiation of a service
    global_analyzers = await service_factory.get_service("global_analyzers")
    print(f"Global analyzers ready: {global_analyzers is not None}")


async def error_handling_examples():
    """Demonstrate error handling in service factory"""
    print("\n=== Error Handling ===")
    
    try:
        # Try to get non-existent service
        await service_factory.get_service("nonexistent_service")
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Create factory with failing service
    failing_factory = ServiceFactory()
    
    def failing_service():
        raise RuntimeError("Service initialization failed")
    
    failing_factory.registry.register("failing_service", failing_service)
    
    try:
        await failing_factory.get_service("failing_service")
    except RuntimeError as e:
        print(f"Service creation failed as expected: {e}")


async def performance_monitoring():
    """Monitor service creation performance"""
    print("\n=== Performance Monitoring ===")
    
    import time
    
    # Time service creation
    start_time = time.time()
    await service_factory.wait_for_ml_modules()
    ml_load_time = time.time() - start_time
    print(f"ML modules load time: {ml_load_time:.2f}s")
    
    # Time service instantiation (should be fast after ML load)
    start_time = time.time()
    services = await service_factory.create_services_for_language("en")
    service_create_time = time.time() - start_time
    print(f"Service creation time: {service_create_time:.2f}s")
    
    # Time subsequent calls (should be instant due to caching)
    start_time = time.time()
    services2 = await service_factory.create_services_for_language("en")
    cached_time = time.time() - start_time
    print(f"Cached service access time: {cached_time:.4f}s")


class CustomService:
    """Example custom service implementation"""
    
    def __init__(self, dependency1, dependency2=None):
        self.dep1 = dependency1
        self.dep2 = dependency2
        self.initialized = True
    
    async def async_initialize(self):
        """Async initialization if needed"""
        await asyncio.sleep(0.1)  # Simulate async work
        self.async_ready = True
    
    def process(self, data):
        return f"Processed {data} with {self.dep1}"


async def advanced_service_patterns():
    """Advanced service factory patterns"""
    print("\n=== Advanced Patterns ===")
    
    # Create factory for advanced examples
    advanced_factory = ServiceFactory()
    
    # Register base service
    advanced_factory.registry.register(
        "base_service",
        lambda: "base_implementation"
    )
    
    # Register service with async factory
    async def create_async_service(base):
        service = CustomService(base)
        await service.async_initialize()
        return service
    
    advanced_factory.registry.register(
        "async_service",
        create_async_service,
        dependencies=["base_service"]
    )
    
    # Use async service
    async_service = await advanced_factory.get_service("async_service")
    print(f"Async service ready: {async_service.async_ready}")
    print(f"Process result: {async_service.process('test_data')}")
    
    # Register conditional service
    def create_conditional_service():
        import os
        if os.getenv("USE_ADVANCED_MODE", "false").lower() == "true":
            return "advanced_mode_service"
        else:
            return "standard_mode_service"
    
    advanced_factory.registry.register("conditional_service", create_conditional_service)
    
    conditional = await advanced_factory.get_service("conditional_service")
    print(f"Conditional service: {conditional}")


async def main():
    """Run all examples"""
    try:
        await basic_service_usage()
        await language_specific_services()
        await custom_service_registration()
        await service_lifecycle_management()
        await error_handling_examples()
        await performance_monitoring()
        await advanced_service_patterns()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())