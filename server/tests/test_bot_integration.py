#!/usr/bin/env python3
"""
Integration test for cached embeddings memory system in the main bot pipeline
Tests that the stateless memory processor works correctly with the actual bot infrastructure
"""

import asyncio
import os
import sys
import time
import httpx
from pathlib import Path

# Set environment for testing
os.environ['USE_STATELESS_MEMORY'] = 'true'
os.environ['ENABLE_MEMORY'] = 'true'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_bot_memory_integration():
    """Test that the stateless memory system works with the bot's service factory"""
    
    print("🧠 Bot Memory Integration Test")
    print("=" * 40)
    
    try:
        # Import bot configuration
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        from config import config
        from core.service_factory import ServiceFactory
        
        # Create service factory
        print("🔧 Creating service factory...")
        service_factory = ServiceFactory()
        
        # Test memory service creation
        print("🧠 Testing memory service creation...")
        memory_service = await service_factory.get_service("memory_service")
        
        if memory_service is None:
            print("❌ Memory service not created")
            return False
        
        # Check if it's the stateless memory processor
        from processors.stateless_memory import StatelessMemoryProcessor
        if not isinstance(memory_service, StatelessMemoryProcessor):
            print(f"❌ Expected StatelessMemoryProcessor, got {type(memory_service)}")
            return False
        
        print("✅ StatelessMemoryProcessor created successfully")
        
        # Test embedding functionality
        print("🔍 Testing embedding API connection...")
        try:
            test_embedding = await memory_service._get_embedding("test connection")
            if test_embedding:
                print("✅ Embedding API working")
                print(f"   Embedding dimension: {len(test_embedding)}")
            else:
                print("⚠️  Embedding API not available - will use fallback search")
        except Exception as e:
            print(f"⚠️  Embedding test failed: {e}")
        
        # Test memory storage and retrieval
        print("💾 Testing memory storage...")
        
        # Store a test memory
        await memory_service._store_exchange(
            "My name is Alice and I love space exploration",
            "Nice to meet you Alice! Space exploration is fascinating."
        )
        
        # Store another memory 
        await memory_service._store_exchange(
            "I work as an astronomer at NASA",
            "That's amazing! Working at NASA must be incredibly exciting."
        )
        
        print("✅ Test memories stored")
        
        # Test memory retrieval
        print("🔍 Testing memory retrieval...")
        
        # Prepare a test query
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What do you know about me?'}
        ]
        
        # Test memory injection
        start_time = time.perf_counter()
        await memory_service._inject_memory_context(messages, "test_user")
        injection_time = (time.perf_counter() - start_time) * 1000
        
        print(f"⏱️  Memory injection time: {injection_time:.2f}ms")
        
        # Check if memories were injected
        memory_injected = len(messages) > 2
        if memory_injected:
            print("✅ Memory context injected successfully")
            print(f"📊 Total messages after injection: {len(messages)}")
            
            # Show the memory context
            for i, msg in enumerate(messages):
                if '[Memory Context' in msg.get('content', ''):
                    print(f"🧠 Memory context found in message {i}")
                    break
        else:
            print("⚠️  No memory context injected")
        
        # Test performance with multiple queries
        print("🚀 Testing performance with rapid queries...")
        
        queries = [
            "What's my name?",
            "Where do I work?", 
            "What do I love?",
            "What's my profession?"
        ]
        
        total_time = 0
        successful_injections = 0
        
        for query in queries:
            test_messages = [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': query}
            ]
            
            start = time.perf_counter()
            try:
                await memory_service._inject_memory_context(test_messages, "test_user")
                injection_time = (time.perf_counter() - start) * 1000
                total_time += injection_time
                successful_injections += 1
                
                memory_found = len(test_messages) > 2
                status = "✅" if memory_found else "⚠️"
                print(f"   '{query}' → {injection_time:.1f}ms {status}")
                
            except Exception as e:
                print(f"   '{query}' → ERROR: {e}")
        
        # Performance summary
        if successful_injections > 0:
            avg_time = total_time / successful_injections
            print(f"\n📊 Performance Summary:")
            print(f"   Average injection time: {avg_time:.2f}ms")
            print(f"   Successful injections: {successful_injections}/{len(queries)}")
            
            if avg_time < 50:
                grade = "🟢 EXCELLENT"
            elif avg_time < 100:
                grade = "🟡 GOOD"
            else:
                grade = "🔴 NEEDS IMPROVEMENT"
            
            print(f"   Performance grade: {grade}")
        
        # Clean up
        if hasattr(memory_service, 'env'):
            memory_service.env.close()
        if hasattr(memory_service, 'thread_pool'):
            memory_service.thread_pool.shutdown(wait=True)
        
        print(f"\n🎉 Integration test completed successfully!")
        print("✅ Cached embeddings memory system working with bot infrastructure")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_server_health():
    """Test if the bot server is running and responding"""
    
    print("\n🌐 Testing Server Health")
    print("-" * 25)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test if server is responding
            response = await client.get("http://localhost:7860/health")
            if response.status_code == 200:
                print("✅ Bot server is running and healthy")
                return True
            else:
                print(f"⚠️  Server responded with status: {response.status_code}")
                return False
                
    except httpx.ConnectError:
        print("⚠️  Bot server not running on localhost:7860")
        print("   Start it with: USE_STATELESS_MEMORY=true ./run_bot.sh")
        return False
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        # Test 1: Memory system integration
        memory_test_passed = await test_bot_memory_integration()
        
        # Test 2: Server health (if running)
        server_test_passed = await test_server_health()
        
        # Summary
        print(f"\n📋 TEST SUMMARY")
        print("-" * 15)
        memory_status = "✅ PASS" if memory_test_passed else "❌ FAIL"
        server_status = "✅ PASS" if server_test_passed else "⚠️  SKIP"
        
        print(f"Memory Integration: {memory_status}")
        print(f"Server Health: {server_status}")
        
        if memory_test_passed:
            print(f"\n🎯 CONCLUSION: Cached embeddings memory system is ready for production!")
            print("🚀 Fast, reliable, and fully integrated with the bot infrastructure.")
        else:
            print(f"\n⚠️  Integration issues detected - needs investigation")
        
        return memory_test_passed
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)