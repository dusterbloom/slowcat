#!/usr/bin/env python3
"""
SurrealDB Setup and Test Script

This script:
1. Starts SurrealDB server automatically
2. Tests the connection and schema setup
3. Runs basic functionality tests
4. Benchmarks against SQLite performance
5. Provides easy environment switching

Usage:
    python test_surrealdb_setup.py [--start-server] [--benchmark] [--cleanup]
"""

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import argparse
from pathlib import Path
from loguru import logger

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from memory import create_smart_memory_system
    from memory.surreal_memory import SurrealMemory, create_surreal_memory_system
    from memory.facts_graph import FactsGraph, extract_facts_from_text
    IMPORTS_OK = True
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.info("Make sure you're in the server/ directory and have activated .venv")
    IMPORTS_OK = False


class SurrealDBTestRunner:
    """Test runner for SurrealDB memory system"""
    
    def __init__(self):
        self.surreal_process = None
        self.surreal_url = "ws://localhost:8000/rpc"
        self.temp_data_dir = None
        
    async def start_surreal_server(self, data_path: str = None):
        """Start SurrealDB server"""
        try:
            # Check if SurrealDB is installed
            result = subprocess.run(['surreal', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("SurrealDB not found. Install with: brew install surrealdb/tap/surreal")
                return False
            
            logger.info(f"Found SurrealDB: {result.stdout.strip()}")
            
            # Create data directory - use rocksdb:// format for persistent storage
            if data_path is None:
                self.temp_data_dir = tempfile.mkdtemp(prefix="surrealdb_test_")
                data_path = f"rocksdb://{self.temp_data_dir}"
            
            logger.info(f"Starting SurrealDB server with data path: {data_path}")
            
            # Start SurrealDB server with correct syntax
            cmd = [
                'surreal', 'start',
                data_path,  # Use rocksdb:// format
                '--bind', '127.0.0.1:8000',
                '--user', 'root',  # Correct flags
                '--pass', 'root'
            ]
            
            self.surreal_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            logger.info("Waiting for SurrealDB server to start...")
            for i in range(10):
                await asyncio.sleep(1)
                
                # Check if process is still running
                if self.surreal_process.poll() is not None:
                    stdout, stderr = self.surreal_process.communicate()
                    logger.error(f"SurrealDB server failed to start: {stderr}")
                    return False
                
                # Try to connect
                try:
                    from surrealdb import AsyncSurreal
                    db = AsyncSurreal(self.surreal_url)
                    # Skip authentication for unauthenticated server
                    await db.use('slowcat', 'memory')
                    await db.close()
                    logger.info("✅ SurrealDB server is ready")
                    return True
                except Exception as e:
                    logger.debug(f"Connection attempt {i+1} failed: {e}")
                    continue
            
            logger.error("❌ SurrealDB server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start SurrealDB server: {e}")
            return False
    
    def stop_surreal_server(self):
        """Stop SurrealDB server"""
        if self.surreal_process:
            logger.info("Stopping SurrealDB server...")
            self.surreal_process.terminate()
            try:
                self.surreal_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.surreal_process.kill()
            self.surreal_process = None
        
        # Cleanup temp directory
        if self.temp_data_dir:
            import shutil
            shutil.rmtree(self.temp_data_dir, ignore_errors=True)
            self.temp_data_dir = None
    
    async def test_surreal_memory_basic(self):
        """Test basic SurrealDB memory functionality"""
        logger.info("🧪 Testing SurrealDB memory basic functionality")
        
        try:
            # Create SurrealDB memory system
            memory = create_surreal_memory_system(self.surreal_url)
            await memory.connect()
            
            # Test facts operations
            test_facts = [
                {'subject': 'user', 'predicate': 'name', 'value': 'Alex'},
                {'subject': 'user', 'predicate': 'pet', 'value': 'Potola', 'species': 'dog'},
                {'subject': 'user', 'predicate': 'location', 'value': 'San Francisco'},
                {'subject': 'user', 'predicate': 'hobby', 'value': 'coding'},
            ]
            
            logger.info("Inserting test facts...")
            for fact in test_facts:
                await memory.reinforce_or_insert(fact)
            
            # Test fact search
            logger.info("Testing fact search...")
            dog_facts = await memory.search_facts("dog")
            assert len(dog_facts) > 0, "Dog facts not found"
            logger.info(f"✅ Found {len(dog_facts)} dog facts")
            
            # Test top facts
            top_facts = await memory.get_top_facts(5)
            assert len(top_facts) > 0, "Top facts not found"
            logger.info(f"✅ Retrieved {len(top_facts)} top facts")
            
            # Test tape operations
            logger.info("Testing tape operations...")
            await memory.add_entry("user", "Hello, this is a test message", "alex")
            await memory.add_entry("assistant", "Hi Alex! How can I help you today?", "alex")
            await memory.add_entry("user", "What's my dog's name?", "alex")
            await memory.add_entry("assistant", "Your dog's name is Potola!", "alex")
            
            # Test tape search
            recent_entries = await memory.get_recent(limit=5)
            assert len(recent_entries) > 0, "Recent entries not found"
            logger.info(f"✅ Retrieved {len(recent_entries)} recent entries")
            
            # Test tape search
            dog_entries = await memory.search_tape("dog")
            assert len(dog_entries) > 0, "Dog conversation not found"
            logger.info(f"✅ Found {len(dog_entries)} dog-related conversations")
            
            # Test time travel
            logger.info("Testing time travel queries...")
            today_entries = await memory.time_travel_query("today", limit=10)
            logger.info(f"✅ Found {len(today_entries)} entries from today")
            
            # Test stats
            stats = await memory.get_stats()
            logger.info(f"✅ Memory stats: {stats}")
            
            # Test decay
            logger.info("Testing decay...")
            await memory.apply_decay()
            logger.info("✅ Decay applied successfully")
            
            await memory.close()
            logger.info("✅ SurrealDB basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SurrealDB basic test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_environment_toggle(self):
        """Test environment variable toggle between SQLite and SurrealDB"""
        logger.info("🔄 Testing environment toggle")
        
        try:
            # Test SQLite mode (default)
            os.environ.pop('USE_SURREALDB', None)
            sqlite_memory = create_smart_memory_system()
            logger.info("✅ SQLite memory system created")
            
            # Store some facts in SQLite
            facts_stored = sqlite_memory.store_facts("My cat's name is Whiskers and I live in Portland")
            logger.info(f"✅ Stored {facts_stored} facts in SQLite")
            
            sqlite_stats = sqlite_memory.get_stats()
            logger.info(f"SQLite stats: {sqlite_stats}")
            sqlite_memory.close()
            
            # Test SurrealDB mode
            os.environ['USE_SURREALDB'] = 'true'
            os.environ['SURREALDB_URL'] = self.surreal_url
            
            surreal_memory = create_smart_memory_system()
            logger.info("✅ SurrealDB memory system created via environment toggle")
            
            # Store some facts in SurrealDB
            surreal_facts_stored = surreal_memory.store_facts("I have a dog named Potola and work at Anthropic")
            logger.info(f"✅ Stored {surreal_facts_stored} facts in SurrealDB")
            
            surreal_stats = surreal_memory.get_stats()
            logger.info(f"SurrealDB stats: {surreal_stats}")
            surreal_memory.close()
            
            logger.info("✅ Environment toggle test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment toggle test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup environment
            os.environ.pop('USE_SURREALDB', None)
            os.environ.pop('SURREALDB_URL', None)
    
    async def benchmark_performance(self):
        """Benchmark SurrealDB vs SQLite performance"""
        logger.info("📊 Running performance benchmark")
        
        try:
            # Prepare test data
            test_facts = []
            for i in range(100):
                test_facts.extend([
                    {'subject': f'user_{i}', 'predicate': 'name', 'value': f'User{i}'},
                    {'subject': f'user_{i}', 'predicate': 'age', 'value': str(20 + i % 50)},
                    {'subject': f'user_{i}', 'predicate': 'city', 'value': f'City{i % 10}'},
                ])
            
            logger.info(f"Benchmarking with {len(test_facts)} facts...")
            
            # Benchmark SQLite
            logger.info("Benchmarking SQLite...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                sqlite_start = time.time()
                
                facts_graph = FactsGraph(f"{tmp_dir}/benchmark.db")
                for fact in test_facts:
                    facts_graph.reinforce_or_insert(fact)
                
                # Test searches
                for i in range(20):
                    facts_graph.search_facts(f"User{i}")
                
                facts_graph.close()
                sqlite_time = time.time() - sqlite_start
            
            # Benchmark SurrealDB
            logger.info("Benchmarking SurrealDB...")
            surreal_start = time.time()
            
            memory = create_surreal_memory_system(self.surreal_url)
            await memory.connect()
            
            for fact in test_facts:
                await memory.reinforce_or_insert(fact)
            
            # Test searches
            for i in range(20):
                await memory.search_facts(f"User{i}")
            
            await memory.close()
            surreal_time = time.time() - surreal_start
            
            # Report results
            logger.info("📊 Benchmark Results:")
            logger.info(f"  SQLite time: {sqlite_time:.2f}s")
            logger.info(f"  SurrealDB time: {surreal_time:.2f}s")
            logger.info(f"  Speedup: {sqlite_time/surreal_time:.2f}x {'(SurrealDB faster)' if surreal_time < sqlite_time else '(SQLite faster)'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self, start_server=True, benchmark=False):
        """Run all tests"""
        if not IMPORTS_OK:
            logger.error("❌ Cannot run tests due to import errors")
            return False
        
        try:
            if start_server:
                if not await self.start_surreal_server():
                    return False
            
            success = True
            
            # Basic functionality test
            if not await self.test_surreal_memory_basic():
                success = False
            
            # Environment toggle test
            if not await self.test_environment_toggle():
                success = False
            
            # Performance benchmark
            if benchmark:
                if not await self.benchmark_performance():
                    success = False
            
            if success:
                logger.info("🎉 All tests passed! SurrealDB memory system is ready.")
                logger.info("🚀 To use SurrealDB in your bot:")
                logger.info("   export USE_SURREALDB=true")
                logger.info("   export SURREALDB_URL=ws://localhost:8000/rpc")
                logger.info("   ./run_bot.sh")
            else:
                logger.error("❌ Some tests failed. Check the logs above.")
            
            return success
            
        finally:
            if start_server:
                self.stop_surreal_server()


def create_env_file():
    """Create .env.surreal file for easy testing"""
    env_content = """# SurrealDB Environment Configuration
USE_SURREALDB=true
SURREALDB_URL=ws://localhost:8000/rpc
SURREALDB_NAMESPACE=slowcat
SURREALDB_DATABASE=memory

# Copy existing .env settings here
# OPENAI_BASE_URL=http://localhost:1234/v1
# ENABLE_VOICE_RECOGNITION=true
# ENABLE_MEMORY=true
# ENABLE_MCP=true
"""
    
    env_path = Path("server/.env.surreal")
    env_path.write_text(env_content)
    logger.info(f"📝 Created {env_path}")
    logger.info("   To use: cp .env.surreal .env")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="SurrealDB Setup and Test")
    parser.add_argument('--start-server', action='store_true', 
                       help='Start SurrealDB server automatically')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--cleanup', action='store_true',
                       help='Stop any running SurrealDB servers')
    parser.add_argument('--create-env', action='store_true',
                       help='Create .env.surreal configuration file')
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_file()
        return
    
    if args.cleanup:
        logger.info("🧹 Cleaning up any running SurrealDB servers...")
        subprocess.run(['pkill', '-f', 'surreal'], capture_output=True)
        logger.info("✅ Cleanup complete")
        return
    
    runner = SurrealDBTestRunner()
    
    try:
        success = await runner.run_all_tests(
            start_server=args.start_server,
            benchmark=args.benchmark
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        runner.stop_surreal_server()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test runner failed: {e}")
        runner.stop_surreal_server()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())