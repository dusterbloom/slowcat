"""
Test migration plan for moving existing tests to the new test structure

This script helps migrate existing tests to the organized test structure:
- Unit tests → tests/unit/
- Integration tests → tests/integration/
- Performance tests → tests/performance/
- Voice recognition tests → tests/integration/voice/
"""

import os
import shutil
from pathlib import Path
from loguru import logger

# Define test file mapping
TEST_MIGRATIONS = {
    # Current location → New location
    "test_tools.py": "tests/unit/test_tools.py",
    "test_file_tools.py": "tests/unit/test_file_tools.py", 
    "test_llm_tools.py": "tests/unit/test_llm_tools.py",
    "test_llm_direct.py": "tests/unit/test_llm_direct.py",
    "test_models_search.py": "tests/unit/test_models_search.py",
    
    # Voice recognition tests
    "test_voice_recognition.py": "tests/integration/voice/test_voice_recognition.py",
    "test_voice_recognition_live.py": "tests/integration/voice/test_voice_recognition_live.py",
    
    # Integration tests from tests/ directory
    "tests/test_integration.py": "tests/integration/test_integration.py",
    "tests/test_memory.py": "tests/integration/test_memory.py", 
    "tests/test_memory_search.py": "tests/integration/test_memory_search.py",
    "tests/test_direct_search.py": "tests/integration/test_direct_search.py",
    "tests/test_model_function_support.py": "tests/integration/test_model_function_support.py",
    
    # Performance tests
    "tests/test_performance_optimizations.py": "tests/performance/test_optimizations.py",
    "tests/test_streaming_tts.py": "tests/performance/test_streaming_tts.py",
    
    # Debug scripts → tools
    "tests/debug_bot.py": "tests/tools/debug_bot.py",
    "tests/debug_memory_tools.py": "tests/tools/debug_memory_tools.py", 
    "tests/debug_voice_recognition.py": "tests/tools/debug_voice_recognition.py",
}

# Test directories to create
TEST_DIRECTORIES = [
    "tests/unit",
    "tests/integration", 
    "tests/integration/voice",
    "tests/performance",
    "tests/tools",
    "tests/fixtures",
    "tests/conftest"
]


def create_test_directories():
    """Create organized test directory structure"""
    logger.info("Creating test directory structure...")
    
    base_dir = Path(__file__).parent.parent
    
    for directory in TEST_DIRECTORIES:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for test packages
        if not directory.endswith("conftest"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Test package"""')
        
        logger.info(f"✅ Created directory: {directory}")


def migrate_test_files():
    """Migrate test files to new structure"""
    logger.info("Migrating test files...")
    
    base_dir = Path(__file__).parent.parent
    migrated_count = 0
    
    for old_path, new_path in TEST_MIGRATIONS.items():
        old_file = base_dir / old_path
        new_file = base_dir / new_path
        
        if old_file.exists():
            # Ensure target directory exists
            new_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to new location
            shutil.copy2(old_file, new_file)
            logger.info(f"✅ Migrated: {old_path} → {new_path}")
            migrated_count += 1
        else:
            logger.warning(f"⚠️ File not found: {old_path}")
    
    logger.info(f"Migrated {migrated_count} test files")


def create_conftest_files():
    """Create pytest conftest.py files with common fixtures"""
    logger.info("Creating conftest.py files...")
    
    base_dir = Path(__file__).parent.parent
    
    # Root conftest.py
    root_conftest = base_dir / "tests" / "conftest.py"
    root_conftest.write_text('''"""
Root test configuration and fixtures
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

# Add server directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    with patch('config.config') as mock_cfg:
        # Set up default mock values
        mock_cfg.network.server_host = "localhost"
        mock_cfg.network.server_port = 7860
        mock_cfg.network.llm_base_url = "http://localhost:1234/v1"
        mock_cfg.default_language = "en"
        mock_cfg.memory.enabled = True
        mock_cfg.voice_recognition.enabled = True
        mock_cfg.video.enabled = False
        mock_cfg.mcp.enabled = True
        yield mock_cfg


@pytest.fixture
def mock_logger():
    """Mock logger to capture log messages"""
    with patch('loguru.logger') as mock_log:
        yield mock_log


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory for tests"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
''')
    
    # Unit test conftest.py
    unit_conftest = base_dir / "tests" / "unit" / "conftest.py"
    unit_conftest.write_text('''"""
Unit test fixtures and configuration
"""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_service_factory():
    """Mock ServiceFactory for unit tests"""
    factory = Mock()
    factory.get_service = AsyncMock()
    factory.wait_for_ml_modules = AsyncMock()
    factory.wait_for_global_analyzers = AsyncMock()
    factory.create_services_for_language = AsyncMock()
    factory.registry = Mock()
    return factory


@pytest.fixture
def mock_ml_modules():
    """Mock ML modules dictionary"""
    return {
        'WhisperSTTServiceMLX': Mock(),
        'MLXModel': Mock(),
        'KokoroTTSService': Mock(),
        'LLMWithToolsService': Mock(),
        'OpenAILLMService': Mock(),
        'AutoEnrollVoiceRecognition': Mock(),
        'SileroVADAnalyzer': Mock(),
        'LocalSmartTurnAnalyzerV2': Mock()
    }


@pytest.fixture
def mock_webrtc_connection():
    """Mock WebRTC connection"""
    connection = Mock()
    connection.pc_id = "test-connection-id"
    connection.get_answer.return_value = {
        "pc_id": "test-connection-id",
        "sdp": "mock-sdp",
        "type": "answer"
    }
    return connection
''')
    
    # Integration test conftest.py
    integration_conftest = base_dir / "tests" / "integration" / "conftest.py"
    integration_conftest.write_text('''"""
Integration test fixtures and configuration
"""

import pytest
import asyncio
from unittest.mock import patch
from core.service_factory import ServiceFactory
from core.pipeline_builder import PipelineBuilder


@pytest.fixture(scope="module")
async def service_factory():
    """Real ServiceFactory instance for integration tests"""
    # Mock heavy dependencies to avoid loading actual ML models
    with patch('core.service_factory.importlib.import_module') as mock_import:
        # Setup mock modules to avoid actual ML loading
        mock_import.return_value = Mock()
        
        factory = ServiceFactory()
        # Initialize with mocked ML modules
        await factory.get_service("ml_loader")
        return factory


@pytest.fixture
async def pipeline_builder(service_factory):
    """PipelineBuilder with real ServiceFactory"""
    return PipelineBuilder(service_factory)


@pytest.fixture
def integration_config():
    """Configuration for integration tests"""
    return {
        "test_language": "en",
        "test_llm_model": "test-model",
        "mock_ml_modules": True,
        "enable_logging": True
    }
''')
    
    logger.info("✅ Created conftest.py files")


def create_test_runner():
    """Create test runner script"""
    logger.info("Creating test runner...")
    
    base_dir = Path(__file__).parent.parent
    test_runner = base_dir / "run_tests.py"
    
    test_runner.write_text('''#!/usr/bin/env python3
"""
Test runner for Slowcat server tests
Provides organized test execution with different test categories
"""

import sys
import subprocess
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v", "--tb=short"
    ])
    return result.returncode == 0


def run_integration_tests():
    """Run integration tests"""
    print("🔗 Running integration tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v", "--tb=short", "-s"
    ])
    return result.returncode == 0


def run_performance_tests():
    """Run performance tests"""
    print("⚡ Running performance tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/performance/", 
        "-v", "--tb=short"
    ])
    return result.returncode == 0


def run_all_tests():
    """Run all tests"""
    print("🚀 Running all tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v", "--tb=short"
    ])
    return result.returncode == 0


def run_tests_with_coverage():
    """Run tests with coverage report"""
    print("📊 Running tests with coverage...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=core",
        "--cov=processors", 
        "--cov=services",
        "--cov=tools",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Slowcat Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    success = True
    
    if args.unit:
        success &= run_unit_tests()
    elif args.integration:
        success &= run_integration_tests()
    elif args.performance:
        success &= run_performance_tests()
    elif args.coverage:
        success &= run_tests_with_coverage()
    else:
        success &= run_all_tests()
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
''')
    
    test_runner.chmod(0o755)  # Make executable
    logger.info("✅ Created test runner")


def create_github_workflow():
    """Create GitHub Actions workflow for tests"""
    logger.info("Creating GitHub Actions workflow...")
    
    base_dir = Path(__file__).parent.parent
    workflow_dir = base_dir / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_file = workflow_dir / "tests.yml"
    workflow_file.write_text('''name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: macos-latest
    
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        cd server
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run unit tests
      run: |
        cd server
        python run_tests.py --unit
    
    - name: Run integration tests (non-ML)
      run: |
        cd server  
        # Run integration tests that don't require ML models
        python -m pytest tests/integration/ -k "not voice" -v
    
    - name: Generate coverage report
      run: |
        cd server
        python run_tests.py --coverage
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./server/htmlcov/coverage.xml
        flags: unittests
        name: codecov-umbrella
''')
    
    logger.info("✅ Created GitHub Actions workflow")


def main():
    """Run complete test migration"""
    logger.info("🚀 Starting test migration...")
    
    try:
        create_test_directories()
        migrate_test_files()
        create_conftest_files()
        create_test_runner()
        create_github_workflow()
        
        logger.info("✅ Test migration completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review migrated test files for any needed updates")
        logger.info("2. Run: python run_tests.py --unit")
        logger.info("3. Update imports in migrated tests if needed")
        logger.info("4. Consider removing old test files after verification")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
''')

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analyze current bot.py structure and identify refactoring opportunities", "status": "completed", "priority": "high"}, {"id": "2", "content": "Design service factory pattern for dependency injection", "status": "completed", "priority": "high"}, {"id": "3", "content": "Create core service abstractions and interfaces", "status": "completed", "priority": "high"}, {"id": "4", "content": "Implement pipeline builder with dependency injection", "status": "completed", "priority": "high"}, {"id": "5", "content": "Create migration strategy maintaining backward compatibility", "status": "completed", "priority": "medium"}, {"id": "6", "content": "Consolidate test organization under tests/ directory", "status": "completed", "priority": "medium"}, {"id": "7", "content": "Create code examples for each refactoring step", "status": "in_progress", "priority": "medium"}]