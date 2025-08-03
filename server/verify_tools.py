#!/usr/bin/env python3
"""
Tool verification script for Slowcat
Tests that all tools are properly defined and can be executed
"""

import asyncio
import sys
import os
from typing import Dict, Any
from loguru import logger

# Add local pipecat to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

# Import tool components
from tools import get_tools, get_tool_names, execute_tool_call
from tools.formatters import format_tool_response_for_voice
from tools.definitions import ALL_FUNCTION_SCHEMAS

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

async def test_tool_definitions():
    """Test that all tools are properly defined"""
    print_header("Testing Tool Definitions")
    
    # Test get_tools()
    tools_schema = get_tools()
    print_success(f"get_tools() returned ToolsSchema with {len(tools_schema.standard_tools)} tools")
    
    # Test get_tool_names()
    tool_names = get_tool_names()
    print_success(f"get_tool_names() returned {len(tool_names)} tool names")
    print(f"  Tools: {', '.join(tool_names)}")
    
    # Verify each tool schema
    print("\nVerifying individual tool schemas:")
    for schema in ALL_FUNCTION_SCHEMAS:
        try:
            assert hasattr(schema, 'name'), f"Missing 'name' in schema"
            assert hasattr(schema, 'description'), f"Missing 'description' in {schema.name}"
            assert hasattr(schema, 'properties'), f"Missing 'properties' in {schema.name}"
            print_success(f"Schema for '{schema.name}' is valid")
        except AssertionError as e:
            print_error(f"Schema validation failed: {e}")
            return False
    
    return True

async def test_tool_execution():
    """Test executing sample tools"""
    print_header("Testing Tool Execution")
    
    # Test cases for different tools
    test_cases = [
        {
            "name": "get_current_time",
            "args": {"format": "human", "timezone": "UTC"},
            "description": "Get current time"
        },
        {
            "name": "calculate",
            "args": {"expression": "2 + 2"},
            "description": "Basic calculation"
        },
        {
            "name": "remember_information",
            "args": {"key": "test_key", "value": "test_value"},
            "description": "Store test information"
        },
        {
            "name": "recall_information",
            "args": {"key": "test_key"},
            "description": "Retrieve test information"
        },
        {
            "name": "list_files",
            "args": {"directory": ".", "pattern": "*.py"},
            "description": "List Python files"
        }
    ]
    
    success_count = 0
    for test in test_cases:
        try:
            print(f"\nTesting: {test['description']}...")
            result = await execute_tool_call(test['name'], test['args'])
            
            # Check if result indicates error
            if isinstance(result, dict) and "error" in result:
                print_error(f"{test['name']} returned error: {result['error']}")
            else:
                print_success(f"{test['name']} executed successfully")
                print(f"  Result type: {type(result).__name__}")
                
                # Test voice formatting
                voice_output = format_tool_response_for_voice(test['name'], result)
                print(f"  Voice output: {voice_output[:100]}...")
                success_count += 1
                
        except Exception as e:
            print_error(f"{test['name']} failed with exception: {e}")
    
    print(f"\n{success_count}/{len(test_cases)} tools executed successfully")
    return success_count == len(test_cases)

async def test_lm_studio_format():
    """Test that tools can be formatted for LM Studio"""
    print_header("Testing LM Studio Compatibility")
    
    try:
        # Import OpenAI types to verify format compatibility
        from openai.types.chat import ChatCompletionToolParam
        
        # Convert our tools to OpenAI format
        tools_schema = get_tools()
        
        # Check that we can access standard_tools
        if hasattr(tools_schema, 'standard_tools'):
            print_success(f"ToolsSchema has {len(tools_schema.standard_tools)} standard tools")
            
            # Verify first tool structure
            if tools_schema.standard_tools:
                first_tool = tools_schema.standard_tools[0]
                print_success(f"First tool: {first_tool.name}")
                print(f"  Description: {first_tool.description[:50]}...")
                print(f"  Properties: {list(first_tool.properties.keys())}")
                print(f"  Required: {first_tool.required}")
        else:
            print_error("ToolsSchema missing standard_tools attribute")
            return False
            
        return True
        
    except ImportError:
        print_warning("OpenAI types not available for format verification")
        return True
    except Exception as e:
        print_error(f"Format testing failed: {e}")
        return False

async def test_llm_service():
    """Test that LLMWithToolsService can be instantiated"""
    print_header("Testing LLM Service Integration")
    
    try:
        from services.llm_with_tools import LLMWithToolsService
        
        # Set a dummy API key to satisfy OpenAI client
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
        
        # Create service instance
        service = LLMWithToolsService(
            api_key="sk-dummy-key-for-testing",
            model="test-model",
            base_url="http://localhost:1234/v1",
            max_tokens=1000
        )
        
        print_success("LLMWithToolsService instantiated successfully")
        
        # Check that tools are registered
        tools = service.get_tools_schema()
        print_success(f"Service has {len(tools.standard_tools)} tools available")
        
        # Clean up
        del os.environ["OPENAI_API_KEY"]
        
        return True
        
    except Exception as e:
        print_error(f"Service testing failed: {e}")
        return False

async def main():
    """Run all verification tests"""
    print_header("Slowcat Tool Verification")
    print("Testing tool definitions, execution, and LM Studio compatibility\n")
    
    # Run all tests
    results = []
    
    results.append(await test_tool_definitions())
    results.append(await test_tool_execution())
    results.append(await test_lm_studio_format())
    results.append(await test_llm_service())
    
    # Summary
    print_header("Test Summary")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print_success(f"All {total} test suites passed!")
        print("\n🎉 Tools are properly configured and ready for use with LM Studio")
    else:
        print_error(f"{passed}/{total} test suites passed")
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())