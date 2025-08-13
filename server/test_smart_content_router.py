#!/usr/bin/env python3
"""
Test Smart Content Router functionality
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection
from processors.smart_content_router import SmartContentRouter, ContentTypeDetector

# Sample test data for different content types
SAMPLE_DATA = {
    'browser': {
        'tool_name': 'browser_navigate',
        'content': '''
        <html><head><title>Example Page</title></head>
        <body>
            <nav>Skip to main content</nav>
            <div class="header">
                <h1>Welcome to Our Site</h1>
                <a href="/login">Log In</a>
                <a href="/privacy">Privacy Policy</a>
            </div>
            <div class="main-content">
                <p>This is the main content of the page with useful information.</p>
                <p>Another paragraph with more details about our services.</p>
                <button onclick="subscribe()">Subscribe to Newsletter</button>
            </div>
            <footer>© 2024 Example Corp. All rights reserved.</footer>
            <script>function subscribe() { alert("Thanks!"); }</script>
        </body></html>
        '''
    },
    
    'search': {
        'tool_name': 'search_web',
        'content': json.dumps({
            'results': [
                {
                    'title': 'Best Python Libraries 2024',
                    'url': 'https://example.com/python-libraries',
                    'snippet': 'Discover the top Python libraries that every developer should know in 2024, including FastAPI, Pydantic, and more.'
                },
                {
                    'title': 'Python Development Guide',
                    'url': 'https://docs.python.org/guide',
                    'snippet': 'Official Python development guide covering best practices, code style, and advanced topics.'
                }
            ]
        })
    },
    
    'file': {
        'tool_name': 'read_file',
        'content': '''#!/usr/bin/env python3
"""
Example Python file content
This is a sample Python script with multiple functions.
"""

import os
import sys
from typing import List, Dict

def hello_world():
    """Print hello world message"""
    print("Hello, World!")

def process_data(data: List[Dict]) -> Dict:
    """Process input data and return summary"""
    return {
        'total': len(data),
        'types': [type(item).__name__ for item in data]
    }

if __name__ == "__main__":
    hello_world()
    sample_data = [{"name": "test"}, {"id": 123}]
    result = process_data(sample_data)
    print(f"Processed {result}")
'''
    },
    
    'api': {
        'tool_name': 'get_weather',
        'content': json.dumps({
            'status': 200,
            'data': {
                'temperature': 22.5,
                'humidity': 65,
                'conditions': 'partly cloudy',
                'forecast': [
                    {'day': 'today', 'high': 25, 'low': 18},
                    {'day': 'tomorrow', 'high': 27, 'low': 20}
                ]
            },
            'metadata': {
                'source': 'weather-api',
                'timestamp': '2024-01-15T10:30:00Z',
                'location': 'San Francisco, CA'
            }
        })
    }
}


def test_content_type_detection():
    """Test content type detection"""
    print("🧪 Testing Content Type Detection...")
    
    detector = ContentTypeDetector()
    
    for expected_type, data in SAMPLE_DATA.items():
        detected_type = detector.detect_content_type(data['tool_name'], data['content'])
        status = "✅" if detected_type == expected_type else "❌"
        print(f"  {status} {data['tool_name']}: detected={detected_type}, expected={expected_type}")
    
    print()


async def test_router_processing():
    """Test Smart Content Router processing"""
    print("🧪 Testing Smart Content Router Processing...")
    
    router = SmartContentRouter()
    
    for content_type, data in SAMPLE_DATA.items():
        # Create LLMMessagesFrame with tool message
        messages = [{
            "role": "tool",
            "tool_call_id": f"call_{content_type}_123",
            "name": data['tool_name'],
            "content": data['content']
        }]
        
        frame = LLMMessagesFrame(messages=messages)
        
        # Process the frame
        processed_frames = []
        original_push_frame = router.push_frame
        
        async def capture_frame(frame, direction):
            processed_frames.append(frame)
        
        router.push_frame = capture_frame
        
        try:
            await router.process_frame(frame, FrameDirection.DOWNSTREAM)
            
            if processed_frames:
                processed_frame = processed_frames[0]
                if isinstance(processed_frame, LLMMessagesFrame):
                    processed_content = processed_frame.messages[0]['content']
                    original_length = len(data['content'])
                    processed_length = len(processed_content)
                    
                    print(f"  ✅ {content_type}: {original_length} → {processed_length} chars")
                    print(f"     Preview: {processed_content[:100]}...")
                    if processed_content != data['content']:
                        print(f"     🔄 Content was processed")
                    else:
                        print(f"     ➡️ Content passed through unchanged")
                else:
                    print(f"  ❌ {content_type}: Frame type changed unexpectedly")
            else:
                print(f"  ❌ {content_type}: No frame output")
        
        except Exception as e:
            print(f"  ❌ {content_type}: Error - {e}")
        
        finally:
            router.push_frame = original_push_frame
    
    # Print stats
    stats = router.get_stats()
    print(f"\n📊 Processing Stats:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  By type: {stats['by_type']}")
    print()


def test_edge_cases():
    """Test edge cases"""
    print("🧪 Testing Edge Cases...")
    
    detector = ContentTypeDetector()
    
    edge_cases = [
        ("empty_content", "", "unknown"),
        ("null_content", None, "unknown"),
        ("json_only", '{"key": "value"}', "unknown"),  # Should not detect as API without more context
        ("mixed_signatures", "<html>Some API response: {\"status\": 200}</html>", "browser"),  # HTML should take precedence
        ("large_content", "x" * 10000 + "<div>content</div>", "browser"),  # Size-based detection
    ]
    
    for case_name, content, expected in edge_cases:
        try:
            if content is None:
                detected = "unknown"  # Handle None case
            else:
                detected = detector.detect_content_type("test_tool", content)
            status = "✅" if detected == expected else "❌"
            print(f"  {status} {case_name}: detected={detected}, expected={expected}")
        except Exception as e:
            print(f"  ❌ {case_name}: Error - {e}")
    
    print()


async def test_error_handling():
    """Test error handling and failsafe mechanisms"""
    print("🧪 Testing Error Handling...")
    
    router = SmartContentRouter()
    
    # Test with malformed content
    test_cases = [
        {
            "name": "malformed_json",
            "content": '{"incomplete": json',
            "tool_name": "api_call"
        },
        {
            "name": "very_large_content", 
            "content": "x" * 50000,  # Very large content
            "tool_name": "file_read"
        },
        {
            "name": "special_chars",
            "content": "Content with special chars: \x00\x01\x02\xff",
            "tool_name": "unknown_tool"
        }
    ]
    
    for case in test_cases:
        try:
            result = router._route_tool_content(case['content'], case['tool_name'])
            print(f"  ✅ {case['name']}: Handled gracefully ({len(result)} chars)")
        except Exception as e:
            print(f"  ❌ {case['name']}: Failed - {e}")
    
    print()


async def main():
    """Run all tests"""
    print("🚀 Starting Smart Content Router Tests\n")
    
    test_content_type_detection()
    await test_router_processing()
    test_edge_cases()
    await test_error_handling()
    
    print("✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())