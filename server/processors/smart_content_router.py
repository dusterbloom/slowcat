"""
Smart Content Router

Intelligently routes tool results to appropriate content processors based on 
tool type and content characteristics. This processor acts as a dispatcher 
that enhances tool results with specialized processing while maintaining 
frame structure.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class ContentTypeDetector:
    """Detects content type from tool results"""
    
    # Tool name patterns for different content types
    TOOL_PATTERNS = {
        'browser': ['browser_', 'navigate_', 'click_', 'scroll_', 'web_'],
        'search': ['search_', 'google_', 'bing_', 'duckduckgo_'],
        'file': ['read_', 'write_', 'ls_', 'find_', 'grep_', 'file_'],
        'api': ['get_', 'post_', 'put_', 'delete_', 'fetch_', 'api_']
    }
    
    # Content signatures for detection
    CONTENT_SIGNATURES = {
        'browser': [
            r'<html[^>]*>',
            r'<div[^>]*>',
            r'<a[^>]*href=',
            r'<button[^>]*>',
            r'<input[^>]*>',
            r'Skip to',
            r'Cookie',
            r'Privacy Policy'
        ],
        'search': [
            r'"results":\s*\[',
            r'"url":\s*"https?://',
            r'"title":\s*"[^"]*"',
            r'"snippet":\s*"[^"]*"',
            r'Found \d+ results',
            r'Search results for'
        ],
        'file': [
            r'^/[^/\s]*(/[^/\s]*)*$',  # Unix paths
            r'^[A-Za-z]:\\',  # Windows paths
            r'^\s*\d+\s+[rwx-]{9}',  # ls -l output
            r'directory',
            r'file not found',
            r'permission denied'
        ],
        'api': [
            r'^{[^}]*"[^"]*":[^}]*}$',  # JSON response pattern
            r'"status":\s*(200|404|500)',
            r'"data":\s*[{\[]',
            r'"error":\s*',
            r'"response":\s*',
            r'HTTP/\d\.\d'
        ]
    }
    
    def detect_content_type(self, tool_name: str, content: str) -> str:
        """
        Detect content type based on tool name and content
        
        Args:
            tool_name: Name of the tool that produced the content
            content: The actual content to analyze
            
        Returns:
            Content type: 'browser', 'search', 'file', 'api', or 'unknown'
        """
        if not content or not isinstance(content, str):
            return 'unknown'
        
        # First check tool name patterns
        tool_name_lower = tool_name.lower()
        for content_type, patterns in self.TOOL_PATTERNS.items():
            if any(pattern in tool_name_lower for pattern in patterns):
                logger.debug(f"📝 Content type '{content_type}' detected from tool name: {tool_name}")
                return content_type
        
        # Then check content signatures
        content_sample = content[:2000].lower()  # Check first 2K chars for performance
        
        for content_type, signatures in self.CONTENT_SIGNATURES.items():
            matches = sum(1 for sig in signatures if re.search(sig, content_sample, re.IGNORECASE | re.MULTILINE))
            if matches >= 2:  # Require at least 2 signature matches
                logger.debug(f"📝 Content type '{content_type}' detected from signatures ({matches} matches)")
                return content_type
        
        # Special case: very large content is likely web content
        if len(content) > 5000 and any(indicator in content_sample for indicator in ['html', 'body', 'div', 'href']):
            logger.debug("📝 Content type 'browser' detected from size and HTML indicators")
            return 'browser'
        
        return 'unknown'


class BrowserContentProcessor:
    """Processes browser/web content"""
    
    def process(self, content: str, tool_name: str) -> str:
        """Extract meaningful content from web pages"""
        if not content:
            return content
        
        logger.debug(f"🌐 Processing browser content from {tool_name}")
        
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract meaningful text from HTML
        content = re.sub(r'<[^>]+>', ' ', content)  # Remove HTML tags
        content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', content)  # Remove HTML entities
        
        # Clean up navigation and boilerplate
        lines = content.split('\n')
        meaningful_lines = []
        
        skip_patterns = [
            r'^(skip to|log in|subscribe|search|menu|navigation)',
            r'(privacy|cookie|terms|advertisement|©)',
            r'^[A-Z\s]{10,}$',  # All caps navigation
            r'^\s*$'  # Empty lines
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue
                
            # Skip navigation/boilerplate
            if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                continue
                
            meaningful_lines.append(line)
            
            # Limit processed content
            if len(meaningful_lines) >= 20:
                break
        
        result = ' '.join(meaningful_lines)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        if len(result) > 1000:
            result = result[:1000] + "... [web content continues]"
        
        logger.debug(f"🌐 Browser content processed: {len(content)} → {len(result)} chars")
        return result


class SearchResultProcessor:
    """Processes search results"""
    
    def process(self, content: str, tool_name: str) -> str:
        """Extract and format search results"""
        if not content:
            return content
            
        logger.debug(f"🔍 Processing search results from {tool_name}")
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                if isinstance(results, list) and len(results) > 0:
                    formatted = []
                    for i, result in enumerate(results[:5]):  # Limit to top 5
                        title = result.get('title', 'No title')
                        url = result.get('url', '')
                        snippet = result.get('snippet', result.get('description', ''))
                        
                        formatted.append(f"{i+1}. {title}")
                        if snippet:
                            formatted.append(f"   {snippet[:200]}...")
                        if url:
                            formatted.append(f"   URL: {url}")
                        formatted.append("")  # Empty line
                    
                    result = '\n'.join(formatted)
                    logger.debug(f"🔍 Formatted {len(results)} search results")
                    return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Fallback: clean up raw search content
        lines = content.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            
            # Look for result patterns
            if re.match(r'^\d+\.', line) or 'http' in line.lower():
                cleaned.append(line)
            elif len(cleaned) > 0:  # Context after a numbered result
                cleaned.append(f"   {line}")
            
            if len(cleaned) >= 20:  # Limit lines
                break
        
        result = '\n'.join(cleaned)
        if len(result) > 800:
            result = result[:800] + "... [more search results]"
            
        logger.debug(f"🔍 Search results processed: {len(content)} → {len(result)} chars")
        return result


class FileContentProcessor:
    """Processes file operations"""
    
    def process(self, content: str, tool_name: str) -> str:
        """Format file operation results"""
        if not content:
            return content
            
        logger.debug(f"📁 Processing file content from {tool_name}")
        
        # Handle directory listings
        if 'ls' in tool_name.lower() or re.search(r'^\s*total \d+', content):
            lines = content.split('\n')[:20]  # Limit directory listings
            result = '\n'.join(line.strip() for line in lines if line.strip())
            if len(lines) >= 20:
                result += "\n... [more files]"
            return result
        
        # Handle file content - summarize if too long
        if len(content) > 1500:
            lines = content.split('\n')
            if len(lines) > 30:
                # Show first 15 and last 5 lines
                result = '\n'.join(lines[:15])
                result += f"\n... [{len(lines) - 20} lines omitted] ..."
                result += '\n' + '\n'.join(lines[-5:])
                logger.debug(f"📁 File content summarized: {len(lines)} lines → preview")
                return result
            else:
                # Truncate long lines
                result = content[:1500] + "... [file continues]"
                logger.debug(f"📁 File content truncated: {len(content)} → {len(result)} chars")
                return result
        
        return content


class APIResponseProcessor:
    """Processes API responses"""
    
    def process(self, content: str, tool_name: str) -> str:
        """Format API responses"""
        if not content:
            return content
            
        logger.debug(f"🔌 Processing API response from {tool_name}")
        
        # Try to parse and format JSON
        try:
            data = json.loads(content)
            # Pretty format but limit depth
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Limit size
            if len(formatted) > 800:
                # Try to extract key information
                if isinstance(data, dict):
                    key_fields = ['status', 'data', 'result', 'message', 'error']
                    summary = {}
                    for field in key_fields:
                        if field in data:
                            value = data[field]
                            if isinstance(value, (str, int, float, bool)):
                                summary[field] = value
                            elif isinstance(value, (list, dict)) and len(str(value)) < 200:
                                summary[field] = value
                            else:
                                summary[field] = f"[{type(value).__name__}]"
                    
                    if summary:
                        formatted = json.dumps(summary, indent=2, ensure_ascii=False)
                        formatted += "\n... [response contains additional fields]"
                
                # Final truncation if still too long
                if len(formatted) > 800:
                    formatted = formatted[:800] + "... [API response continues]"
            
            logger.debug(f"🔌 API response formatted: {len(content)} → {len(formatted)} chars")
            return formatted
            
        except json.JSONDecodeError:
            pass
        
        # Handle non-JSON API responses
        if len(content) > 800:
            content = content[:800] + "... [API response continues]"
            
        return content


class PassthroughProcessor:
    """Default processor that passes content through unchanged"""
    
    def process(self, content: str, tool_name: str) -> str:
        """Pass content through unchanged"""
        logger.debug(f"➡️ Passing through unknown content from {tool_name}")
        return content


class SmartContentRouter(FrameProcessor):
    """
    Smart Content Router for tool results
    
    Routes tool results to appropriate content processors based on:
    - Tool name patterns
    - Content signatures and characteristics
    - Content size and verbosity indicators
    
    Maintains LLMMessagesFrame structure while enhancing tool content.
    """
    
    def __init__(self):
        super().__init__()
        self.detector = ContentTypeDetector()
        self.processors = {
            'browser': BrowserContentProcessor(),
            'search': SearchResultProcessor(), 
            'file': FileContentProcessor(),
            'api': APIResponseProcessor(),
            'unknown': PassthroughProcessor()
        }
        self.stats = {
            'processed': 0,
            'by_type': {'browser': 0, 'search': 0, 'file': 0, 'api': 0, 'unknown': 0}
        }
        logger.info("🧠 Smart Content Router initialized")
    
    def _route_tool_content(self, content: str, tool_name: str) -> str:
        """
        Route content to appropriate processor
        
        Args:
            content: Tool result content
            tool_name: Name of the tool that generated the content
            
        Returns:
            Processed content
        """
        if not content:
            return content
        
        try:
            # Detect content type
            content_type = self.detector.detect_content_type(tool_name, content)
            
            # Route to appropriate processor
            processor = self.processors.get(content_type, self.processors['unknown'])
            enhanced_content = processor.process(content, tool_name)
            
            # Update stats
            self.stats['processed'] += 1
            self.stats['by_type'][content_type] += 1
            
            # Log routing decision
            if enhanced_content != content:
                reduction = len(content) - len(enhanced_content)
                logger.info(f"🧠 Routed to {content_type}: {tool_name} "
                           f"({len(content)}→{len(enhanced_content)} chars, {reduction:+d})")
            else:
                logger.debug(f"🧠 Routed to {content_type}: {tool_name} (no changes)")
            
            return enhanced_content
            
        except Exception as e:
            logger.error(f"🚨 Content routing failed for {tool_name}: {e}")
            return content  # Failsafe: return original content
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            enhanced_messages = []
            messages_modified = False
            
            for msg in frame.messages:
                if msg.get("role") == "tool":
                    # Get tool information
                    tool_call_id = msg.get("tool_call_id", "")
                    tool_name = msg.get("name", "unknown_tool")
                    original_content = msg.get("content", "")
                    
                    # Route content to appropriate processor
                    enhanced_content = self._route_tool_content(original_content, tool_name)
                    
                    if enhanced_content != original_content:
                        # Create enhanced message
                        enhanced_msg = {**msg, "content": enhanced_content}
                        enhanced_messages.append(enhanced_msg)
                        messages_modified = True
                    else:
                        enhanced_messages.append(msg)
                else:
                    # Pass through non-tool messages unchanged
                    enhanced_messages.append(msg)
            
            if messages_modified:
                # Send enhanced frame
                enhanced_frame = LLMMessagesFrame(messages=enhanced_messages)
                await self.push_frame(enhanced_frame, direction)
                return
        
        # Pass through all other frames unchanged
        await self.push_frame(frame, direction)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_processed': self.stats['processed'],
            'by_type': dict(self.stats['by_type'])
        }