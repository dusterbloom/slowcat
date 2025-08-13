"""
Unified LLM service with proper Pipecat tool/function calling support
Uses FunctionSchema and ToolsSchema for OpenAI-compatible tool calling with LM Studio
"""

from typing import Optional, Dict
import json
import asyncio
from loguru import logger
from openai import NOT_GIVEN
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from tools.handlers import execute_tool_call
from tools.formatters import format_tool_response_for_voice
from tools.definitions import ALL_FUNCTION_SCHEMAS, get_tools

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    FunctionCallResultFrame,
    FunctionCallInProgressFrame,
)
from .custom_tool_parser import CustomToolParser
class LLMWithToolsService(OpenAILLMService):
    """
    Unified LLM service with proper Pipecat function calling support.
    Extends OpenAILLMService to handle tool calls with LM Studio.
    """
    
    def __init__(self, **kwargs):
        """Initialize the tool-enabled LLM service"""
        super().__init__(**kwargs)
        
        logger.info("🚀 Initializing LLMWithToolsService")
        logger.info(f"📍 Base URL: {kwargs.get('base_url')}")
        logger.info(f"🤖 Model: {kwargs.get('model')}")
        
        # Register all function handlers
        self._register_function_handlers()
        
        # Store tools schema for reference
        self._tools_schema = get_tools()
        logger.info(f"🛠️ Loaded {len(self._tools_schema.standard_tools)} tools")
        
        # Audio player reference (will be set by pipeline builder)
        self._audio_player = None
        
        # MCP tool manager (will be set by pipeline builder)
        self._mcp_tool_manager = None
    
    def set_audio_player(self, audio_player):
        """Set the audio player reference for music control"""
        self._audio_player = audio_player
        logger.info("🎵 Audio player reference set in LLM service")
    
    def set_mcp_tool_manager(self, mcp_tool_manager):
        """Set the MCP tool manager - registration will be called explicitly"""
        self._mcp_tool_manager = mcp_tool_manager
        logger.info("🔧 MCP tool manager reference set in LLM service")
    
    async def _register_mcp_tools(self):
        """Register all discovered MCP tools as function handlers (bulletproof startup)"""
        try:
            if not self._mcp_tool_manager:
                logger.warning("⚠️ MCP tool manager not set, skipping registration")
                return
            
            # Ensure tools are discovered
            await self._mcp_tool_manager.refresh_if_stale()
            
            # Get discovered tools 
            mcp_tools = await self._mcp_tool_manager.discover_tools()
            
            logger.info(f"🔧 Registering {len(mcp_tools)} MCP tools as function handlers")
            
            # Register each MCP tool
            for tool_name, description in mcp_tools.items():
                handler = self._create_mcp_tool_handler(tool_name)
                self.register_function(tool_name, handler)
                logger.debug(f"   ✅ Registered MCP tool: {tool_name}")
            
            logger.info(f"🎯 Successfully registered {len(mcp_tools)} MCP tools!")
            
        except Exception as e:
            logger.error(f"❌ Failed to register MCP tools: {e}")
    
    def _create_mcp_tool_handler(self, tool_name: str):
        """Create a function handler for an MCP tool"""
        async def mcp_handler(function_name: str, tool_call_id: str, arguments: dict, llm: any, context: any, result_callback: any):
            logger.info(f"🔧 Executing MCP tool: {tool_name} with args: {arguments}")
            try:
                # Call the tool via MCPO
                result = await self._mcp_tool_manager.call_tool(tool_name, arguments)
                await result_callback({"result": result})
            except Exception as e:
                logger.error(f"❌ MCP tool {tool_name} failed: {e}")
                await result_callback({"error": str(e)})
        
        return mcp_handler
    
    
    def _should_route_to_mcp(self, function_name: str) -> bool:
        """
        Determine if a tool should be routed to MCP or handled locally
        
        Local tools: calculate, extract_url_text, music_*, *_timed_task, get_current_time
        MCP tools: memory_*, filesystem_*, browser_*, run_javascript, etc.
        """
        if not self._mcp_tool_manager:
            return False
            
        # Local-only tools (as specified by user)
        local_tools = {
            "calculate", "get_current_time", "extract_url_text", "search_web_free",
            "play_music", "pause_music", "skip_song", "stop_music", 
            "queue_music", "search_music", "get_now_playing", 
            "set_volume", "create_playlist", "get_music_stats",
            "start_timed_task", "check_task_status", "stop_timed_task",
            "add_to_timed_task", "get_active_tasks"
        }
        
        # If it's in local tools, don't route to MCP
        if function_name in local_tools:
            return False
        
        # MCP tool patterns
        mcp_patterns = [
            "memory_", "filesystem_", "brave_", "browser_", 
            "run_javascript", "execute_javascript"
        ]
        
        # Check if function matches MCP patterns
        for pattern in mcp_patterns:
            if function_name.startswith(pattern) or pattern.rstrip("_") == function_name:
                return True
        
        # Check if tool exists in MCP manager's manifest
        if hasattr(self._mcp_tool_manager, 'tool_manifest'):
            return function_name in self._mcp_tool_manager.tool_manifest
        
        # Default to local for unknown tools
        return False
    
    def _register_function_handlers(self):
        """Register function handlers using Pipecat's register_function method"""
        
        # Register LOCAL tools with handlers
        for function_schema in ALL_FUNCTION_SCHEMAS:
            function_name = function_schema.name
            
            self.register_function(
                function_name,
                self._create_handler(function_name),
                cancel_on_interruption=True
            )
            
            logger.debug(f"✅ Registered LOCAL handler for: {function_name}")
        
        logger.info(f"🎯 Successfully registered {len(ALL_FUNCTION_SCHEMAS)} local function handlers")
    
    # MCP tools are now handled natively by LM Studio - no Pipecat registration needed
    
    # MCP pass-through handlers removed - LM Studio handles MCP tools natively
    
    def _create_handler(self, function_name: str):
        """
        Create an async handler for a specific function
        
        Args:
            function_name: Name of the function to handle
            
        Returns:
            Async handler function
        """
        async def handler(params: FunctionCallParams):
            """Handler that executes tool and returns result with enhanced error handling"""
            logger.info(f"🎯 Executing function: {function_name}")
            logger.debug(f"📥 Arguments: {params.arguments}")
            
            try:
                # Validate arguments
                if not isinstance(params.arguments, dict):
                    logger.warning(f"Invalid arguments type for {function_name}: {type(params.arguments)}")
                    params.arguments = {}
                
                # 🚀 SIMPLIFIED ROUTING: Only handle LOCAL tools here
                # MCP tools should be handled NATIVELY by LM Studio - they shouldn't reach here
                try:
                    if self._should_route_to_mcp(function_name):
                        logger.error(f"🚨 MCP tool {function_name} reached local handler - this shouldn't happen!")
                        logger.info(f"💡 MCP tools should be handled by LM Studio natively via tool_choice='auto'")
                        result = {
                            "error": f"MCP tool {function_name} reached local handler",
                            "solution": "This tool should be handled by LM Studio's native MCP integration"
                        }
                    else:
                        logger.info(f"🏠 Executing local tool: {function_name}")
                        result = await asyncio.wait_for(
                            execute_tool_call(function_name, params.arguments),
                            timeout=30.0  # 30 second timeout for tool execution
                        )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Timeout executing {function_name}")
                    error_msg = f"Tool execution timed out after 30 seconds"
                    await params.result_callback(error_msg)
                    return
                
                # Check if result contains music control data
                if isinstance(result, dict) and "_music_control" in result:
                    music_control = result["_music_control"]
                    logger.info(f"🎵 Tool returned music control: {music_control['command']}")
                    
                    # Import and push MusicControlFrame
                    from processors.audio_player_real import MusicControlFrame
                    control_frame = MusicControlFrame(
                        music_control["command"],
                        music_control.get("data", {})
                    )
                    
                    # Push frame directly to audio player if available
                    if self._audio_player:
                        await self._audio_player.push_frame(control_frame)
                        logger.info(f"🎵 Pushed control frame to audio player")
                    else:
                        logger.warning("🎵 No audio player reference, cannot send music control")
                    
                    # Handle special commands
                    if self._audio_player:
                        if music_control["command"] == "queue_multiple":
                            # Queue each song individually
                            for song in music_control["data"].get("songs", []):
                                await self._audio_player.push_frame(MusicControlFrame("queue", song))
                        elif music_control["command"] == "create_playlist":
                            # Queue all songs and potentially start playing
                            songs = music_control["data"].get("songs", [])
                            for song in songs:
                                await self._audio_player.push_frame(MusicControlFrame("queue", song))
                            
                            # If nothing playing and we should start
                            if music_control["data"].get("start_playing") and songs:
                                first_song = songs[0]
                                await self._audio_player.push_frame(MusicControlFrame("play", first_song))
                    
                    # Remove the special key from result before sending to LLM
                    result = {k: v for k, v in result.items() if k != "_music_control"}
                
                # Format result for voice if needed
                formatted_result = format_tool_response_for_voice(function_name, result)
                
                # Sanitize repeated phrases / whitespace
                import re
                if isinstance(formatted_result, str):
                    # Collapse repeated sentences
                    formatted_result = re.sub(r'\b(\w+\b)(?:\s+\1\b)+', r'\1', formatted_result, flags=re.IGNORECASE)
                    # Remove excessive whitespace
                    formatted_result = re.sub(r'\s{2,}', ' ', formatted_result).strip()
                
                # Use formatted result for LLM context to avoid bloating with raw JSON
                result_str = formatted_result
                
                # Also log the original (possibly truncated) JSON elsewhere for debugging
                if isinstance(result, (dict, list)):
                    raw_result_str = json.dumps(result, ensure_ascii=False)
                    if len(raw_result_str) > 4000:
                        raw_result_str = raw_result_str[:3900] + "... [truncated]"
                        logger.warning(f"⚠️ Truncated large raw result for {function_name}")
                    logger.debug(f"Full raw tool result for {function_name}: {raw_result_str}")
                
                # Send sanitized result back through callback
                await params.result_callback(result_str)
                
                logger.info(f"✅ Function '{function_name}' completed successfully")
                logger.debug(f"📤 Result (first 200 chars): {result_str[:200]}...")
                
                # Optional: Queue voice-formatted response
                # This allows the LLM to use the raw result while providing
                # a pre-formatted voice response
                if hasattr(params, 'llm') and formatted_result != result_str:
                    logger.debug(f"🎤 Voice format available: {formatted_result[:100]}...")
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON error in function '{function_name}': {e}")
                error_msg = f"Error: Invalid JSON response from {function_name}"
                await params.result_callback(error_msg)
            except KeyError as e:
                logger.error(f"❌ Missing required parameter in '{function_name}': {e}")
                error_msg = f"Error: Missing required parameter '{e}' for {function_name}"
                await params.result_callback(error_msg)
            except Exception as e:
                logger.error(f"❌ Unexpected error in function '{function_name}': {e}", exc_info=True)
                error_msg = f"Error executing {function_name}: {str(e)}"
                await params.result_callback(error_msg)
        
        return handler
    
    async def _process_context(self, context):
        """
        Override to ensure tools are properly set and to handle the initial
        greeting where no user message is present.
        """
        has_user_message = any(
            msg.get("role") == "user" for msg in context.messages
        )

        # If there's no user message (i.e., the bot is speaking first) and tools are present,
        # create a temporary context object without tools for this specific API call.
        if not has_user_message and context.tools is not NOT_GIVEN:
            logger.debug("No user message in context. Creating temporary tool-less context for this turn.")
            # Create a new, temporary context instance without tools.
            tool_less_context = OpenAILLMContext(
                messages=context.messages,
                tools=NOT_GIVEN
            )
            # Process the temporary context. The original context remains unchanged for future turns.
            return await super()._process_context(tool_less_context)
        else:
            # For all other cases (or if no tools are defined), process the original context.
            return await super()._process_context(context)

    def _get_completion_kwargs(self, context) -> dict:
        """Override to add tool_choice='auto' for MCP tool calling and log full request"""
        kwargs = super()._get_completion_kwargs(context)
        
        # 🚀 CRITICAL: Add tool_choice="auto" when tools are present
        if context.tools != NOT_GIVEN and context.tools:
            kwargs["tool_choice"] = "auto"
            logger.info("🔧 Added tool_choice='auto' for native MCP integration")
        
        # 🔍 DEBUG: Log the complete request being sent to LM Studio
        debug_kwargs = dict(kwargs)
        if 'messages' in debug_kwargs:
            debug_kwargs['messages'] = f"[{len(debug_kwargs['messages'])} messages]"
        if 'tools' in debug_kwargs and debug_kwargs['tools']:
            if hasattr(debug_kwargs['tools'], 'standard_tools'):
                tool_names = [t.name for t in debug_kwargs['tools'].standard_tools]
                debug_kwargs['tools'] = f"[{len(tool_names)} tools: {tool_names}]"
            elif isinstance(debug_kwargs['tools'], list):
                tool_names = [getattr(t, 'name', str(t)) for t in debug_kwargs['tools']]
                debug_kwargs['tools'] = f"[{len(tool_names)} tools: {tool_names}]"
        
        logger.info(f"📤 LM Studio Request Parameters: {debug_kwargs}")
        
        return kwargs

    async def _stream_chat_completions(self, context):
        """Override to add debugging and push immediate feedback on tool calls."""
        logger.debug("🌊 Starting streaming chat completion")
        logger.info(f"📝 Context has {len(context.messages)} messages")
        
        # Check tools - context.tools might be a list or ToolsSchema
        if context.tools != NOT_GIVEN:
            if hasattr(context.tools, 'standard_tools'):
                logger.info(f"🛠️ Tools in context: {len(context.tools.standard_tools)} tools")
            elif isinstance(context.tools, list):
                logger.info(f"🛠️ Tools in context: {len(context.tools)} tools")
            else:
                logger.info(f"🛠️ Tools in context: {type(context.tools)}")
        else:
            logger.info("🛠️ Tools in context: None")
        
        # Debug: Log message roles to check for alternation issues
        roles = [msg.get("role", "unknown") for msg in context.messages]
        logger.debug(f"Message roles sequence: {roles}")
        
        # Check for consecutive same-role messages
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1] and roles[i] in ["user", "assistant"]:
                logger.warning(f"⚠️ Consecutive {roles[i]} messages at positions {i-1} and {i}")
        
        # Log the last user message
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                logger.info(f"👤 Last user message: {msg.get('content', '')[:100]}...")
                break
        
        # Clean context messages before sending to parent
        cleaned_context = self._clean_context_for_llm(context)
        
        logger.info("🚀 About to call parent _stream_chat_completions")
        # Get stream from parent with cleaned context
        stream = await super()._stream_chat_completions(cleaned_context)
        logger.info("✅ Parent _stream_chat_completions completed, got stream")
        
        # For now, just return the parent stream to fix the async bug
        # TODO: Re-add tool call feedback functionality later
        return stream
    
    async def process_frame(self, frame: Frame, direction=None):
        """Override to intercept and parse custom tool call formats"""
        
        # Log all frames for debugging
        frame_type = type(frame).__name__
        # logger.info(f"🔍 LLMWithTools processing {frame_type}: {getattr(frame, 'text', str(frame)[:100])}...")
        
        # Log all text frames for debugging
        if isinstance(frame, TextFrame):
            logger.info(f"📝 LLMWithTools processing TextFrame: {frame.text[:100]}...")
        
        # Check if this is a text frame that might contain custom tool calls
        if isinstance(frame, TextFrame) and frame.text and '[' in frame.text:
            # Try to parse custom tool format
            tool_calls, remaining_text = CustomToolParser.parse_content_for_tools(frame.text)
            
            if tool_calls:
                logger.info(f"🔧 Intercepted custom tool calls in text: {frame.text[:100]}...")
                
                # Send immediate feedback
                await self.push_frame(TextFrame("Let me check that for you."))
                
                # Convert to proper format and execute
                for tool_call in tool_calls:
                    try:
                        # Create function call params
                        function_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        
                        # Execute the tool
                        logger.info(f"🔨 Executing parsed tool: {function_name} with {arguments}")
                        result = await execute_tool_call(function_name, arguments)
                        
                        # Format and send result
                        formatted_result = format_tool_response_for_voice(function_name, result)
                        
                        # If there's remaining text, combine it with the result
                        if remaining_text.strip():
                            formatted_result = f"{remaining_text} {formatted_result}"
                        
                        await self.push_frame(TextFrame(formatted_result))
                        
                    except Exception as e:
                        logger.error(f"Error executing parsed tool call: {e}")
                        await self.push_frame(TextFrame(f"I had trouble with that request: {str(e)}"))
                
                # Don't process the original frame further
                return
        
        # Otherwise, process normally
        await super().process_frame(frame, direction)

    def get_tools_schema(self) -> ToolsSchema:
        """
        Get the ToolsSchema object for use in context
        
        Returns:
            ToolsSchema containing all available tools
        """
        return self._tools_schema
    
    def create_context_aggregator(self, context):
        """
        Override the context aggregator creation to add logging
        """
        logger.info("🔧 LLMWithToolsService creating context aggregator")
        logger.info(f"📝 Context type: {type(context)}")
        logger.info(f"📝 Context messages count: {len(context.messages) if hasattr(context, 'messages') else 'N/A'}")
        
        # Create the aggregator using the parent method
        aggregator = super().create_context_aggregator(context)
        
        logger.info(f"✅ Context aggregator created: {type(aggregator)}")
        logger.info(f"📊 User aggregator: {type(aggregator.user())}")
        logger.info(f"📊 Assistant aggregator: {type(aggregator.assistant())}")
        
        return aggregator

    def _clean_context_for_llm(self, context):
        """Clean context messages to remove tool JSON metadata while preserving meaning"""
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        
        cleaned_messages = []
        
        logger.info(f"🧹 Cleaning context with {len(context.messages)} messages")
        
        for i, msg in enumerate(context.messages):
            role = msg.get("role", "")
            
            if role == "assistant" and "tool_calls" in msg:
                logger.info(f"🚫 Message {i}: Cleaning assistant tool_calls JSON")
                # For assistant messages with tool_calls, ONLY include the actual text content
                # NOT the tool call descriptions - those shouldn't be spoken by TTS
                content = msg.get("content", "")
                if content and content.strip():
                    # Only include the actual response text, not tool call metadata
                    cleaned_msg = {"role": "assistant", "content": content.strip()}
                    cleaned_messages.append(cleaned_msg)
                    logger.info(f"✅ Kept only text content: {content.strip()}")
                else:
                    # No actual text content in tool call message - skip it entirely
                    logger.info("✅ Skipped tool-call-only message (no text content)")
                    # Don't add anything to cleaned_messages
                
            elif role == "tool":
                logger.info(f"➡️ Message {i}: Passing through tool result for Smart Content Router")
                # Pass tool results through unchanged - let Smart Content Router handle them
                # This preserves rich content for proper processing downstream
                cleaned_messages.append(msg)
                logger.debug(f"✅ Tool result preserved for downstream processing")
                
            else:
                # User, system, or other messages - pass through unchanged
                cleaned_messages.append(msg)
        
        # Create new context with cleaned messages
        cleaned_context = OpenAILLMContext(
            messages=cleaned_messages,
            tools=context.tools if hasattr(context, 'tools') else None
        )
        
        logger.info(f"🧹 Context cleaned: {len(context.messages)} → {len(cleaned_messages)} messages")
        return cleaned_context