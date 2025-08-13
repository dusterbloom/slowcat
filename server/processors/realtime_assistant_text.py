from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, StartFrame, EndFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger
import json


class RealtimeAssistantText(FrameProcessor):
    """
    Processor that intercepts assistant TextFrames and routes them to multiple destinations:
    1. TTS service for audio generation (individual fragments for real-time)
    2. WebSocket output for real-time client display  
    3. Assistant context aggregator for conversation history (complete response only)
    
    This prevents text fragmentation in the context while maintaining real-time TTS.
    """

    def __init__(self, assistant_aggregator=None):
        super().__init__()
        self.assistant_aggregator = assistant_aggregator
        self._started = False
        self._accumulating_response = False
        self._accumulated_text = ""
    
    def _clean_tool_calls_for_context(self, messages):
        """Clean tool calls JSON for context aggregator while preserving meaning"""
        cleaned_messages = []
        
        for m in messages:
            role = m.get("role", "")
            
            if role == "assistant" and "tool_calls" in m:
                # Convert tool calls to readable text
                tool_calls = m.get("tool_calls", [])
                call_descriptions = []
                for call in tool_calls:
                    func_name = call.get("function", {}).get("name", "unknown_function")
                    try:
                        args = json.loads(call.get("function", {}).get("arguments", "{}"))
                        arg_strs = [f"{k}={v}" for k, v in args.items()]
                        call_desc = f"Called {func_name}({', '.join(arg_strs)})"
                    except:
                        call_desc = f"Called {func_name}"
                    call_descriptions.append(call_desc)
                
                # Include any text content plus the tool call description
                content_parts = []
                if m.get("content"):
                    content_parts.append(m["content"])
                if call_descriptions:
                    content_parts.extend(call_descriptions)
                
                if content_parts:
                    cleaned_content = ". ".join(filter(None, content_parts))
                    cleaned_messages.append({"role": "assistant", "content": cleaned_content})
                    
            elif role == "tool":
                # Clean tool results - remove JSON quotes if present
                content = m.get("content", "")
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]  # Remove surrounding quotes
                cleaned_messages.append({"role": "tool", "content": content})
                
            else:
                # Pass through other messages unchanged
                cleaned_messages.append(m)
        
        return cleaned_messages

    async def process_frame(self, frame: Frame, direction=None):
        # Always call super().process_frame first
        await super().process_frame(frame, direction)

        # Handle LLM response start - begin accumulating
        if isinstance(frame, LLMFullResponseStartFrame):
            self._accumulating_response = True
            self._accumulated_text = ""
            logger.debug("🟢 Starting LLM response accumulation")
            # Forward control frames to aggregator
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward LLMFullResponseStartFrame to aggregator: {e}")
            await self.push_frame(frame, direction)
            return

        # Handle LLM response end - send accumulated text to context
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._accumulating_response and self._accumulated_text.strip():
                logger.debug(f"🔵 LLM response complete: '{self._accumulated_text[:50]}...' ({len(self._accumulated_text)} chars)")
                # Send the complete accumulated text as a single frame to context aggregator
                if self.assistant_aggregator and self._started:
                    accumulated_frame = TextFrame(text=self._accumulated_text.strip())
                    try:
                        await self.assistant_aggregator.process_frame(accumulated_frame, direction)
                        logger.debug("✅ Complete response forwarded to context aggregator")
                    except Exception as e:
                        logger.error(f"Failed to forward accumulated response to aggregator: {e}")
            
            # Reset accumulation state
            self._accumulating_response = False
            self._accumulated_text = ""
            
            # Forward control frame to aggregator
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward LLMFullResponseEndFrame to aggregator: {e}")
            await self.push_frame(frame, direction)
            return

        # Handle StartFrame
        if isinstance(frame, StartFrame):
            self._started = True
            # Forward to assistant aggregator
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward StartFrame to aggregator: {e}")
            # Continue pipeline flow
            await self.push_frame(frame, direction)
            return

        # Handle EndFrame
        if isinstance(frame, EndFrame):
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward EndFrame to aggregator: {e}")
            await self.push_frame(frame, direction)
            return

        # Handle assistant TextFrames (not user transcriptions)
        if isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
            # If we're accumulating a response, add this fragment to our buffer
            if self._accumulating_response:
                self._accumulated_text += frame.text
                # logger.debug(f"📝 Accumulated fragment: '{frame.text}' (total: {len(self._accumulated_text)} chars)")
                # DO NOT forward individual fragments to context aggregator
            else:
                # Not in accumulation mode - forward individual frame (fallback behavior)
                if self.assistant_aggregator and self._started:
                    try:
                        await self.assistant_aggregator.process_frame(frame, direction)
                        logger.debug("⚠️ Non-accumulated TextFrame forwarded to aggregator")
                    except Exception as e:
                        logger.error(f"Failed to forward TextFrame to assistant aggregator: {e}")

            # Always continue pipeline flow - TextFrame goes to TTS for real-time audio
            # This is crucial for maintaining real-time TTS playback
            await self.push_frame(frame, direction)
            return

        # Handle LLMMessagesFrame - clean tool calls before forwarding to context aggregator
        if isinstance(frame, LLMMessagesFrame) and self.assistant_aggregator and self._started:
            # Clean the messages for context while passing original to pipeline
            cleaned_messages = self._clean_tool_calls_for_context(frame.messages)
            if cleaned_messages != frame.messages:
                cleaned_frame = LLMMessagesFrame(messages=cleaned_messages)
                try:
                    await self.assistant_aggregator.process_frame(cleaned_frame, direction)
                    logger.debug("🧹 Cleaned LLMMessagesFrame forwarded to context aggregator")
                except Exception as e:
                    logger.error(f"Failed to forward cleaned LLMMessagesFrame to aggregator: {e}")
            else:
                # No cleaning needed, forward original
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward LLMMessagesFrame to aggregator: {e}")
            
            # Continue pipeline flow with original frame
            await self.push_frame(frame, direction)
            return

        # For all other frames, pass through the pipeline
        await self.push_frame(frame, direction)