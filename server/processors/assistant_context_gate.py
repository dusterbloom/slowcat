# processors/assistant_context_gate.py
import json, re
from typing import Any, List, Tuple
from loguru import logger
from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame
import pipecat.processors.frame_processor as pcfp

FrameDirection = pcfp.FrameDirection

_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*[\s\S]*?```", re.MULTILINE)
_PUNCT_R = set(",.!?:;)]}'”»")
_PUNCT_L = set(",.!?:;([{'“«")

def _find_json_object_spans(s: str) -> List[Tuple[int, int]]:
    spans=[]; i=0; d=0; st=None; ins=False; esc=False
    while i<len(s):
        ch=s[i]
        if ins:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=='"': ins=False
        else:
            if ch=='"': ins=True
            elif ch=='{':
                if d==0: st=i
                d+=1
            elif ch=='}':
                if d>0:
                    d-=1
                    if d==0 and st is not None:
                        spans.append((st,i+1)); st=None
        i+=1
    return spans

def _looks_like_proto(obj: Any) -> bool:
    if not isinstance(obj, dict): return False
    if str(obj.get("protocol","")).lower().startswith("tts"): return True
    sig={"message_id","chunk_index","total_chunks","message_type","is_final","timestamp","text"}
    return len(sig.intersection(obj.keys()))>=3

def _maybe_space(buf: List[str], nxt: str):
    if not buf or not nxt: return
    prev=buf[-1]
    if not prev: return
    a=prev[-1]; b=nxt[0]
    # Only add space if we're joining two words that need separation
    # Be more conservative to avoid over-spacing
    if (a.isalnum() and b.isalnum()) or (a in ".,!?;:" and b.isalnum()):
        buf.append(" ")

def _is_pure_json(t: str) -> bool:
    t=(t or "").strip()
    if not (t.startswith("{") or t.startswith("[")): return False
    try: json.loads(t); return True
    except: return False

def _sanitize_blob_text(s: str) -> str:
    if not s: return ""
    s=_JSON_FENCE_RE.sub("", s)
    spans=_find_json_object_spans(s)
    if not spans:
        t=s.strip()
        return t if (t and not _is_pure_json(t)) else ""
    out: List[str]=[]; idx=0
    for a,b in spans:
        if idx<a:
            pre=s[idx:a].strip()
            if pre: _maybe_space(out, pre); out.append(pre)
        chunk=s[a:b]
        try: obj=json.loads(chunk)
        except: pass
        else:
            if _looks_like_proto(obj):
                # Skip protocol messages entirely - don't extract text from them
                pass  
            else:
                # For non-protocol JSON, we might want to preserve some content
                # But for now, skip all JSON to keep text clean
                pass
        idx=b
    if idx<len(s):
        tail=s[idx:].strip()
        if tail: _maybe_space(out, tail); out.append(tail)
    txt="".join(out)
    # More conservative whitespace cleanup - preserve natural spacing
    txt=re.sub(r"[ \t]{2,}"," ",txt)  # Only collapse multiple spaces, not single spaces
    txt=re.sub(r"\s+\n","\n",txt)
    txt=re.sub(r"\n\s+","\n",txt)
    return txt.strip()

def _clean_assistant_text(text: str) -> str:
    if not text: return ""
    t=_sanitize_blob_text(text)
    return "" if _is_pure_json(t) else t

class AssistantContextGate(pcfp.FrameProcessor):
    """Place immediately BEFORE context_aggregator.assistant(). Swallows polluted assistant frames and injects one clean frame."""
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction); return

        if isinstance(frame, LLMMessagesFrame):
            logger.info(f"🎯 AssistantContextGate processing LLMMessagesFrame with {len(frame.messages)} messages")
            cleaned_messages = []
            messages_modified = False
            
            for i, m in enumerate(frame.messages):
                role = m.get("role", "")
                logger.debug(f"  Message {i}: role={role}, has_tool_calls={'tool_calls' in m}, has_tool_call_id={'tool_call_id' in m}")
                
                if role == "assistant":
                    # Handle assistant messages - convert tool_calls JSON to clean text
                    if "tool_calls" in m:
                        logger.info("🚫 BLOCKING assistant message with tool_calls JSON!")
                        # Convert tool calls to readable text for context
                        tool_calls = m.get("tool_calls", [])
                        call_descriptions = []
                        for call in tool_calls:
                            func_name = call.get("function", {}).get("name", "unknown_function")
                            try:
                                args = json.loads(call.get("function", {}).get("arguments", "{}"))
                                # Format arguments cleanly
                                arg_strs = [f"{k}={v}" for k, v in args.items()]
                                call_desc = f"Called {func_name}({', '.join(arg_strs)})"
                            except:
                                call_desc = f"Called {func_name}"
                            call_descriptions.append(call_desc)
                        
                        # Include any text content plus the tool call description
                        content_parts = []
                        if m.get("content"):
                            content_parts.append(_clean_assistant_text(m["content"]))
                        if call_descriptions:
                            content_parts.extend(call_descriptions)
                        
                        if content_parts:
                            cleaned_content = ". ".join(filter(None, content_parts))
                            cleaned_messages.append({"role": "assistant", "content": cleaned_content})
                            logger.info(f"✅ Converted to: {cleaned_content}")
                        
                        messages_modified = True
                    else:
                        # Regular assistant message - process normally
                        content = m.get("content") or ""
                        cleaned_text = _clean_assistant_text(content)
                        if cleaned_text:
                            cleaned_messages.append({"role": "assistant", "content": cleaned_text})
                
                elif role == "tool":
                    logger.info("🚫 BLOCKING tool result with tool_call_id!")
                    # Clean up tool results - remove metadata, keep just the result content
                    content = m.get("content", "")
                    # Clean the content but keep the tool role for LLM context
                    cleaned_content = _clean_assistant_text(content)  # Remove JSON quotes if present
                    if cleaned_content:
                        cleaned_messages.append({"role": "tool", "content": cleaned_content})
                        logger.info(f"✅ Cleaned tool result: {cleaned_content}")
                    messages_modified = True
                
                else:
                    # User, system, or other messages - pass through unchanged
                    cleaned_messages.append(m)
            
            if messages_modified:
                logger.info(f"🧹 AssistantContextGate modified {len(frame.messages)} → {len(cleaned_messages)} messages")
                # Send cleaned frame if we made changes
                if cleaned_messages:
                    await self.push_frame(LLMMessagesFrame(messages=cleaned_messages), direction)
                # Swallow original polluted frame
                logger.info("🗑️ SWALLOWED original polluted frame")
                return
            
            # No changes needed - pass through original frame
            logger.debug("➡️ Passing through unmodified frame")
            await self.push_frame(frame, direction); return

        if isinstance(frame, TextFrame):
            # Check frame type by class name to avoid import issues
            frame_class_name = frame.__class__.__name__
            
            # Skip TranscriptionFrames (user input) - pass through unchanged
            if frame_class_name == 'TranscriptionFrame':
                await self.push_frame(frame, direction)
                return
            
            # Block TTSTextFrames completely - they're just for the client display
            if frame_class_name == 'TTSTextFrame':
                # Silently block TTSTextFrames - they're protocol messages for the client
                return  # Don't pass through at all
            
            # For regular TextFrames from the assistant, pass through
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
