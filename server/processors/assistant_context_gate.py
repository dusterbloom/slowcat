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
    if (not a.isspace() and a not in _PUNCT_R) and (not b.isspace() and b not in _PUNCT_L):
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
                t=(obj.get("text") or "").strip()
                if t: _maybe_space(out, t); out.append(t)
        idx=b
    if idx<len(s):
        tail=s[idx:].strip()
        if tail: _maybe_space(out, tail); out.append(tail)
    txt="".join(out)
    txt=re.sub(r"[ \t]+"," ",txt)
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
            buf: List[str]=[]; saw_assistant=False
            for m in frame.messages:
                if m.get("role") == "assistant":
                    saw_assistant=True
                    t=_clean_assistant_text(m.get("content") or "")
                    if t: _maybe_space(buf, t); buf.append(t)
            if saw_assistant:
                stitched="".join(buf).strip()
                if stitched:
                    await self.push_frame(LLMMessagesFrame(messages=[{"role":"assistant","content": stitched}]), direction)
                # swallow original polluted frame
                return
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
