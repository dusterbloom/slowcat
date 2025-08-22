import asyncio
from typing import Callable, Awaitable, Dict, Optional

_finalizers: Dict[str, Callable[[], Awaitable[Optional[str]]]] = {}

def register_session_finalizer(pc_id: Optional[str], coro: Callable[[], Awaitable[Optional[str]]]):
    if not pc_id or not callable(coro):
        return
    _finalizers[pc_id] = coro

async def notify_session_finalizer(pc_id: Optional[str]):
    if not pc_id:
        return
    coro = _finalizers.pop(pc_id, None)
    if coro is None:
        return
    try:
        await coro()
    except Exception:
        # Best-effort; avoid raising during transport cleanup
        pass
