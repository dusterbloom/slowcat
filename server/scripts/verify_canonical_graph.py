#!/usr/bin/env python3
import asyncio
import os
import sys
from loguru import logger


def rows(res):
    try:
        if isinstance(res, list):
            r = res[0]
            return r.get('result') if isinstance(r, dict) else res
    except Exception:
        return []
    return []


async def main() -> int:
    os.environ.setdefault('DSPY_EXTRACTION_MODEL', 'qwen2.5-7b-instruct')
    try:
        from processors.surreal_message_store import SurrealMessageStore
        from memory.surreal_connection import get_surreal_connection
    except Exception as e:
        logger.error(f"Imports failed: {e}")
        return 2

    speaker = os.getenv('VERIFY_SPEAKER', 'verify_user')
    text = os.getenv('VERIFY_TEXT', "My dog's name is Potola.")

    store = SurrealMessageStore(speaker_id=speaker, auto_create_session=True)
    await store._handle_user_message(text)
    await asyncio.sleep(2.5)

    conn = get_surreal_connection()
    await conn.ensure_connected()

    sess = await conn.db.query(
        "SELECT * FROM sessions WHERE speaker_id = $sp ORDER BY start_time DESC LIMIT 1",
        {"sp": speaker},
    )
    sr = rows(sess)
    assert sr, "No session"
    sid = sr[0]['id']

    msgs = rows(await conn.db.query("SELECT <-message_belongs_to<-messages[*] FROM $sid", {"sid": sid}))
    assert msgs, "No message_belongs_to"
    mid = msgs[0]['id']

    knows = rows(await conn.db.query("SELECT <-knowledge_from<-knowledge[*] FROM $sid", {"sid": sid}))
    assert knows, "No knowledge_from"
    kid = knows[0]['id']

    about = rows(await conn.db.query("SELECT ->knowledge_about->entity[*] FROM $kid", {"kid": kid}))
    assert about, "No knowledge_about"

    involves = rows(await conn.db.query("SELECT ->session_involves->entity[*] FROM $sid", {"sid": sid}))
    assert involves, "No session_involves"

    mentions = rows(await conn.db.query("SELECT <-entity_mentioned_in<-entity[*] FROM $sid", {"sid": sid}))
    assert mentions, "No entity_mentioned_in"

    mcontains = rows(await conn.db.query("SELECT ->message_contains->knowledge[*] FROM $mid", {"mid": mid}))
    assert mcontains, "No message_contains"

    print("OK:", dict(messages=len(msgs), knowledge=len(knows), about=len(about),
                    involves=len(involves), mentions=len(mentions), message_contains=len(mcontains)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(asyncio.run(main()))
    except AssertionError as e:
        logger.error(e)
        sys.exit(3)
    except Exception as e:
        logger.error(f"Verifier failed: {e}")
        sys.exit(4)

