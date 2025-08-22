# Voice Mood Capture – Notes and Safe Stash Plan

Created: 2025-08-21 (Late)
Status: Proposal + Safe staging guidance

## Context

- Idea: Capture paralinguistic “mood” features from the user’s voice (energy, pitch, arousal) per utterance and store alongside tape entries.
- Branch: Not the right branch right now; late; we still need to test the latest staged changes before merging broader work.
- Goal: Park only the new, untracked files so the working tree for tracked files remains intact for testing.

## What Was Added (Untracked)

- `server/processors/mood_analyzer.py` (new):
  - Buffers user audio during a VAD turn; computes energy, ZCR, F0 (autocorr), arousal; classifies mood (neutral/calm/engaged/excited/stressed).
  - On stop speaking, attaches a compact meta dict to the most recent tape entry.
  - Helper: `attach_mood_analyzer(tee, vad_bridge, tape_store)` to wire it quickly.

- `server/scripts/precache_models.py` (new):
  - Pre-caches embedding and optional cross‑encoder models for DTH.
  - Optional utility; does not affect core flow.

These two are safe to stash independently without impacting tracked files.

## Minimal DB Support (Tracked change already present)

- `server/memory/tape_store.py`: Added `entry_meta` table and `add_entry_meta()/get_entry_meta()` helpers. This is already in tracked changes and can remain for testing; it’s dormant unless the analyzer writes to it.

## Why Stash Now

- Not on the right branch; avoid accidental commit of new features.
- We want to test currently staged (tracked) changes first (e.g., DTH advancements) without carrying along new processors/scripts.

## Safe Stash Plan

- Stash only untracked files:
  - `server/processors/mood_analyzer.py`
  - `server/scripts/precache_models.py`

Examples:
- `git stash push -m "stash: mood analyzer + precache (untracked)" -- server/processors/mood_analyzer.py server/scripts/precache_models.py`

This keeps tracked edits in place for testing.

## Next Steps (When Ready)

1. Testing
   - Basic: Verify analyzer attaches meta after user turns and `tape_store.get_entry_meta(ts)` returns data.
   - Performance: Ensure buffering/analysis stays under ~5–10ms per utterance on typical machines.
2. Optional Enhancements
   - Speaking rate: fuse STT word count and duration.
   - SurrealDB path: upsert mood into `tape.metadata` when using SurrealDB backend.
   - SER model: optionally swap heuristics for a pretrained speech emotion recognizer.
3. UI/Prompting
   - Surface recent mood in UI or prompt for tone-adaptive responses (guardrails required).

## Rollback/Notes

- This doc is a pointer. No migration needed; the `entry_meta` table is additive.
- Re-apply the stashed files when the branch is correct: `git stash list` → `git stash apply <ref>`.

---

Author: Peppi (late-night note)
