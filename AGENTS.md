# Repository Guidelines

## Project Structure & Module Organization
- `server/`: Python voice agent backend. Key areas: `core/` (service factory, pipelines), `processors/` (audio, memory, music, context), `services/` (STT/LLM/TTS), `config/`, `tests/`, and utility scripts in `scripts/`.
- `client/`: Next.js (TypeScript) web UI in `src/app/`. ESLint and Tailwind configured.
- `docs/`: Architecture, setup, and feature guides. Start with `docs/QUICK_START.md`.
- `assets/`, `data/`: Static assets and example data.

## Build, Test, and Development Commands
- Server (Python):
  - `cd server && ./run_bot.sh`: Start local agent backend.
  - `cd server && python -m pytest`: Run tests (discovers `tests/` and `test_*.py`).
  - `cd server && pip install -r requirements.txt`: Install dependencies.
- Client (Next.js):
  - `cd client && npm run dev`: Start the web UI.
  - `cd client && npm run build && npm start`: Production build/start.
  - `cd client && npm run lint`: Lint client code.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, prefer type hints and docstrings (see `server/core/*.py`). Modules and files use `snake_case.py`; classes use `PascalCase`.
- TypeScript/React: Components `PascalCase.tsx` under `client/src/app/`; hooks/utilities `camelCase.ts`. Keep JSX lean and colocate styles in `globals.css` or component‑level CSS as appropriate.
- Imports: Prefer relative within feature areas; avoid circular deps across `core/`, `processors/`, and `services/`.

## Testing Guidelines
- Framework: `pytest` (server). Place unit tests in `server/tests/` and name files `test_*.py`.
- Scope: Favor small, deterministic tests around processors and services. Add fixtures for audio/text samples under `server/tests/fixtures/` when needed.
- Run: `cd server && python -m pytest -q` (use `-k <pattern>` to filter).

## Commit & Pull Request Guidelines
- Commits: Follow Conventional Commits seen in history, e.g., `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`.
- PRs: Include clear description, rationale, and scope; link issues; add before/after notes (logs, screenshots for UI); list test coverage and manual verification steps.
- Checks: Ensure `pytest` passes and `npm run lint` is clean before requesting review.

## Security & Configuration Tips
- Copy `server/.env.example` to `server/.env` and adjust ports/keys. Do not commit `.env`.
- macOS microphone permissions are required for local voice input.

## Architecture Overview (Memory & Context)
- Facts Graph: `server/memory/facts_graph.py` stores structured facts in SQLite with fidelity levels S4→S0 and natural decay. Core APIs: `reinforce_or_insert`, `get_facts`, `search_facts`, `decay_facts`. Tunables: `DECAY_HALF_LIFE_S`, `PROMOTE_THRESH`, `DEMOTE_THRESH`.
- Fact Extraction: `server/memory/spacy_fact_extractor.py` parses utterances (spaCy) to subject–predicate–value facts; integrated via `extract_facts_from_text()` and `SmartMemorySystem.store_facts(text)`.
- Retrieval & Routing: `server/memory/query_classifier.py` and `query_router.py` classify intent and route to stores (`facts`, `tape`, `embeddings`). Usage:
  - `from memory import create_smart_memory_system`
  - `mem = create_smart_memory_system(); await mem.process_query("what's my dog's name?")`
- Context Management: `server/processors/smart_context_manager.py` builds a fixed 4096‑token context using `TokenBudget` (system prompt, relevant facts, recent window, current input). It asynchronously extracts and stores facts, injects only curated messages (`LLMMessagesFrame`), and prevents unbounded growth. See also `processors/context_filter.py` (blocks streaming frames from polluting context).
- Extend/Modify:
  - New facts: adapt extractor rules or map additional entities in `spacy_fact_extractor.py`.
  - Decay/retention: tune constants in `facts_graph.py`.
- Budgeting: adjust `TokenBudget` in `smart_context_manager.py`.
- New stores/routes: plug into `create_query_router()`.

## Environment Flags (Context, Sessions, STT)

- USER_ID: Optional static user id when speaker recognition is disabled. Empty uses detected speaker id.
- FACTS_DB_PATH: Absolute path to the facts DB (ensures session/facts persistence across runs).
- PIPELINE_IDLE_TIMEOUT_SECS: 0 disables idle cancellation; set seconds to enable.
- SHERPA_RULE2_MIN_TRAILING_SILENCE: Endpoint detector trailing silence (s) for Sherpa; 1.0 recommended.
- SHERPA_MIN_FINAL_WORDS / SHERPA_ENDPOINT_DEBOUNCE_MS: Short-final debounce (words threshold, hold ms).

### SmartContextManager Feature Toggles

- ENABLE_SPELLING_HINTS (default: false)
  - Detects spelled-out names (e.g., “S E R R A M A N N A”) and adds a compact hint so the model treats it as a proper name.
  - Off by default for general deployments.

- ENABLE_LOCATION_SPELLING_HINTS (default: false)
  - Location-focused variant; when the conversation is about location/town names, applies stronger hints to stop re-asking and treat the spelled sequence as a place name to confirm.

- ENABLE_GREETING_FALLBACK (default: false)
  - If the previous-session summary is too noisy/empty, injects a simple “Hello!” on connect to reset tone.

### TapeStore Noise Guard (defaults enforce it)

- TAPE_MIN_USEFUL_WORDS (default: 3)
- TAPE_MIN_USEFUL_LEN (default: 15)

Persist a turn to TapeStore only if any of the following holds:
- It is a question (ends with “?”), or
- It contains at least TAPE_MIN_USEFUL_WORDS alpha words, or
- Its length is at least TAPE_MIN_USEFUL_LEN characters.

Set both to 0 to disable filtering (not recommended).
