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
