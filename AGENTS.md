# Repository Guidelines

## Project Structure & Module Organization
- `server/`: Python voice-agent backend. Key areas: `core/` (service factory, pipelines), `processors/` (audio, memory, music, context), `services/` (STT/LLM/TTS), `config/`, `tests/`, and utility scripts in `scripts/`.
- `client/`: Next.js (TypeScript) web UI in `src/app/` with ESLint and Tailwind.
- `docs/`: Architecture, setup, feature guides (start with `docs/QUICK_START.md`).
- `assets/`, `data/`: Static assets and example data.

## Build, Test, and Development Commands
- Backend run: `cd server && ./run_bot.sh` (start local agent backend).
- Backend tests: `cd server && python -m pytest -q` (use `-k <pattern>` to filter).
- Backend deps: `cd server && pip install -r requirements.txt`.
- Client dev: `cd client && npm run dev` (Next.js dev server).
- Client prod: `cd client && npm run build && npm start`.
- Client lint: `cd client && npm run lint`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, prefer type hints and docstrings. Files use `snake_case.py`; classes use `PascalCase`. Keep imports local to feature areas and avoid circular deps across `core/`, `processors/`, `services/`.
- TypeScript/React: Components `PascalCase.tsx` under `client/src/app/`; hooks/utilities `camelCase.ts`. Keep JSX lean; colocate styles in `globals.css` or component‑level CSS.

## Testing Guidelines
- Framework: `pytest` in `server/`.
- Location/naming: place unit tests in `server/tests/` as `test_*.py`.
- Scope: prefer small, deterministic tests for processors/services; add fixtures under `server/tests/fixtures/`.
- Run: `cd server && python -m pytest -q`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`).
- PRs: include clear description, rationale, linked issues, before/after notes (logs/screenshots for UI), and test coverage or steps.
- Checks: ensure `pytest` passes and `npm run lint` is clean before requesting review.

## Security & Configuration Tips
- Copy `server/.env.example` to `server/.env` and set keys/ports. Never commit `.env`.
- macOS: grant microphone permission for local voice input.
- Useful env flags: `USER_ID`, `FACTS_DB_PATH`, `PIPELINE_IDLE_TIMEOUT_SECS`.

## Architecture Overview (Memory & Context)
- Memory: `server/memory/facts_graph.py` stores facts with decay/promote thresholds; facts extracted via `spacy_fact_extractor.py`.
- Retrieval/routing: `server/memory/query_router.py` orchestrates facts/tape/embeddings; create via `create_smart_memory_system()`.
- Context: `server/processors/smart_context_manager.py` builds a fixed token budget and filters noisy frames.

