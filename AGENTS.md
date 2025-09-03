# Repository Guidelines

## Project Structure & Module Organization
- server/: Python voice‑agent backend. Key areas: core/ (service factory, pipelines), processors/ (audio, memory, music, context), services/ (STT/LLM/TTS), config/, tests/, and scripts/.
- client/: Next.js (TypeScript) web UI in src/app/ with ESLint and Tailwind.
- docs/: Architecture, setup, feature guides (start at docs/QUICK_START.md).
- assets/, data/: Static assets and example data.

## Build, Test, and Development Commands
- Backend deps: `cd server && pip install -r requirements.txt` — install Python deps.
- Backend run: `cd server && ./run_bot.sh` — start local agent backend.
- Backend tests: `cd server && python -m pytest -q` (use `-k <pattern>` to filter).
- Client dev: `cd client && npm run dev` — start Next.js dev server.
- Client prod: `cd client && npm run build && npm start` — build and serve.
- Client lint: `cd client && npm run lint` — run ESLint.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, prefer type hints/docstrings. Files use snake_case.py; classes use PascalCase. Keep imports local to feature areas; avoid circular deps across core/, processors/, services/.
- TypeScript/React: Components `PascalCase.tsx` under `client/src/app/`; hooks/utilities `camelCase.ts`. Keep JSX lean; use Tailwind and `globals.css` for styles.

## Testing Guidelines
- Framework: pytest in `server/`.
- Location/naming: place unit tests in `server/tests/` as `test_*.py`; fixtures in `server/tests/fixtures/`.
- Run: `cd server && python -m pytest -q`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`).
- PRs: include description, rationale, linked issues, before/after notes (logs/screenshots for UI), and tests or verification steps. Ensure `pytest` passes and `npm run lint` is clean.

## Security & Configuration Tips
- Copy `server/.env.example` to `server/.env` and set keys/ports. Never commit `.env`.
- macOS: grant microphone permission for local voice input.
- Useful env flags: `USER_ID`, `FACTS_DB_PATH`, `PIPELINE_IDLE_TIMEOUT_SECS`.

## Architecture & Workflow Notes
- Memory & context: see `server/memory/facts_graph.py`, `spacy_fact_extractor.py`, `server/memory/query_router.py` (via `create_smart_memory_system()`), and `server/processors/smart_context_manager.py`.
- Tasks via Backlog.md: use the CLI only. Examples: `backlog task list --plain`, `backlog task 42 --plain`, `backlog task edit 42 --check-ac 1`. Never edit `backlog/tasks/` files directly.

