# Broker GPT — Agent Notes

## Way of Work
- Fail fast: missing config/data or API failures must surface clear exceptions rather than silent fallbacks.
- Validate inputs/outputs: check required columns for every CSV read or written by the engine/server.
- Keep behaviour deterministic; avoid hidden environment switches or implicit defaults.
- Prefer simple, well-documented logic backed by references or existing conventions; no speculative trading algorithms.

## Scope Guardrails
- The repository contains the data engine (`scripts/engine`), supporting fetchers/indicators, and a Playwright-based TCBS portfolio scraper (`scripts/scrapers/tcbs.py`). Do not reintroduce order generation, policy overlays, tuners, or patch aggregators.
- All generated analytics live under `out/`. Portfolio sources and fill history stay under `data/`. Never mutate the source portfolio CSVs inside the engine.
- Paths must resolve relative to the repo root unless explicitly absolute; keep configuration in `config/data_engine.yaml` unless requirements change.

## Testing & Tooling
- Run `./broker.sh tests` after touching Python code.
- Keep dependencies minimal; do not add heavy packages unless strictly required for data processing.
- Avoid long-running external calls in tests; use fakes/mocks where possible.

## Documentation
- Reflect behavioural changes in `README.md` (usage-facing) and `SYSTEM_DESIGN.md` (architecture details) whenever logic or contracts change.
- Keep instructions focused on the current lightweight workflow; remove stale references when cleaning up features.

### Prompt Sample
- Canonical sample prompt lives at `prompts/SAMPLE_PROMPT.md` (README only links to it).
- When presets/schema change, update this file to instruct reading the `PresetDescription` column from each `out/presets/<preset>.csv` and adjust example file list if needed.
- Do not re-embed the long prompt into `README.md` to avoid drift.
