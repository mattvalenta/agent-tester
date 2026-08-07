# Changelog

Material changes to architecture, integrations, data contracts, runtime,
deployment, safety, and shared knowledge. Newest first.

This is not a Git log. Routine code churn, dependency bumps, UI tweaks, and
additions to the flow allowlist do not belong here unless they change a contract,
a boundary, or an operational procedure. Date entries only when the date is known.

## 2026-08-07 — Shared agent knowledge base established

Added `AGENTS.md` as the portable shared contract, `CLAUDE.md` as a single-line
pointer to it, and `docs/agent/` containing `ARCHITECTURE.md`, `OPERATIONS.md`,
`INVARIANTS.md`, `OPEN_QUESTIONS.md`, and this changelog. These files are
committed so every developer and coding agent receives and updates the same
context rather than relying on local or private notes.

Documented from current code and config: the three-hop data flow (browser →
Express → GitHub Contents API, `python3` converter subprocess, and OpenRouter),
the HTTP contract, the prompt assembly order, the ephemeral in-memory session
model with its 20-message cap, environment variable and secret *names*, the
read-only relationship to `mattvalenta/voice_agent_prompts`, the absence of
tests, CI, deployment, and Python manifests, and the safety and authorization
boundaries governing edits here.

No source, workflow, manifest, schema, or runtime behavior was changed.
