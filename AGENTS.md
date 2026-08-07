# AGENTS.md — agent-tester

Portable shared contract for human developers and coding agents working in
`mattvalenta/agent-tester`. Default branch: `main`.

## Identity

A local **voice-agent flow tester**: a developer tool that reproduces the
production Pipecat single-prompt setup in a browser chat window so a flow can be
exercised against an LLM without placing a phone call.

It takes a flow JSON definition, converts it to a single system prompt with the
same Python converter used in production, stitches it together with the shared
`ANLIS.md` and `guardrails.md` prompts, and then relays typed turns to an LLM
through OpenRouter. Output is an on-screen transcript with per-response comments
that can be downloaded as JSON.

Repository class: **TOOLING**. It is a testing harness, not a product surface —
it serves no end users, holds no persistent data, and is not deployed.

## Ownership and boundaries

- Owner: Matt (`mattvalenta`). Treat him as the approver for anything below.
- **Owned here:** the Express server and its HTTP API, the browser UI, the flow
  allowlist, and the local copy of the prompt converter.
- **Not owned here:** flow JSON and stitch prompts (owned by
  `mattvalenta/voice_agent_prompts`), the production Pipecat runtime, model
  hosting (OpenRouter), and the models themselves.
- Do not add product behavior, persistence, auth, or deployment to this tool
  without explicit approval; scope creep changes what this repository *is*.

## Architecture map

Single Node process, one Python subprocess per conversion, no database.

```
browser (public/index.html)
   │  fetch /api/flows, /api/session, /api/chat, /api/comment, /api/transcript
   ▼
server.js  ── Express, in-memory `sessions` object
   ├─► GitHub Contents API ──► mattvalenta/voice_agent_prompts
   │        flow JSON + stitch_prompts/{ANLIS,guardrails}.md   (read-only)
   ├─► python3 subprocess ──► prompt_flow_converter.py
   │        flow JSON (stdin) → system prompt text (stdout)
   └─► OpenRouter /api/v1/chat/completions
            [system prompt, history, user turn] → assistant turn
```

Session assembly happens once per session; every chat turn replays the stored
system prompt plus history. See [docs/agent/ARCHITECTURE.md](docs/agent/ARCHITECTURE.md).

## Important paths

| Path | Role |
| --- | --- |
| `server.js` | Express app, session store, all three integrations, flow allowlist |
| `prompt_flow_converter.py` | Flow JSON → single system prompt; array and dict flow formats |
| `public/index.html` | Entire UI — markup, styles, and client script in one file |
| `package.json` | `npm start` → `node server.js`; deps: express, cors, dotenv |
| `package-lock.json` | Lockfile v3; use `npm ci` for reproducible installs |
| `docs/agent/` | Durable knowledge base (this document links it below) |

There is no test directory, no CI workflow, no Dockerfile, no schema, and no
Python manifest in this repository.

## Integrations and data flow

All three integrations are **outbound only**; nothing calls into this service.

1. **GitHub Contents API** — HTTPS GET to
   `https://api.github.com/repos/mattvalenta/voice_agent_prompts/contents/<path>`,
   base64 body decoded to UTF-8. Auth is a bearer token read once at process
   start from `gh auth token`; if that command fails the token is an empty
   string and requests will fail at call time. Source of truth for flows and
   stitch prompts is `voice_agent_prompts`, never this repository. No retry and
   no caching; a non-OK status fails `POST /api/session` with 500.
2. **OpenRouter** — HTTPS POST to
   `https://openrouter.ai/api/v1/chat/completions` with bearer
   `OPENROUTER_API_KEY`, `temperature: 0.7`, and identifying `HTTP-Referer` /
   `X-Title` headers. **Not idempotent** — every call is a billable inference.
   No retry; failures surface as 500 on `POST /api/chat` and the turn is not
   appended to history.
3. **`prompt_flow_converter.py` subprocess** — `python3 -c` spawned with
   `cwd` at the repository root, flow JSON on stdin, prompt text on stdout. A
   non-zero exit rejects the conversion and fails session creation.

Environment variables read: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `PORT`
(loaded via `dotenv`). Names only — never commit values. `.env` is gitignored.

## Data ownership

This repository owns **no durable data**. Sessions live in a plain in-memory
object and are lost on restart; history is truncated to the last 20 messages;
transcripts exist only when a user downloads them client-side. Flow content is
fetched fresh on every session and is authoritative upstream. Consequently there
is nothing here to migrate, back up, or delete.

## Runtime and deployment

Local developer runtime only: one long-lived `node server.js` process listening
on `PORT` (default `3100`), serving `public/` statically. It is not containerized,
not scheduled, and has no deploy pipeline or hosted environment. Python 3 must be
on `PATH` for conversion to work.

## Setup, run, and targeted verification

```bash
npm ci                     # or: npm install
# .env (untracked): OPENROUTER_API_KEY=…  [optional] OPENROUTER_MODEL=…  PORT=…
gh auth login              # supplies the GitHub token via `gh auth token`
npm start                  # http://localhost:3100
```

Prefer the **smallest deterministic check** that covers the change, not a full
run. There is no test suite to invoke.

- `node --check server.js` after any server edit.
- `python3 -m py_compile prompt_flow_converter.py` after any converter edit.
- Converter behavior: pipe a small inline flow dict through
  `convert_generic_flow_to_prompt` and read the output. This imports `loguru`,
  which no manifest in this repository declares — see
  [docs/agent/OPEN_QUESTIONS.md](docs/agent/OPEN_QUESTIONS.md).
- End-to-end checks reach OpenRouter and spend money. Run them deliberately, one
  session at a time, and never in a loop.

Full procedures live in [docs/agent/OPERATIONS.md](docs/agent/OPERATIONS.md).

## Safety and invariants

- Repository edits run through `/Users/matt/.local/bin/claude-harness` with
  explicit allowed paths and the smallest targeted deterministic checks. There is
  no reviewer-agent requirement and no automatic full-suite mandate.
- **Explicit authorization from Matt is required** for commits, pushes,
  deployments, data or schema writes, credential changes, outbound
  communications, and any other externally visible effect. Making a change is
  never permission to publish it.
- Keep the tool read-only toward `voice_agent_prompts`. It fetches; it must not
  write, open PRs, or mutate upstream flows.
- The `FLOWS` and `PROMPTS` maps in `server.js` are an allowlist. Do not replace
  them with caller-supplied paths — that would turn `POST /api/session` into an
  arbitrary-file reader against the token's full GitHub scope.
- Never log, echo, or persist `OPENROUTER_API_KEY` or the `gh` token; record
  secret *names* only in code and docs.
- The server has no authentication and trusts every caller. Bind it to localhost
  and do not expose it publicly.
- Keep this harness faithful to production: prompt assembly order is
  ANLIS → guardrails → flow prompt. Changing it invalidates test results.

Full list with rationale: [docs/agent/INVARIANTS.md](docs/agent/INVARIANTS.md).

## Documentation policy

Documentation is part of the implementation. When a change materially alters
architecture, integrations, data contracts, runtime or deployment behavior,
operational procedures, or durable constraints, update AGENTS.md and the relevant
docs/agent/*.md files in the same change. Keep AGENTS.md concise and
current-state; record significant dated changes in docs/agent/CHANGELOG.md. Do
not record routine code churn or weaken safety, authorization, or governance
rules without Matt's explicit approval.

## Knowledge base index

- [docs/agent/ARCHITECTURE.md](docs/agent/ARCHITECTURE.md) — components, HTTP
  contract, conversion pipeline, prompt assembly, integration contracts.
- [docs/agent/OPERATIONS.md](docs/agent/OPERATIONS.md) — setup, running,
  fixtures, targeted verification, failure modes, destructive boundaries.
- [docs/agent/INVARIANTS.md](docs/agent/INVARIANTS.md) — durable rules that must
  hold, each with its reason and blast radius.
- [docs/agent/CHANGELOG.md](docs/agent/CHANGELOG.md) — dated material changes,
  newest first.
- [docs/agent/OPEN_QUESTIONS.md](docs/agent/OPEN_QUESTIONS.md) — unproved
  assumptions awaiting confirmation.
