# Operations

How to set up, run, verify, and stay inside the safe boundaries of this tool.
Companion to [ARCHITECTURE.md](ARCHITECTURE.md) and [INVARIANTS.md](INVARIANTS.md).

## Prerequisites

- Node.js with `npm` (Express 4, cors, dotenv; lockfile v3).
- `python3` on `PATH`, with `loguru` importable — see
  [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).
- GitHub CLI (`gh`) authenticated, with read access to
  `mattvalenta/voice_agent_prompts`. The server runs `gh auth token` once at
  startup; if it fails, the token is empty and every flow fetch returns 401.
- An OpenRouter account and API key.

## Setup and run

```bash
npm ci
npm start          # listens on PORT, default 3100
```

`.env` at the repository root is untracked and supplies:

| Variable | Required | Effect |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | Bearer credential for chat completions |
| `OPENROUTER_MODEL` | No | Default model; falls back to the managed preset `@preset/dumb-llm-openrouter` |
| `PORT` | No | Listen port; falls back to `3100` |

Record names only. Never paste values into commits, issues, logs, or docs.

Startup prints the URL and the default model. Changing `.env` or either source
file requires a restart — there is no watcher.

## Using the tester

1. Open `http://localhost:<PORT>`.
2. Pick a flow from the dropdown (populated from `GET /api/flows`) and a model.
   The model list includes a `Custom...` option that accepts any OpenRouter model
   id; the per-session choice overrides `OPENROUTER_MODEL`.
3. **Start Session** fetches the prompts, converts the flow, and reveals the
   assembled system prompt under **Show Prompt** — read it first when a run looks
   wrong, since most surprises originate in conversion rather than the model.
4. Type turns. Use **💬 Comment** to annotate any agent response; comments are
   stored on the message and included in the export.
5. **Download Transcript** saves `transcript_<sessionId>_<flow>.json` with the
   flow name, messages, comments, and an `exportedAt` timestamp.
6. **Reset** clears the browser view only. It does not delete the server-side
   session — `DELETE /api/session/:sessionId` does that, and everything is
   discarded on restart regardless.

## Inputs, fixtures, and outputs

**Inputs.** Flow JSON and the two stitch prompts, all fetched live from
`mattvalenta/voice_agent_prompts` at session start. There are no fixtures
committed to this repository — no sample flows, no golden prompts, no recorded
responses. The `FLOWS` map in `server.js` is the fixture *catalog*: adding a flow
to the tool means adding its filename to that map (both key and value are the
same path today), which is what most commits in this repository have done.

**Outputs.** Three, none of them persistent:

- The assembled system prompt, returned by `POST /api/session` and shown in the
  side panel.
- Agent turns with their `finishReason`, rendered in the chat pane. A
  `finishReason` other than `stop` is displayed inline and usually means a length
  cap or filter, not a flow problem.
- Downloaded transcript JSON, written wherever the browser saves files.

**Promotion path.** Findings flow *out* of this tool by hand. A prompt or flow
defect confirmed here is fixed in `voice_agent_prompts`; a converter defect is
fixed in `prompt_flow_converter.py` here and must then be reconciled with the
production copy of the converter. Nothing in this repository promotes, publishes,
or writes anything automatically.

## Targeted verification

There is no test suite, no CI, and no lint configuration. Run the smallest
deterministic check that covers what changed:

| Change | Check |
| --- | --- |
| `server.js` | `node --check server.js` |
| `prompt_flow_converter.py` | `python3 -m py_compile prompt_flow_converter.py` |
| Converter output | Feed a small flow dict to `convert_generic_flow_to_prompt` and inspect the text |
| `public/index.html` | Load the page and use the affected control |
| Flow allowlist | `GET /api/flows` and confirm the new entry starts a session |

Prefer syntax and unit-level checks over end-to-end runs. A full session costs a
GitHub round trip plus a billable inference per turn, so it is a deliberate act,
not a default verification step. Never script it in a loop.

## Failure modes

| Symptom | Likely cause |
| --- | --- |
| `GitHub API error: 401` | `gh auth token` failed at startup, or the token lost repo access. Re-auth, then **restart** — the token is read once |
| `GitHub API error: 404` | Filename in `FLOWS` / `PROMPTS` no longer exists upstream |
| `Flow conversion failed: …` | Python missing, `loguru` not installed, or malformed flow JSON. stderr is passed through in the message |
| `OpenRouter API error: …` | Bad or missing key, unknown model id, or upstream rate limit. The turn is not recorded |
| `Session not found` | Server restarted, or the session was deleted. Sessions are in memory only |
| Agent forgets earlier turns | History is capped at the last 20 messages by design |

## Destructive boundaries

Actions this tool must never take, and that agents working here must not add:

- **No writes to `voice_agent_prompts`** or any other repository. The GitHub
  integration is read-only. Fixes are proposed in the owning repository through
  its own review process.
- **No commits, pushes, merges, tags, or releases** without Matt's explicit
  approval, per [../../AGENTS.md](../../AGENTS.md).
- **No deployment.** This tool has no hosted environment and should not acquire
  one incidentally.
- **No credential changes.** Do not rotate, re-scope, or re-auth `gh`, and do not
  move `OPENROUTER_API_KEY` between accounts or files.
- **No outbound communication** on a developer's behalf, and no exposing the
  server beyond localhost — it has no authentication.
- **No unbounded inference.** Batch or looped session runs spend real money;
  clear them with Matt first.
