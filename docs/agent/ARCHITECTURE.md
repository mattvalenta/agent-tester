# Architecture

Depth behind the map in [../../AGENTS.md](../../AGENTS.md). Current state as read
from `server.js`, `prompt_flow_converter.py`, and `public/index.html`.

## Components

| Component | File | Responsibility |
| --- | --- | --- |
| HTTP server | `server.js` | Serves the UI, owns sessions, calls GitHub / Python / OpenRouter |
| Prompt converter | `prompt_flow_converter.py` | Turns a flow config into one structured system prompt |
| Browser client | `public/index.html` | Flow and model pickers, chat pane, prompt pane, comments, transcript download |

No framework beyond Express, no build step, no bundler. The client is plain
`fetch` against same-origin `/api/*`; `cors()` is enabled but nothing external
consumes the API today.

## Request lifecycle

### `POST /api/session`

1. Reject the request unless `flowFile` is a key of the `FLOWS` allowlist.
2. Fetch three documents from GitHub **in parallel**: `stitch_prompts/ANLIS.md`,
   `stitch_prompts/guardrails.md`, and the mapped flow JSON.
3. `JSON.parse` the flow, then convert it in a Python subprocess.
4. Strip `[break]` markers from the converted text. These are pipeline-level
   speech cues, not LLM instructions, so the tester removes them to match what
   the model actually sees. The stripping lives in `server.js`, not in the
   converter.
5. Concatenate `ANLIS + "\n\n" + guardrails + "\n\n" + flowPrompt` into the
   session's system prompt. **This order is the production contract.**
6. Mint a session id (`Date.now()` base36 + random suffix) and store
   `{ flowFile, model, flow, systemPrompt, messages: [] }` in the `sessions`
   object.
7. Respond with `sessionId`, `systemPromptLength`, and the full `systemPrompt`
   (the UI renders it in the side panel).

### `POST /api/chat`

Builds the message array exactly the way Pipecat does — system message, then
stored history, then the new user turn — and posts it to OpenRouter with the
session's model and `temperature: 0.7`. On success it appends the user turn and,
if content is non-empty, the assistant turn (with an empty `comment` field), then
truncates `messages` to the **last 20 entries**. It returns `agentResponse`,
`messageCount`, and `finishReason`, and logs a one-line summary with a 100-char
preview to stdout.

Because a failed call throws before the history push, a failed turn leaves the
session unchanged and is safe to retry manually.

### Other endpoints

| Endpoint | Behavior |
| --- | --- |
| `GET /api/flows` | Returns the allowlist keys; drives the UI dropdown |
| `POST /api/comment` | Writes `comment` onto `messages[messageIndex]`; 404 if session or message is missing |
| `GET /api/transcript/:sessionId` | Returns `flowFile`, `messages`, `exportedAt`; the client saves it as a JSON file |
| `DELETE /api/session/:sessionId` | Deletes the key and always reports `{ deleted: true }` |

JSON bodies are accepted up to 50 MB.

## Prompt converter

`convert_generic_flow_to_prompt(flow_config)` dispatches on the shape of
`nodes` and emits the same document skeleton either way:
`# SYSTEM PROMPT — SINGLE-PROMPT VOICE AGENT`, an optional bolded `meta` line, an
optional `## ROLE`, `## CONVERSATION FLOW` with numbered `### Stage N` headers,
and a fixed `## BEHAVIORAL GUIDELINES` block (short turns, always end on a
question, avoid special characters because the output is spoken).

**Array format** — `nodes` is a list. Ordering comes from a depth-first walk that
starts at the node typed `initial` (falling back to the first node), follows
`edges`, then follows `next_node_id` and every `decision` branch including
`default_next_node_id`; any node still unvisited is appended. Each stage renders
description, role/task messages, routing instructions, tool schemas, pre/post
actions, and outgoing edge labels.

**Dict format** — `nodes` is keyed by node name with a top-level `initial_node`,
which is hoisted to the front. There are no edges, so terminal nodes are inferred:
type `end`, or (only when some function is a dict) a non-initial node with no
`next_node_id` and no `decision`.

A function is *routing* if it carries `decision` or `next_node_id`; otherwise it
renders as a tool schema with required parameters and a property table. Property
rendering surfaces `type`, `enum`, `pattern`, `format`, `default`, `items`, any
unrecognized keys, and the description, and unescapes JSON-escaped bold markers
in property names.

Results are memoized in `_generic_flow_prompt_cache` keyed by an MD5 of the
sorted-JSON flow config, with `clear_flow_prompt_cache()` to reset it. Note the
cache is **process-lifetime**: because `server.js` spawns a fresh `python3` for
every conversion, the cache never survives a request in this tool. It matters in
the long-lived production process, not here.

The module imports `loguru` for its progress logging — the only dependency, and
one no manifest in this repository declares.

## Integration contracts

| Integration | Direction | Transport | Contract | Source of truth | Retry / idempotency | Failure effect | Owning repo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GitHub Contents API | Outbound | HTTPS GET, bearer token | `{ content: base64 }` decoded to UTF-8 | `mattvalenta/voice_agent_prompts` | None; read-only so safe to repeat | 500 from `/api/session`; no session created | `voice_agent_prompts` |
| OpenRouter | Outbound | HTTPS POST, bearer key | OpenAI-style `choices[0].message.content` + `finish_reason` | OpenRouter | None; **not idempotent**, each call bills | 500 from `/api/chat`; history untouched | OpenRouter (external) |
| Prompt converter | Outbound, local | `spawn('python3', ['-c', …])`, JSON on stdin / text on stdout | Non-zero exit ⇒ rejection carrying stderr | This repository | Deterministic; safe to repeat | 500 from `/api/session` | `agent-tester` |

Missing or unreadable optional fields degrade quietly inside the converter — it
uses `.get(...)` defaults throughout rather than validating input, so a malformed
flow yields a thin prompt rather than an error.

## Trust and credentials

The GitHub token is resolved **once**, at module load, by shelling out to
`gh auth token`; failure is swallowed and yields `''`. That token carries the
developer's full `gh` scope, which is why the `FLOWS` / `PROMPTS` allowlist must
stay closed to caller input. `OPENROUTER_API_KEY` comes from the environment via
`dotenv`. Neither value is ever written to disk or returned to the client.

The server itself is unauthenticated and has no rate limiting; its only intended
listener is a browser on the same machine.
