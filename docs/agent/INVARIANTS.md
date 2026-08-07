# Invariants

Rules that must hold for this tool to remain trustworthy. Each states what to
preserve, why, and what breaks if it is violated. Weakening any safety,
authorization, or governance rule below requires Matt's explicit approval.

## Fidelity to production

**1. Prompt assembly order is ANLIS → guardrails → flow prompt.**
`POST /api/session` joins the three documents in that order with blank lines
between them. The whole point of the tester is that the model sees what it sees
in production; reordering or dropping a section silently invalidates every
conclusion drawn from a session.

**2. Chat requests keep the Pipecat message shape** — one system message, then
conversation history, then the current user turn. Do not fold history into the
system message, inject scaffolding turns, or reformat roles.

**3. `[break]` markers are stripped before the prompt reaches the model.**
They are pipeline-level speech cues. Leaving them in feeds the model instructions
it never receives in production.

**4. Converter changes must stay compatible with both flow formats.**
`prompt_flow_converter.py` handles array-based (`nodes` as a list, with `edges`)
and dict-based (`nodes` keyed by name, with `initial_node`) layouts. Both are in
active use across the allowlisted flows, and the file mirrors production logic —
diverging from that copy makes tester results misleading. See
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) on which copy is authoritative.

## Security

**5. `FLOWS` and `PROMPTS` are a closed allowlist.**
`POST /api/session` accepts only keys of `FLOWS`. The GitHub token comes from the
developer's own `gh` session and carries their full scope, so accepting a
caller-supplied path would turn this endpoint into an arbitrary-file reader
against every repository they can see. Add entries by editing the map; never by
interpolating request input into a GitHub path.

**6. Credentials never leave the process.**
`OPENROUTER_API_KEY` and the `gh` token are used in request headers only. Do not
log them, return them in responses, write them to transcripts, or commit them.
Documentation and code record secret *names* only. `.env` stays gitignored.

**7. The server stays on localhost.**
There is no authentication, no rate limiting, and no per-user isolation; every
caller can read any session and spend the key. It must not be bound to a public
interface or tunneled out without a deliberate access-control design.

**8. The GitHub integration is read-only.** This tool fetches flows and prompts;
it never writes to `mattvalenta/voice_agent_prompts`. Corrections go through that
repository's own process.

## Runtime and data

**9. Sessions are ephemeral and in-memory by design.**
No database, no session files, no cross-restart durability. Adding persistence
introduces data ownership, retention, and cleanup obligations this tool has
deliberately avoided — treat it as a design change requiring approval, not a
convenience fix.

**10. History stays bounded.** `session.messages` is truncated to the last 20
entries after each turn to keep context and cost in check. Raising the cap raises
the per-turn bill on every subsequent request.

**11. A failed inference must not corrupt history.** Turns are appended only
after a successful OpenRouter response, so a failure leaves the session
replayable. Preserve that ordering.

**12. Every conversion runs in a subprocess with `cwd` at the repository root.**
That is how `prompt_flow_converter` resolves on `sys.path`. Moving the converter,
or changing the spawn's working directory, breaks session creation.

## Governance

**13. Repository edits run through `/Users/matt/.local/bin/claude-harness`** with
explicit allowed paths and the smallest targeted deterministic checks. There is
no reviewer-agent requirement and no automatic full-suite mandate — but staying
inside declared paths is not optional.

**14. External effects require explicit authorization from Matt**: commits,
pushes, deployments, data or schema writes, credential changes, outbound
communications, and anything else visible outside the working copy. Completing a
change is not permission to ship it.

**15. Inference spend is a real effect.** Manual, one-at-a-time sessions are
routine; batch runs, sweeps, and automated loops are not, and need approval
first.

**16. Documentation ships with the change.** Material changes to architecture,
integrations, data contracts, runtime or deployment behavior, operations, or
these invariants update `AGENTS.md` and the relevant `docs/agent/*.md` in the same
change, with a dated entry in [CHANGELOG.md](CHANGELOG.md).
