# Open questions

Things the current repository does not settle. Each is an unproved assumption,
not a fact — resolve it against reality, then move the answer into the document
that owns it and delete the entry here.

**1. Which copy of `prompt_flow_converter.py` is authoritative?**
`server.js` calls it "the production Python prompt converter," and its docstring
describes Pipecat flow conversion, but no manifest, submodule, or sync script
ties this file to a production source. If production holds the original, this is
a vendored copy that can silently drift and quietly invalidate test results. Need
to know: where the canonical copy lives, and how divergence gets caught.

**2. `loguru` has no declared dependency.**
`prompt_flow_converter.py` imports it, and conversion fails without it, yet there
is no `requirements.txt`, `pyproject.toml`, or documented Python version in this
repository. Today it works only because the developer's environment happens to
have it. Adding a Python manifest would fix this — it is a source change, so it
needs Matt's approval rather than a docs edit.

**3. Does the production runtime also strip `[break]` markers?**
The tester strips them in `server.js` after conversion. Whether production
strips them at the same point, earlier, or relies on the TTS layer to consume
them determines whether this harness is faithful here.

**4. Is `mattvalenta/voice_agent_prompts` public or private?**
The token path (`gh auth token`) implies private, but that is inference. It
affects whether a new contributor needs to be granted access before the tool will
run at all.

**5. Which allowlisted flows are current?**
`FLOWS` lists twelve entries spanning several generations
(`internet_agent_v6` through `v7_compact`, `product_demo_v5` through `v13`,
plus experimental variants). Nothing marks which are live, which are historical
comparisons, and which are dead. Without that, a tester cannot tell whether a
result reflects the shipping agent.

**6. Is there an intended retention or review path for transcripts?**
Downloads land in the browser's download directory and leave this system's
control entirely. If they are meant to be collected, compared, or reviewed
somewhere, that destination is undocumented — and if the conversations can carry
customer-like content, a handling rule is needed.
