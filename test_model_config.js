// Static checks on the OpenRouter model configuration contract.
//
// This file is deliberately read-only: it parses source text and never requires
// server.js, so it starts no listener, spawns no python3, and makes no network
// or OpenRouter calls.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_MODEL = '@preset/dumb-llm-openrouter';
const PREVIOUS_DEFAULT_MODEL = 'inclusionai/ling-3.0-flash';

const serverSource = fs.readFileSync(path.join(__dirname, 'server.js'), 'utf8');
const envExample = fs.readFileSync(path.join(__dirname, '.env.example'), 'utf8');
const indexHtml = fs.readFileSync(
  path.join(__dirname, 'public', 'index.html'),
  'utf8'
);

// The options inside `<select id="model-select">`, in document order.
function modelOptions() {
  const select = indexHtml.match(
    /<select\s+id="model-select"\s*>([\s\S]*?)<\/select>/
  );
  assert.ok(select, 'expected a #model-select element in public/index.html');
  return [...select[1].matchAll(/<option\s+value="([^"]*)"([^>]*)>/g)].map(
    ([, value, attrs]) => ({ value, selected: /\bselected\b/.test(attrs) })
  );
}

// `const MODEL = process.env.OPENROUTER_MODEL || '<default>';`
const modelAssignment =
  /const\s+MODEL\s*=\s*process\.env\.OPENROUTER_MODEL\s*\|\|\s*(['"])([^'"]+)\1\s*;/;

test('server.js still reads the model from OPENROUTER_MODEL', () => {
  const match = serverSource.match(modelAssignment);
  assert.ok(match, 'expected MODEL to fall back from process.env.OPENROUTER_MODEL');
});

test('server.js defaults to the new model', () => {
  const [, , defaultModel] = serverSource.match(modelAssignment);
  assert.equal(defaultModel, DEFAULT_MODEL);
});

test('server.js no longer defaults to the previous model', () => {
  assert.ok(
    !serverSource.includes(PREVIOUS_DEFAULT_MODEL),
    `server.js should not reference ${PREVIOUS_DEFAULT_MODEL}`
  );
});

test('.env.example documents OPENROUTER_MODEL with the new default', () => {
  const line = envExample
    .split('\n')
    .find((l) => l.startsWith('OPENROUTER_MODEL='));
  assert.ok(line, 'expected an OPENROUTER_MODEL entry in .env.example');
  assert.equal(line.trim(), `OPENROUTER_MODEL=${DEFAULT_MODEL}`);
});

test('the browser model picker defaults to the new model', () => {
  const selected = modelOptions().filter((o) => o.selected);
  assert.equal(selected.length, 1, 'expected exactly one selected model option');
  assert.equal(selected[0].value, DEFAULT_MODEL);
});

test('server, .env.example, and the browser picker agree on the default', () => {
  const [, , serverDefault] = serverSource.match(modelAssignment);
  const envDefault = envExample
    .split('\n')
    .find((l) => l.startsWith('OPENROUTER_MODEL='))
    .trim()
    .slice('OPENROUTER_MODEL='.length);
  const browserDefault = modelOptions().find((o) => o.selected).value;
  assert.equal(serverDefault, envDefault);
  assert.equal(browserDefault, envDefault);
});

test('the default is a managed OpenRouter preset id', () => {
  assert.match(DEFAULT_MODEL, /^@preset\//);
});

test('the browser labels the default option as a managed preset', () => {
  const option = indexHtml.match(
    /<option\s+value="@preset\/dumb-llm-openrouter"[^>]*>([^<]*)</
  );
  assert.ok(option, 'expected the preset option in #model-select');
  assert.match(option[1], /preset/i);
});

test('the per-session model still overrides the default', () => {
  // POST /api/session stores the caller's choice; POST /api/chat replays it.
  assert.match(serverSource, /const\s+useModel\s*=\s*model\s*\|\|\s*MODEL\s*;/);
  assert.match(serverSource, /model:\s*session\.model\s*\|\|\s*MODEL\s*,/);
});

test('the browser model picker drops the retired option', () => {
  const values = modelOptions().map((o) => o.value);
  // Exact match only: a `:free` or otherwise suffixed variant is not retired.
  assert.ok(
    !values.includes(PREVIOUS_DEFAULT_MODEL),
    `#model-select should not offer ${PREVIOUS_DEFAULT_MODEL}`
  );
});

test('the browser model picker keeps the other choices', () => {
  const values = modelOptions().map((o) => o.value);
  for (const kept of [
    'openai/gpt-oss-20b:free',
    'openai/gpt-oss-120b:free',
    'google/gemini-3.1-flash-lite-preview',
    'x-ai/grok-4.1-fast',
    '__custom__',
  ]) {
    assert.ok(values.includes(kept), `#model-select should still offer ${kept}`);
  }
});

test('.env.example names the other env vars without carrying a key value', () => {
  const apiKeyLine = envExample
    .split('\n')
    .find((l) => l.startsWith('OPENROUTER_API_KEY='));
  assert.ok(apiKeyLine, 'expected an OPENROUTER_API_KEY entry in .env.example');
  assert.equal(apiKeyLine.trim(), 'OPENROUTER_API_KEY=');
  assert.match(envExample, /^PORT=/m);
});
