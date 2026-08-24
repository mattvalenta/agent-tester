// Static checks on the OpenRouter model configuration contract.
//
// This file is deliberately read-only: it parses source text and never requires
// server.js, so it starts no listener, spawns no python3, and makes no network
// or OpenRouter calls.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_MODEL = 'inclusionai/ling-3.0-flash';
const PREVIOUS_DEFAULT_MODEL = 'openai/gpt-oss-120b';

const serverSource = fs.readFileSync(path.join(__dirname, 'server.js'), 'utf8');
const envExample = fs.readFileSync(path.join(__dirname, '.env.example'), 'utf8');

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

test('.env.example names the other env vars without carrying a key value', () => {
  const apiKeyLine = envExample
    .split('\n')
    .find((l) => l.startsWith('OPENROUTER_API_KEY='));
  assert.ok(apiKeyLine, 'expected an OPENROUTER_API_KEY entry in .env.example');
  assert.equal(apiKeyLine.trim(), 'OPENROUTER_API_KEY=');
  assert.match(envExample, /^PORT=/m);
});
