require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const MODEL = process.env.OPENROUTER_MODEL || 'openai/gpt-oss-120b';
const API_KEY = process.env.OPENROUTER_API_KEY;
const PORT = process.env.PORT || 3100;

// Bridge to production Python prompt converter
function convertFlowToPrompt(flowJson) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', ['-c', `
import json, sys
sys.path.insert(0, '.')
from prompt_flow_converter import convert_generic_flow_to_prompt
flow = json.load(sys.stdin)
result = convert_generic_flow_to_prompt(flow)
# Strip [break] markers — pipeline-level, not LLM instructions
result = result.replace('\\n[break]\\n', '\\n').replace('[break]', '')
sys.stdout.write(result)
`], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (data) => { stdout += data.toString(); });
    py.stderr.on('data', (data) => { stderr += data.toString(); });

    py.on('close', (code) => {
      if (code === 0) resolve(stdout.trim());
      else reject(new Error(`Flow conversion failed: ${stderr}`));
    });

    py.stdin.write(JSON.stringify(flowJson));
    py.stdin.end();
  });
}

// Fetch flow JSON from GitHub
const GITHUB_TOKEN = (() => {
  try {
    return require('child_process').execSync('gh auth token', { encoding: 'utf-8' }).trim();
  } catch { return ''; }
})();

const GITHUB_API_BASE = 'https://api.github.com/repos/mattvalenta/voice_agent_prompts/contents';

const FLOWS = {
  'internet_agent_v6.json': 'internet_agent_v6.json',
  'internet_agent_v7.json': 'internet_agent_v7.json',
  'internet_agent_v7_compact.json': 'internet_agent_v7_compact.json',
  'internet_agent_experimental.json': 'internet_agent_experimental.json',
  'internet_agent_experimental_minimal.json': 'internet_agent_experimental_minimal.json',
  'equity_agent_v5.json': 'equity_agent_v5.json',
  'inbound_agent_rebuilt_v4.json': 'inbound_agent_rebuilt_v4.json',
  'product_demo_v5.json': 'product_demo_v5.json',
  'product_demo_v10.json': 'product_demo_v10.json',
  'product_demo_v11.json': 'product_demo_v11.json',
};

const PROMPTS = {
  'ANLIS.md': 'stitch_prompts/ANLIS.md',
  'guardrails.md': 'stitch_prompts/guardrails.md',
};

async function fetchFromGitHub(githubPath) {
  const res = await fetch(`${GITHUB_API_BASE}/${githubPath}`, {
    headers: {
      'Authorization': `Bearer ${GITHUB_TOKEN}`,
      'Accept': 'application/vnd.github.v3+json',
    }
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  const data = await res.json();
  return Buffer.from(data.content, 'base64').toString('utf-8');
}

// In-memory sessions — each holds: flow, systemPrompt, messages[]
const sessions = {};

app.get('/api/flows', (req, res) => {
  res.json({ flows: Object.keys(FLOWS) });
});

app.post('/api/session', async (req, res) => {
  try {
    const { flowFile, model } = req.body;
    if (!flowFile || !FLOWS[flowFile]) return res.status(400).json({ error: 'Unknown flow' });

    const useModel = model || MODEL;

    // Fetch prompts and flow in parallel
    const [anlis, guardrails, flowContent] = await Promise.all([
      fetchFromGitHub(PROMPTS['ANLIS.md']),
      fetchFromGitHub(PROMPTS['guardrails.md']),
      fetchFromGitHub(FLOWS[flowFile]),
    ]);

    const flow = JSON.parse(flowContent);

    // Build system prompt using production converter
    const flowPrompt = await convertFlowToPrompt(flow);
    const systemPrompt = `${anlis}\n\n${guardrails}\n\n${flowPrompt}`;

    const sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
    sessions[sessionId] = {
      flowFile,
      model: useModel,
      flow,
      systemPrompt,
      messages: [],
    };

    res.json({ sessionId, systemPromptLength: systemPrompt.length, systemPrompt });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/chat', async (req, res) => {
  try {
    const { sessionId, userMessage } = req.body;
    if (!sessionId || !userMessage) return res.status(400).json({ error: 'Missing fields' });

    const session = sessions[sessionId];
    if (!session) return res.status(404).json({ error: 'Session not found' });

    // Build messages exactly like Pipecat:
    // 1. System message (the full flow prompt)
    // 2. Conversation history (user/assistant alternating)
    // 3. Current user message
    const messages = [
      { role: 'system', content: session.systemPrompt },
      ...session.messages,
      { role: 'user', content: userMessage },
    ];

    // Call OpenRouter with session-specific model
    const apiRes = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`,
        'HTTP-Referer': 'http://localhost:3100',
        'X-Title': 'Agent Tester',
      },
      body: JSON.stringify({
        model: session.model || MODEL,
        messages,
        temperature: 0.7,
      }),
    });

    if (!apiRes.ok) {
      const errText = await apiRes.text();
      throw new Error(`OpenRouter API error: ${apiRes.status} ${errText}`);
    }

    const data = await apiRes.json();
    const content = data.choices?.[0]?.message?.content || '';
    const finishReason = data.choices?.[0]?.finish_reason || 'unknown';

    // Log for debugging
    console.log(`LLM | finish=${finishReason} | len=${content.length} | preview=${content.substring(0, 100)}`);

    // Update conversation history
    session.messages.push({ role: 'user', content: userMessage });
    if (content) {
      session.messages.push({ role: 'assistant', content, comment: '' });
    }

    // Limit history to prevent context bloat (keep last 20 messages)
    if (session.messages.length > 20) {
      session.messages = session.messages.slice(-20);
    }

    res.json({
      agentResponse: content,
      messageCount: session.messages.length,
      finishReason,
    });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/comment', (req, res) => {
  const { sessionId, messageIndex, comment } = req.body;
  const session = sessions[sessionId];
  if (!session) return res.status(404).json({ error: 'Session not found' });
  if (session.messages[messageIndex]) {
    session.messages[messageIndex].comment = comment;
    res.json({ success: true });
  } else {
    res.status(404).json({ error: 'Message not found' });
  }
});

app.get('/api/transcript/:sessionId', (req, res) => {
  const session = sessions[req.params.sessionId];
  if (!session) return res.status(404).json({ error: 'Session not found' });
  res.json({
    flowFile: session.flowFile,
    messages: session.messages,
    exportedAt: new Date().toISOString(),
  });
});

app.delete('/api/session/:sessionId', (req, res) => {
  delete sessions[req.params.sessionId];
  res.json({ deleted: true });
});

app.listen(PORT, () => {
  console.log(`Agent Tester running on http://localhost:${PORT}`);
  console.log(`Model: ${MODEL}`);
});
