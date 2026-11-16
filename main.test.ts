import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { Server } from 'node:http';
import { Readable } from 'node:stream';
import { test } from 'node:test';
import type { TestContext } from 'node:test';
import { briefDocuments, catalogOf, clientModels, createRouter, directory, discover, frameText, readFrames, serve } from './main.ts';

const env = 'http://simulator.invalid';
type Json = Record<string, any>;
type Wire = 'openai' | 'anthropic' | 'gemini';

function row(id: string, wire: Wire, auth = 'none') {
  const secured = auth !== 'none';
  return { id, wire, auth, key_env: secured ? id.toUpperCase() + '_TEST_KEY' : '',
    key: secured ? 'local-fixture' : '', path: '/p/' + id };
}

function spec(provider: string, model: string, rating_milli: number, extras: Json = {}) {
  return { provider, model, rating_milli, free: true, context_length: 32_000,
    tools: true, vision: false, json: false, ...extras };
}

function json(value: unknown, status = 200): Response {
  return Response.json(value, { status });
}

function sse(chunks: Array<[string | undefined, Json | '[DONE]']>): Response {
  return new Response(chunks.map(([event, data]) => frameText({
    event, data: typeof data === 'string' ? data : JSON.stringify(data),
  })).join(''), { headers: { 'content-type': 'text/event-stream' } });
}

function completion(model: unknown, text = 'answer', stop = 'stop') {
  return { id: 'chat-fixture', object: 'chat.completion', created: 1, model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: stop }],
    usage: { prompt_tokens: 8, completion_tokens: 5, total_tokens: 13 } };
}

function anthropicMessage(model: unknown, text = 'answer', stop = 'end_turn') {
  return { type: 'message', id: 'msg_fixture', model, role: 'assistant', content: [{ type: 'text', text }],
    stop_reason: stop, stop_sequence: null, usage: { input_tokens: 8, output_tokens: 5 } };
}

function ratingCatalog(ratings: ReturnType<typeof spec>[]) {
  return ratings.map(rating => ({ id: rating.model, free: rating.free, context_length: rating.context_length }));
}

async function install(t: TestContext, rows: ReturnType<typeof row>[],
  ratings: ReturnType<typeof spec>[], respond: (id: string, payload: Json, options: RequestInit, url: URL) => Response,
  work = () => ({ done: false })) {
  const calls: Array<{ id: string; payload: Json }> = [];
  const catalogs: Array<{ id: string; cursor: string | null }> = [];
  t.mock.method(globalThis, 'fetch', async (source: string, options: RequestInit = {}) => {
    const url = new URL(source);
    assert.equal(url.origin, env);
    const id = url.pathname.split('/')[2];
    if (url.pathname === '/work') return json(work());
    const provider = rows.find(item => item.id === id);
    assert.ok(provider);
    const expectedAuth: Record<string, [string, string]> = {
      bearer: ['authorization', 'Bearer local-fixture'], 'x-api-key': ['x-api-key', 'local-fixture'],
      'x-goog-api-key': ['x-goog-api-key', 'local-fixture'],
    };
    const headers = new Headers(options.headers);
    for (const header of ['authorization', 'x-api-key', 'x-goog-api-key']) {
      const auth = expectedAuth[provider.auth];
      assert.equal(headers.get(header), auth?.[0] === header ? auth[1] : null);
    }
    if (options.method === 'POST') {
      const payload = JSON.parse(options.body as string);
      calls.push({ id, payload });
      return respond(id, payload, options, url);
    }
    const models = ratingCatalog(ratings.filter(rating => rating.provider === id));
    catalogs.push({ id, cursor: url.searchParams.get('pageToken') });
    if (provider.wire === 'gemini') {
      assert.equal(url.pathname, `/p/${id}/v1beta/models`);
      const entries = models.map(({ id, context_length, ...rest }) => ({ name: 'models/' + id, inputTokenLimit: context_length, ...rest }));
      if (entries.length > 1 && !url.searchParams.has('pageToken')) {
        return json({ models: entries.slice(0, 1), nextPageToken: 'next' });
      }
      return json({ models: url.searchParams.has('pageToken') ? entries.slice(1) : entries });
    }
    assert.equal(url.pathname, `/p/${id}/v1/models`);
    if (provider.wire === 'anthropic') assert.equal(headers.get('anthropic-version'), '2023-06-01');
    return json({ data: models });
  });
  const providers = directory(rows, env);
  await Promise.all(providers.map(discover));
  return { providers, calls, catalogs };
}

test('brief, native catalog shapes, auth, pagination, paid flags, and deduplicated free model lists', async t => {
  const rows = [row('alpha', 'openai'), row('bravo', 'anthropic', 'bearer'), row('charlie', 'gemini', 'x-goog-api-key')];
  const ratings = [spec('alpha', 'common', 100), spec('bravo', 'common', 105),
    spec('charlie', 'first', 120), spec('charlie', 'paid', 800, { free: false })];
  const documents = briefDocuments(`Brief\r\n{just prose}\n${JSON.stringify({ providers: rows })}\n${JSON.stringify({ ratings })}\n`);
  assert.equal(documents.length, 2);
  assert.throws(() => directory([{ ...rows[0], path: 'https://outside.invalid' }], env), /outside/);
  const { providers, catalogs } = await install(t, rows, ratings, () => { throw Error('Unexpected inference'); });
  assert.deepEqual(catalogs.filter(item => item.id === 'charlie').map(item => item.cursor), [null, 'next']);
  assert.deepEqual(catalogOf(providers).providers[2], { id: 'charlie', base_url: '/p/charlie',
    key_env: 'CHARLIE_TEST_KEY', wire: 'gemini', models: [{ id: 'first', free: true, context_length: 32_000 },
      { id: 'paid', free: false, context_length: 32_000 }] });
  assert.deepEqual(clientModels(providers, 'openai').data.map((model: Json) => model.id), ['common', 'first']);
  assert.equal(clientModels(providers, 'anthropic').last_id, 'first');
  const result = spawnSync(process.execPath, ['main.ts', '--list'], { encoding: 'utf8', env: {} });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, '{"providers":[]}');

  const simulatedCatalogs = Object.fromEntries(providers.map(provider => [provider.id,
    provider.wire === 'gemini' ? { models: provider.models } : { data: provider.models },
  ]));
  const preload = `globalThis.fetch = async address => {
    const path = new URL(address).pathname;
    const catalogs = ${JSON.stringify(simulatedCatalogs)};
    return Response.json(path === '/instance' ? { providers: ${JSON.stringify(rows)} } : catalogs[path.split('/')[2]]);
  };`;
  const trialList = spawnSync(process.execPath, ['--import', 'data:text/javascript,' + encodeURIComponent(preload), 'main.ts', '--list'],
    { encoding: 'utf8', env: { MAKEFUN_ENV_URL: env } });
  assert.equal(trialList.status, 0, trialList.stderr);
  assert.equal(trialList.stdout, JSON.stringify(catalogOf(providers)));
});

test('routes to the highest-rated free model when a provider has models on both sides of a competitor', async t => {
  const rows = [row('top', 'openai', 'x-api-key'), row('middle', 'openai')];
  const ratings = [spec('top', 'large', 150), spec('middle', 'mid', 140), spec('top', 'small', 130),
    spec('middle', 'paid', 999, { free: false })];
  const { providers, calls } = await install(t, rows, ratings, (_id, body) => json(completion(body.model)));
  const route = createRouter(env, providers, ratings, []);
  const answer = await route('openai', { messages: [{ role: 'user', content: 'hello' }] }, {});
  assert.deepEqual(calls.map(call => [call.id, call.payload.model]), [['top', 'large']]);
  assert.equal(answer.status, 200);
});

test('handles transient 429, tries one model per provider, and retires an exhausted provider', async t => {
  const rows = [row('best', 'openai', 'bearer'), row('backup', 'openai')];
  const ratings = [spec('best', 'high', 150), spec('best', 'low', 149), spec('backup', 'fallback', 140)];
  let topCalls = 0;
  const { providers, calls } = await install(t, rows, ratings, (id, body) => {
    if (id === 'best' && ++topCalls === 1) return json({ error: { message: 'Transient refusal' } }, 429);
    if (id === 'best' && topCalls === 3) return json({ error: { message: 'Free tier quota exhausted' } }, 429);
    return json(completion(body.model));
  });
  const route = createRouter(env, providers, ratings, []);
  for (let i = 0; i < 4; i++) {
    const answer = await route('openai', { messages: [{ role: 'user', content: 'hello' }] }, {});
    assert.equal(answer.status, 200);
  }
  assert.deepEqual(calls.map(call => call.id), ['best', 'backup', 'best', 'best', 'backup', 'backup']);
});

test('a scheduled 402 can recover and an explicit outage retires the provider', async t => {
  const rows = [row('first', 'openai'), row('second', 'openai')];
  const ratings = [spec('first', 'primary', 150), spec('second', 'backup', 140)];
  let primary = 0;
  const { providers, calls } = await install(t, rows, ratings, (id, body) => {
    if (id === 'first' && ++primary === 1) return json({ error: { message: 'Provider refused this call' } }, 402);
    if (id === 'first' && primary === 3) return json({ error: { message: 'Provider is out' } }, 402);
    return json(completion(body.model));
  });
  const route = createRouter(env, providers, ratings, []);
  const body = { messages: [{ role: 'user', content: 'fixture' }] };
  for (let index = 0; index < 4; index++) assert.equal((await route('openai', body, {})).status, 200);
  assert.deepEqual(calls.map(call => call.id), ['first', 'second', 'first', 'first', 'second', 'second']);
});

test('uses the manifest context floor and filters image, JSON and tool capabilities', async t => {
  const rows = [row('native', 'openai'), row('claude', 'anthropic'), row('gem', 'gemini')];
  const ratings = [spec('native', 'small', 170, { vision: true, json: true }),
    spec('claude', 'larger', 160, { context_length: 180_000, vision: true, json: true }),
    spec('gem', 'large', 140, { context_length: 180_000, vision: true, json: true })];
  const { providers, calls } = await install(t, rows, ratings, (id, _body, _opts, url) => {
    assert.equal(id, 'gem');
    assert.equal(url.pathname, '/p/gem/v1beta/models/large:generateContent');
    return json({ candidates: [{ content: { parts: [{ text: '{"ok":true}' }] }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 21, candidatesTokenCount: 8, totalTokenCount: 29 } });
  }, () => ({ done: false, next: 'context' }));
  const route = createRouter(env, providers, ratings, [
    { id: 'plain', api: 'openai', stream: false, weight: 1, requires: {} },
    { id: 'context', api: 'openai', stream: false, weight: 2, requires: { min_context: 100_000 } },
  ]);
  const schema = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };
  const answer = await route('openai', {
    messages: [{ role: 'system', content: 'policy' }, { role: 'user', content: [
      { type: 'text', text: 'describe this' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj' } },
    ] }],
    tools: [{ type: 'function', function: { name: 'lookup', parameters: { type: 'object' } } }],
    tool_choice: { type: 'function', function: { name: 'lookup' } },
    response_format: { type: 'json_schema', json_schema: { name: 'decision', schema } },
    max_completion_tokens: 1024, stop: ['first', 'second'],
  }, {});
  assert.equal(answer.status, 200);
  assert.equal((answer.body as Json).choices[0].message.content, '{"ok":true}');
  assert.deepEqual((answer.body as Json).usage, { prompt_tokens: 21, completion_tokens: 8, total_tokens: 29 });
  const sent = calls[0].payload;
  assert.deepEqual(sent.contents[0].parts[1], { inlineData: { mimeType: 'image/png', data: 'YWJj' } });
  assert.deepEqual(sent.systemInstruction, { parts: [{ text: 'policy' }] });
  assert.deepEqual(sent.generationConfig, { maxOutputTokens: 1024, stopSequences: ['first', 'second'],
    responseMimeType: 'application/json', responseJsonSchema: schema });
  assert.deepEqual(sent.toolConfig.functionCallingConfig, { mode: 'ANY', allowedFunctionNames: ['lookup'] });
});

test('translates OpenAI tool history and Anthropic streamed tool calls, usage, and stop reasons', async t => {
  const rows = [row('claude', 'anthropic', 'bearer')];
  const ratings = [spec('claude', 'message-model', 150)];
  const args = '{"city":"Zürich","limit":1}';
  const { providers, calls } = await install(t, rows, ratings, (_id, body, options) => {
    const headers = new Headers(options.headers);
    assert.equal(headers.get('anthropic-version'), '2023-06-01');
    return sse([
      ['message_start', { type: 'message_start', message: { ...anthropicMessage(body.model), content: [],
        stop_reason: null, usage: { input_tokens: 34, output_tokens: 0 } } }],
      ['content_block_start', { type: 'content_block_start', index: 0,
        content_block: { type: 'tool_use', id: 'call-back', name: 'search', input: {} } }],
      ['content_block_delta', { type: 'content_block_delta', index: 0,
        delta: { type: 'input_json_delta', partial_json: args.slice(0, 12) } }],
      ['content_block_delta', { type: 'content_block_delta', index: 0,
        delta: { type: 'input_json_delta', partial_json: args.slice(12) } }],
      ['content_block_stop', { type: 'content_block_stop', index: 0 }],
      ['message_delta', { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 16 } }],
      ['message_stop', { type: 'message_stop' }],
    ]);
  });
  const body = {
    messages: [{ role: 'system', content: 'system one' }, { role: 'system', content: 'system two' },
      { role: 'user', content: 'search' }, { role: 'assistant', content: null,
        tool_calls: [{ id: 'prior', type: 'function', function: { name: 'search', arguments: '{"city":"Paris"}' } }] },
      { role: 'tool', tool_call_id: 'prior', content: '10 hits' }, { role: 'user', content: 'now Zürich' }],
    max_tokens: 321, stop: 'END', stream: true, stream_options: { include_usage: true }, parallel_tool_calls: false,
    tools: [{ type: 'function', function: { name: 'search', parameters: { type: 'object' } } }],
  };
  const answer = await createRouter(env, providers, ratings, [])('openai', body, {});
  assert.equal(answer.status, 200);
  assert.ok(Array.isArray(answer.body));
  assert.equal(answer.body.at(-1)?.data, '[DONE]');
  const chunks = answer.body.slice(0, -1).map(frame => JSON.parse(frame.data));
  assert.equal(chunks[0].choices[0].delta.role, 'assistant');
  assert.equal(chunks.at(-2).choices[0].finish_reason, 'tool_calls');
  assert.deepEqual(chunks.at(-1).usage, { prompt_tokens: 34, completion_tokens: 16, total_tokens: 50 });
  const tools = chunks.flatMap(chunk => chunk.choices[0]?.delta.tool_calls ?? []);
  assert.equal(tools[0].id, 'call-back');
  assert.equal(tools.map(tool => tool.function.arguments).join(''), args);
  const sent = calls[0].payload;
  assert.equal(sent.max_tokens, 321);
  assert.deepEqual(sent.stop_sequences, ['END']);
  assert.deepEqual(sent.system.map((block: Json) => block.text), ['system one', 'system two']);
  assert.equal(sent.tool_choice.disable_parallel_tool_use, true);
  assert.deepEqual(sent.messages[1].content[0], { type: 'tool_use', id: 'prior', name: 'search', input: { city: 'Paris' } });
  assert.deepEqual(sent.messages[2].content, [
    { type: 'tool_result', tool_use_id: 'prior', content: [{ type: 'text', text: '10 hits' }] },
    { type: 'text', text: 'now Zürich' },
  ]);
});

test('translates OpenAI SSE into ordered Anthropic SSE and preserves output limits and exact token counts', async t => {
  const rows = [row('upstream', 'openai')];
  const ratings = [spec('upstream', 'completion-model', 150, { vision: true })];
  const { providers, calls } = await install(t, rows, ratings, (_id, body) => {
    const meta = { id: 'chat-fixture', object: 'chat.completion.chunk', created: 1, model: body.model };
    return sse([
      [undefined, { ...meta, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }],
      [undefined, { ...meta, choices: [{ index: 0, delta: { content: 'wide 🐋' }, finish_reason: null }] }],
      [undefined, { ...meta, choices: [{ index: 0, delta: { content: ' world' }, finish_reason: null }] }],
      [undefined, { ...meta, choices: [{ index: 0, delta: {}, finish_reason: 'length' }] }],
      [undefined, { ...meta, choices: [], usage: { prompt_tokens: 8, completion_tokens: 5, total_tokens: 13 } }],
      [undefined, '[DONE]'],
    ]);
  });
  const body = { model: 'auto', stream: true, system: [{ type: 'text', text: 'system' }], max_tokens: 5,
    messages: [{ role: 'user', content: [{ type: 'text', text: 'user' },
      { type: 'image', source: { type: 'url', url: 'https://example.invalid/image.png' } }] }] };
  const reply = await createRouter(env, providers, ratings, [])('anthropic', body, { 'anthropic-version': '2023-06-01' });
  assert.equal(reply.status, 200);
  assert.ok(Array.isArray(reply.body));
  const frames = readFrames(reply.body.map(frameText).join('').replaceAll('\n', '\r\n'));
  assert.deepEqual(frames.map(frame => frame.event), ['message_start', 'content_block_start', 'content_block_delta',
    'content_block_stop', 'message_delta', 'message_stop']);
  assert.equal(JSON.parse(frames[0].data).message.usage.input_tokens, 8);
  assert.equal(JSON.parse(frames[2].data).delta.text, 'wide 🐋 world');
  assert.deepEqual(JSON.parse(frames[4].data), { type: 'message_delta', delta: { stop_reason: 'max_tokens', stop_sequence: null },
    usage: { output_tokens: 5 } });
  assert.deepEqual(calls[0].payload.stream_options, { include_usage: true });
  assert.equal(calls[0].payload.max_tokens, 5);
  assert.equal(calls[0].payload.messages[1].content[1].image_url.url, 'https://example.invalid/image.png');
});

test('fails over on an interrupted event stream and produces repeatable bytes', async t => {
  const rows = [row('gem', 'gemini'), row('native', 'anthropic', 'x-api-key')];
  const ratings = [spec('gem', 'best', 200), spec('native', 'stable', 180)];
  const { providers, calls } = await install(t, rows, ratings, (id, body, options, url) => {
    if (id === 'gem') {
      assert.equal(url.pathname, '/p/gem/v1beta/models/best:streamGenerateContent');
      assert.equal(url.searchParams.get('alt'), 'sse');
      return sse([[undefined, { candidates: [{ content: { parts: [{ text: 'partial' }] } }] }]]);
    }
    assert.equal(new Headers(options.headers).get('anthropic-version'), 'fixture-version');
    const msg = anthropicMessage(body.model, 'fallback');
    return sse([
      ['message_start', { type: 'message_start', message: { ...msg, content: [], stop_reason: null } }],
      ['content_block_start', { type: 'content_block_start', index: 0, content_block: msg.content[0] }],
      ['content_block_stop', { type: 'content_block_stop', index: 0 }],
      ['message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 5 } }],
      ['message_stop', { type: 'message_stop' }],
    ]);
  });
  const body = { max_tokens: 10, stream: true, messages: [{ role: 'user', content: 'question' }] };
  const run = () => createRouter(env, providers.map(provider => ({ ...provider })), ratings, [])(
    'anthropic', body, { 'anthropic-version': 'fixture-version' });
  const first = await run();
  assert.equal(first.status, 200);
  assert.equal(JSON.stringify(first), JSON.stringify(await run()));
  assert.deepEqual(calls.map(call => call.id), ['gem', 'native', 'gem', 'native']);
});

test('fails over on missing final usage and includes translated usage when the client omits the option', async t => {
  const rows = [row('upstream', 'openai'), row('gem', 'gemini')];
  const ratings = [spec('upstream', 'best', 200), spec('gem', 'backup', 180)];
  const { providers, calls } = await install(t, rows, ratings, id => {
    if (id === 'upstream') {
      return sse([
        [undefined, { choices: [{ index: 0, delta: { content: 'unmetered' }, finish_reason: null }] }],
        [undefined, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }],
        [undefined, '[DONE]'],
      ]);
    }
    return sse([
      [undefined, { candidates: [{ content: { parts: [{ text: 'metered' }] }, finishReason: 'STOP' }] }],
      [undefined, { usageMetadata: { promptTokenCount: 19, candidatesTokenCount: 7, totalTokenCount: 29 } }],
    ]);
  });
  const reply = await createRouter(env, providers, ratings, [])('openai', {
    stream: true, messages: [{ role: 'user', content: 'question' }],
  }, {});
  assert.deepEqual(calls.map(call => call.id), ['upstream', 'gem']);
  assert.deepEqual(calls[0].payload.stream_options, { include_usage: true });
  assert.equal(reply.status, 200);
  assert.ok(Array.isArray(reply.body));
  const chunks = reply.body.slice(0, -1).map(frame => JSON.parse(frame.data));
  assert.equal(chunks.map(chunk => chunk.choices[0]?.delta.content ?? '').join(''), 'metered');
  assert.equal(chunks.at(-2).choices[0].finish_reason, 'stop');
  assert.deepEqual(chunks.at(-1).choices, []);
  assert.deepEqual(chunks.at(-1).usage, { prompt_tokens: 19, completion_tokens: 7, total_tokens: 29 });
  assert.equal(reply.body.at(-1)?.data, '[DONE]');
});

test('registers the exact serialized catalog, serves both HTTP APIs, and closes when work is done', async t => {
  const rows = [row('unit', 'openai')];
  const ratings = [spec('unit', 'demo', 100)];
  const providers = directory(rows, env);
  providers[0].models = [{ id: 'demo', free: true, context_length: 32_000 }, { id: 'paid', free: false, context_length: null }];
  let server: Server | undefined;
  let closed = false;
  let finished = false;
  let workload: Promise<void> | undefined;
  const port = 8123;

  t.mock.method(Server.prototype, 'listen', function (this: Server, requestedPort: number, host: string, ready: () => void) {
    assert.equal(requestedPort, 0);
    assert.equal(host, '0.0.0.0');
    server = this;
    Object.defineProperty(this, 'listening', { get: () => !closed });
    ready();
    return this;
  });
  t.mock.method(Server.prototype, 'address', () => ({ address: '0.0.0.0', family: 'IPv4', port }));
  t.mock.method(Server.prototype, 'close', function (this: Server, callback: () => void) {
    closed = true;
    callback();
    return this;
  });

  function request(method: string, url: string, headers: Record<string, string>, payload?: string) {
    return new Promise<{ status: number; data: Json; stream?: string }>((resolve, reject) => {
      const requestStream = Object.assign(Readable.from(payload === undefined ? [] : [Buffer.from(payload)]), { method, url, headers });
      let status = 0;
      let contentType = '';
      const writes: string[] = [];
      server!.emit('request', requestStream, {
        writeHead(code: number, responseHeaders: Record<string, string>) {
          status = code;
          contentType = responseHeaders['content-type'];
        },
        write(chunk: string) { writes.push(chunk); },
        end(text: string) {
          if (contentType.startsWith('text/event-stream')) resolve({ status, data: {}, stream: writes.join('') });
          else resolve({ status, data: JSON.parse(text) });
        },
        destroy: reject,
      });
    });
  }

  const body = JSON.stringify({ model: 'auto', messages: [{ role: 'user', content: 'question' }], max_tokens: 11 });
  const nativeUsage = { prompt_tokens: 8, completion_tokens: 5, total_tokens: 13,
    prompt_tokens_details: { cached_tokens: 2 }, completion_tokens_details: { reasoning_tokens: 1 } };
  t.mock.method(globalThis, 'fetch', async (source: string, options: RequestInit) => {
    const url = new URL(source);
    assert.equal(url.origin, env);
    if (url.pathname === '/register') {
      assert.equal(options.method, 'POST');
      assert.equal(options.body, `{"protocol_version":1,"port":${port},"catalog":${JSON.stringify(catalogOf(providers))}}`);
      workload = (async () => {
        const oa = await request('GET', '/v1/models', { authorization: 'bearer client' });
        assert.equal(oa.data.object, 'list');
        assert.deepEqual(oa.data.data.map((model: Json) => model.id), ['demo']);
        const an = await request('GET', '/v1/models', { 'x-api-key': 'client' });
        assert.equal(an.data.has_more, false);
        assert.deepEqual(an.data.data.map((model: Json) => model.id), ['demo']);
        const invalid = await request('POST', '/v1/messages', {}, '{');
        assert.equal(invalid.status, 400);
        assert.equal(invalid.data.type, 'error');
        assert.equal((await request('GET', '/unknown', {})).status, 404);
        const completionReply = await request('POST', '/v1/chat/completions', {}, body);
        assert.equal(completionReply.status, 200);
        assert.equal(completionReply.data.choices[0].message.content, 'served');
        const messageReply = await request('POST', '/v1/messages', {}, body);
        assert.equal(messageReply.status, 200);
        assert.deepEqual(messageReply.data.content, [{ type: 'text', text: 'served' }]);
        assert.equal(messageReply.data.stop_reason, 'end_turn');
        const stream = await request('POST', '/v1/chat/completions', {}, JSON.stringify({ ...JSON.parse(body), stream: true }));
        assert.equal(stream.status, 200);
        assert.ok(stream.stream);
        const frames = readFrames(stream.stream);
        assert.equal(JSON.parse(frames[0].data).choices[0].delta.content, 'served');
        assert.equal(JSON.parse(frames[1].data).choices[0].finish_reason, 'stop');
        assert.deepEqual(JSON.parse(frames[2].data), { choices: [], usage: nativeUsage });
        assert.equal(frames.length, 4);
        assert.equal(frames.at(-1)?.data, '[DONE]');
      })().finally(() => { finished = true; });
      // Both attach a rejection handler immediately and await the outcome below.
      void workload.catch(() => undefined);
      return json({ ok: true, requests: 3 });
    }
    if (url.pathname === '/work') return json({ done: finished });
    assert.equal(url.pathname, '/p/unit/v1/chat/completions');
    const payload = JSON.parse(options.body as string);
    assert.equal(payload.model, 'demo');
    if (payload.stream) {
      const chunks: Array<[undefined, Json | '[DONE]']> = [
        [undefined, { choices: [{ index: 0, delta: { role: 'assistant', content: 'served' }, finish_reason: null }] }],
        [undefined, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }],
      ];
      if (payload.stream_options?.include_usage === true) chunks.push([undefined, { choices: [], usage: nativeUsage }]);
      chunks.push([undefined, '[DONE]']);
      return sse(chunks);
    }
    return json(completion('demo', 'served'));
  });

  await serve(env, providers, ratings, []);
  assert.ok(workload);
  await workload;
  assert.ok(closed);
});
