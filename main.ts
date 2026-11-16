import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

type Wire = 'openai' | 'anthropic' | 'gemini';
type ClientWire = Exclude<Wire, 'gemini'>;
// The three provider APIs carry vendor-specific extensions in otherwise JSON records.
type Json = Record<string, any>;
type Requirements = { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
type ManifestEntry = { id: string; api: ClientWire; stream: boolean; requires: Requirements; weight: number };
type RatedModel = Requirements & {
  provider: string; model: string; rating_milli: number; context_length: number; free: boolean;
};
type Model = { id: string; free: boolean; context_length: number | null };
type Provider = {
  id: string; url: string; base_url: string; key_env: string; wire: Wire;
  headers: Record<string, string>; models: Model[]; failures: number; retryAt: number; retired: boolean;
};
type Candidate = { provider: Provider; model: Model; rating: RatedModel };
type Image = { kind: 'image'; url: string; detail?: string };
type ToolCall = { kind: 'call'; id: string; name: string; args: string };
type Part =
  | { kind: 'text'; text: string }
  | Image
  | ToolCall
  | { kind: 'result'; id: string; parts: Part[]; error?: boolean }
  | { kind: 'refusal'; text: string };
type Turn = { role: 'user' | 'assistant'; parts: Part[] };
type Tool = { name: string; description?: string; schema?: Json; strict?: boolean };
type Format = { schema?: Json; name?: string; strict?: boolean; description?: string };
type Conversation = {
  system: string[]; turns: Turn[]; tools: Tool[]; choice?: string;
  parallel?: boolean; format?: Format; stops?: string[]; max?: number;
  temperature?: number; topP?: number; topK?: number; stream: boolean;
};
type Usage = { input?: number; output?: number; total?: number };
type Answer = {
  id?: string; created?: number; parts: Part[]; stop?: string; sequence?: string;
  usage: Usage;
};

const catalogPath = { openai: '/v1/models', anthropic: '/v1/models', gemini: '/v1beta/models' };
const messagePath = { openai: '/v1/chat/completions', anthropic: '/v1/messages' };
const defaultAnthropicVersion = '2023-06-01';
const maxRequestBytes = 16 * 1024 * 1024;
const maxAnswerBytes = 8 * 1024 * 1024;
const longContextTokens = 200_000;
const authHeaders: Record<string, string> = { bearer: 'authorization', 'x-api-key': 'x-api-key', 'x-goog-api-key': 'x-goog-api-key' };

const isObject = (value: unknown): value is Json => !!value && typeof value === 'object' && !Array.isArray(value);
const list = (value: unknown): Json[] => Array.isArray(value) ? value.filter(isObject) : [];
const number = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const string = (value: unknown): string | undefined => typeof value === 'string' ? value : undefined;

function briefDocuments(text: string): Json[] {
  const documents: Json[] = [];
  // The brief embeds labeled JSON documents on their own lines.
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    try {
      const document = JSON.parse(trimmed);
      if (isObject(document)) documents.push(document);
    } catch {
      // An unrelated prose line can start with a brace.
    }
  }
  return documents;
}

function sameOriginUrl(origin: string, address: string): URL {
  const result = new URL(address, origin);
  if (result.origin !== new URL(origin).origin || result.username || result.password) {
    throw new Error('Provider address is outside the trial environment');
  }
  return result;
}

function directory(rows: Json[], origin: string): Provider[] {
  return rows.map(row => {
    const wire = row.wire as Wire;
    if (!Object.hasOwn(catalogPath, wire) || typeof row.id !== 'string') {
      throw new Error('Invalid provider directory entry');
    }
    const keyEnv = string(row.key_env) ?? '';
    const auth = string(row.auth) ?? 'none';
    const key = string(row.key);
    const address = string(row.path) ?? string(row.base_url);
    if (!address) throw new Error('Provider directory has no address');
    const url = sameOriginUrl(origin, address);
    const headers: Record<string, string> = {};
    if (auth !== 'none') {
      if (!Object.hasOwn(authHeaders, auth) || !key) throw new Error('Invalid provider authentication configuration');
      headers[authHeaders[auth]] = auth === 'bearer' ? `Bearer ${key}` : key;
    }
    if (wire === 'anthropic') headers['anthropic-version'] = defaultAnthropicVersion;
    return {
      id: row.id, url: url.href.replace(/\/$/, ''), base_url: url.pathname,
      key_env: keyEnv, wire, headers, models: [], failures: 0, retryAt: 0, retired: false,
    };
  });
}

async function fetchTrial(url: string, init: RequestInit = {}, timeout = 4_000): Promise<Response> {
  return fetch(url, { ...init, redirect: 'error', signal: AbortSignal.timeout(timeout) });
}

async function getJson(url: string, init?: RequestInit): Promise<Json> {
  const response = await fetchTrial(url, init);
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`Trial endpoint returned HTTP ${response.status}`);
  }
  const value: unknown = await response.json();
  if (!isObject(value)) throw new Error('Trial endpoint returned an invalid JSON object');
  return value;
}

async function discover(provider: Provider): Promise<void> {
  const models = new Map<string, Model>();
  const visited = new Set<string>();
  let cursor: string | undefined;
  do {
    const url = new URL(provider.url + catalogPath[provider.wire]);
    if (cursor) url.searchParams.set(provider.wire === 'gemini' ? 'pageToken' : 'after_id', cursor);
    const page = await getJson(url.href, { headers: provider.headers });
    for (const item of list(provider.wire === 'gemini' ? page.models : page.data)) {
      const id = string(item.id) ?? string(item.name)?.replace(/^models\//, '');
      if (id && typeof item.free === 'boolean') {
        models.set(id, {
          id, free: item.free,
          context_length: number(item.context_length) ?? number(item.inputTokenLimit) ?? null,
        });
      }
    }
    cursor = provider.wire === 'gemini' ? string(page.nextPageToken)
      : page.has_more ? string(page.last_id) ?? string(list(page.data).at(-1)?.id) : undefined;
    if (cursor && visited.has(cursor)) throw new Error('Provider catalog repeated its pagination cursor');
    if (cursor) visited.add(cursor);
  } while (cursor);
  provider.models = [...models.values()];
}

function catalogOf(providers: Provider[]) {
  return {
    providers: providers.map(({ id, base_url, key_env, wire, models }) => ({ id, base_url, key_env, wire, models })),
  };
}

function clientModels(providers: Provider[], wire: ClientWire): Json {
  const owners = new Map<string, string>();
  for (const provider of providers) {
    for (const model of provider.models) {
      if (model.free && !owners.has(model.id)) owners.set(model.id, provider.id);
    }
  }
  if (wire === 'openai') {
    const data = [...owners].map(([id, owned_by]) => ({ id, object: 'model', created: 0, owned_by }));
    return { object: 'list', data };
  }
  const data = [...owners.keys()].map(id => ({ id, type: 'model', display_name: id, created_at: '1970-01-01T00:00:00Z' }));
  return { data, has_more: false, first_id: data[0]?.id ?? null, last_id: data.at(-1)?.id ?? null };
}

function anthropicParts(content: unknown): Part[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  return list(content).flatMap((block): Part[] => {
    switch (block.type) {
      case 'text': return typeof block.text === 'string' ? [{ kind: 'text', text: block.text }] : [];
      case 'image': {
        const source = block.source;
        if (source?.type === 'base64' && typeof source.data === 'string') {
          return [{ kind: 'image', url: `data:${source.media_type};base64,${source.data}` }];
        }
        if (source?.type === 'url' && typeof source.url === 'string') return [{ kind: 'image', url: source.url }];
        return [];
      }
      case 'tool_use':
        return [{ kind: 'call', id: block.id, name: block.name, args: JSON.stringify(block.input ?? {}) }];
      case 'tool_result':
        return [{ kind: 'result', id: block.tool_use_id, parts: anthropicParts(block.content), error: block.is_error }];
      default: return [];
    }
  });
}

function openaiParts(content: unknown): Part[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  return list(content).flatMap((block): Part[] => {
    if (block.type === 'text' && typeof block.text === 'string') return [{ kind: 'text', text: block.text }];
    if (block.type === 'image_url' && typeof block.image_url?.url === 'string') {
      return [{ kind: 'image', url: block.image_url.url, detail: block.image_url.detail }];
    }
    if (block.type === 'refusal' && typeof block.refusal === 'string') return [{ kind: 'refusal', text: block.refusal }];
    return [];
  });
}

function parseConversation(body: Json, wire: ClientWire): Conversation {
  const openai = wire === 'openai';
  const conversation: Conversation = {
    system: openai ? [] : (typeof body.system === 'string' ? [body.system] : list(body.system).map(block => block.text).filter(text => typeof text === 'string')),
    turns: [], tools: [], stream: body.stream === true,
    max: number(body.max_tokens) ?? number(body.max_completion_tokens),
    temperature: number(body.temperature), topP: number(body.top_p), topK: number(body.top_k),
  };
  const stops = openai ? body.stop : body.stop_sequences;
  if (typeof stops === 'string') conversation.stops = [stops];
  if (Array.isArray(stops)) conversation.stops = stops.filter(stop => typeof stop === 'string');
  for (const message of list(body.messages)) {
    if (openai && ['system', 'developer'].includes(message.role)) {
      conversation.system.push(...openaiParts(message.content).flatMap(part => part.kind === 'text' ? [part.text] : []));
      continue;
    }
    if (openai && ['tool', 'function'].includes(message.role)) {
      const result: Part = { kind: 'result', id: message.tool_call_id ?? message.name, parts: openaiParts(message.content) };
      const previous = conversation.turns.at(-1);
      if (previous?.role === 'user' && previous.parts.length && previous.parts.every(part => part.kind === 'result')) previous.parts.push(result);
      else conversation.turns.push({ role: 'user', parts: [result] });
      continue;
    }
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    const parts = openai ? openaiParts(message.content) : anthropicParts(message.content);
    if (openai) {
      const calls = list(message.tool_calls);
      if (isObject(message.function_call)) calls.push({ id: message.function_call.name, function: message.function_call });
      for (const call of calls) {
        if (typeof call.function?.name === 'string') {
          parts.push({ kind: 'call', id: call.id, name: call.function.name, args: call.function.arguments ?? '{}' });
        }
      }
      if (typeof message.refusal === 'string') parts.push({ kind: 'refusal', text: message.refusal });
    }
    const previous = conversation.turns.at(-1);
    if (openai && message.role === 'user' && previous?.role === 'user' && previous.parts.length && previous.parts.every(part => part.kind === 'result')) {
      previous.parts.push(...parts);
    } else {
      conversation.turns.push({ role: message.role, parts });
    }
  }
  conversation.tools = (openai ? list(body.tools).flatMap(tool => isObject(tool.function) ? [tool.function] : []).concat(list(body.functions)) : list(body.tools))
    .filter(tool => typeof tool.name === 'string')
    .map(tool => ({ name: tool.name, description: tool.description, schema: openai ? tool.parameters : tool.input_schema, strict: tool.strict }));
  const choice = body.tool_choice ?? body.function_call;
  if (typeof choice === 'string') conversation.choice = choice;
  if (isObject(choice)) {
    const name = openai ? choice.function?.name ?? choice.name : choice.name;
    conversation.choice = string(name) ?? (choice.type === 'any' ? 'required' : string(choice.type));
    if (!openai && typeof choice.disable_parallel_tool_use === 'boolean') conversation.parallel = !choice.disable_parallel_tool_use;
  }
  if (openai && typeof body.parallel_tool_calls === 'boolean') conversation.parallel = body.parallel_tool_calls;

  const format = openai ? body.response_format : body.output_config?.format ?? body.output_format;
  if (format?.type === 'json_schema') {
    conversation.format = openai ? {
      schema: format.json_schema?.schema, name: format.json_schema?.name,
      description: format.json_schema?.description, strict: format.json_schema?.strict,
    } : { schema: format.schema };
  } else if (format?.type === 'json_object') {
    conversation.format = {};
  }
  return conversation;
}

function decodeImage(url: string): { mime: string; data: string } | undefined {
  const match = /^data:([^;,]+)(?:;[^;,]+)*;base64,([\s\S]*)$/i.exec(url);
  if (match) return { mime: match[1], data: match[2] };
}

function argsObject(args: string): Json {
  const parsed: unknown = JSON.parse(args);
  if (!isObject(parsed)) throw new Error('Tool arguments must be a JSON object');
  return parsed;
}

function textOf(parts: Part[]): string {
  return parts.flatMap(part => part.kind === 'text' || part.kind === 'refusal' ? [part.text] : []).join('\n');
}

function encodeAnthropicPart(part: Part): Json {
  switch (part.kind) {
    case 'text':
    case 'refusal':
      return { type: 'text', text: part.text };
    case 'image': {
      const image = decodeImage(part.url);
      return { type: 'image', source: image ? { type: 'base64', media_type: image.mime, data: image.data } : { type: 'url', url: part.url } };
    }
    case 'call':
      return { type: 'tool_use', id: part.id, name: part.name, input: argsObject(part.args) };
    case 'result':
      return { type: 'tool_result', tool_use_id: part.id, content: part.parts.map(encodeAnthropicPart), is_error: part.error };
  }
}

function openaiContent(parts: Part[]): string | Json[] {
  if (parts.every(part => part.kind === 'text') && parts.length <= 1) return textOf(parts);
  return parts.flatMap((part): Json[] => {
    if (part.kind === 'image') return [{ type: 'image_url', image_url: { url: part.url, detail: part.detail } }];
    if (part.kind === 'text') return [{ type: 'text', text: part.text }];
    return [];
  });
}

function openaiCall(part: ToolCall): Json {
  return { id: part.id, type: 'function', function: { name: part.name, arguments: part.args } };
}

function toolCalls(parts: Part[]): ToolCall[] {
  return parts.filter((part): part is ToolCall => part.kind === 'call');
}

function toOpenai(conversation: Conversation, model: string): Json {
  const messages: Json[] = conversation.system.map(content => ({ role: 'system', content }));
  for (const turn of conversation.turns) {
    for (const part of turn.parts) {
      if (part.kind === 'result') messages.push({ role: 'tool', tool_call_id: part.id, content: openaiContent(part.parts) });
    }
    const contentParts = turn.parts.filter(part => part.kind === 'text' || part.kind === 'image');
    const calls = toolCalls(turn.parts).map(openaiCall);
    const refusal = turn.parts.filter(part => part.kind === 'refusal').map(part => part.text).join('');
    if (contentParts.length || calls.length || refusal || turn.parts.length === 0) {
      messages.push({ role: turn.role, content: contentParts.length || !calls.length ? openaiContent(contentParts) : null,
        tool_calls: calls.length ? calls : undefined, refusal: refusal || undefined });
    }
  }
  const { choice, format } = conversation;
  return {
    model, messages, stream: conversation.stream,
    temperature: conversation.temperature, top_p: conversation.topP, max_tokens: conversation.max, stop: conversation.stops,
    tools: conversation.tools.length ? conversation.tools.map(tool => ({
      type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.schema, strict: tool.strict },
    })) : undefined,
    tool_choice: !choice || ['auto', 'required', 'none'].includes(choice) ? choice : { type: 'function', function: { name: choice } },
    parallel_tool_calls: conversation.parallel,
    response_format: format?.schema ? { type: 'json_schema', json_schema: {
      name: format.name ?? 'response', schema: format.schema, strict: format.strict, description: format.description,
    } } : format ? { type: 'json_object' } : undefined,
  };
}

function toAnthropic(conversation: Conversation, model: string): Json {
  const { choice, parallel, format } = conversation;
  const toolChoice = choice === 'required' ? { type: 'any' } : !choice || ['auto', 'none'].includes(choice) ?
    { type: choice ?? 'auto' } : { type: 'tool', name: choice };
  return {
    model, stream: conversation.stream, max_tokens: conversation.max ?? 4096,
    system: conversation.system.length ? conversation.system.map(text => ({ type: 'text', text })) : undefined,
    messages: conversation.turns.map(turn => ({ role: turn.role, content: turn.parts.map(encodeAnthropicPart) })),
    temperature: conversation.temperature, top_p: conversation.topP, top_k: conversation.topK, stop_sequences: conversation.stops,
    tools: conversation.tools.length ? conversation.tools.map(tool => ({
      name: tool.name, description: tool.description, input_schema: tool.schema ?? { type: 'object' }, strict: tool.strict,
    })) : undefined,
    tool_choice: choice || parallel !== undefined ? { ...toolChoice, disable_parallel_tool_use: parallel === undefined ? undefined : !parallel } : undefined,
    output_config: format?.schema ? { format: { type: 'json_schema', schema: format.schema } } : undefined,
  };
}

function geminiImage(part: Image): Json {
  const image = decodeImage(part.url);
  return image ? { inlineData: { mimeType: image.mime, data: image.data } } : { fileData: { fileUri: part.url } };
}

function toGemini(conversation: Conversation): Json {
  const calls = new Map<string, string>();
  const { choice, format } = conversation;
  return {
    systemInstruction: conversation.system.length ? { parts: conversation.system.map(text => ({ text })) } : undefined,
    contents: conversation.turns.map(turn => ({
      role: turn.role === 'assistant' ? 'model' : 'user',
      parts: turn.parts.map((part): Json => {
        switch (part.kind) {
          case 'text':
          case 'refusal':
            return { text: part.text };
          case 'image':
            return geminiImage(part);
          case 'call':
            calls.set(part.id, part.name);
            return { functionCall: { id: part.id, name: part.name, args: argsObject(part.args) } };
          case 'result': {
            const name = calls.get(part.id);
            if (!name) throw new Error('Tool result has no matching call');
            let result: unknown = textOf(part.parts);
            try { result = JSON.parse(result as string); } catch { /* Plain text is also a tool result. */ }
            const images = part.parts.filter((item): item is Image => item.kind === 'image');
            return { functionResponse: {
              id: part.id, name, response: part.error ? { error: result } : isObject(result) ? result : { output: result },
              parts: images.length ? images.map(geminiImage) : undefined,
            } };
          }
        }
      }),
    })),
    generationConfig: {
      maxOutputTokens: conversation.max, temperature: conversation.temperature, topP: conversation.topP, topK: conversation.topK,
      stopSequences: conversation.stops, responseMimeType: format ? 'application/json' : undefined, responseJsonSchema: format?.schema,
    },
    tools: conversation.tools.length ? [{ functionDeclarations: conversation.tools.map(tool => ({
      name: tool.name, description: tool.description, parametersJsonSchema: tool.schema,
    })) }] : undefined,
    toolConfig: choice ? { functionCallingConfig: {
      mode: choice === 'none' ? 'NONE' : choice === 'auto' ? 'AUTO' : 'ANY',
      allowedFunctionNames: ['none', 'auto', 'required'].includes(choice) ? undefined : [choice],
    } } : undefined,
  };
}

type Frame = { event?: string; data: string };

function readFrames(text: string): Frame[] {
  const frames: Frame[] = [];
  let event: string | undefined;
  let data: string[] = [];
  for (const line of text.split(/\r\n|\n|\r/)) {
    if (line === '') {
      if (data.length) frames.push({ event, data: data.join('\n') });
      event = undefined;
      data = [];
      continue;
    }
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
    if (field === 'event') event = value;
    if (field === 'data') data.push(value);
  }
  if (data.length) frames.push({ event, data: data.join('\n') });
  return frames;
}

function frameText(frame: Frame): string {
  const event = frame.event ? `event: ${frame.event}\n` : '';
  return event + frame.data.split('\n').map(line => `data: ${line}\n`).join('') + '\n';
}

function jsonFrame(value: Json, event?: string): Frame {
  return { event, data: JSON.stringify(value) };
}

function tokenCount(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value;
}

function readVendorUsage(wire: Wire, usage: Json | undefined, answer: Answer): void {
  if (!usage) return;
  const [input, output, total] = {
    openai: ['prompt_tokens', 'completion_tokens', 'total_tokens'],
    anthropic: ['input_tokens', 'output_tokens', ''],
    gemini: ['promptTokenCount', 'candidatesTokenCount', 'totalTokenCount'],
  }[wire];
  answer.usage.input = tokenCount(usage[input]) ?? answer.usage.input;
  answer.usage.output = tokenCount(usage[output]) ?? answer.usage.output;
  answer.usage.total = tokenCount(usage[total]) ?? answer.usage.total;
}

function readOpenaiAnswer(answer: Answer, events: Json[], stream: boolean, id: string): void {
  const pieces = new Map<string, Part>();
  for (const event of events) {
    if (event.error) throw new Error('Provider sent a stream error');
    answer.id = string(event.id) ?? answer.id;
    answer.created = number(event.created) ?? answer.created;
    if (!stream) readVendorUsage('openai', event.usage, answer);
    const choice = list(event.choices).find(row => row.index === 0) ?? list(event.choices)[0];
    if (!choice) continue;
    answer.stop = string(choice.finish_reason) ?? answer.stop;
    const message = stream ? choice.delta : choice.message;
    if (!isObject(message)) continue;
    if (typeof message.content === 'string' && message.content !== '') {
      let text = pieces.get('text');
      if (text?.kind !== 'text') {
        text = { kind: 'text', text: '' };
        pieces.set('text', text);
      }
      text.text += message.content;
    } else if (Array.isArray(message.content)) {
      for (const part of openaiParts(message.content)) pieces.set(`content-${pieces.size}`, part);
    }
    if (typeof message.refusal === 'string') {
      let refusal = pieces.get('refusal');
      if (refusal?.kind !== 'refusal') {
        refusal = { kind: 'refusal', text: '' };
        pieces.set('refusal', refusal);
      }
      refusal.text += message.refusal;
    }
    const calls = list(message.tool_calls);
    if (isObject(message.function_call)) calls.push({ index: 0, function: message.function_call });
    for (const [position, call] of calls.entries()) {
      const index = number(call.index) ?? position;
      const key = `tool-${index}`;
      let part = pieces.get(key);
      if (part?.kind !== 'call') {
        part = { kind: 'call', id: `${id}-tool-${index}`, name: '', args: '' };
        pieces.set(key, part);
      }
      if (typeof call.id === 'string') part.id = call.id;
      part.name += string(call.function?.name) ?? '';
      part.args += string(call.function?.arguments) ?? '';
    }
  }
  const last = events.at(-1);
  if (stream && Array.isArray(last?.choices) && last.choices.length === 0) {
    readVendorUsage('openai', last.usage, answer);
  }
  answer.parts = [...pieces.values()];
}

function readAnthropicAnswer(answer: Answer, events: Json[], stream: boolean): void {
  const blocks = new Map<number, Json>();
  for (const event of events) {
    if (event.type === 'error') throw new Error('Provider sent a stream error');
    if (!stream || event.type === 'message_start') {
      const message = stream ? event.message : event;
      answer.id = string(message?.id);
      answer.stop = string(message?.stop_reason);
      answer.sequence = string(message?.stop_sequence);
      const usage = stream ? { input_tokens: message?.usage?.input_tokens } : message?.usage;
      readVendorUsage('anthropic', usage, answer);
      for (const [index, block] of list(message?.content).entries()) blocks.set(index, { ...block });
    } else if (event.type === 'content_block_start') {
      blocks.set(event.index, { ...event.content_block });
    } else if (event.type === 'content_block_delta') {
      const block = blocks.get(event.index);
      if (!block) throw new Error('Provider emitted a delta before its content block');
      if (event.delta?.type === 'text_delta') block.text = (block.text ?? '') + event.delta.text;
      if (event.delta?.type === 'input_json_delta') block.args = (block.args ?? '') + event.delta.partial_json;
    } else if (event.type === 'message_delta') {
      answer.stop = string(event.delta?.stop_reason) ?? answer.stop;
      answer.sequence = string(event.delta?.stop_sequence) ?? answer.sequence;
      readVendorUsage('anthropic', event.usage, answer);
    }
  }
  answer.parts = [...blocks].sort(([a], [b]) => a - b).flatMap(([, block]): Part[] => {
    if (block.type === 'tool_use' && typeof block.args === 'string') {
      return [{ kind: 'call', id: block.id, name: block.name, args: block.args }];
    }
    return anthropicParts([block]);
  });
}

function readGeminiAnswer(answer: Answer, events: Json[], id: string): void {
  for (const event of events) {
    if (event.error) throw new Error('Provider sent a stream error');
    answer.id = string(event.responseId) ?? answer.id;
    readVendorUsage('gemini', event.usageMetadata, answer);
    const candidate = list(event.candidates).find(row => row.index === 0) ?? list(event.candidates)[0];
    if (!candidate) {
      if (event.promptFeedback?.blockReason) answer.stop = 'SAFETY';
      continue;
    }
    answer.stop = string(candidate.finishReason) ?? answer.stop;
    for (const part of list(candidate.content?.parts)) {
      if (part.thought === true) continue;
      if (typeof part.text === 'string' && part.text !== '') {
        const previous = answer.parts.at(-1);
        if (previous?.kind === 'text') previous.text += part.text;
        else answer.parts.push({ kind: 'text', text: part.text });
      }
      if (isObject(part.functionCall)) {
        const call = part.functionCall;
        answer.parts.push({ kind: 'call', id: call.id ?? `${id}-tool-${answer.parts.length}`, name: call.name, args: JSON.stringify(call.args ?? {}) });
      }
    }
  }
}

function decodeAnswer(wire: Wire, events: Json[], streamed: boolean, id: string): Answer {
  const answer: Answer = { parts: [], usage: {} };
  if (wire === 'openai') readOpenaiAnswer(answer, events, streamed, id);
  if (wire === 'anthropic') readAnthropicAnswer(answer, events, streamed);
  if (wire === 'gemini') readGeminiAnswer(answer, events, id);
  if (!answer.stop) throw new Error('Provider answer ended without a stop reason');
  return answer;
}

function reason(answer: Answer, wire: ClientWire): string {
  const stop = answer.stop ?? '';
  if (['max_tokens', 'length', 'MAX_TOKENS'].includes(stop)) return wire === 'openai' ? 'length' : 'max_tokens';
  if (['refusal', 'content_filter', 'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'RECITATION', 'IMAGE_SAFETY', 'MODEL_ARMOR'].includes(stop)) {
    return wire === 'openai' ? 'content_filter' : 'refusal';
  }
  if (['tool_use', 'tool_calls', 'function_call'].includes(stop) || toolCalls(answer.parts).length) return wire === 'openai' ? 'tool_calls' : 'tool_use';
  if (wire === 'anthropic' && (stop === 'stop_sequence' || answer.sequence !== undefined)) return 'stop_sequence';
  if (wire === 'anthropic' && ['pause_turn', 'model_context_window_exceeded'].includes(stop)) return stop;
  return wire === 'openai' ? 'stop' : 'end_turn';
}

function asOpenaiUsage({ input, output, total }: Usage): Json | undefined {
  if (input === undefined || output === undefined) return undefined;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: total ?? input + output };
}

function* textDeltas(text: string): Generator<string> {
  let delta = '';
  for (const character of text) {
    delta += character;
    if (delta.length >= 192) {
      yield delta;
      delta = '';
    }
  }
  if (delta) yield delta;
}

function openaiResponse(answer: Answer, model: string, id: string, stream: boolean): Json | Frame[] {
  const meta = { id: answer.id ?? `chatcmpl-${id}`, created: answer.created ?? 0, model };
  const calls = toolCalls(answer.parts).map(openaiCall);
  const text = answer.parts.filter(part => part.kind === 'text').map(part => part.text).join('');
  const refusal = answer.parts.filter(part => part.kind === 'refusal').map(part => part.text).join('');
  const stop = reason(answer, 'openai');
  const usage = asOpenaiUsage(answer.usage);
  if (!stream) {
    const message = { role: 'assistant', content: text || (calls.length || refusal ? null : ''),
      tool_calls: calls.length ? calls : undefined, refusal: refusal || undefined };
    return { ...meta, object: 'chat.completion', choices: [{ index: 0, message, finish_reason: stop }], usage };
  }
  const chunkMeta = { ...meta, object: 'chat.completion.chunk' };
  const chunk = (delta: Json, finish_reason: string | null = null) => jsonFrame({
    ...chunkMeta, choices: [{ index: 0, delta, finish_reason }], usage: null,
  });
  const frames = [chunk({ role: 'assistant', content: '' })];
  for (const content of textDeltas(text)) frames.push(chunk({ content }));
  for (const value of textDeltas(refusal)) frames.push(chunk({ refusal: value }));
  for (const [index, call] of calls.entries()) {
    frames.push(chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.function.name, arguments: '' } }] }));
    for (const args of textDeltas(call.function.arguments)) frames.push(chunk({ tool_calls: [{ index, function: { arguments: args } }] }));
  }
  frames.push(chunk({}, stop));
  frames.push(jsonFrame({ ...chunkMeta, choices: [], usage }));
  frames.push({ data: '[DONE]' });
  return frames;
}

function anthropicResponse(answer: Answer, model: string, id: string, stream: boolean): Json | Frame[] {
  const meta = { id: answer.id ?? `msg_${id}`, type: 'message', role: 'assistant', model };
  const content = answer.parts.map(encodeAnthropicPart);
  const stop = { stop_reason: reason(answer, 'anthropic'), stop_sequence: answer.sequence ?? null };
  const { input, output } = answer.usage;
  if (input === undefined || output === undefined) throw new Error('Provider answer omitted token usage');
  const usage = { input_tokens: input, output_tokens: output };
  if (!stream) return { ...meta, content, ...stop, usage };
  const frame = (type: string, data: Json = {}) => jsonFrame({ type, ...data }, type);
  const frames = [frame('message_start', { message: {
    ...meta, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
  } })];
  for (const [index, block] of content.entries()) {
    if (block.type === 'text') {
      frames.push(frame('content_block_start', { index, content_block: { type: 'text', text: '' } }));
      for (const text of textDeltas(block.text)) frames.push(frame('content_block_delta', { index, delta: { type: 'text_delta', text } }));
    } else if (block.type === 'tool_use') {
      frames.push(frame('content_block_start', { index, content_block: { ...block, input: {} } }));
      for (const partial_json of textDeltas(JSON.stringify(block.input))) {
        frames.push(frame('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json } }));
      }
    } else {
      frames.push(frame('content_block_start', { index, content_block: block }));
    }
    frames.push(frame('content_block_stop', { index }));
  }
  frames.push(frame('message_delta', { delta: stop, usage: { output_tokens: usage.output_tokens } }), frame('message_stop'));
  return frames;
}

class ProviderError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`Provider returned HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function cappedText(response: Response): Promise<string> {
  if (!response.body) return '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let text = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return text + decoder.decode();
      size += value.byteLength;
      if (size > maxAnswerBytes) {
        await reader.cancel();
        throw new Error('Provider answer exceeded the response limit');
      }
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
}

async function invoke(candidate: Candidate, wire: ClientWire, body: Json, headers: IncomingMessage['headers'],
  conversation: Conversation, id: string): Promise<Json | Frame[]> {
  const { provider, model } = candidate;
  const native = provider.wire === wire;
  const stream = conversation.stream;
  const upstreamHeaders = { ...provider.headers, 'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json' };
  let payload: Json;
  let endpoint: string;
  if (provider.wire === 'gemini') {
    endpoint = `/v1beta/models/${encodeURIComponent(model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    payload = toGemini(conversation);
  } else {
    endpoint = messagePath[provider.wire];
    if (native) {
      payload = { ...body, model: model.id };
    } else {
      payload = provider.wire === 'openai' ? toOpenai(conversation, model.id) : toAnthropic(conversation, model.id);
    }
    if (provider.wire === 'anthropic') {
      const version = string(headers['anthropic-version']);
      const beta = string(headers['anthropic-beta']);
      if (version) upstreamHeaders['anthropic-version'] = version;
      if (beta && wire === 'anthropic') upstreamHeaders['anthropic-beta'] = beta;
    } else if (wire === 'openai' && typeof headers['openai-beta'] === 'string') {
      upstreamHeaders['openai-beta'] = headers['openai-beta'];
    }
  }
  if (stream && provider.wire === 'openai') {
    const options = isObject(payload.stream_options) ? payload.stream_options : {};
    payload.stream_options = { ...options, include_usage: true };
  }
  const response = await fetchTrial(provider.url + endpoint, {
    method: 'POST', headers: upstreamHeaders, body: JSON.stringify(payload),
  }, 7_500);
  const text = await cappedText(response);
  if (!response.ok) throw new ProviderError(response.status, text.slice(0, 2048));
  if (stream && !/^text\/event-stream\b/i.test(response.headers.get('content-type') ?? '')) {
    throw new Error('Provider returned a non-SSE response to a streaming request');
  }
  const frames = stream ? readFrames(text) : [];
  if (stream && provider.wire === 'openai' && frames.at(-1)?.data !== '[DONE]') throw new Error('OpenAI stream is incomplete');
  if (stream && provider.wire === 'anthropic' && frames.at(-1)?.event !== 'message_stop') throw new Error('Anthropic stream is incomplete');
  const events = stream ? frames.filter(frame => frame.data !== '[DONE]').map(frame => JSON.parse(frame.data)) : [JSON.parse(text)];
  const answer = decodeAnswer(provider.wire, events, stream, id);
  if (stream && (answer.usage.input === undefined || answer.usage.output === undefined)) {
    throw new Error('Provider stream omitted final token usage');
  }
  if (native) return stream ? frames : events[0];
  return wire === 'openai' ? openaiResponse(answer, model.id, id, stream)
    : anthropicResponse(answer, model.id, id, stream);
}

function requestRequirements(conversation: Conversation, manifest: Requirements): Requirements {
  let vision = false;
  let tools = conversation.tools.length > 0;
  let bytes = conversation.system.reduce((sum, text) => sum + Buffer.byteLength(text), 0);
  const visit = (part: Part) => {
    if (part.kind === 'image') vision = true;
    if (part.kind === 'text' || part.kind === 'refusal') bytes += Buffer.byteLength(part.text);
    if (part.kind === 'call') {
      tools = true;
      bytes += Buffer.byteLength(part.args);
    }
    if (part.kind === 'result') {
      tools = true;
      part.parts.forEach(visit);
    }
  };
  for (const turn of conversation.turns) turn.parts.forEach(visit);
  for (const tool of conversation.tools) bytes += Buffer.byteLength(JSON.stringify(tool));
  if (conversation.format?.schema) bytes += Buffer.byteLength(JSON.stringify(conversation.format.schema));
  const estimate = Math.ceil(bytes / 4) + (conversation.max ?? 0) + 4 * conversation.turns.length;
  return {
    tools: tools || manifest.tools, vision: vision || manifest.vision,
    json: !!conversation.format || manifest.json, min_context: Math.max(estimate, manifest.min_context ?? 0),
  };
}

function supports({ model, rating }: Candidate, requirements: Requirements): boolean {
  const context = Math.min(model.context_length ?? Infinity, rating.context_length);
  if (context < (requirements.min_context ?? 0)) return false;
  if (requirements.tools && !rating.tools || requirements.vision && !rating.vision || requirements.json && !rating.json) return false;
  return true;
}

function accepts(candidate: Candidate, requirements: Requirements, conversation: Conversation, wire: ClientWire, body: Json): boolean {
  const { provider } = candidate;
  if (provider.retired || !supports(candidate, requirements)) return false;
  if (wire === 'openai' && conversation.format && provider.wire === 'anthropic') return false;
  if (wire === 'openai' && (body.n > 1 || body.logprobs === true) && provider.wire !== wire) return false;
  return true;
}

function updateHealth(provider: Provider, error: unknown, ordinal: number): void {
  if (error instanceof ProviderError) {
    const { status, detail } = error;
    if (status === 401 || status === 403 || ((status === 402 || status === 429) &&
      /budget.{0,30}(spent|exhausted)|quota.{0,30}(spent|exhausted)|\bis out\b|outage/i.test(detail))) {
      provider.retired = true;
      return;
    }
    if (status === 400 || /concurrent_requests/.test(detail)) return;
  }
  provider.failures += 1;
  // A schedule miss can be followed immediately by a success. Repeated failures
  // defer the next probe by request count, with an escape when all alternatives fail.
  provider.retryAt = ordinal + Math.min(4, provider.failures);
}

function bestPerProvider(candidates: Candidate[], wire: ClientWire, ordinal: number): Candidate[] {
  const ready = (provider: Provider) => Number(provider.retryAt <= ordinal);
  const ordered = candidates.sort((a, b) =>
    ready(b.provider) - ready(a.provider) ||
    b.rating.rating_milli - a.rating.rating_milli ||
    Number(b.provider.wire === wire) - Number(a.provider.wire === wire) ||
    a.provider.failures - b.provider.failures ||
    (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
    (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  const selected = new Set<string>();
  return ordered.filter(({ provider }) => {
    if (selected.has(provider.id)) return false;
    selected.add(provider.id);
    return true;
  });
}

function ratedCandidates(providers: Provider[], ratings: RatedModel[]): Candidate[] {
  const byProvider = new Map<string, Map<string, RatedModel>>();
  for (const rating of ratings) {
    let models = byProvider.get(rating.provider);
    if (!models) {
      models = new Map();
      byProvider.set(rating.provider, models);
    }
    if (rating.free === true) models.set(rating.model, rating);
  }
  return providers.flatMap(provider => provider.models.flatMap(model => {
    const rating = byProvider.get(provider.id)?.get(model.id);
    return model.free && rating ? [{ provider, model, rating }] : [];
  }));
}

type Routed = { status: number; body: Json | Frame[] };

function apiError(wire: ClientWire, status: number, message: string): Routed {
  const type = status === 400 ? 'invalid_request_error' : status === 404 ? 'not_found_error'
    : wire === 'anthropic' ? 'overloaded_error' : 'service_unavailable';
  const error = wire === 'openai' ? { type, message, param: null, code: type } : { type, message };
  return { status, body: wire === 'anthropic' ? { type: 'error', error } : { error } };
}

function createRouter(origin: string, providers: Provider[], ratings: RatedModel[], manifest: ManifestEntry[]) {
  const candidates = ratedCandidates(providers, ratings);
  const byId = new Map(manifest.map(entry => [entry.id, entry]));
  const bestById = new Map(manifest.map(entry => [entry.id, Math.max(0, ...candidates
    .filter(candidate => supports(candidate, entry.requires)).map(candidate => candidate.rating.rating_milli))]));
  const bound = manifest.reduce((total, entry) => total + entry.weight * bestById.get(entry.id)!, 0);
  // Spend up to 1.5% of the weighted quality bound, leaving 0.5% for fallbacks.
  const discretionaryLoss = Math.floor(3 * bound / 200);
  const tried = new Set<Provider>();
  let weightedLoss = 0;
  let ordinal = 0;

  return async (wire: ClientWire, body: Json, headers: IncomingMessage['headers']): Promise<Routed> => {
    if (!Array.isArray(body.messages) || !body.messages.length) return apiError(wire, 400, 'messages must be a nonempty array');
    if (wire === 'anthropic' && (!Number.isInteger(body.max_tokens) || body.max_tokens <= 0)) {
      return apiError(wire, 400, 'max_tokens must be a positive integer');
    }
    const conversation = parseConversation(body, wire);
    const sequence = ordinal++;
    const work = await getJson(origin + '/work').catch(() => null);
    const entry = byId.get(work?.next) ?? manifest[sequence];
    const brief = entry?.api === wire && entry.stream === conversation.stream ? entry : undefined;
    const requirements = requestRequirements(conversation, brief?.requires ?? {});
    const eligible = candidates.filter(candidate => accepts(candidate, requirements, conversation, wire, body));
    const bestRating = brief ? bestById.get(brief.id) ?? 0 : 0;
    const weight = brief?.weight ?? 0;
    const loss = (rating: number) => weight * Math.max(0, bestRating - rating);
    const ordered = bestPerProvider(eligible, wire, sequence);
    if (bestRating && !requirements.vision && (requirements.min_context ?? 0) < longContextTokens) {
      const freshIndex = ordered.findIndex(({ provider }) => !tried.has(provider));
      if (freshIndex > 0 && weightedLoss + loss(ordered[freshIndex].rating.rating_milli) <= discretionaryLoss) {
        ordered.unshift(ordered.splice(freshIndex, 1)[0]);
      }
    }
    const id = `makefun-${sequence + 1}`;
    for (const candidate of ordered) {
      const { provider } = candidate;
      tried.add(provider);
      try {
        const reply = await invoke(candidate, wire, body, headers, conversation, id);
        weightedLoss += loss(candidate.rating.rating_milli);
        provider.failures = 0;
        provider.retryAt = 0;
        return { status: 200, body: reply };
      } catch (error) {
        updateHealth(provider, error, sequence);
      }
    }
    weightedLoss += loss(0);
    return apiError(wire, 503, eligible.length ? 'All eligible free providers failed' : 'No eligible free model is available');
  };
}

async function clientBody(request: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size <= maxRequestBytes) chunks.push(chunk);
  }
  if (size > maxRequestBytes) throw new Error('Request exceeds the body limit');
  const body: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!isObject(body)) throw new Error('Request must be a JSON object');
  return body;
}

function send(response: ServerResponse, answer: Routed): void {
  if (Array.isArray(answer.body)) {
    response.writeHead(answer.status, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' });
    for (const frame of answer.body) response.write(frameText(frame));
    response.end();
  } else {
    response.writeHead(answer.status, { 'content-type': 'application/json; charset=utf-8' });
    response.end(JSON.stringify(answer.body));
  }
}

async function serve(origin: string, providers: Provider[], ratings: RatedModel[], manifest: ManifestEntry[]): Promise<void> {
  const route = createRouter(origin, providers, ratings, manifest);
  let queue: Promise<unknown> = Promise.resolve();
  const server = createServer(async (request, response) => {
    const path = request.url?.split('?')[0];
    const modelWire = /^Bearer\s/i.test(String(request.headers.authorization ?? '')) ? 'openai'
      : request.headers['x-api-key'] !== undefined ? 'anthropic' : 'openai';
    const wire = path === '/v1/messages' ? 'anthropic' : path === '/v1/chat/completions' ? 'openai' : modelWire;
    if (request.method === 'GET' && path === '/v1/models') {
      send(response, { status: 200, body: clientModels(providers, modelWire) });
      return;
    }
    if (request.method !== 'POST' || path !== messagePath[wire]) {
      send(response, apiError(wire, 404, 'Unknown route'));
      return;
    }
    let body: Json;
    try {
      body = await clientBody(request);
    } catch {
      send(response, apiError(wire, 400, 'Expected a JSON request object within the body limit'));
      return;
    }
    // Serialize inference to keep every upstream within its one-call limit.
    const task = queue.then(() => route(wire, body, request.headers));
    queue = task.catch(() => undefined);
    try {
      send(response, await task);
    } catch {
      if (!response.headersSent) send(response, apiError(wire, 503, 'Could not complete the request'));
      else response.destroy();
    }
  });

  try {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => {
        server.off('error', reject);
        resolve();
      });
    });
    const bound = server.address();
    if (!bound || typeof bound === 'string') throw new Error('Could not determine server port');
    const registered = await getJson(origin + '/register', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port: bound.port, catalog: catalogOf(providers) }),
    });
    if (registered.ok !== true) throw new Error('Trial environment did not accept registration');
    let workErrors = 0;
    for (;;) {
      const work = await getJson(origin + '/work').catch(error => {
        workErrors += 1;
        if (workErrors >= 3) throw error;
        return null;
      });
      if (work) {
        workErrors = 0;
        if (work.done === true) break;
      }
      await delay(100);
    }
    await queue;
  } finally {
    if (server.listening) await new Promise<void>(resolve => server.close(() => resolve()));
  }
}

async function main(): Promise<void> {
  const promptPath = process.env.MAKEFUN_PROMPT_FILE;
  const envUrl = process.env.MAKEFUN_ENV_URL;
  const print = process.argv.includes('--list');
  if (!envUrl) {
    if (!print) throw new Error('Set MAKEFUN_ENV_URL to the trial environment');
    process.stdout.write(JSON.stringify(catalogOf([])));
    return;
  }
  const url = new URL(envUrl);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Invalid MAKEFUN_ENV_URL');
  const origin = url.origin;
  const documents = promptPath ? briefDocuments(await readFile(promptPath, 'utf8')) : [];
  const providersDoc = documents.find(doc => Array.isArray(doc.providers));
  const ratings = (documents.find(doc => Array.isArray(doc.ratings))?.ratings ?? []) as RatedModel[];
  const workloadDoc = documents.find(doc => Array.isArray(doc.manifest));
  const instance = providersDoc && workloadDoc ? null : await getJson(origin + '/instance');
  const providers = directory(list(providersDoc?.providers ?? instance?.providers), origin);
  const manifest = (workloadDoc?.manifest ?? instance?.workload?.manifest ?? []) as ManifestEntry[];
  const discovered = await Promise.allSettled(providers.map(provider => discover(provider)));
  for (const [index, result] of discovered.entries()) {
    if (result.status === 'rejected') process.stderr.write(`Could not discover catalog for ${providers[index].id}\n`);
  }
  if (print) {
    process.stdout.write(JSON.stringify(catalogOf(providers)));
    return;
  }
  await serve(origin, providers, ratings, manifest);
}

export { briefDocuments, catalogOf, clientModels, createRouter, directory, discover, frameText, readFrames, serve };

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    process.stderr.write(`Aggregator: ${error instanceof Error ? error.message : 'unexpected failure'}\n`);
    process.exitCode = 1;
  });
}
