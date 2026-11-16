# Free model aggregator

This entry for Make.fun's `free-model-aggregator-01` uses Node's standard library. The challenge entrypoint is `main.ts`, which targets the kit's Node 22.23.4 runtime with built-in TypeScript stripping.

## Run

Run the server inside a trial with its environment URL and brief path.

```sh
MAKEFUN_ENV_URL="$TRIAL_ENV" MAKEFUN_PROMPT_FILE="$TRIAL_BRIEF" node main.ts
```

It discovers models through the provider APIs, registers the catalog and an ephemeral port with `/register`, and closes its listener when `/work` reports completion. All outgoing HTTP goes to the configured trial origin.

`node main.ts --list` prints the same compact catalog JSON used for registration when run with the same brief and provider catalogs. With the trial variables unset, it prints `{"providers":[]}`.

| Route | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions JSON or SSE |
| `POST /v1/messages` | Anthropic Messages JSON or SSE |
| `GET /v1/models` | Free model IDs in OpenAI format for Bearer auth or Anthropic format for `x-api-key` |

## Routing

Provider catalogs determine free status. Only models marked free there and rated in the brief can serve inference. The router uses request fields and the current manifest entry to filter by tools, images, JSON support, and context. It orders available providers by the best eligible model's rating, favoring the client's native wire when ratings tie. Each request makes at most one inference attempt per provider. Provider health is updated from observed responses, with request-count cooldowns for repeated failures and retirement for explicit exhaustion or outages.

Requests use the provider's configured auth scheme and preserve the native payload on same-wire calls. Across wires the router translates system messages, text and images, tool definitions and history, generation controls, and structured output options that the destination represents. It reads the provider's stream through completion before writing SSE to the client, so an interrupted upstream response can still fail over before any partial answer is sent. This buffers the upstream stream and delays the first client event.

For the challenge, OpenAI streams include the provider's final token usage immediately before `[DONE]`, even if the client omits `stream_options.include_usage`. The router requests usage on every upstream OpenAI stream and tries another provider if a stream completes without the required counts.

## Test

```sh
node --test main.test.ts
node main.ts --list
```

The test suite runs in memory with synthetic catalogs and provider responses. See [QA.md](QA.md) for local practice scores and runtime limitations.

## License

This entry is available under the [MIT License](LICENSE). Copyright (c) 2026 Hasanuddin Sn.
