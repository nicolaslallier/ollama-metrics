# Ollama Proxy Sidecar

A lightweight sidecar proxy for [Ollama](https://ollama.com/) that shapes outgoing **requests**, repairs incoming **responses**, and exposes Prometheus **metrics** for monitoring your LLM deployments.

![cover image](.docs/cover.png)

## Overview

Ollama Proxy Sidecar sits between your applications and Ollama, forwarding every
request upstream and doing three things along the way:

- **Request** - apply configured defaults and overrides before the request
  reaches Ollama.
- **Response** - repair invalid UTF-8 and keep client-visible output identical
  to Ollama's own.
- **Metrics** - expose Prometheus metrics on tokens, latency and loaded models.

Apart from the request fields you deliberately configure, it is transparent:
clients see the response Ollama would have sent, streaming included. Ollama
itself needs no modification.

Token usage is accounted for on both of Ollama's dialects: the native
`/api/generate` and `/api/chat` endpoints, and the OpenAI-compatible
`/v1/chat/completions` and `/v1/completions` endpoints, and the Anthropic-compatible
`/v1/messages` endpoint (what Claude Code speaks). All other paths are
proxied through unchanged, with request duration recorded.

## Features

The three jobs are independent, and all have safe defaults - the sidecar works
out of the box without setting a single variable.

### Request

Applied to `POST /v1/chat/completions` before the body is forwarded upstream.

- **Defaults** (`OLLAMA_PROXY_REQUEST_DEFAULTS`) - fields merged *under* the
  client's body, so any client can opt out by sending its own value. Use this
  for a fleet-wide preference.
- **Overrides** (`OLLAMA_PROXY_REQUEST_OVERRIDES`) - fields merged *over* the
  client's body, so the configured value always wins. Use this for a value
  clients must not change.
- **Usage injection** - streaming `/v1` requests get
  `stream_options.include_usage` added when the client did not set it. Ollama
  otherwise reports no token counts on that dialect, so this is what makes
  `/v1` token metrics possible at all.

Precedence when the same field is set in more than one place:
**overrides > client request > defaults**.

### Response

- **UTF-8 repair** (`OLLAMA_SANITIZE_UTF8_RESPONSE`, default `true`) - invalid
  byte sequences from the model are replaced with U+FFFD. A consumer that puts
  the reply into protobuf/gRPC drops the entire message over a single bad byte;
  repairing it costs one character. Set to `false` to forward bytes untouched.
- **Reasoning promotion** (`OLLAMA_PROMOTE_REASONING_TO_CONTENT`, default
  `true`) - a non-streaming `/v1` reply whose assistant message has empty
  `content` but non-empty `reasoning` gets the reasoning copied into `content`.
  A thinking model routinely spends its final turn entirely in `reasoning`, and
  a client reading only `content` then gets a successful, well-formed response
  carrying nothing. Choices with `tool_calls` are left alone, and `reasoning` is
  copied rather than moved, so a client reading both fields sees exactly what
  the model sent. Streaming responses are not touched - deciding that content
  stayed empty would mean buffering the whole stream. Set to `false` to forward
  the empty content as-is.
- **Client output unchanged** - the usage chunk the sidecar asked for on the
  client's behalf is withheld on the way back, so the stream a client sees is
  the one it would have got talking to Ollama directly. A client that requested
  usage itself has its chunk forwarded.
- **Streaming preserved** - responses are relayed and flushed chunk by chunk,
  never buffered to completion.

### Metrics

- Prometheus-compatible endpoint at `/metrics`.
- Token usage, request duration, time per token, time to first token, and
  loaded-model and RAM gauges - see [Available Metrics](#available-metrics).
- Recorded across both API dialects, with per-model and per-endpoint labels.
- Ships with a pre-built Grafana dashboard.

## Usage

### Environment Variables

| Variable | Default | Applies to | Description |
| --- | --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | - | Upstream Ollama address |
| `PORT` | `8080` | - | Port the sidecar listens on |
| `DEBUG_MODE` | `false` | - | Log request input and response output bodies |
| `OLLAMA_PROXY_REQUEST_DEFAULTS` | `{}` | Request | JSON object merged *under* `POST /v1/chat/completions`; client values take precedence, so each client can opt out |
| `OLLAMA_PROXY_REQUEST_OVERRIDES` | `{}` | Request | JSON object merged *over* `POST /v1/chat/completions`; configured values win over client values |
| `OLLAMA_SANITIZE_UTF8_RESPONSE` | `true` | Response | Replace invalid UTF-8 in upstream responses with U+FFFD; `false` forwards response bytes unchanged |
| `OLLAMA_PROMOTE_REASONING_TO_CONTENT` | `true` | Response | Copy `reasoning` into an empty `content` on non-streaming `/v1/chat/completions`; `false` leaves the empty content alone |

When the same field is set in more than one place, precedence is
**overrides > client request > defaults**.

`OLLAMA_SANITIZE_UTF8_RESPONSE` and `OLLAMA_PROMOTE_REASONING_TO_CONTENT` only
accept what Go's `ParseBool` reads (`true`/`false`, `1`/`0`, `t`/`f`). Anything
else is treated as a typo and logged as a warning, leaving the repair enabled -
the protection is never switched off by accident.

Reasoning promotion is a floor, not a fix. What it delivers is the model's chain
of thought, so it will not follow whatever output contract the caller asked for:
it turns a silently dropped answer into a badly formatted one, which is strictly
better and still worth fixing in the prompt. The higher the thinking level, the
longer that promoted text is and the less it resembles the requested shape.

### Docker

```bash
docker run -d --name ollama-proxy \
  -e OLLAMA_HOST=http://ollama:11434 \
  -e 'OLLAMA_PROXY_REQUEST_DEFAULTS={"reasoning_effort":"none"}' \
  -p 8080:8080 \
  ghcr.io/norskhelsenett/ollama-proxy:latest
```

### Local Development

```bash
# Run directly
go run main.go

# Build and run
go build -o ollama-proxy
./ollama-proxy
```

## Metrics

Access Prometheus metrics at http://localhost:8080/metrics

### Available Metrics

| Metric                               | Labels                  | Description                                                   |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------- |
| `ollama_prompt_tokens_total`         | `model`                 | Total number of prompt tokens sent to the model               |
| `ollama_generated_tokens_total`      | `model`                 | Total number of tokens generated by the model                 |
| `ollama_request_duration_seconds`    | `api_endpoint`, `model` | Duration of Ollama requests in seconds                        |
| `ollama_time_per_token_seconds`      | `model`                 | Time per generated token (seconds per token)                  |
| `ollama_time_to_first_token_seconds` | `api_endpoint`, `model` | Time from request arrival to the first streamed content token |
| `ollama_loaded_models`               | –                       | Number of models currently loaded in memory                   |
| `ollama_model_loaded`                | `model`                 | Indicator (1/0) if a model is loaded                          |
| `ollama_model_ram_mb`                | `model`                 | RAM usage in MB for each loaded model                         |

The histogram series (`_bucket`, `_count`, `_sum`) for a given label set only
appear once a request has produced them — they are lazily created children of a
Prometheus `*Vec`, so an idle sidecar exposing no `ollama_request_duration_*`
series is expected.

### The `api_endpoint` label

The endpoint label is deliberately named `api_endpoint`, not `endpoint`: the
Prometheus Operator stamps its own `endpoint="<service port name>"` on every
scraped series, and a collision makes Prometheus rename the app's label to
`exported_endpoint`. Values are normalised paths — `chat`, `generate`,
`v1/chat/completions`.

### Streaming and the two dialects

- **Native `/api/*`** — Ollama reports `prompt_eval_count`, `eval_count` and
  `eval_duration` in the final chunk, so `ollama_time_per_token_seconds` is
  taken straight from `eval_duration / eval_count`.
- **OpenAI-compatible `/v1/*`** — non-streaming responses always carry a `usage`
  object. Streaming responses carry one *only* if the request set
  `stream_options.include_usage`, so the sidecar injects that option when the
  client did not, reads the usage from the final chunk, and then withholds that
  chunk from the client — client-visible output is unchanged. A client that asks
  for usage itself has its chunk forwarded untouched.
- `ollama_time_per_token_seconds` is recorded for `/v1` only when streaming, and
  is measured across the generation window (first content chunk to last) rather
  than the whole request. Prompt evaluation of a large context can dwarf
  generation, which would make a whole-request figure meaningless; for
  non-streaming `/v1` the metric is skipped rather than recorded wrongly.
- **Anthropic-compatible `/v1/messages`** — usage always arrives, streamed or
  not, so nothing is injected and the response passes untouched. The prompt
  count is `input_tokens + cache_read_input_tokens +
  cache_creation_input_tokens`, since `input_tokens` excludes cached tokens.
  When streaming, Ollama's `message_start` carries a placeholder input count
  that `message_delta` corrects, so the last non-zero value wins.
- `ollama_time_to_first_token_seconds` is recorded for streaming requests only.

## Prometheus & Grafana Setup

A pre-configured Prometheus and Grafana setup is available in the `prometheus/` directory:

```bash
cd prometheus
docker-compose up -d
```

This will start:
- Prometheus for metrics collection
- Grafana with pre-configured dashboard

Access Grafana at http://localhost:3000 (default credentials: admin/admin)

## Building from Source

```bash
# Clone the repository
git clone https://github.com/NorskHelsenett/ollama-proxy.git
cd ollama-proxy

# Build
docker build -t ollama-proxy .

# Or build locally
go build -o ollama-proxy

# Run the tests (they stand up a stub Ollama; no cluster or GPU needed)
go test ./...
```

## Screenshots

![grafana dashboard](.docs/dashboard.png)

## License

[MIT License](LICENSE)

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
