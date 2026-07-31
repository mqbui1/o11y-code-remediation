# o11y-code-remediation

Standalone service that turns Splunk AlwaysOn/snapshot **CPU and memory
profiling** data (and trace **exception** events) into LLM-generated
unified-diff code fixes, with a small dashboard UI for reviewing hotspots
and exceptions.

Extracted from the profiling/remediation subsystem of
[`autonomous-o11y-agent`](https://github.com/mqbui1/autonomous-o11y-agent),
where it was originally built and validated. It has zero coupling to that
project's agent/specialist code — the intent of pulling it out into its own
repo is to make it independently deployable and to eventually contribute it
into Splunk Observability Cloud itself, layered on top of the existing real
AlwaysOn Profiler + call-graph data path (see "Background" below).

## What it does

- Accepts OTLP/HTTP JSON exports for **traces** (exception events) and
  **logs** (AlwaysOn / snapshot profiling records), routing them into
  in-memory stores (`streaming/profiling_store.py`, `snapshot_store.py`,
  `exception_store.py`).
- Serves a profiling dashboard UI (`GET /profiling`) with flamegraphs, call
  graphs, memory allocation views, and exception lists.
- Generates an LLM-based unified-diff code fix for a CPU/memory hotspot or
  exception (`POST /api/fix`), using the real source code read live from a
  running Docker container (`GET /api/source`, via the Docker Engine API —
  no GitHub/CI dependency).
- Generates plain-language narrative summaries of a service's profiling
  state (`POST /api/profiling/narrative/<service>`).

## Background

Splunk Observability Cloud already has a real AlwaysOn Profiler + call-graph
backend (Kafka → Druid → REST API → APM UI) that collects and visualizes
this exact profiling data. That backend's API surface
(`/v2/apm/profiling/*`, `/v2/call-graphs/*`) is confirmed **internal-only** —
not reachable from the public customer-facing gateway — so this service
implements its own lightweight OTLP-to-store decode path instead of calling
that backend directly. It listens on the same `/v1/traces` and `/v1/logs`
paths any OTel Collector already exports to, so it can run as a secondary
fan-out target alongside a normal Splunk export pipeline with no changes to
instrumented services.

## Running it

```bash
cp .env.example .env   # fill in LLM provider credentials
docker compose up --build
```

The receiver listens on port `4318` (OTLP/HTTP) by default. Point your OTel
Collector's `otlphttp` exporter at it — see
`deploy/otelcol-config-snippet.yml` for the exact config to add alongside
your existing pipeline.

Dashboard: `http://localhost:4318/profiling`

## Configuration

LLM provider is selected via `LLM_PROVIDER`:

- `bedrock` (default): `BEDROCK_MODEL_ID`, `AWS_DEFAULT_REGION` (+ standard
  AWS credential env vars / instance role)
- `ollama` / `openai`: `OPENAI_BASE_URL` (or `OLLAMA_BASE_URL`),
  `OPENAI_API_KEY` (or `OLLAMA_MODEL`), `OPENAI_MODEL`

`/api/source` and `/api/fix`'s source-reading path require
`/var/run/docker.sock` mounted read-only (already wired in
`docker-compose.yml`) so the service can read source files out of the
running application containers by name.

## Project layout

```
receiver/            Flask app (routes), fix generator, narrative generator,
                      Docker-socket-based source reader, dashboard UI
streaming/            OTLP decode pipeline + in-memory stores
providers/            LLM provider abstraction (Bedrock, OpenAI-compatible)
config.py             Env-var-based provider selection
deploy/               otelcol config snippet
```

## Status / not yet done

- In-memory-only stores — data resets on restart (no persistence layer yet).
- Human-triggered fix generation only — no autonomous detection loop or
  GitHub PR creation yet (read-only source access only).
- Memory-profile fix generation exists in the prompt schema but is less
  battle-tested than the CPU-hotspot path.
