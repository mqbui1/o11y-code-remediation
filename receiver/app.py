"""
Standalone OTLP receiver + profiling/remediation API.

Runs a Flask server (default port 4318) that:
  - Accepts OTLP/HTTP JSON exports for traces (exception events) and logs
    (AlwaysOn / snapshot profiling records), routing them into
    streaming.profiling_store / snapshot_store / exception_store.
  - Serves the profiling dashboard UI (GET /profiling) and its supporting
    JSON APIs (flamegraphs, call graphs, memory allocation views, exceptions).
  - Serves POST /api/fix — generates an LLM-based unified-diff code fix for a
    CPU/memory hotspot or exception, using receiver.fix_generator.
  - Serves GET /api/source — reads source lines from a running Docker
    container for a given service+file+line, via receiver.source_reader.

Point your OTel Collector's otlphttp exporter at this service's /v1/traces
and /v1/logs (encoding: json, compression: none) alongside your normal
Splunk export path — see deploy/otelcol-config-snippet.yml.
"""

import json
import logging
import os
import threading

from flask import Flask, Response, request, send_file

from streaming import pipeline
from streaming import profiling_store as ps
from streaming import snapshot_store as ss
from streaming import exception_store as es
from receiver.fix_generator import generate_fix
from receiver.narrative_generator import generate_narrative
from receiver.source_reader import read_source
from config import get_llm_provider

logger = logging.getLogger(__name__)


def create_app(environment: str = "") -> Flask:
    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)

    # ── OTLP ingestion ──────────────────────────────────────────────────────

    @app.post("/v1/traces")
    def receive_traces():
        payload = _parse_body()
        if payload is None:
            return Response("Bad Request", status=400)
        resource_spans = payload.get("resourceSpans", [])
        if resource_spans:
            try:
                pipeline.process_resource_spans(resource_spans)
            except Exception as exc:
                logger.error("Error processing traces: %s", exc, exc_info=True)
        return Response(json.dumps({"partialSuccess": {}}), status=200, mimetype="application/json")

    @app.post("/v1/logs")
    def receive_logs():
        payload = _parse_body()
        if payload is None:
            return Response("Bad Request", status=400)
        resource_logs = payload.get("resourceLogs", [])
        if resource_logs:
            try:
                pipeline.process_resource_logs(resource_logs)
            except Exception as exc:
                logger.error("Error processing logs: %s", exc, exc_info=True)
        return Response(json.dumps({"partialSuccess": {}}), status=200, mimetype="application/json")

    @app.get("/v1/logs")
    def receive_logs_get():
        # Some collectors do a GET probe — return 200 to avoid 404 noise
        return Response(json.dumps({"partialSuccess": {}}), status=200, mimetype="application/json")

    @app.get("/healthz")
    def healthz():
        return Response("ok", status=200)

    # ── Profiling APIs ───────────────────────────────────────────────────────

    @app.get("/api/profiling/status")
    def profiling_status():
        try:
            env = environment or request.args.get("environment", "")
            services = ps.get_services(env) if env else []
            flamegraphs = {}
            for svc in services:
                frames = ps.get_flamegraph(svc, env)
                flamegraphs[svc] = {"frame_count": len(frames), "top_frame": frames[0] if frames else None}
            return Response(
                json.dumps({
                    "environment": env,
                    "profiling_services": services,
                    "exception_services": es.services(),
                    "flamegraphs": flamegraphs,
                }),
                status=200, mimetype="application/json",
            )
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/callgraph/<service>/<trace_id>")
    def callgraph_lookup(service, trace_id):
        try:
            methods = ss.get_slowest_methods(service, trace_id, limit=10)
            return Response(
                json.dumps({"service": service, "trace_id": trace_id, "slowest_methods": methods, "found": bool(methods)}),
                status=200, mimetype="application/json",
            )
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/memory-snapshot/<service>/<trace_id>")
    def memory_snapshot(service, trace_id):
        """Return trace-correlated heap allocation frames for a specific trace."""
        try:
            frames = ss.get_allocations(service, trace_id)
            return Response(
                json.dumps({"service": service, "trace_id": trace_id, "found": bool(frames), "frames": frames}),
                status=200, mimetype="application/json",
            )
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/snapshot")
    def snapshot_debug():
        try:
            with ss._store._lock:
                keys = []
                for (svc, tid), recs in ss._store._records.items():
                    arrived_at = max((r["ts"] for r in recs), default=0)
                    keys.append({"service": svc, "trace_id": tid, "record_count": len(recs), "arrived_at": arrived_at})
            return Response(json.dumps({"total_traces": len(keys), "traces": keys}), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/flamegraph/<service>")
    def flamegraph_data(service):
        """Return all AlwaysOn CPU frames for a service (for the UI icicle chart)."""
        try:
            env = environment or request.args.get("environment", "")
            since = float(request.args.get("since", 0) or 0)
            until = float(request.args.get("until", 0) or 0)
            frames = ps.get_flamegraph(service, env, since=since, until=until)
            return Response(json.dumps({"service": service, "environment": env, "frames": frames}), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/hotspots/<service>")
    def hotspots_data(service):
        """Return aggregated method hotspots across all snapshot traces for a service."""
        try:
            since = float(request.args.get("since", 0) or 0)
            until = float(request.args.get("until", 0) or 0)
            data = ss.get_hotspots(service, since=since, until=until)
            return Response(json.dumps(data), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/memory/<service>")
    def memory_flamegraph(service):
        try:
            env = environment or request.args.get("environment", "")
            since = float(request.args.get("since", 0) or 0)
            until = float(request.args.get("until", 0) or 0)
            frames = ps.get_memory_flamegraph(service, env, since=since, until=until)
            return Response(json.dumps({"frames": frames}), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/profiling/diff/<service>")
    def flamegraph_diff(service):
        try:
            env = environment or request.args.get("environment", "")
            window = int(request.args.get("window", 300) or 300)
            baseline = int(request.args.get("baseline_offset", 900) or 900)
            diff = ps.get_flamegraph_diff(service, env, window_seconds=window, baseline_offset=baseline)
            return Response(json.dumps({"diff": diff}), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.post("/api/profiling/narrative/<service>")
    def service_narrative(service):
        try:
            env = environment or request.args.get("environment", "")
            cpu_frames = ps.get_flamegraph(service, env)
            memory_frames = ps.get_memory_flamegraph(service, env)
            diff = ps.get_flamegraph_diff(service, env)
            exceptions = es.list_recent(service=service, limit=500)
            snapshot_count = ss.count_for_service(service)

            result = generate_narrative(get_llm_provider(), {
                "service": service,
                "cpu_frames": cpu_frames,
                "memory_frames": memory_frames,
                "exceptions": exceptions,
                "diff": diff,
                "snapshot_count": snapshot_count,
            })
            return Response(json.dumps(result), status=200, mimetype="application/json")
        except Exception as exc:
            logger.error("Narrative error: %s", exc, exc_info=True)
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/exceptions")
    def exceptions_list():
        """Return recent exception summaries (newest first). Optional ?service= filter."""
        try:
            svc = request.args.get("service") or None
            limit = int(request.args.get("limit", 200) or 200)
            data = es.list_recent(service=svc, limit=limit)
            for item in data:
                item["has_snapshot"] = ss.has_data(item["service"], item["trace_id"])
                item["has_alloc_data"] = ss.has_allocation_data(item["service"], item["trace_id"])
            return Response(json.dumps({"exceptions": data, "count": len(data)}), status=200, mimetype="application/json")
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/api/exceptions/<service>/<trace_id>")
    def exception_detail(service, trace_id):
        """Return full exception records (with parsed frames) for a specific trace."""
        try:
            records = es.get(service, trace_id)
            has_snapshot = ss.has_data(service, trace_id)
            return Response(
                json.dumps({"service": service, "trace_id": trace_id, "records": records, "has_snapshot": has_snapshot}),
                status=200, mimetype="application/json",
            )
        except Exception as exc:
            return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")

    @app.get("/profiling")
    def profiling_ui():
        """Serve the profiling flamegraph + snapshot UI."""
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling_ui.html")
        return send_file(ui_path, mimetype="text/html")

    # ── Remediation APIs ─────────────────────────────────────────────────────

    @app.post("/api/fix")
    def code_fix():
        """
        Generate a code fix for a profiling hotspot or exception using the LLM.

        POST body (JSON, CPU/memory hotspot): service, blocking_fn, blocking_file,
        blocking_line, self_time_ms, app_fn, app_file, app_line, source_lines.

        POST body (exception, no source_lines): service, blocking_fn/app_fn,
        exc_message, exc_stacktrace.
        """
        body = request.get_json(force=True, silent=True) or {}
        if not body:
            return Response(json.dumps({"error": "Request body required"}), status=400, mimetype="application/json")
        try:
            result = generate_fix(get_llm_provider(), body)
        except Exception as exc:
            logger.error("Fix generation error: %s", exc, exc_info=True)
            result = {"error": str(exc)}
        return Response(json.dumps(result), status=200, mimetype="application/json")

    @app.get("/api/source")
    def source_view():
        """
        Return source lines for a file inside a service's Docker container.
        Query params: service, file, line (1-based), context (lines around target).
        """
        svc = request.args.get("service", "")
        file = request.args.get("file", "")
        line = int(request.args.get("line", 0) or 0)
        ctx = int(request.args.get("context", 25) or 25)
        if not svc or not file:
            return Response(json.dumps({"error": "service and file params required"}), status=400, mimetype="application/json")
        data = read_source(svc, file, line=line, context=ctx)
        return Response(json.dumps(data), status=200, mimetype="application/json")

    return app


_protobuf_warned = False


def _parse_body() -> dict | None:
    """
    Parse OTLP/HTTP body. Only JSON is supported.

    Add `encoding: json` and `compression: none` to your OTel Collector's
    otlphttp exporter for this service — see deploy/otelcol-config-snippet.yml.
    """
    global _protobuf_warned
    try:
        ct = request.content_type or ""
        if "json" in ct or not ct:
            return request.get_json(force=True, silent=True)
        if "protobuf" in ct or "octet-stream" in ct:
            if not _protobuf_warned:
                logger.warning(
                    "Received protobuf-encoded payload (content-type: %s). "
                    "This receiver only supports JSON encoding. Add 'encoding: json' "
                    "to your otlphttp exporter config. (Logs once per process.)",
                    ct,
                )
                _protobuf_warned = True
            return {}
        logger.debug("Received unrecognised content-type: %s — skipping", ct)
        return {}
    except Exception as exc:
        logger.warning("Failed to parse request body: %s", exc)
        return None


def start_receiver(port: int = 4318, host: str = "0.0.0.0", environment: str = "") -> threading.Thread:
    """Start the receiver in a daemon thread. Returns the thread."""
    app = create_app(environment=environment)

    def _serve():
        logger.info("Profiling/remediation receiver listening on %s:%d", host, port)
        try:
            from werkzeug.serving import make_server
            srv = make_server(host, port, app, threaded=True)
            srv.serve_forever()
        except Exception as exc:
            logger.error("Receiver failed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_serve, daemon=True, name="profiling-receiver")
    thread.start()
    return thread
