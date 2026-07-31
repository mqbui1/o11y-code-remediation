"""
Profiling + exception routing pipeline.

Extracted from autonomous-o11y-agent's streaming/pipeline.py, keeping ONLY the
code paths that feed the profiling/remediation subsystem:

  - process_resource_logs():  routes AlwaysOn/snapshot profiling log records
    (com.splunk.sourcetype == "otel.profiling") to profiling_store / snapshot_store.
  - process_resource_spans(): extracts exception events from trace spans and
    routes them to exception_store (used by the memory/exception views and by
    fix_generator's exception-analysis path).

Deliberately dropped (out of scope for this project, lived in the original
pipeline.py alongside these): PII scanning, attribute-gap checking,
cardinality tracking, service-discovery tracking, log-error-burst tracking,
and the AgentConfig-based alert dispatcher. None of those are needed to
receive/store/serve profiling data or generate a code fix.
"""

import logging

from . import profiling_store, snapshot_store, exception_store

logger = logging.getLogger(__name__)


def process_resource_spans(resource_spans: list[dict]) -> None:
    """
    Process a resourceSpans array from an OTLP/HTTP JSON payload.

    Extracts exception span-events and routes them to exception_store, so the
    UI/fix-generator can look up a real stack trace + code location for a
    failed trace_id.
    """
    # First pass: build span_id -> attrs lookup across the batch so exception
    # spans can resolve their parent span's code location.
    span_lookup: dict[str, dict] = {}
    for rs in resource_spans:
        for scope in rs.get("scopeSpans", []):
            for span in scope.get("spans", []):
                sid = span.get("spanId", "")
                if sid:
                    span_lookup[sid] = _parse_attributes(span.get("attributes", []))

    for rs in resource_spans:
        resource_attrs = _parse_attributes(rs.get("resource", {}).get("attributes", []))
        service = resource_attrs.get("service.name", "unknown")

        for scope in rs.get("scopeSpans", []):
            for span in scope.get("spans", []):
                span_name = span.get("name", "unknown")
                span_attrs = _parse_attributes(span.get("attributes", []))

                trace_id = span.get("traceId", "")
                if not trace_id:
                    continue
                for event in span.get("events", []):
                    if event.get("name") != "exception":
                        continue
                    event_attrs = _parse_attributes(event.get("attributes", []))
                    exc_type = event_attrs.get("exception.type", "")
                    exc_msg = event_attrs.get("exception.message", "")
                    exc_stack = event_attrs.get("exception.stacktrace", "")
                    if not (exc_type or exc_stack):
                        continue

                    span_code_frame = _extract_code_frame(span_attrs)
                    parent_code_frame = None
                    parent_sid = span.get("parentSpanId", "")
                    if parent_sid and parent_sid in span_lookup:
                        parent_code_frame = _extract_code_frame(span_lookup[parent_sid])

                    exception_store.observe(
                        service=service,
                        trace_id=trace_id,
                        span_name=span_name,
                        exc_type=exc_type,
                        exc_message=exc_msg,
                        stacktrace=exc_stack,
                        span_code_frame=span_code_frame,
                        parent_code_frame=parent_code_frame,
                    )


def process_resource_logs(resource_logs: list[dict]) -> None:
    """
    Process a resourceLogs array from an OTLP/HTTP JSON payload.

    Routes AlwaysOn Profiling / snapshot profiling records (marked with
    com.splunk.sourcetype == "otel.profiling") to profiling_store or
    snapshot_store. All other log records are ignored — this service does
    not do general log monitoring, only profiling.
    """
    for rl in resource_logs:
        resource_attrs = _parse_attributes(rl.get("resource", {}).get("attributes", []))
        service = resource_attrs.get("service.name", "unknown")

        for scope in rl.get("scopeLogs", []):
            for record in scope.get("logRecords", []):
                record_attrs = _parse_attributes(record.get("attributes", []))
                if record_attrs.get("com.splunk.sourcetype") != "otel.profiling":
                    continue

                body_container = record.get("body", {})
                body = body_container.get("stringValue", "") or str(body_container)
                data_format = record_attrs.get("profiling.data.format", "")
                data_type = record_attrs.get("profiling.data.type", "cpu")
                instr_source = record_attrs.get("profiling.instrumentation.source", "continuous")
                environment = resource_attrs.get("deployment.environment", "unknown")

                if instr_source == "snapshot":
                    # Snapshot profiling: pprof call-graph OR json-alloc-v1 heap data
                    snapshot_store.observe(
                        service=service,
                        body=body,
                        data_format=data_format,
                        data_type=data_type,
                    )
                else:
                    # AlwaysOn (continuous) profiling: index by service+environment
                    profiling_store.observe(
                        service=service,
                        environment=environment,
                        data_type=data_type,
                        body=body,
                        data_format=data_format,
                    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_attributes(attr_list: list[dict]) -> dict:
    """
    Convert OTLP attribute list to a flat dict.

    OTLP attribute format:
      [{"key": "service.name", "value": {"stringValue": "payment"}}, ...]
    """
    result = {}
    for item in attr_list:
        key = item.get("key", "")
        value_container = item.get("value", {})
        for vtype in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if vtype in value_container:
                result[key] = value_container[vtype]
                break
        else:
            if "arrayValue" in value_container:
                result[key] = value_container["arrayValue"]
            elif "kvlistValue" in value_container:
                result[key] = value_container["kvlistValue"]
    return result


def _extract_code_frame(attrs: dict) -> dict | None:
    """
    Extract a code location frame from span attributes.

    Checks code.filepath / code.lineno / code.function (OTel semantic conventions).
    Returns None if no filepath is present.
    """
    filepath = attrs.get("code.filepath") or attrs.get("code.namespace") or ""
    if not filepath:
        return None
    try:
        line = int(attrs.get("code.lineno", 0) or 0)
    except (ValueError, TypeError):
        line = 0
    return {
        "file": str(filepath),
        "line": line,
        "function": str(attrs.get("code.function", "") or ""),
    }
