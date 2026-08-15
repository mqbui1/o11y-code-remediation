"""
Prototype: profiling data sourced from Splunk's real internal profiling `api`
service (profiling-backend/api), instead of self-decoded OTLP/pprof.

This is reachable ONLY from inside the internal network the `api` service
runs on (confirmed 404 from the public gateway/mesh, but a real 200 when
called directly against the `api` container) — see memory/profiling.md for
the full investigation. It requires no bearer token, only an
`X-SF-OrgId` header (auth happens upstream of this service, not here).

Endpoints used (see profiling-backend/api/src/main/java/.../rest/):
  GET /v2/services            -> list of active services in a time window
  GET /v2/table                -> FrameTableResponse{frames:[{frame,value,count}]}
                                   frame is a single composite string
                                   "functionName(fileName:lineNumber)" (same
                                   format our own pprof decoder produces) —
                                   parsed here to recover function/file/line.
  GET /v2/call-graphs/{service}/{traceId}/slow-methods
                                -> real measured totalSelfTimeMs per method,
                                   but className/methodName only — the API
                                   itself throws away file/line
                                   (SlowMethodQueryImpl.parseFrame strips the
                                   "(file:line)" suffix before returning).

Output shapes intentionally match streaming/profiling_store.py and
streaming/snapshot_store.py so callers (receiver/app.py, fix_generator.py)
can swap data sources without changing their own code.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("INTERNAL_API_BASE_URL", "http://api:8080")
_ORG_ID = os.environ.get("INTERNAL_API_ORG_ID", "code-remediation-dev")
_TIMEOUT_SECONDS = 5

# Matches the pprof-derived frame format: "functionName(fileName:lineNumber)"
_FRAME_RE = re.compile(r"^(?P<function>.*)\((?P<file>[^():]*):(?P<line>\d+)\)$")


def _get(path: str, params: dict) -> dict | list | None:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
    url = f"{_BASE_URL}{path}?{query}"
    req = urllib.request.Request(url, headers={"X-SF-OrgId": _ORG_ID})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("internal api %s -> HTTP %s: %s", url, exc.code, exc.read()[:500])
        return None
    except Exception as exc:
        logger.warning("internal api %s -> error: %s", url, exc)
        return None


def _parse_frame(frame: str) -> tuple[str, str, int]:
    """Split "functionName(fileName:lineNumber)" -> (function, file, line)."""
    m = _FRAME_RE.match(frame or "")
    if not m:
        return frame or "", "", 0
    return m.group("function"), m.group("file"), int(m.group("line"))


def get_services(environment: str, lookback_seconds: int = 1800) -> list[str]:
    now_ms = int(time.time() * 1000)
    data = _get("/v2/services", {
        "from": now_ms - lookback_seconds * 1000,
        "to": now_ms,
        "environment": environment,
    })
    if not data:
        return []
    return sorted({row.get("serviceName", "") for row in data if row.get("serviceName")})


def _get_flamegraph(service: str, environment: str, profiler_type: str,
                     since: float = 0, until: float = 0) -> list[dict]:
    default_lookback_ms = 1800 * 1000
    now_ms = int(time.time() * 1000)
    since_ms = int(since * 1000) if since else now_ms - default_lookback_ms
    until_ms = int(until * 1000) if until else now_ms

    data = _get("/v2/table", {
        "from": since_ms,
        "to": until_ms,
        "serviceName": service,
        "environment": environment,
        "profilerType": profiler_type,
        "n": 30,
    })
    if not data or not data.get("frames"):
        return []

    frames = []
    total = 0
    for row in data["frames"]:
        function, file_, line = _parse_frame(row.get("frame", ""))
        samples = row.get("count", 0) or 0
        if not function or samples <= 0:
            continue
        total += samples
        frames.append({"function": function, "file": file_, "line": line, "samples": samples})

    total = total or 1
    for f in frames:
        f["pct_cpu"] = round(f["samples"] / total * 100, 1)
    frames.sort(key=lambda x: -x["samples"])
    return frames[:30]


def get_flamegraph(service: str, environment: str, since: float = 0, until: float = 0) -> list[dict]:
    return _get_flamegraph(service, environment, "CPU", since=since, until=until)


def get_memory_flamegraph(service: str, environment: str, since: float = 0, until: float = 0) -> list[dict]:
    return _get_flamegraph(service, environment, "MEMORY", since=since, until=until)


def get_slowest_methods(service: str, trace_id: str, limit: int = 5) -> list[dict]:
    """
    Real measured self-time via /v2/call-graphs/.../slow-methods.

    NOTE: no file/line here — the API discards it (see module docstring).
    Shape approximates streaming/snapshot_store.py's get_slowest_methods,
    minus file/line/exit_call/app_frame fields that only the self-decoded
    pipeline can provide.
    """
    now_ms = int(time.time() * 1000)
    data = _get(f"/v2/call-graphs/{urllib.parse.quote(service)}/{urllib.parse.quote(trace_id)}/slow-methods", {
        "from": now_ms - 1800 * 1000,
        "to": now_ms,
        "limit": limit,
    })
    if not data or not data.get("methods"):
        return []
    return [
        {
            "method": m.get("methodName", ""),
            "class": m.get("className", ""),
            "self_time_ms": m.get("totalSelfTimeMs", 0),
            "sample_count": m.get("sampleCount", 0),
            "file": None,   # not available from this endpoint, see docstring
            "line": None,
        }
        for m in data["methods"][:limit]
    ]
