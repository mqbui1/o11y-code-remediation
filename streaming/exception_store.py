"""
Exception Store — captures exception stack traces from error spans.

When a span carries a span event named "exception", the pipeline extracts
exception.type, exception.message, and exception.stacktrace and indexes them
by (service, trace_id) for correlation with snapshot profiling data.

Unlike profiling (which is sampled at 10%), exception stacks are deterministic —
captured on every error, giving the exact file:line at the moment of failure.

For Node.js services with NODE_OPTIONS=--enable-source-maps, the stacktrace
already references original TypeScript file paths — no VLQ decoding needed.
For Python, paths are absolute container paths readable via Docker socket.
"""

import re
import threading
import time
from collections import defaultdict, deque

_WINDOW_SECONDS = 1800   # 30-minute retention (matches snapshot_store)
_MAXLEN_PER_TRACE = 10   # max exception events per trace

# Path fragments that indicate library / framework code — same logic as snapshot_store.
_SKIP_PATTERNS = (
    "/usr/lib/", "/usr/local/lib/", "site-packages", "/venv/", "dist-packages",
    "node_modules", "node:",
    "next/dist", "<", "opentelemetry", "splunk_otel", "grpc/",
    "/.next/server/webpack", "webpack-runtime", ".runtime.prod.js",
    "webpack:", "/_next/", "concurrent/futures", "asyncio/",
)


# ── Language-specific stack trace parsers ─────────────────────────────────────

# Python:  File "/app/foo.py", line 42, in func_name
_PY_RE = re.compile(r'\s+File "(.+?)", line (\d+), in (.+)')

# Node.js: at funcName (/app/src/foo.ts:42:10)
#          at Object.method (/app/src/foo.ts:42:10)
#          at /app/src/foo.ts:42:10   (anonymous)
_NODE_RE = re.compile(r'\s+at (?:(.+?) \()?(.+?):(\d+):\d+\)?$')

# Java:    at com.example.Foo.method(Foo.java:42)
_JAVA_RE = re.compile(r'\s+at ([\w.$<>/]+)\(([\w.$]+\.java):(\d+)\)')


def _parse_stacktrace(stacktrace: str) -> list[dict]:
    """
    Parse a language-specific exception stacktrace string into a list of frames.
    Returns [{function, file, line}, ...] from outermost to innermost.
    Skips library/framework frames.
    """
    if not stacktrace:
        return []

    frames = []
    for line in stacktrace.splitlines():
        # Python
        m = _PY_RE.match(line)
        if m:
            frames.append({
                "file":     m.group(1),
                "line":     int(m.group(2)),
                "function": m.group(3).strip(),
            })
            continue

        # Node.js
        m = _NODE_RE.match(line)
        if m:
            func  = (m.group(1) or "").strip() or "anonymous"
            file_ = m.group(2)
            lno   = int(m.group(3))
            if not any(p in file_ for p in ("node:", "node_modules")):
                frames.append({"file": file_, "line": lno, "function": func})
            continue

        # Java
        m = _JAVA_RE.match(line)
        if m:
            frames.append({
                "function": m.group(1),
                "file":     m.group(2),
                "line":     int(m.group(3)),
            })

    return frames


def _find_app_frame(frames: list[dict]) -> dict | None:
    """Return the innermost frame that looks like app code (not library/framework)."""
    for frame in reversed(frames):          # innermost = closest to error
        f = frame.get("file", "")
        if not f:
            continue
        if any(p in f for p in _SKIP_PATTERNS):
            continue
        return frame
    # No user-code frame found — return None rather than a library frame
    return None


# ── Store ──────────────────────────────────────────────────────────────────────

_EVICTION_INTERVAL = 60   # run eviction at most once per 60 seconds


class ExceptionStore:
    """Thread-safe store for exception records keyed by (service, trace_id_hex)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (service, trace_id_hex) → deque of exception dicts
        self._records: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAXLEN_PER_TRACE)
        )
        self._last_eviction: float = 0.0

    def observe(
        self,
        service:           str,
        trace_id:          str,
        span_name:         str,
        exc_type:          str,
        exc_message:       str,
        stacktrace:        str,
        span_code_frame:   dict | None = None,
        parent_code_frame: dict | None = None,
    ) -> None:
        tid_hex = trace_id.replace("-", "").lower()
        frames    = _parse_stacktrace(stacktrace)
        app_frame = _find_app_frame(frames)

        # Fallback 1: span's own code.filepath / code.lineno attrs
        if not app_frame and span_code_frame and span_code_frame.get("file"):
            f = span_code_frame["file"]
            if not any(p in f for p in _SKIP_PATTERNS):
                app_frame = {**span_code_frame, "inferred": True, "inferred_from": "span"}

        # Fallback 2: parent span's code attrs — the app code that initiated the call
        if not app_frame and parent_code_frame and parent_code_frame.get("file"):
            f = parent_code_frame["file"]
            if not any(p in f for p in _SKIP_PATTERNS):
                app_frame = {**parent_code_frame, "inferred": True, "inferred_from": "parent_span"}

        ts = time.time()

        with self._lock:
            self._records[(service, tid_hex)].append({
                "ts":          ts,
                "span_name":   span_name,
                "exc_type":    exc_type,
                "exc_message": exc_message,
                "stacktrace":  stacktrace,
                "frames":      frames,
                "app_frame":   app_frame,
            })
            # Time-gated eviction — at most once per EVICTION_INTERVAL seconds
            if ts - self._last_eviction >= _EVICTION_INTERVAL:
                cutoff  = ts - _WINDOW_SECONDS
                expired = [k for k, recs in self._records.items()
                           if all(r["ts"] < cutoff for r in recs)]
                for k in expired:
                    del self._records[k]
                self._last_eviction = ts

    def get(self, service: str, trace_id: str) -> list[dict]:
        """Return all exception records for a specific trace."""
        tid_hex = trace_id.replace("-", "").lower()
        cutoff  = time.time() - _WINDOW_SECONDS
        with self._lock:
            records = list(self._records.get((service, tid_hex), []))
        return [r for r in records if r["ts"] >= cutoff]

    def has_data(self, service: str, trace_id: str) -> bool:
        tid_hex = trace_id.replace("-", "").lower()
        with self._lock:
            return (service, tid_hex) in self._records

    def services(self) -> list[str]:
        """Return all service names that have unexpired exception records."""
        cutoff = time.time() - _WINDOW_SECONDS
        with self._lock:
            return sorted({
                svc for (svc, _), recs in self._records.items()
                if any(r["ts"] >= cutoff for r in recs)
            })

    def list_recent(self, service: str | None = None, limit: int = 200) -> list[dict]:
        """
        Return recent exception summaries sorted newest-first.
        If service is given, filter to that service only.
        """
        cutoff  = time.time() - _WINDOW_SECONDS
        results = []
        with self._lock:
            for (svc, tid), recs in self._records.items():
                if service and svc != service:
                    continue
                for rec in recs:
                    if rec["ts"] < cutoff:
                        continue
                    results.append({
                        "service":     svc,
                        "trace_id":    tid,
                        "ts":          rec["ts"],
                        "span_name":   rec["span_name"],
                        "exc_type":    rec["exc_type"],
                        "exc_message": (rec["exc_message"] or "")[:200],
                        "app_frame":   rec["app_frame"],
                    })
        results.sort(key=lambda x: -x["ts"])
        # Deduplicate: one entry per (service, trace_id) — keep most recent
        seen:    set[tuple[str, str]] = set()
        deduped: list[dict]           = []
        for r in results:
            key = (r["service"], r["trace_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:limit]


# ── Module-level singleton ────────────────────────────────────────────────────

_store = ExceptionStore()


def observe(
    service:           str,
    trace_id:          str,
    span_name:         str,
    exc_type:          str,
    exc_message:       str,
    stacktrace:        str,
    span_code_frame:   dict | None = None,
    parent_code_frame: dict | None = None,
) -> None:
    _store.observe(service, trace_id, span_name, exc_type, exc_message, stacktrace,
                   span_code_frame=span_code_frame, parent_code_frame=parent_code_frame)


def get(service: str, trace_id: str) -> list[dict]:
    return _store.get(service, trace_id)


def has_data(service: str, trace_id: str) -> bool:
    return _store.has_data(service, trace_id)


def services() -> list[str]:
    return _store.services()


def list_recent(service: str | None = None, limit: int = 200) -> list[dict]:
    return _store.list_recent(service=service, limit=limit)
