"""
Local Snapshot (Call Graph) Profiling store.

Receives pprof-gzip-base64 log records from the OTel Collector fan-out
(profiling.instrumentation.source = "snapshot") and indexes stack-trace
samples by (service, trace_id) so that get_slowest_methods() can return
trace-correlated profiling data without calling the Splunk API.

Data flow:
  CallgraphsSpanProcessor (SDK)
    → stamps splunk.snapshot.profiling=True on spans
    → starts ProfilingContext(instrumentation_source="snapshot")
    → exports pprof OTLP log with per-sample trace_id label
    → OTel Collector logs pipeline → otlphttp/agent (JSON)
    → otlp_receiver /v1/logs → pipeline.process_resource_logs()
    → snapshot_store.observe()
    → profiling_tools.get_slowest_methods() (local, no API call)
"""

import base64
import gzip
import threading
import time
from collections import defaultdict, deque

# Path fragments that indicate library / framework code (not user application code).
# _find_app_frame() skips frames matching any of these to surface the first
# app-code caller above a blocking/hot library frame.
_LIBRARY_PATH_PATTERNS = (
    # Python — stdlib and common frameworks
    "/venv/lib/",
    "site-packages",
    "/usr/lib/python",
    "lib/python",           # /usr/local/lib/python3.x/...
    "threading.py",
    "asyncio/",
    "_asyncio",
    "grpc/",
    "opentelemetry/",
    "splunk_otel/",
    "google/protobuf/",
    "concurrent/futures",
    "<",                    # <string>, <frozen importlib._bootstrap>
    # Node.js — built-ins, packages, compiled bundles
    "node:",                # node:internal/async_hooks, node:fs, node:http2, etc.
    "node_modules/",        # npm packages
    "/dist/compiled/",      # webpack/Next.js compiled bundles
    ".runtime.prod.js",     # minified production runtimes
    ".runtime.dev.js",      # minified dev runtimes
    "webpack:",             # webpack internal module references
    "/_next/",              # Next.js public static asset path (client bundles)
    "/.next/server/webpack",# webpack runtime/chunks inside Next.js server build
    "/.next/server/chunks/__",  # double-underscore internal Next.js chunks
    ".chunk.js",            # explicitly named webpack chunk files
    "_chunks/",             # Next.js/webpack chunks directory
    "link-of-the-server",   # Next.js internal server bundle name pattern
    "/compiled/",           # compiled output directories
    ".min.js",              # minified JS files
    # NOTE: "/.next/" intentionally NOT here — server-side page/api files like
    # /.next/server/pages/api/cart.js are app code that source-maps back to TypeScript.
)

# Keep snapshot records for up to 30 minutes; ring buffer depth per trace
_WINDOW_SECONDS = 1800
_MAXLEN_PER_TRACE = 20


class SnapshotStore:
    """
    Thread-safe store for snapshot profiling samples, keyed by (service, trace_id_hex).

    Stores two kinds of records:
      _records       — pprof CPU call-graph samples (data_type="cpu")
      _alloc_records — json-alloc-v1 heap allocation diffs (data_type="allocation")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (service, trace_id_hex) → deque of {ts, frames: list[dict]}
        self._records: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAXLEN_PER_TRACE)
        )
        # (service, trace_id_hex) → deque of {ts, frames: list[dict]}
        # frames: [{function, file, line, size_bytes, count}]
        self._alloc_records: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAXLEN_PER_TRACE)
        )

    def observe(self, service: str, body: str, data_format: str,
                data_type: str = "cpu") -> None:
        if data_format == "json-alloc-v1" or data_type == "allocation":
            self._observe_allocation(service, body)
            return
        if data_format != "pprof-gzip-base64" or not body:
            return

        samples = _decode_snapshot_pprof(body)   # {trace_id_hex: [frame_dicts]}
        if not samples:
            return

        ts = time.time()
        with self._lock:
            for trace_id_hex, frames in samples.items():
                self._records[(service, trace_id_hex)].append({"ts": ts, "frames": frames})
            # Evict expired entries lazily
            cutoff = ts - _WINDOW_SECONDS
            expired = [k for k, recs in self._records.items()
                       if all(r["ts"] < cutoff for r in recs)]
            for k in expired:
                del self._records[k]

    def _observe_allocation(self, service: str, body: str) -> None:
        """Store a json-alloc-v1 record from heap_snapshot_collector/processor."""
        if not body:
            return
        try:
            data = __import__("json").loads(body)
        except Exception:
            return
        trace_id_hex = (data.get("trace_id") or "").replace("-", "").lower()
        frames = data.get("frames") or []
        if not trace_id_hex or not frames:
            return
        ts = time.time()
        with self._lock:
            self._alloc_records[(service, trace_id_hex)].append(
                {"ts": ts, "frames": frames}
            )
            # Lazy eviction
            cutoff = ts - _WINDOW_SECONDS
            expired = [k for k, recs in self._alloc_records.items()
                       if all(r["ts"] < cutoff for r in recs)]
            for k in expired:
                del self._alloc_records[k]

    def get_allocations(self, service: str, trace_id: str) -> list[dict]:
        """
        Return aggregated allocation frames for a trace, ranked by size_bytes desc.
        Returns empty list if no allocation data exists for this trace.
        """
        tid_hex = trace_id.replace("-", "").lower()
        cutoff  = time.time() - _WINDOW_SECONDS

        with self._lock:
            records = list(self._alloc_records.get((service, tid_hex), []))

        agg: dict[str, dict] = {}
        for rec in records:
            if rec["ts"] < cutoff:
                continue
            for frame in rec["frames"]:
                key = f"{frame.get('file', '')}:{frame.get('function', '')}:{frame.get('line', 0)}"
                if key in agg:
                    agg[key]["size_bytes"] += frame.get("size_bytes", 0)
                    agg[key]["count"]      += frame.get("count", 0)
                else:
                    agg[key] = {
                        "function":   frame.get("function", "unknown"),
                        "file":       frame.get("file", "unknown"),
                        "line":       frame.get("line", 0),
                        "size_bytes": frame.get("size_bytes", 0),
                        "count":      frame.get("count", 0),
                    }

        if not agg:
            return []

        results = sorted(agg.values(), key=lambda x: -x["size_bytes"])
        total   = sum(r["size_bytes"] for r in results) or 1
        for r in results:
            r["pct"] = round(r["size_bytes"] / total * 100, 1)
        return results

    def has_allocation_data(self, service: str, trace_id: str) -> bool:
        tid_hex = trace_id.replace("-", "").lower()
        with self._lock:
            return (service, tid_hex) in self._alloc_records

    def get_slowest_methods(
        self, service: str, trace_id: str, limit: int = 5
    ) -> list[dict]:
        """
        Return up to `limit` frames ranked by sample count (proxy for self-time).
        Each result includes an `app_frame` field pointing to the first non-library
        caller above the hot frame — the exact line of app code responsible.
        Returns empty list if no snapshot data exists for this trace.
        """
        tid_hex = trace_id.replace("-", "").lower()
        cutoff = time.time() - _WINDOW_SECONDS

        with self._lock:
            records = list(self._records.get((service, tid_hex), []))

        # Aggregate sample counts per (function, file, line); collect full stacks
        counts: dict[str, dict] = {}
        stacks_per_key: dict[str, list] = defaultdict(list)
        for rec in records:
            if rec["ts"] < cutoff:
                continue
            for frame in rec["frames"]:
                key = f"{frame['file']}:{frame['function']}:{frame['line']}"
                if key in counts:
                    counts[key]["sample_count"] += 1
                else:
                    counts[key] = {
                        "method": frame["function"],
                        "class": _infer_class(frame["file"]),
                        "file": frame["file"],
                        "line": frame["line"],
                        "self_time_ms": 0.0,
                        "sample_count": 1,
                        "exit_call": None,
                        "exit_call_action": None,
                    }
                if frame.get("stack"):
                    stacks_per_key[key].append(frame["stack"])

        if not counts:
            return []

        result = sorted(counts.values(), key=lambda x: -x["sample_count"])[:limit]

        for m in result:
            # Estimate self_time_ms: each snapshot sample ≈ 10ms
            m["self_time_ms"] = round(m["sample_count"] * 10.0, 1)
            # Surface the first app-code caller above this library/blocking frame
            key = f"{m['file']}:{m['method']}:{m['line']}"
            m["app_frame"] = _find_app_frame(stacks_per_key.get(key, []))

        return result

    def has_data(self, service: str, trace_id: str) -> bool:
        tid_hex = trace_id.replace("-", "").lower()
        with self._lock:
            return (service, tid_hex) in self._records

    def get_hotspots(
        self, service: str, since: float = 0, until: float = 0, limit: int = 25
    ) -> dict:
        """
        Aggregate snapshot profiling data across ALL traces for a service in a time window.

        Returns method hotspots ranked by contribution (% of total samples), mirroring
        Dynatrace's Method Hotspots "Stacktrace samples" + "Contribution" view.

        Each entry includes:
          method, file, line, total_samples, traces_affected, total_traces,
          contribution_pct, avg_self_time_ms, worst_trace_id, app_frame
        """
        now = time.time()
        cutoff_lo = max(since if since > 0 else 0, now - _WINDOW_SECONDS)
        cutoff_hi = until if until > 0 else float("inf")

        # Snapshot relevant records under the lock, then process outside it
        with self._lock:
            relevant = [
                (tid, [dict(r) for r in recs])
                for (svc, tid), recs in self._records.items()
                if svc == service
            ]

        # Single-pass aggregation: count samples per (method, file, line) across traces
        agg: dict[str, dict] = {}
        active_traces: set[str] = set()
        total_samples = 0

        for tid, records in relevant:
            trace_active = False
            for rec in records:
                if rec["ts"] < cutoff_lo or rec["ts"] > cutoff_hi:
                    continue
                trace_active = True
                for frame in rec["frames"]:
                    key = f"{frame['file']}:{frame['function']}:{frame['line']}"
                    total_samples += 1
                    if key not in agg:
                        agg[key] = {
                            "method":   frame["function"],
                            "file":     frame["file"],
                            "line":     frame["line"],
                            "total_samples": 0,
                            "per_trace": defaultdict(int),
                            "stacks":   [],
                        }
                    agg[key]["total_samples"] += 1
                    agg[key]["per_trace"][tid] += 1
                    if frame.get("stack") and len(agg[key]["stacks"]) < 5:
                        agg[key]["stacks"].append(frame["stack"])
            if trace_active:
                active_traces.add(tid)

        if not agg:
            return {"methods": [], "total_samples": 0, "total_traces": 0}

        total_traces = len(active_traces)
        results = []
        for m in agg.values():
            traces_hit = len(m["per_trace"])
            worst_tid = max(m["per_trace"], key=m["per_trace"].__getitem__)
            results.append({
                "method":           m["method"],
                "file":             m["file"],
                "line":             m["line"],
                "total_samples":    m["total_samples"],
                "traces_affected":  traces_hit,
                "total_traces":     total_traces,
                "contribution_pct": round(m["total_samples"] / total_samples * 100, 1) if total_samples else 0,
                "avg_self_time_ms": round(m["total_samples"] / traces_hit * 10.0, 1),
                "worst_trace_id":   worst_tid,
                "app_frame":        _find_app_frame(m["stacks"]),
            })

        results.sort(key=lambda x: -x["total_samples"])
        return {
            "methods":       results[:limit],
            "total_samples": total_samples,
            "total_traces":  total_traces,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_store = SnapshotStore()


def observe(service: str, body: str, data_format: str,
            data_type: str = "cpu") -> None:
    _store.observe(service, body, data_format, data_type=data_type)


def get_slowest_methods(service: str, trace_id: str, limit: int = 5) -> list[dict]:
    return _store.get_slowest_methods(service, trace_id, limit)


def has_data(service: str, trace_id: str) -> bool:
    return _store.has_data(service, trace_id)


def get_hotspots(service: str, since: float = 0, until: float = 0, limit: int = 25) -> dict:
    return _store.get_hotspots(service, since=since, until=until, limit=limit)


def get_allocations(service: str, trace_id: str) -> list[dict]:
    return _store.get_allocations(service, trace_id)


def has_allocation_data(service: str, trace_id: str) -> bool:
    return _store.has_allocation_data(service, trace_id)


def count_for_service(service: str) -> int:
    """Return the number of unique snapshot traces for a service in the last 30 min."""
    cutoff = time.time() - _WINDOW_SECONDS
    with _store._lock:
        return sum(
            1 for (svc, _), recs in _store._records.items()
            if svc == service and any(r["ts"] >= cutoff for r in recs)
        )


# ── pprof decoder (extract per-sample trace_id labels) ───────────────────────
#
# pprof wire format fields used here:
#   Profile.sample       field 2  → Sample.location_id field 1, Sample.label field 3
#   Profile.location     field 4  → Location.id field 1, Location.line field 4 (Line.function_id field 1)
#   Profile.function     field 5  → Function.id 1, .name 2, .filename 4, .start_line 5
#   Profile.string_table field 6
#   Label.key field 1, Label.str field 2 (string table index)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while pos < len(data):
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("truncated varint")


def _parse_fields(data: bytes) -> dict[int, list]:
    from collections import defaultdict as _dd
    fields: dict[int, list] = _dd(list)
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_num, wire_type = tag >> 3, tag & 0x7
        if wire_type == 0:
            val, pos = _read_varint(data, pos)
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            val = data[pos:pos + length]; pos += length
        elif wire_type == 1:
            val = int.from_bytes(data[pos:pos + 8], "little"); pos += 8
        elif wire_type == 5:
            val = int.from_bytes(data[pos:pos + 4], "little"); pos += 4
        else:
            break
        fields[field_num].append((wire_type, val))
    return fields


def _packed_varints(data: bytes) -> list[int]:
    result, pos = [], 0
    while pos < len(data):
        v, pos = _read_varint(data, pos)
        result.append(v)
    return result


def _is_library_frame(frame: dict) -> bool:
    """True if the frame belongs to a library/framework rather than app code."""
    import re as _re
    file_ = frame.get("file", "")
    if not file_:
        return True
    if any(p in file_ for p in _LIBRARY_PATH_PATTERNS):
        return True
    # Heuristic: webpack/Next.js hash-named bundle files
    # e.g. "link-of-the-server-_Rfe8u.js", "app-_abc123.js", "page-CHx9f_jO.js"
    basename = file_.rsplit("/", 1)[-1]
    if basename.endswith(".js") and _re.search(r"[-_][A-Za-z0-9_]{4,}\.js$", basename):
        # Only treat as compiled if the base name has a hash-like suffix
        # and is NOT a simple descriptive name (e.g. "server.js", "index.js" are fine)
        if _re.search(r"[-_][A-Z][A-Za-z0-9_]{3,}\.js$", basename):
            return True
    return False


def _find_app_frame(stacks: list[list[dict]]) -> dict | None:
    """
    Given a list of full call stacks (each innermost-first), vote on the most
    common first-app-code frame across all stacks.

    Skips frame[0] (already the self-time frame) and walks callers outward until
    a non-library frame is found. The winning candidate (most votes) is returned
    as {"function", "file", "line"}.
    """
    if not stacks:
        return None
    votes: dict[str, int] = defaultdict(int)
    candidates: dict[str, dict] = {}
    for stack in stacks:
        for frame in stack[1:]:   # skip innermost self-time frame
            if not _is_library_frame(frame):
                key = f"{frame['file']}:{frame['function']}:{frame['line']}"
                votes[key] += 1
                candidates[key] = frame
                break  # first app frame per stack
    if not votes:
        return None
    best = max(votes, key=lambda k: votes[k])
    return candidates[best]


def _infer_class(file_path: str) -> str:
    """Best-effort class name from a Python file path (e.g. src/cart/repo.py → cart.repo)."""
    if not file_path:
        return ""
    parts = file_path.replace("\\", "/").split("/")
    name = parts[-1].removesuffix(".py") if parts else ""
    return name


def _decode_snapshot_pprof(b64: str) -> dict[str, list[dict]]:
    """
    Decode a pprof-gzip-base64 snapshot profile.

    Returns {trace_id_hex: [{"function": ..., "file": ..., "line": ...}]}
    where each list entry is the innermost (self-time) frame of one sample.
    """
    try:
        raw = gzip.decompress(base64.b64decode(b64))
        fields = _parse_fields(raw)

        # String table (field 6) — do NOT pre-seed; let first entry be index 0
        strings: list[str] = []
        for wt, val in fields.get(6, []):
            if wt == 2 and isinstance(val, bytes):
                strings.append(val.decode("utf-8", errors="replace"))

        def s(idx: int) -> str:
            return strings[idx] if 0 <= idx < len(strings) else ""

        # Find the "trace_id" key index in string table
        try:
            trace_id_key_idx = strings.index("trace_id")
        except ValueError:
            return {}   # no trace_id labels → not a snapshot profile

        # Functions: field 5 → {id: (name, file, start_line)}
        fns: dict[int, tuple[str, str, int]] = {}
        for wt, val in fields.get(5, []):
            if wt != 2 or not isinstance(val, bytes):
                continue
            f = _parse_fields(val)
            fid   = f[1][0][1] if f.get(1) else 0
            name  = s(f[2][0][1] if f.get(2) else 0)
            fname = s(f[4][0][1] if f.get(4) else 0)
            sline = f[5][0][1] if f.get(5) else 0
            fns[fid] = (name, fname, sline)

        # Locations: field 4 → {id: function_id} (first Line only)
        locs: dict[int, tuple[int, int]] = {}  # loc_id → (function_id, line_number)
        for wt, val in fields.get(4, []):
            if wt != 2 or not isinstance(val, bytes):
                continue
            l = _parse_fields(val)
            lid = l[1][0][1] if l.get(1) else 0
            for lwt, lval in l.get(4, []):          # Line messages
                if lwt == 2 and isinstance(lval, bytes):
                    lf = _parse_fields(lval)
                    locs[lid] = (lf[1][0][1] if lf.get(1) else 0, lf[2][0][1] if lf.get(2) else 0)
                    break

        # Samples: field 2
        result: dict[str, list[dict]] = defaultdict(list)
        for wt, val in fields.get(2, []):
            if wt != 2 or not isinstance(val, bytes):
                continue
            sv = _parse_fields(val)

            # Extract trace_id from labels (field 3 of Sample = repeated Label)
            trace_id_hex: str | None = None
            for lwt, lval in sv.get(3, []):        # each Label is an embedded message
                if lwt != 2 or not isinstance(lval, bytes):
                    continue
                label_f = _parse_fields(lval)
                key_idx = label_f[1][0][1] if label_f.get(1) else 0
                if key_idx == trace_id_key_idx:
                    str_idx = label_f[2][0][1] if label_f.get(2) else 0
                    trace_id_hex = s(str_idx)
                    break

            if not trace_id_hex:
                continue

            # Collect ALL location_ids — pprof encodes the full call stack here,
            # innermost (self-time) frame first, callers after.
            loc_ids: list[int] = []
            for lwt, lval in sv.get(1, []):
                if lwt == 2 and isinstance(lval, bytes):
                    loc_ids.extend(_packed_varints(lval))
                elif lwt == 0:
                    loc_ids.append(lval)

            if not loc_ids:
                continue

            # Resolve each location to a frame dict
            stack: list[dict] = []
            for lid in loc_ids:
                loc_entry = locs.get(lid)
                if not loc_entry:
                    continue
                fid, call_line = loc_entry
                if fid not in fns:
                    continue
                name, file_, start_line = fns[fid]
                line = call_line or start_line
                if line > 0x7FFFFFFF:
                    line = start_line if start_line <= 0x7FFFFFFF else 0
                if not name or name == "(idle)":
                    continue
                stack.append({"function": name, "file": file_, "line": line})

            if not stack:
                continue

            # Store innermost frame + full stack for app-caller detection
            result[trace_id_hex].append({**stack[0], "stack": stack})

        return dict(result)

    except Exception:
        return {}
