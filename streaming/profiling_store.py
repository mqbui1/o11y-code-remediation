"""
Local AlwaysOn Profiling store.

Captures profiling OTLP log records fanned out from the OTel Collector and
decodes pprof frames in-process — no Splunk API required.

The Splunk /v2/apm/profiling REST API is not publicly accessible, so this
module provides Tier B/A profiling data directly from the agent's OTLP receiver.

Data flow:
  @splunk/otel service
    → OTLP HTTP (port 4318, encoding: json) → OTel Collector
    → otlphttp/agent fan-out → o11y-agent OTLP receiver
    → process_resource_logs() → profiling_store.observe()
    → tools/profiling_tools.py (get_cpu_flamegraph, get_profiling_services)
    → performance specialist (Tier B — file:line:function from profiling)
"""

import base64
import gzip
import logging
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 1800  # keep profiles from last 30 minutes
_MAXLEN = 180           # ring buffer depth per (service, env)
                        # @ 10s sampling interval = 30 min of history

_PROFILING_SOURCETYPE = "otel.profiling"


class ProfilingStore:
    """Thread-safe ring buffer for profiling records, keyed by (service, environment)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (service, env) → deque of {ts, data_type, frames: list[dict]}
        self._records: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAXLEN)
        )

    def observe(
        self,
        service: str,
        environment: str,
        data_type: str,
        body: str,
        data_format: str,
    ) -> None:
        frames: list[dict] = []
        if data_format == "pprof-gzip-base64" and body:
            frames = _decode_pprof(body)
        with self._lock:
            self._records[(service, environment)].append({
                "ts": time.time(),
                "data_type": data_type,
                "frames": frames,
            })

    def get_services(self, environment: str) -> list[str]:
        cutoff = time.time() - _WINDOW_SECONDS
        with self._lock:
            return sorted(
                svc
                for (svc, env), records in self._records.items()
                if env == environment and any(r["ts"] > cutoff for r in records)
            )

    def get_flamegraph(
        self, service: str, environment: str, since: float = 0, until: float = 0
    ) -> list[dict]:
        """Merge CPU frames from records within the given time window."""
        default_cutoff = time.time() - _WINDOW_SECONDS
        since_ts = max(since, default_cutoff) if since else default_cutoff
        until_ts = until if until else float("inf")
        with self._lock:
            records = list(self._records.get((service, environment), []))

        combined: dict[str, dict] = {}
        for rec in records:
            if rec["ts"] < since_ts or rec["ts"] > until_ts or rec["data_type"] != "cpu":
                continue
            for frame in rec["frames"]:
                key = f"{frame.get('file', '')}:{frame.get('function', '')}"
                if key in combined:
                    combined[key]["samples"] += frame["samples"]
                else:
                    combined[key] = dict(frame)

        if not combined:
            return []

        total = sum(f["samples"] for f in combined.values()) or 1
        result = sorted(combined.values(), key=lambda x: -x["samples"])
        for f in result:
            f["pct_cpu"] = round(f["samples"] / total * 100, 1)
        return result[:30]


    def get_memory_flamegraph(
        self, service: str, environment: str, since: float = 0, until: float = 0
    ) -> list[dict]:
        """Merge memory/heap allocation frames from records in the given time window."""
        default_cutoff = time.time() - _WINDOW_SECONDS
        since_ts = max(since, default_cutoff) if since else default_cutoff
        until_ts = until if until else float("inf")
        with self._lock:
            records = list(self._records.get((service, environment), []))

        combined: dict[str, dict] = {}
        for rec in records:
            if rec["ts"] < since_ts or rec["ts"] > until_ts:
                continue
            if rec["data_type"] not in ("heap", "allocation", "memory"):
                continue
            for frame in rec["frames"]:
                key = f"{frame.get('file', '')}:{frame.get('function', '')}"
                if key in combined:
                    combined[key]["samples"] += frame["samples"]
                else:
                    combined[key] = dict(frame)

        if not combined:
            return []

        total = sum(f["samples"] for f in combined.values()) or 1
        result = sorted(combined.values(), key=lambda x: -x["samples"])
        for f in result:
            f["pct_cpu"] = round(f["samples"] / total * 100, 1)
        return result[:30]

    def get_flamegraph_diff(
        self,
        service: str,
        environment: str,
        window_seconds: int = 300,
        baseline_offset: int = 900,
    ) -> list[dict]:
        """
        Compare the recent CPU flamegraph against a baseline window.

        current  = [now - window_seconds, now]
        baseline = [now - baseline_offset - window_seconds, now - baseline_offset]

        Returns functions with |delta| >= 2 pct_cpu, sorted by |delta| desc.
        """
        now = time.time()
        current  = self.get_flamegraph(service, environment,
                                       since=now - window_seconds, until=now)
        baseline = self.get_flamegraph(service, environment,
                                       since=now - baseline_offset - window_seconds,
                                       until=now - baseline_offset)
        if not current or not baseline:
            return []

        curr_map = {f"{f['file']}:{f['function']}": f["pct_cpu"] for f in current}
        base_map = {f"{f['file']}:{f['function']}": f["pct_cpu"] for f in baseline}

        # Build a lookup from either window for function metadata
        meta: dict[str, dict] = {}
        for f in baseline + current:
            meta[f"{f['file']}:{f['function']}"] = f

        diffs = []
        for key in set(curr_map) | set(base_map):
            before = base_map.get(key, 0.0)
            after  = curr_map.get(key, 0.0)
            delta  = round(after - before, 1)
            if abs(delta) < 2.0:
                continue
            fn = meta[key]
            diffs.append({
                "function":   fn.get("function", ""),
                "file":       fn.get("file", ""),
                "before_pct": round(before, 1),
                "after_pct":  round(after, 1),
                "delta_pct":  delta,
            })

        return sorted(diffs, key=lambda x: -abs(x["delta_pct"]))[:20]


# ── Module-level singleton ────────────────────────────────────────────────────

_store = ProfilingStore()


def observe(
    service: str,
    environment: str,
    data_type: str,
    body: str,
    data_format: str,
) -> None:
    _store.observe(service, environment, data_type, body, data_format)


def get_services(environment: str) -> list[str]:
    return _store.get_services(environment)


def get_flamegraph(service: str, environment: str, since: float = 0, until: float = 0) -> list[dict]:
    return _store.get_flamegraph(service, environment, since=since, until=until)


def get_memory_flamegraph(service: str, environment: str, since: float = 0, until: float = 0) -> list[dict]:
    return _store.get_memory_flamegraph(service, environment, since=since, until=until)


def get_flamegraph_diff(
    service: str, environment: str,
    window_seconds: int = 300,
    baseline_offset: int = 900,
) -> list[dict]:
    return _store.get_flamegraph_diff(service, environment,
                                      window_seconds=window_seconds,
                                      baseline_offset=baseline_offset)


# ── Minimal pprof decoder (pure stdlib — no protobuf dependency) ──────────────
#
# pprof wire format (subset needed for flamegraph):
#   Profile.sample_type  field 1
#   Profile.sample       field 2  → Sample.location_id field 1, Sample.value field 2
#   Profile.location     field 4  → Location.id field 1, Location.line field 4
#     Line.function_id   field 1
#   Profile.function     field 5  → Function.id 1, .name 2, .filename 4, .start_line 5
#   Profile.string_table field 6


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
    fields: dict[int, list] = defaultdict(list)
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


def _decode_pprof(b64: str) -> list[dict]:
    """Decode pprof-gzip-base64 into [{function, file, line, samples}] sorted desc."""
    try:
        raw = gzip.decompress(base64.b64decode(b64))
        fields = _parse_fields(raw)

        # string_table: field 6
        # pprof spec: string_table[0] is always "" and IS included in the proto wire bytes.
        # Do NOT pre-seed — let the first field-6 entry provide index 0 naturally.
        strings: list[str] = []
        for wt, val in fields.get(6, []):
            if wt == 2 and isinstance(val, bytes):
                strings.append(val.decode("utf-8", errors="replace"))

        def s(idx: int) -> str:
            return strings[idx] if 0 <= idx < len(strings) else ""

        # functions: field 5 → id → (name, file, start_line)
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

        # locations: field 4 → id → first function_id
        locs: dict[int, int] = {}
        for wt, val in fields.get(4, []):
            if wt != 2 or not isinstance(val, bytes):
                continue
            l = _parse_fields(val)
            lid = l[1][0][1] if l.get(1) else 0
            for lwt, lval in l.get(4, []):
                if lwt == 2 and isinstance(lval, bytes):
                    lf = _parse_fields(lval)
                    locs[lid] = lf[1][0][1] if lf.get(1) else 0
                    break

        # samples: field 2 → accumulate count per function
        counts: dict[int, int] = defaultdict(int)
        for wt, val in fields.get(2, []):
            if wt != 2 or not isinstance(val, bytes):
                continue
            sv = _parse_fields(val)
            count = 0
            for vwt, vval in sv.get(2, []):
                if vwt == 2 and isinstance(vval, bytes):
                    count += sum(_packed_varints(vval))
                elif vwt == 0:
                    count += vval
            # Splunk AlwaysOn pprof omits sample values — each sample = 1 CPU unit
            if count == 0:
                count = 1
            for lwt, lval in sv.get(1, []):
                if lwt == 2 and isinstance(lval, bytes):
                    for lid in _packed_varints(lval):
                        fid = locs.get(lid)
                        if fid:
                            counts[fid] += count
                elif lwt == 0:
                    fid = locs.get(lval)
                    if fid:
                        counts[fid] += count

        frames = []
        for fid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            name, file_, sline = fns.get(fid, ("", "", 0))
            # Skip empty names, the V8 idle marker, and zero-sample frames
            if not name or name == "(idle)" or cnt <= 0:
                continue
            frames.append({"function": name, "file": file_, "line": sline, "samples": cnt})
        return frames[:30]

    except Exception as _exc:
        logger.debug("pprof decode error: %s", _exc)
        return []
