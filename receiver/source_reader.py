"""
Source code reader for Docker containers.

Reads source files from running containers via the Docker Engine API over the
Unix socket (/var/run/docker.sock). Zero external dependencies — uses stdlib
socket, http.client, and tarfile only.

The Docker Engine's GET /containers/{name}/archive endpoint returns a tar
stream for a given path, which tarfile can read directly from bytes.

Requires the Docker socket to be mounted into the agent container:
  /var/run/docker.sock:/var/run/docker.sock:ro  (in docker-compose.yml)
"""

import io
import json
import logging
import re
import socket
import tarfile
import urllib.parse
from http.client import HTTPConnection

logger = logging.getLogger(__name__)

_DOCKER_SOCKET = "/var/run/docker.sock"


class _UnixSocketHTTPConnection(HTTPConnection):
    """HTTPConnection that dials a Unix domain socket instead of TCP."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


def _read_container_file(container_name: str, file_path: str) -> str | None:
    """
    Fetch a single file from a running Docker container.

    Uses GET /containers/{name}/archive?path=... which returns a tar stream.
    Returns file contents as a string, or None on any error.
    """
    try:
        encoded = urllib.parse.quote(file_path, safe="")
        conn = _UnixSocketHTTPConnection(_DOCKER_SOCKET)
        conn.request("GET", f"/containers/{container_name}/archive?path={encoded}")
        resp = conn.getresponse()
        if resp.status != 200:
            logger.debug(
                "Docker archive API: container=%s file=%s status=%d",
                container_name, file_path, resp.status,
            )
            return None
        data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = tar.getmembers()
            if not members:
                return None
            f = tar.extractfile(members[0])
            return f.read().decode("utf-8", errors="replace") if f else None
    except FileNotFoundError:
        logger.debug("Docker socket not found at %s", _DOCKER_SOCKET)
        return None
    except Exception as exc:
        logger.debug("Error reading %s:%s — %s", container_name, file_path, exc)
        return None


# ── Source map resolution (pure stdlib, no external deps) ────────────────────
#
# Source Map v3 spec: mappings is a string of VLQ-encoded segments separated
# by ',' (within a line) and ';' (between generated lines).  Each segment has
# 1 or 4–5 values: [genCol, srcIdx, origLine, origCol, ?namesIdx].
# All values are deltas relative to the previous segment's state.

_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_TABLE = {c: i for i, c in enumerate(_B64)}


def _vlq_decode(s: str) -> list[int]:
    """Decode one VLQ group (all segments in a comma-less token)."""
    result, i = [], 0
    while i < len(s):
        v = shift = 0
        while True:
            if i >= len(s):
                break
            digit = _B64_TABLE.get(s[i], 0)
            i += 1
            v |= (digit & 0x1F) << shift
            shift += 5
            if not (digit & 0x20):
                break
        result.append(-(v >> 1) if (v & 1) else (v >> 1))
    return result


def _sourcemap_lookup(mappings: str, compiled_line: int) -> tuple[int, int, int]:
    """
    Walk the source map to find (source_index, original_line, original_col)
    for a given 1-based compiled line number.

    Returns (0, 0, 0) when the line cannot be resolved.
    """
    # Accumulated state across ALL segments (deltas are global, not per-line)
    src_idx = orig_line = orig_col = gen_col = 0
    best = (0, 0, 0)
    target = compiled_line - 1  # 0-indexed

    for line_num, line_str in enumerate(mappings.split(";")):
        gen_col = 0   # generated column resets at each ';'
        if not line_str:
            if line_num == target:
                return best
            continue
        for seg in line_str.split(","):
            if not seg:
                continue
            vals = _vlq_decode(seg)
            if len(vals) >= 1:
                gen_col += vals[0]
            if len(vals) >= 4:
                src_idx  += vals[1]
                orig_line += vals[2]
                orig_col  += vals[3]
                if line_num == target:
                    best = (src_idx, orig_line + 1, orig_col)  # 1-based line
        if line_num == target and best != (0, 0, 0):
            return best
        if line_num > target:
            break

    return best


def _clean_source_path(raw: str) -> str:
    """Strip webpack:// protocol and leading ./ from source map source paths."""
    raw = re.sub(r"^webpack://[^/]*/", "", raw)   # webpack://frontend/./foo → ./foo
    raw = re.sub(r"^\./", "", raw)                 # ./pages/api/cart.ts → pages/api/cart.ts
    return raw or raw


def _best_app_source(sources: list[str], sources_content: list[str | None]) -> int:
    """
    Return the index of the most likely app-code source file.
    Prefers .ts/.tsx files that are not in node_modules / Next.js internals.
    Falls back to the source with the most content.
    """
    _skip = ("node_modules", "next/dist", "webpack/runtime", "webpack-runtime",
             "@opentelemetry", "@grpc", "external commonjs")

    candidates = []
    for i, src in enumerate(sources):
        path = _clean_source_path(src)
        if not path:
            continue
        if any(s in src for s in _skip):
            continue
        score = 0
        if path.endswith(".ts") or path.endswith(".tsx"):
            score += 10
        if path.endswith(".js") or path.endswith(".jsx"):
            score += 5
        content = sources_content[i] if i < len(sources_content) else None
        score += min(len(content or ""), 5000) // 100  # longer = more likely primary
        candidates.append((score, i))

    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return 0


def _try_sourcemap_resolve(service: str, js_file: str, line: int, context: int) -> dict | None:
    """
    Try to resolve a compiled .js file to its original TypeScript source via
    an adjacent .js.map file.  Returns a read_source()-compatible dict on
    success, or None if no usable source map is found.
    """
    map_raw = _read_container_file(service, js_file + ".map")
    if not map_raw:
        return None

    try:
        sm = json.loads(map_raw)
    except Exception:
        return None

    sources: list[str] = sm.get("sources") or []
    sources_content: list[str | None] = sm.get("sourcesContent") or []
    mappings: str = sm.get("mappings", "")

    if not sources or not mappings:
        return None

    # Resolve compiled line → original file + line via VLQ mapping
    src_idx, orig_line = 0, 0
    if line > 0:
        src_idx, orig_line, _ = _sourcemap_lookup(mappings, line)
        src_idx = max(0, min(src_idx, len(sources) - 1))

    # If VLQ lookup landed on a non-app source (webpack runtime, node_modules, etc.),
    # fall back to heuristically selecting the primary app source file.
    resolved_path = _clean_source_path(sources[src_idx])
    _non_app = ("node_modules", "next/dist", "external commonjs", "webpack/runtime")
    if not resolved_path or any(s in sources[src_idx] for s in _non_app):
        src_idx = _best_app_source(sources, sources_content)
        orig_line = 0   # can't pinpoint a line without a valid mapping

    source_path = _clean_source_path(sources[src_idx])
    if not source_path:
        return None

    # Use embedded sourcesContent (present in production builds with source maps)
    content: str | None = None
    if src_idx < len(sources_content):
        content = sources_content[src_idx]

    if not content:
        # Fallback: try fetching the original file from the container
        guessed = f"/app/{source_path}" if not source_path.startswith("/") else source_path
        content = _read_container_file(service, guessed)

    if not content:
        return None

    all_lines = content.splitlines()
    total = len(all_lines)

    if orig_line > 0:
        start = max(0, orig_line - context - 1)
        end   = min(total, orig_line + context)
    else:
        start, end = 0, min(total, 50)

    lines = [
        {"n": i + 1, "text": all_lines[i], "hot": (i + 1 == orig_line)}
        for i in range(start, end)
    ]

    return {
        "service":        service,
        "file":           source_path,
        "compiled_file":  js_file,
        "compiled_line":  line,
        "target_line":    orig_line,
        "total_lines":    total,
        "lines":          lines,
        "source_mapped":  True,
    }


# Virtual/protocol paths that don't correspond to real files on disk.
_VIRTUAL_PATH_PREFIXES = ("node:", "webpack:", "electron:", "v8-compile-cache:", "<")


def _is_virtual_path(file_path: str) -> bool:
    return any(file_path.startswith(p) for p in _VIRTUAL_PATH_PREFIXES)


def read_source(
    service: str,
    file_path: str,
    line: int = 0,
    context: int = 25,
) -> dict:
    """
    Return source lines for a service + file + target line.

    Args:
        service:   Service name (used as Docker container name).
        file_path: Absolute path inside the container (e.g. /app/server.py).
        line:      Target (hot) line number, 1-based. 0 = return first 50 lines.
        context:   Number of lines before and after the target line to include.

    Returns a dict:
        service, file, target_line, total_lines, lines[]
        Each line entry: {n, text, hot}
        On error: adds an "error" key explaining what failed.
        On virtual path: adds "virtual": true with an informational message.
    """
    # Reject empty, placeholder, or virtual paths
    if not file_path or file_path in ("unknown", "?", "<unknown>"):
        return {
            "service": service,
            "file": file_path,
            "target_line": line,
            "lines": [],
            "virtual": True,
            "info": "No source file available — profiling frame has no file path.",
        }

    # Virtual paths (Node.js built-ins, webpack internals, etc.) have no source on disk.
    if _is_virtual_path(file_path):
        return {
            "service": service,
            "file": file_path,
            "target_line": line,
            "lines": [],
            "virtual": True,
            "info": f"'{file_path}' is a runtime built-in — no source file on disk.",
        }

    # For compiled JS files, try source map resolution first so the UI shows
    # original TypeScript instead of minified bundle output.
    if file_path.endswith(".js"):
        resolved = _try_sourcemap_resolve(service, file_path, line, context)
        if resolved is not None:
            return resolved

    raw = _read_container_file(service, file_path)

    if raw is None:
        return {
            "service": service,
            "file": file_path,
            "target_line": line,
            "lines": [],
            "error": (
                f"Could not read '{file_path}' from container '{service}'. "
                "Make sure /var/run/docker.sock is mounted read-only in the "
                "o11y-agent container (see deploy/docker-compose.yml)."
            ),
        }

    all_lines = raw.splitlines()
    total = len(all_lines)

    if line > 0:
        start = max(0, line - context - 1)
        end = min(total, line + context)
    else:
        start, end = 0, min(total, 50)

    lines = [
        {"n": i + 1, "text": all_lines[i], "hot": (i + 1 == line)}
        for i in range(start, end)
    ]

    return {
        "service": service,
        "file": file_path,
        "target_line": line,
        "total_lines": total,
        "lines": lines,
    }
