"""
LLM-based code fix generator for profiling hotspots.

Takes a profiling finding (blocking frame + app caller + source context) and
generates a concrete unified diff using the configured LLM provider.

No tools are needed — a single converse() call with a structured prompt.
The LLM returns JSON with: issue_type, issue_summary, explanation, diff,
estimated_impact, confidence.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert performance engineer. Given profiling data showing a hot or blocking
code path, plus source context, you identify the root cause and generate a concrete fix
as a standard unified diff.

Respond with ONLY a valid JSON object — no markdown code fences, no preamble, no trailing text.
Required fields:
{
  "issue_type": "sync_io_in_hot_path|n_plus_one|cpu_hotspot|lock_contention|redundant_computation|other",
  "issue_summary": "one-line problem description (max 80 chars)",
  "why_issue": "2-3 sentences explaining WHY this is a performance problem: what the code is doing wrong, what the runtime cost is, and how it manifests under load (e.g. thread starvation, latency stacking, O(N) queries)",
  "what_fix_solves": "2-3 sentences explaining exactly what the fix eliminates or improves: the mechanism of the fix, what operation is removed or made cheaper, and the resulting behaviour change",
  "explanation": "1-2 sentences summarising the change for a changelog or PR description",
  "diff": "unified diff string — use a/ and b/ prefixes, proper @@ hunks",
  "estimated_impact": "concrete estimate e.g. 'Eliminates ~10ms per request'",
  "confidence": "high|medium|low"
}
If not enough context to write a diff, set diff to "" and explain in why_issue."""

_ISSUE_META = {
    "sync_io_in_hot_path":    ("Sync I/O in Hot Path",    "#f59e0b"),
    "n_plus_one":             ("N+1 Query",               "#ef4444"),
    "cpu_hotspot":            ("CPU Hotspot",              "#8b5cf6"),
    "lock_contention":        ("Lock Contention",          "#ec4899"),
    "redundant_computation":  ("Redundant Computation",    "#06b6d4"),
    "other":                  ("Performance Issue",        "#64748b"),
}

_EXC_SYSTEM = """\
You are an expert software reliability engineer. Given an exception stack trace from a
production service, identify the root cause and suggest a concrete fix.

Respond with ONLY a valid JSON object — no markdown code fences, no preamble, no trailing text.
Required fields:
{
  "issue_type": "configuration_error|missing_implementation|network_error|invalid_input|timeout|other",
  "issue_summary": "one-line problem description (max 80 chars)",
  "why_issue": "2-3 sentences explaining WHY this exception occurs: what the code is doing, what assumption is violated, and how it manifests in production",
  "what_fix_solves": "2-3 sentences explaining what the fix does: what configuration change, code fix, or deployment step eliminates the exception",
  "explanation": "1-2 sentences summarising the change for a changelog or PR description",
  "diff": "unified diff string if a code change applies, otherwise empty string",
  "estimated_impact": "concrete outcome e.g. 'Eliminates gRPC UNIMPLEMENTED errors on /Recommend endpoint'",
  "confidence": "high|medium|low"
}
If the fix is purely operational (config/deployment), set diff to "" and explain in what_fix_solves."""

_EXC_ISSUE_META = {
    "configuration_error":    ("Config Error",           "#f59e0b"),
    "missing_implementation": ("Missing Implementation", "#ef4444"),
    "network_error":          ("Network Error",          "#3b82f6"),
    "invalid_input":          ("Invalid Input",          "#8b5cf6"),
    "timeout":                ("Timeout",                "#ec4899"),
    "other":                  ("Exception",              "#64748b"),
}


def _hint_issue_type(blocking_file: str, blocking_fn: str) -> str:
    f  = (blocking_file or "").lower()
    fn = (blocking_fn  or "").lower()
    if ("grpc" in f or "http" in f or "socket" in f or "request" in fn) and \
       any(x in fn for x in ("block", "wait", "send", "recv", "connect")):
        return "sync_io_in_hot_path"
    if any(x in fn for x in ("lock", "acquire", "_wait_once", "condition", "semaphore")):
        return "lock_contention"
    if any(x in f for x in ("db", "database", "query", "cursor", "mysql", "postgres", "sqlite", "redis")):
        return "n_plus_one"
    return "other"


def build_prompt(
    service: str,
    blocking_fn: str,
    blocking_file: str,
    blocking_line: int,
    self_time_ms: float,
    app_fn: str,
    app_file: str,
    app_line: int,
    source_lines: list[dict],
) -> str:
    src = "\n".join(
        f"{'>>>' if l.get('hot') else '   '} {l['n']:4d}  {l['text']}"
        for l in source_lines
    )
    hint = _hint_issue_type(blocking_file, blocking_fn)
    short_app_file = "/".join(app_file.replace("\\", "/").split("/")[-3:])
    return f"""## Profiling Finding

Service:        {service}
Blocking frame: {blocking_fn}()  —  {blocking_file}:{blocking_line}
Self-time:      {self_time_ms}ms per sampled request
App caller:     {app_fn}()  —  {app_file}:{app_line}
Suspected type: {hint}

## Source Context  ({short_app_file})

```
{src}
```

The hot line is marked with >>>. Generate a fix targeting the app code at \
line {app_line} of {short_app_file}. The diff file paths should use \
{short_app_file} (no a/ b/ slash prefix is also acceptable).
"""


def _build_exception_prompt(service: str, exc_type: str, exc_message: str, stacktrace: str) -> str:
    return f"""## Exception Trace

Service:        {service}
Exception type: {exc_type}
Message:        {exc_message}

## Stack Trace

```
{stacktrace}
```

Analyze this exception and generate a concrete fix targeting the root cause, not just the symptoms.
"""


def generate_fix(provider, data: dict) -> dict:
    """
    Call the LLM and return a structured fix dict.

    For profiling hotspots: expects service, blocking_fn, blocking_file, blocking_line,
    self_time_ms, app_fn, app_file, app_line, source_lines.

    For exception-only analysis (no source): expects service, blocking_fn/app_fn,
    exc_message, exc_stacktrace. source_lines must be absent or empty.

    Returns dict with: issue_type, issue_summary, explanation, diff,
    estimated_impact, confidence, label, color.
    On error: returns {error: "..."}.
    """
    has_source = bool(data.get("source_lines"))

    if has_source:
        required = ("service", "blocking_fn", "blocking_file", "blocking_line",
                    "self_time_ms", "app_fn", "app_file", "app_line", "source_lines")
        missing = [k for k in required if k not in data]
        if missing:
            return {"error": f"Missing fields: {missing}"}
        prompt  = build_prompt(**{k: data[k] for k in required})
        system  = _SYSTEM
        meta    = _ISSUE_META
    else:
        prompt  = _build_exception_prompt(
            service     = data.get("service", "unknown"),
            exc_type    = data.get("blocking_fn") or data.get("app_fn") or "Exception",
            exc_message = data.get("exc_message", ""),
            stacktrace  = data.get("exc_stacktrace", ""),
        )
        system  = _EXC_SYSTEM
        meta    = {**_ISSUE_META, **_EXC_ISSUE_META}

    try:
        result = provider.converse(
            system_prompt=system,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            tools=[],
        )
        raw = result.get("text", "").strip()
    except Exception as exc:
        logger.error("LLM fix generation failed: %s", exc)
        return {"error": str(exc)}

    # Strip markdown fences if the LLM wrapped despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {"error": f"LLM response was not JSON. Preview: {raw[:400]}"}

    try:
        out = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}. Preview: {raw[:400]}"}

    issue_type = out.get("issue_type", "other")
    label, color = meta.get(issue_type, meta.get("other", ("Exception", "#64748b")))
    out["label"] = label
    out["color"] = color
    return out
