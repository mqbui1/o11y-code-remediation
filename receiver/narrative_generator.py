"""
AI narrative generator — service health analysis.

Synthesizes CPU profiling, memory profiling, exceptions, and regression
data into a plain-English health summary with severity-ranked findings.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert site reliability engineer analyzing a microservice in production.
Given profiling, exception, and regression data, produce a concise health analysis.

Respond with ONLY a valid JSON object — no markdown fences, no preamble.
{
  "severity": "critical|warning|healthy",
  "score": <integer 0-100, higher = healthier>,
  "summary": "2-3 sentence plain-English paragraph. Be specific: name functions, \
percentages, error counts. State the most important thing the team should act on.",
  "findings": [
    {
      "severity": "critical|warning|info",
      "title": "short title (max 60 chars)",
      "detail": "1-2 sentences with specific numbers and a concrete next step"
    }
  ]
}
2-4 findings. Prioritise by user impact. Skip findings with no data.
If CPU shows only stdlib/framework idle frames, score high and note the service is healthy."""


def _build_prompt(
    service: str,
    cpu_frames: list[dict],
    memory_frames: list[dict],
    exceptions: list[dict],
    diff: list[dict],
    snapshot_count: int,
) -> str:
    lines = [f"## Service: {service}\n"]

    # CPU
    if cpu_frames:
        lines.append("### CPU (AlwaysOn profiling — top 8 functions by self-time)")
        for f in cpu_frames[:8]:
            lines.append(f"  {f['pct_cpu']:5.1f}%  {f['function']}()  {f.get('file','')}")
    else:
        lines.append("### CPU: no profiling data in window")

    # Memory
    if memory_frames:
        lines.append("\n### Memory allocation (top 5)")
        for f in memory_frames[:5]:
            lines.append(f"  {f['pct_cpu']:5.1f}%  {f['function']}()  {f.get('file','')}")
    else:
        lines.append("\n### Memory: no allocation profiling data (requires SPLUNK_PROFILER_MEMORY_ENABLED=true)")

    # Exceptions
    if exceptions:
        counts: dict[str, int] = {}
        for e in exceptions:
            k = (e.get("exc_type") or "Unknown").split(".")[-1]
            counts[k] = counts.get(k, 0) + 1
        lines.append(f"\n### Exceptions — {len(exceptions)} unique traces in last 30 min")
        for exc_type, n in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {n}x  {exc_type}")
    else:
        lines.append("\n### Exceptions: none in last 30 min")

    # Regression diff
    regressions  = [d for d in diff if d["delta_pct"] > 0]
    improvements = [d for d in diff if d["delta_pct"] < 0]
    if regressions:
        lines.append("\n### CPU regressions vs ~15 min ago")
        for d in regressions[:4]:
            lines.append(
                f"  +{d['delta_pct']}%  {d['function']}()  "
                f"(was {d['before_pct']}%, now {d['after_pct']}%)"
            )
    if improvements:
        lines.append("\n### CPU improvements vs ~15 min ago")
        for d in improvements[:2]:
            lines.append(
                f"  {d['delta_pct']}%  {d['function']}()  "
                f"(was {d['before_pct']}%, now {d['after_pct']}%)"
            )
    if not diff:
        lines.append("\n### Regression diff: insufficient history (need ~15 min of data)")

    lines.append(f"\n### Snapshot traces in window: {snapshot_count}")
    return "\n".join(lines)


def generate_narrative(provider, data: dict) -> dict:
    """
    Call the LLM and return a structured health narrative.

    Expected keys in data: service, cpu_frames, memory_frames, exceptions,
    diff, snapshot_count.
    Returns: {severity, score, summary, findings}  or  {error: "..."}
    """
    prompt = _build_prompt(
        service        = data.get("service", "unknown"),
        cpu_frames     = data.get("cpu_frames", []),
        memory_frames  = data.get("memory_frames", []),
        exceptions     = data.get("exceptions", []),
        diff           = data.get("diff", []),
        snapshot_count = data.get("snapshot_count", 0),
    )

    try:
        result = provider.converse(
            system_prompt=_SYSTEM,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            tools=[],
        )
        raw = result.get("text", "").strip()
    except Exception as exc:
        logger.error("Narrative generation failed: %s", exc)
        return {"error": str(exc)}

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"error": f"LLM response was not JSON: {raw[:300]}"}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}"}
