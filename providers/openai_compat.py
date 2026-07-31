"""
OpenAI-compatible provider.

Works with any endpoint that implements the OpenAI Chat Completions API:
  - Galileo Luna (self-hosted)
  - Azure OpenAI
  - Google Vertex AI (via openai compatibility layer)
  - Ollama (local models)
  - Any OpenAI-API-compatible server
"""

import json
import logging
import re
import uuid

from .base import LLMProvider

logger = logging.getLogger(__name__)

_TOOL_CALL_START_RE = re.compile(r'\{\s*"name"\s*:')


def _find_bare_tool_call(text: str) -> dict | None:
    """Recover a {"name": ..., "arguments": {...}} object embedded in narrated
    text, whether or not it's wrapped in <tool_call> tags -- stopgap for models
    whose tool-call output isn't reliably tag-wrapped yet (Ollama's own parser
    only populates message.tool_calls from <tool_call>...</tool_call>, so an
    untagged-but-otherwise-correct call falls through to plain content text
    instead). Uses json.JSONDecoder.raw_decode rather than a regex because
    regex can't correctly balance nested braces in the arguments object (e.g.
    {"arguments": {"filters": {"key": "value"}}} would truncate at the wrong
    "}"). Returns the FIRST match only: observed duplicate/repeated tool-call
    JSON in raw output should collapse to one call, not fan out into several.
    """
    decoder = json.JSONDecoder()
    for m in _TOOL_CALL_START_RE.finditer(text):
        try:
            obj, _end = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
    return None

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAICompatProvider(LLMProvider):
    """
    LLM provider for any OpenAI-compatible Chat Completions endpoint.

    Handles schema conversion from Bedrock toolSpec format to OpenAI function format,
    and maps message/response shapes between the two APIs.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for LLM_PROVIDER=openai. "
                "Install with: pip install openai>=1.0.0"
            )
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def convert_tools(self, tools: list[dict]) -> list:
        """Convert Bedrock toolSpec format → OpenAI function format."""
        converted = []
        for tool in tools:
            spec = tool.get("toolSpec", {})
            schema = spec.get("inputSchema", {}).get("json", {})
            converted.append({
                "type": "function",
                "function": {
                    "name": spec.get("name", ""),
                    "description": spec.get("description", ""),
                    "parameters": schema,
                },
            })
        return converted

    def converse(self, system_prompt: str, messages: list[dict], tools: list[dict], force_tool: str = None) -> dict:
        openai_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            # Bedrock content blocks → OpenAI messages
            for block in content:
                if "text" in block:
                    openai_messages.append({"role": role, "content": block["text"]})
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    openai_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tu["toolUseId"],
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu.get("input", {})),
                            },
                        }],
                    })
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    text = " ".join(
                        c.get("text", "") for c in tr.get("content", []) if "text" in c
                    )
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["toolUseId"],
                        "content": text,
                    })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
        }
        if tools:
            # tools is already in OpenAI format (converted by agent_loop via convert_tools)
            kwargs["tools"] = tools
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}}
                if force_tool else "auto"
            )

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # Ollama often returns finish_reason="stop" even when tool_calls are present.
        # Check for tool_calls on the message directly regardless of finish_reason.
        raw_tool_calls = choice.message.tool_calls or []
        if finish_reason == "tool_calls" or raw_tool_calls:
            tool_uses = []
            for tc in raw_tool_calls:
                name = tc.function.name or ""
                try:
                    input_data = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    input_data = {}
                if name:  # skip malformed tool calls with empty names
                    tool_uses.append({"id": tc.id, "name": name, "input": input_data})
            if tool_uses:
                return {
                    "stop_reason": "tool_use",
                    "text": "",
                    "tool_uses": tool_uses,
                    "raw_message": choice.message,
                }

        text = choice.message.content or ""

        # Stopgap: recover a tool call the model clearly intended but didn't wrap
        # in <tool_call> tags, rather than silently treating it as prose (which
        # would make the specialist narrate fake findings instead of pulling
        # real data).
        parsed = _find_bare_tool_call(text)
        name = (parsed or {}).get("name") or ""
        if name:
            logger.warning(
                "Recovered untagged tool call %r from content via fallback parser "
                "(model did not wrap it in <tool_call> tags)",
                name,
            )
            return {
                "stop_reason": "tool_use",
                "text": "",
                "tool_uses": [{
                    "id": f"fallback-{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "input": parsed.get("arguments") or {},
                }],
                "raw_message": choice.message,
            }

        return {"stop_reason": "end_turn", "text": text, "tool_uses": [], "raw_message": choice.message}

    def format_tool_result(self, tool_use_id: str, content: str) -> dict:
        # OpenAI tool results use Bedrock's toolResult block shape so agent_loop
        # can handle both providers uniformly — the conversion happens in converse()
        return {"toolResult": {"toolUseId": tool_use_id, "content": [{"text": content}]}}
