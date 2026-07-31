"""
Minimal environment-variable-based configuration.

Deliberately not a full AgentConfig dataclass (as in the original
autonomous-o11y-agent) — this service only needs to know which LLM provider
to call for fix/narrative generation. Everything else (profiling data,
exceptions, source reading) is stateless and needs no config at all.
"""

import os

from providers.bedrock import BedrockProvider
from providers.openai_compat import OpenAICompatProvider
from providers.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    """
    Build an LLM provider from environment variables.

    LLM_PROVIDER=bedrock (default):
      BEDROCK_MODEL_ID, AWS_DEFAULT_REGION (default us-west-2)

    LLM_PROVIDER=ollama|openai:
      OPENAI_BASE_URL (falls back to OLLAMA_BASE_URL)
      OPENAI_API_KEY  (falls back to "ollama")
      OPENAI_MODEL    (falls back to OLLAMA_MODEL)
    """
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
    if provider in ("ollama", "openai"):
        return OpenAICompatProvider(
            base_url=os.environ.get("OPENAI_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "")),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            model=os.environ.get("OPENAI_MODEL", os.environ.get("OLLAMA_MODEL", "")),
        )
    return BedrockProvider(
        model_id=os.environ.get("BEDROCK_MODEL_ID", ""),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )
