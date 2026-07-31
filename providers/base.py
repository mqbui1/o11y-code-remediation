"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Callable


class LLMProvider(ABC):
    """
    Minimal interface for an LLM provider that supports tool-calling.

    Both BedrockProvider and OpenAICompatProvider implement this so that
    agent_loop.py is decoupled from AWS Bedrock's API format.
    """

    @abstractmethod
    def converse(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        force_tool: str = None,
    ) -> dict:
        """
        Send a conversation turn to the model.

        force_tool: if set, constrain the model to call this specific tool
        (grammar-enforced by the provider, not just a text hint). Used by
        agent_loop.py to guarantee submit_findings is called on the final
        turn instead of relying on the model to follow a text reminder.

        Returns a normalized response dict:
          {
            "stop_reason": "end_turn" | "tool_use",
            "text": str,                    # non-empty when stop_reason == "end_turn"
            "tool_uses": [                  # non-empty when stop_reason == "tool_use"
              {"id": str, "name": str, "input": dict}
            ],
            "raw_message": dict,            # provider-specific full response (for appending to history)
          }
        """

    @abstractmethod
    def format_tool_result(self, tool_use_id: str, content: str) -> dict:
        """
        Wrap a tool result in the provider's expected message format.
        This is what gets appended to the messages list after tool execution.
        """

    @abstractmethod
    def convert_tools(self, tools: list[dict]) -> list:
        """
        Convert tools from the internal Bedrock toolSpec format to the
        provider's native format.

        Internal format (Bedrock):
          {"toolSpec": {"name": str, "description": str, "inputSchema": {"json": {...}}}}

        OpenAI format:
          {"type": "function", "function": {"name": str, "description": str, "parameters": {...}}}
        """
