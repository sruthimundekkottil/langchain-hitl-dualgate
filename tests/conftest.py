"""Test doubles: a scripted chat model and tools that count their own executions."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per invocation.

    The last message repeats once the script runs out, which keeps a test from hanging
    if the agent loops more than expected.
    """

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        # The script already decides which tools get called.
        return self


def call_then_finish(
    name: str,
    args: dict[str, Any],
    call_id: str = "call_1",
    final: str = "all done",
) -> list[AIMessage]:
    """Script: one tool call, then a plain answer."""
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        AIMessage(content=final),
    ]


class Recorder:
    """Counts tool executions so a test can assert exactly-once."""

    def __init__(self) -> None:
        self.executions: list[dict[str, Any]] = []

    def record(self, name: str, **args: Any) -> None:
        self.executions.append({"tool": name, **args})

    def count(self, name: str | None = None) -> int:
        if name is None:
            return len(self.executions)
        return sum(1 for e in self.executions if e["tool"] == name)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()
